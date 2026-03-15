"""
AutoVoiceEvals — Closed-loop autoresearch runner.

Phase A: Attack agent with adversarial scenarios (Claude generates + judges)
Phase B: Improver Agent analyzes failures, rewrites system prompt
Phase C: Verify improved agent with same + new attacks, measure improvement
"""

import json
import os
import time
from datetime import datetime
from collections import Counter

from . import vapi, llm, graphs


def _run_round(label: str, phase: str, scenarios: list, cfg: dict,
               all_experiments: list, all_failures: set) -> list:
    """Run one round of eval, return (scenario, conversation, eval_result, score) tuples."""
    api_key = cfg["secrets"]["vapi_api_key"]
    assistant_id = cfg["assistant"]["id"]
    max_turns = cfg["run"]["max_turns_per_conversation"]
    results = []

    for i, sc in enumerate(scenarios):
        n = len(all_experiments) + 1
        sid = sc.get("id", f"{label}_{i+1:02d}")
        voice = sc.get("voice_characteristics", {})

        print(f"\n  [{n:02d}] {sid} | {sc.get('persona_name','')} | {sc.get('attack_strategy','')[:60]}")
        if voice:
            print(f"       voice: accent={voice.get('accent','none')}, "
                  f"noise={voice.get('background_noise','quiet')}, "
                  f"pace={voice.get('pace','normal')}")

        conv = vapi.run_conversation(
            api_key, assistant_id, sid,
            sc.get("caller_script", ["Hi, I need an appointment"]), max_turns)

        if conv.error:
            print(f"       ERROR: {conv.error}")
        else:
            print(f"       {len(conv.turns)} turns, {conv.avg_latency_ms:.0f}ms avg")

        try:
            ev = llm.evaluate(conv.transcript, sc)
        except Exception:
            ev = {"csat_score": 50, "passed": False, "summary": "Eval failed",
                  "agent_should_results": [], "agent_should_not_results": [],
                  "issues": [], "failure_modes": ["EVAL_ERROR"], "strengths": [], "weaknesses": []}

        sr = ev.get("agent_should_results", [])
        snr = ev.get("agent_should_not_results", [])
        s_score = sum(1 for c in sr if c.get("passed")) / max(len(sr), 1)
        sn_score = sum(1 for c in snr if c.get("passed")) / max(len(snr), 1)
        lat = 1.0 if conv.avg_latency_ms < 3000 else 0.5
        composite = 0.50 * s_score + 0.35 * sn_score + 0.15 * lat

        p = "PASS" if ev.get("passed") else "FAIL"
        bar = "█" * int(composite * 20) + "░" * (20 - int(composite * 20))
        print(f"       [{p}] {composite:.3f} [{bar}] CSAT={ev.get('csat_score',0)}")

        failures = ev.get("failure_modes", [])
        if failures:
            print(f"       failures: {', '.join(failures[:4])}")
        for fm in failures:
            all_failures.add(fm)

        exp = {
            "round": label, "phase": phase, "scenario_id": sid,
            "scenario": sc, "difficulty": sc.get("difficulty", "B"),
            "persona": sc.get("persona_name", ""), "attack_strategy": sc.get("attack_strategy", ""),
            "voice_characteristics": voice,
            "score": composite, "csat_score": ev.get("csat_score", 50),
            "passed": ev.get("passed", False),
            "should_score": s_score, "should_not_score": sn_score,
            "agent_should_results": sr, "agent_should_not_results": snr,
            "failure_modes": failures, "issues": ev.get("issues", []),
            "strengths": ev.get("strengths", []), "weaknesses": ev.get("weaknesses", []),
            "summary": ev.get("summary", ""),
            "num_turns": len(conv.turns), "avg_latency_ms": conv.avg_latency_ms,
            "transcript": conv.transcript,
        }
        all_experiments.append(exp)
        results.append((sc, conv, ev, composite))

    return results


def run(cfg: dict):
    """Main autoresearch loop."""
    print()
    print("=" * 70)
    print("  AutoVoiceEvals — Closed-Loop Autoresearch")
    print("  Phase A: Attack | Phase B: Improve | Phase C: Verify")
    print("=" * 70)

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    api_key = cfg["secrets"]["vapi_api_key"]
    assistant_id = cfg["assistant"]["id"]
    agent_desc = cfg["assistant"]["description"]
    n_attack = cfg["run"]["attack_rounds"]
    n_verify = cfg["run"]["verify_rounds"]
    n_scenarios = cfg["run"]["scenarios_per_round"]
    top_k = cfg["run"]["top_k_elites"]

    # Init LLM
    llm.init(cfg["secrets"]["anthropic_api_key"], cfg["llm"]["model"],
             timeout_seconds=cfg["llm"]["timeout_seconds"],
             max_retries=cfg["llm"]["max_retries"])

    all_experiments = []
    all_failures = set()
    all_issues = []
    elite_pool = []
    round_stats = []

    original_prompt = vapi.get_assistant(api_key, assistant_id)["model"]["messages"][0]["content"]
    print(f"\n  Assistant: {cfg['assistant']['name']}")
    print(f"  Original prompt: {len(original_prompt)} chars")
    total = (n_attack + n_verify) * n_scenarios
    print(f"  Plan: {n_attack} attack + {n_verify} verify rounds x {n_scenarios} = {total} experiments\n")

    # ─── PHASE A: ATTACK ─────────────────────────────────────────────────
    for rnd in range(1, n_attack + 1):
        label = f"A{rnd}"
        print(f"\n{'━' * 70}")
        print(f"  PHASE A — ATTACK {rnd}/{n_attack}")
        print(f"{'━' * 70}")

        prev_f = sorted(all_failures)
        prev_t = [e["transcript"] for e in sorted(all_experiments, key=lambda x: x["score"])[:2]]

        scenarios = []
        if rnd == 1 or not elite_pool:
            scenarios = llm.generate_scenarios(n_scenarios, rnd, agent_desc, prev_f, prev_t)
        else:
            for j in range(min(top_k * 2, n_scenarios - 1)):
                parent = elite_pool[j % len(elite_pool)]
                try:
                    m = llm.mutate_scenario(parent[0], parent[1], parent[2], f"{label}_M{j+1:02d}")
                    if m:
                        scenarios.append(m)
                except Exception:
                    pass
            need = n_scenarios - len(scenarios)
            if need > 0:
                scenarios.extend(llm.generate_scenarios(need, rnd, agent_desc, prev_f, prev_t))

        results = _run_round(label, "attack", scenarios, cfg, all_experiments, all_failures)
        for r in results:
            all_issues.extend(r[2].get("issues", []))

        scores = [r[3] for r in results]
        csats = [r[2].get("csat_score", 50) for r in results]
        s_vals = [e["should_score"] for e in all_experiments if e["round"] == label]
        sn_vals = [e["should_not_score"] for e in all_experiments if e["round"] == label]

        stat = {"round": label, "phase": "attack",
                "avg_score": sum(scores)/max(len(scores),1),
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "avg_csat": sum(csats)/max(len(csats),1),
                "avg_dimensions": {"agent_should": sum(s_vals)/max(len(s_vals),1),
                                   "agent_should_not": sum(sn_vals)/max(len(sn_vals),1)},
                "unique_failures_cumulative": len(all_failures)}
        round_stats.append(stat)

        elite_pool = [(r[0], r[1].transcript, r[2].get("failure_modes",[]), r[3])
                       for r in sorted(results, key=lambda r: r[3])[:top_k]]

        print(f"\n  {label}: score={stat['avg_score']:.3f} csat={stat['avg_csat']:.0f} failures={len(all_failures)}")

    # ─── PHASE B: IMPROVE ────────────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print(f"  PHASE B — IMPROVER AGENT ({len(all_issues)} issues, {len(all_failures)} failure modes)")
    print(f"{'━' * 70}")

    worst_t = [e["transcript"] for e in sorted(all_experiments, key=lambda x: x["score"])[:3]]
    improvement = llm.improve_prompt(original_prompt, all_issues, sorted(all_failures), worst_t)

    additions = improvement.get("prompt_additions", [])
    improved_prompt = improvement.get("improved_prompt", original_prompt)

    print(f"\n  Generated {len(additions)} prompt additions:")
    for pa in additions[:6]:
        print(f"    [{pa.get('severity','?').upper()}] {pa.get('description','')[:65]}")
    print(f"\n  Improved prompt: {len(improved_prompt)} chars (was {len(original_prompt)})")

    if vapi.update_prompt(api_key, assistant_id, improved_prompt):
        print(f"  Vapi assistant updated.")
    else:
        print(f"  WARNING: Vapi update failed!")

    # ─── PHASE C: VERIFY ─────────────────────────────────────────────────
    for rnd in range(1, n_verify + 1):
        label = f"C{rnd}"
        print(f"\n{'━' * 70}")
        print(f"  PHASE C — VERIFY {rnd}/{n_verify} (improved prompt)")
        print(f"{'━' * 70}")

        prev_f = sorted(all_failures)
        prev_t = [e["transcript"] for e in sorted(all_experiments, key=lambda x: x["score"])[:2]]

        scenarios = []
        if rnd == 1 and elite_pool:
            for j, (sc, _, _, _) in enumerate(elite_pool):
                c = dict(sc); c["id"] = f"{label}_RE{j+1:02d}"; scenarios.append(c)
            need = n_scenarios - len(scenarios)
            if need > 0:
                scenarios.extend(llm.generate_scenarios(need, rnd + n_attack, agent_desc, prev_f, prev_t))
        else:
            for j in range(min(top_k * 2, n_scenarios - 1)):
                parent = elite_pool[j % len(elite_pool)]
                try:
                    m = llm.mutate_scenario(parent[0], parent[1], parent[2], f"{label}_M{j+1:02d}")
                    if m: scenarios.append(m)
                except Exception:
                    pass
            need = n_scenarios - len(scenarios)
            if need > 0:
                scenarios.extend(llm.generate_scenarios(need, rnd + n_attack, agent_desc, prev_f, prev_t))

        results = _run_round(label, "verify", scenarios, cfg, all_experiments, all_failures)
        for r in results:
            all_issues.extend(r[2].get("issues", []))

        scores = [r[3] for r in results]
        csats = [r[2].get("csat_score", 50) for r in results]
        s_vals = [e["should_score"] for e in all_experiments if e["round"] == label]
        sn_vals = [e["should_not_score"] for e in all_experiments if e["round"] == label]

        stat = {"round": label, "phase": "verify",
                "avg_score": sum(scores)/max(len(scores),1),
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "avg_csat": sum(csats)/max(len(csats),1),
                "avg_dimensions": {"agent_should": sum(s_vals)/max(len(s_vals),1),
                                   "agent_should_not": sum(sn_vals)/max(len(sn_vals),1)},
                "unique_failures_cumulative": len(all_failures)}
        round_stats.append(stat)

        elite_pool = [(r[0], r[1].transcript, r[2].get("failure_modes",[]), r[3])
                       for r in sorted(results, key=lambda r: r[3])[:top_k]]

        print(f"\n  {label}: score={stat['avg_score']:.3f} csat={stat['avg_csat']:.0f} failures={len(all_failures)}")

    # ─── Restore original prompt ─────────────────────────────────────────
    print(f"\n  Restoring original prompt...")
    vapi.update_prompt(api_key, assistant_id, original_prompt)

    # ─── Save ────────────────────────────────────────────────────────────
    log = {
        "meta": {
            "version": "autovoiceevals-1.0",
            "timestamp": datetime.now().isoformat(),
            "assistant": cfg["assistant"]["name"],
            "llm": cfg["llm"]["model"],
            "total_experiments": len(all_experiments),
            "unique_failures": len(all_failures),
            "prompt_additions": len(additions),
            "original_prompt_chars": len(original_prompt),
            "improved_prompt_chars": len(improved_prompt),
        },
        "prompt_additions": additions,
        "improved_prompt": improved_prompt,
        "round_stats": round_stats,
        "experiments": all_experiments,
    }
    log_path = os.path.join(out_dir, "experiments.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    # ─── Graphs ──────────────────────────────────────────────────────────
    if cfg["output"]["generate_graphs"]:
        paths = graphs.generate_all(round_stats, all_experiments, out_dir)
        for p in paths:
            print(f"  graph: {p}")

    # ─── Report ──────────────────────────────────────────────────────────
    attack_s = [s for s in round_stats if s["phase"] == "attack"]
    verify_s = [s for s in round_stats if s["phase"] == "verify"]
    a_avg = sum(s["avg_score"] for s in attack_s) / max(len(attack_s), 1)
    v_avg = sum(s["avg_score"] for s in verify_s) / max(len(verify_s), 1)
    a_csat = sum(s["avg_csat"] for s in attack_s) / max(len(attack_s), 1)
    v_csat = sum(s["avg_csat"] for s in verify_s) / max(len(verify_s), 1)

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Experiments: {len(all_experiments)}  |  Unique failures: {len(all_failures)}")
    print(f"  Prompt additions: {len(additions)}")
    print(f"\n  BEFORE (attack):  score={a_avg:.3f}  CSAT={a_csat:.0f}")
    print(f"  AFTER  (verify):  score={v_avg:.3f}  CSAT={v_csat:.0f}")
    d = v_avg - a_avg
    print(f"  Change: {'▲' if d > 0 else '▼'} {abs(d):.3f} ({'+' if d > 0 else ''}{d/max(a_avg,0.01)*100:.1f}%)")
    print(f"\n  Output: {out_dir}/")
    print()
