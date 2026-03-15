"""
AutoVoiceEvals — True Autoresearch Loop

Karpathy's autoresearch pattern applied to voice AI prompt optimization:

  autoresearch           | autovoiceevals
  -----------------------|---------------------------
  train.py               | system prompt
  val_bpb (lower=better) | composite score (higher=better)
  5-min training run     | run N adversarial scenarios
  prepare.py (fixed)     | eval suite (fixed at startup)
  results.tsv            | results.tsv

The loop:
  1. AI proposes ONE change to the system prompt
  2. Apply it to the Vapi assistant
  3. Run the fixed eval suite
  4. If score improved → keep
  5. If worse → revert
  6. Log to results.tsv
  7. Never stop
"""

import json
import os
import time
from datetime import datetime

from . import vapi, llm


def _run_eval(api_key: str, assistant_id: str,
              eval_suite: list[dict], max_turns: int) -> list[dict]:
    """Run the fixed eval suite and return per-scenario results."""
    results = []
    for sc in eval_suite:
        conv = vapi.run_conversation(
            api_key, assistant_id, sc.get("id", "eval"),
            sc.get("caller_script", ["Hi, I need an appointment"]), max_turns)

        try:
            ev = llm.evaluate(conv.transcript, sc)
        except Exception:
            ev = {
                "csat_score": 50, "passed": False, "summary": "Eval failed",
                "agent_should_results": [], "agent_should_not_results": [],
                "issues": [], "failure_modes": ["EVAL_ERROR"],
                "strengths": [], "weaknesses": [],
            }

        sr = ev.get("agent_should_results", [])
        snr = ev.get("agent_should_not_results", [])
        s_score = sum(1 for c in sr if c.get("passed")) / max(len(sr), 1)
        sn_score = sum(1 for c in snr if c.get("passed")) / max(len(snr), 1)
        lat = 1.0 if conv.avg_latency_ms < 3000 else 0.5
        composite = 0.50 * s_score + 0.35 * sn_score + 0.15 * lat

        p = "PASS" if ev.get("passed") else "FAIL"
        bar = "\u2588" * int(composite * 20) + "\u2591" * (20 - int(composite * 20))
        print(f"    [{p}] {composite:.3f} [{bar}] "
              f"CSAT={ev.get('csat_score', 0)} {sc.get('persona_name', '')[:40]}")

        results.append({
            "scenario_id": sc.get("id"),
            "persona": sc.get("persona_name", ""),
            "score": composite,
            "csat_score": ev.get("csat_score", 50),
            "passed": ev.get("passed", False),
            "failure_modes": ev.get("failure_modes", []),
            "issues": ev.get("issues", []),
            "summary": ev.get("summary", ""),
            "strengths": ev.get("strengths", []),
            "weaknesses": ev.get("weaknesses", []),
            "transcript": conv.transcript,
            "num_turns": len(conv.turns),
            "avg_latency_ms": conv.avg_latency_ms,
        })

    return results


def _metrics(results: list[dict]) -> dict:
    """Aggregate metrics from eval results."""
    scores = [r["score"] for r in results]
    csats = [r["csat_score"] for r in results]
    n_passed = sum(1 for r in results if r["passed"])
    failures = set()
    for r in results:
        failures.update(r["failure_modes"])
    return {
        "avg_score": sum(scores) / max(len(scores), 1),
        "avg_csat": sum(csats) / max(len(csats), 1),
        "pass_rate": n_passed / max(len(results), 1),
        "n_passed": n_passed,
        "n_total": len(results),
        "unique_failures": sorted(failures),
    }


def _load_resume_state(out_dir: str) -> dict | None:
    """Load state from a previous autoresearch run for resuming."""
    log_path = os.path.join(out_dir, "autoresearch.json")
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        return json.load(f)


def run(cfg: dict, resume: bool = False):
    """The autoresearch loop. Runs indefinitely until Ctrl+C."""

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    api_key = cfg["secrets"]["vapi_api_key"]
    assistant_id = cfg["assistant"]["id"]
    agent_desc = cfg["assistant"]["description"]
    max_turns = cfg["run"]["max_turns_per_conversation"]

    ar = cfg.get("autoresearch", {})
    n_eval = ar.get("eval_scenarios", 8)
    threshold = ar.get("improvement_threshold", 0.005)
    max_experiments = ar.get("max_experiments", 0)  # 0 = forever

    # Init LLM
    llm.init(
        cfg["secrets"]["anthropic_api_key"], cfg["llm"]["model"],
        timeout_seconds=cfg["llm"]["timeout_seconds"],
        max_retries=cfg["llm"]["max_retries"],
    )

    # ─── Resume check ─────────────────────────────────────────────────
    prev_state = _load_resume_state(out_dir) if resume else None

    if prev_state:
        print()
        print("=" * 70)
        print("  AutoVoiceEvals — Autoresearch Mode (RESUMING)")
        print("=" * 70)

        eval_suite = prev_state["eval_suite"]
        original_prompt = prev_state["original_prompt"]

        # Rebuild history from previous experiments
        history = []
        all_failures = set()
        best_score = 0
        best_prompt = original_prompt
        last_eval = []

        for exp in prev_state["experiments"]:
            h = {
                "experiment": exp["experiment"],
                "score": exp["score"],
                "status": exp["status"],
                "description": exp["description"],
                "prompt_len": exp.get("prompt_len", 0),
            }
            history.append(h)
            if exp["status"] == "keep":
                best_score = exp["score"]
                if exp.get("prompt"):
                    best_prompt = exp["prompt"]
            for r in exp.get("results", []):
                all_failures.update(r.get("failure_modes", []))
            if exp.get("results"):
                last_eval = exp["results"]

        experiment = prev_state["experiments"][-1]["experiment"] if prev_state["experiments"] else 0

        # Apply best prompt to Vapi (in case it was reset)
        vapi.update_prompt(api_key, assistant_id, best_prompt)

        print(f"\n  Resumed from experiment {experiment}")
        print(f"  Best score: {best_score:.3f}")
        print(f"  Best prompt: {len(best_prompt)} chars")
        print(f"  Eval suite: {len(eval_suite)} scenarios")
        print(f"  Failures found: {len(all_failures)}")
        if max_experiments:
            remaining = max_experiments - experiment
            print(f"  Remaining experiments: {remaining}")

        # Rebuild TSV
        results_path = os.path.join(out_dir, "results.tsv")
        # keep existing TSV as-is

        full_log = prev_state

    else:
        # ─── Fresh start ──────────────────────────────────────────────
        print()
        print("=" * 70)
        print("  AutoVoiceEvals — Autoresearch Mode")
        print("  Propose → Eval → Keep/Revert → Repeat Forever")
        print("=" * 70)

        # Read current prompt — this is the artifact we're optimizing
        assistant = vapi.get_assistant(api_key, assistant_id)
        original_prompt = assistant["model"]["messages"][0]["content"]
        best_prompt = original_prompt

        print(f"\n  Assistant: {cfg['assistant']['name']}")
        print(f"  Prompt: {len(original_prompt)} chars")
        print(f"  Eval suite: {n_eval} adversarial scenarios")
        print(f"  Threshold: {threshold}")
        if max_experiments:
            print(f"  Max experiments: {max_experiments}")
        else:
            print(f"  Max experiments: unlimited (Ctrl+C to stop)")

        # Generate eval suite — fixed for the entire run (like prepare.py)
        print(f"\n  Generating eval suite...")
        eval_suite = llm.generate_scenarios(n_eval, 1, agent_desc, [], [])
        print(f"  {len(eval_suite)} scenarios generated:")
        for sc in eval_suite:
            d = sc.get("difficulty", "?")
            print(f"    [{d}] {sc.get('persona_name', '?')} — {sc.get('attack_strategy', '')[:55]}")

        # Tracking
        history = []
        all_failures = set()
        results_path = os.path.join(out_dir, "results.tsv")

        with open(results_path, "w") as f:
            f.write("experiment\tscore\tcsat\tpass_rate\tprompt_len\tstatus\tdescription\n")

        full_log = {
            "meta": {
                "version": "autoresearch-1.0",
                "assistant": cfg["assistant"]["name"],
                "llm": cfg["llm"]["model"],
                "eval_scenarios": n_eval,
                "threshold": threshold,
                "started": datetime.now().isoformat(),
            },
            "eval_suite": eval_suite,
            "original_prompt": original_prompt,
            "experiments": [],
        }

        # ─── Baseline ────────────────────────────────────────────────
        print(f"\n{'━' * 70}")
        print(f"  EXPERIMENT 0: BASELINE")
        print(f"{'━' * 70}\n")

        baseline_results = _run_eval(api_key, assistant_id, eval_suite, max_turns)
        baseline = _metrics(baseline_results)
        best_score = baseline["avg_score"]

        for r in baseline_results:
            all_failures.update(r["failure_modes"])

        print(f"\n  Baseline: score={best_score:.3f}  csat={baseline['avg_csat']:.0f}  "
              f"pass={baseline['n_passed']}/{baseline['n_total']}  "
              f"failures={baseline['unique_failures']}")

        # Log baseline
        with open(results_path, "a") as f:
            f.write(f"0\t{best_score:.6f}\t{baseline['avg_csat']:.1f}\t"
                    f"{baseline['pass_rate']:.3f}\t{len(best_prompt)}\tkeep\tbaseline\n")

        history.append({
            "experiment": 0, "score": best_score, "status": "keep",
            "description": "baseline", "prompt_len": len(best_prompt),
        })

        full_log["experiments"].append({
            "experiment": 0, "timestamp": datetime.now().isoformat(),
            "description": "baseline", "score": best_score,
            "csat": baseline["avg_csat"], "pass_rate": baseline["pass_rate"],
            "status": "keep", "results": baseline_results,
        })

        last_eval = baseline_results
        experiment = 0

    # ─── The loop (runs forever, or until max_experiments) ───────────
    if max_experiments:
        print(f"\n  Starting autoresearch loop ({max_experiments} experiments).\n")
    else:
        print(f"\n  Starting autoresearch loop. Ctrl+C to stop.\n")

    try:
        while True:
            if max_experiments and experiment >= max_experiments:
                print(f"  Reached {max_experiments} experiments. Stopping.\n")
                break
            experiment += 1
            t0 = time.time()

            print(f"{'━' * 70}")
            print(f"  EXPERIMENT {experiment}")
            print(f"{'━' * 70}")

            # 1. AI proposes a change
            proposal = llm.propose_prompt_change(
                best_prompt, last_eval, history, sorted(all_failures))

            description = proposal.get("description", "unknown")
            change_type = proposal.get("change_type", "?")
            reasoning = proposal.get("reasoning", "")
            new_prompt = proposal.get("improved_prompt", best_prompt)

            print(f"  [{change_type}] {description[:70]}")
            if reasoning:
                print(f"  Reasoning: {reasoning[:80]}")
            print(f"  Prompt: {len(best_prompt)} → {len(new_prompt)} chars")

            # Skip if no actual change
            if new_prompt.strip() == best_prompt.strip():
                print(f"  SKIP — no actual change\n")
                with open(results_path, "a") as f:
                    f.write(f"{experiment}\t{best_score:.6f}\t0.0\t0.000\t"
                            f"{len(new_prompt)}\tskip\t{description[:80]}\n")
                history.append({
                    "experiment": experiment, "score": best_score,
                    "status": "skip", "description": description,
                    "prompt_len": len(new_prompt),
                })
                continue

            # 2. Apply proposed prompt to Vapi
            if not vapi.update_prompt(api_key, assistant_id, new_prompt):
                print(f"  ERROR — Vapi update failed, skipping\n")
                continue

            # 3. Run eval suite
            print()
            eval_results = _run_eval(api_key, assistant_id, eval_suite, max_turns)
            m = _metrics(eval_results)
            new_score = m["avg_score"]

            for r in eval_results:
                all_failures.update(r["failure_modes"])

            # 4. Keep or revert
            delta = new_score - best_score

            if delta > threshold:
                # Score improved — keep
                status = "keep"
                best_prompt = new_prompt
                best_score = new_score
                last_eval = eval_results
            elif abs(delta) <= threshold and len(new_prompt) < len(best_prompt) - 20:
                # Score unchanged but prompt is simpler — keep (simplicity criterion)
                status = "keep"
                description += " (simpler)"
                best_prompt = new_prompt
                best_score = new_score
                last_eval = eval_results
            else:
                # Score worse or unchanged — revert
                status = "discard"
                vapi.update_prompt(api_key, assistant_id, best_prompt)

            dt = time.time() - t0
            arrow = "\u25b2" if delta > 0 else "\u25bc" if delta < 0 else "="

            print(f"\n  Result: score={new_score:.3f} ({arrow} {abs(delta):.3f})  "
                  f"csat={m['avg_csat']:.0f}  pass={m['n_passed']}/{m['n_total']}")
            print(f"  → {status.upper()}  (best={best_score:.3f}, "
                  f"prompt={len(best_prompt)} chars, {dt:.0f}s)\n")

            # 5. Log
            with open(results_path, "a") as f:
                f.write(f"{experiment}\t{new_score:.6f}\t{m['avg_csat']:.1f}\t"
                        f"{m['pass_rate']:.3f}\t{len(new_prompt)}\t{status}\t"
                        f"{description[:80]}\n")

            history.append({
                "experiment": experiment, "score": new_score, "status": status,
                "description": description, "prompt_len": len(new_prompt),
            })

            full_log["experiments"].append({
                "experiment": experiment,
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "change_type": change_type,
                "reasoning": reasoning,
                "prompt_len": len(new_prompt),
                "score": new_score,
                "delta": delta,
                "csat": m["avg_csat"],
                "pass_rate": m["pass_rate"],
                "status": status,
                "duration_s": dt,
                "results": eval_results,
                "prompt": new_prompt if status == "keep" else None,
            })

            # Save log after every experiment (crash-safe)
            _save_log(full_log, out_dir)

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 70}")
        print(f"  STOPPED (Ctrl+C)")
        print(f"{'=' * 70}")

    # ─── Final report ─────────────────────────────────────────────────
    kept = [h for h in history if h["status"] == "keep"]
    discarded = [h for h in history if h["status"] == "discard"]
    skipped = [h for h in history if h["status"] == "skip"]

    print(f"\n  Experiments:  {experiment}")
    print(f"  Kept:         {len(kept)}")
    print(f"  Discarded:    {len(discarded)}")
    print(f"  Skipped:      {len(skipped)}")
    print(f"  Baseline:     {history[0]['score']:.3f}")
    print(f"  Best score:   {best_score:.3f}")
    delta_total = best_score - history[0]["score"]
    print(f"  Improvement:  {'+' if delta_total >= 0 else ''}{delta_total:.3f}")
    print(f"  Prompt:       {len(original_prompt)} → {len(best_prompt)} chars")
    print(f"  Failures:     {len(all_failures)} unique modes")

    # Restore original prompt
    print(f"\n  Restoring original prompt on Vapi...")
    vapi.update_prompt(api_key, assistant_id, original_prompt)

    # Save best prompt
    best_path = os.path.join(out_dir, "best_prompt.txt")
    with open(best_path, "w") as f:
        f.write(best_prompt)
    print(f"  Best prompt saved: {best_path}")

    # Save final log
    full_log["meta"]["ended"] = datetime.now().isoformat()
    full_log["meta"]["total_experiments"] = experiment
    full_log["meta"]["best_score"] = best_score
    full_log["meta"]["best_prompt_chars"] = len(best_prompt)
    full_log["meta"]["original_prompt_chars"] = len(original_prompt)
    full_log["best_prompt"] = best_prompt
    _save_log(full_log, out_dir)

    print(f"  Results: {results_path}")
    print(f"  Full log: {os.path.join(out_dir, 'autoresearch.json')}")
    print()


def _save_log(log: dict, out_dir: str):
    path = os.path.join(out_dir, "autoresearch.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=str)
