"""Visualization for pipeline mode results.

Generates PNG charts from round_stats and experiment data.
Only used by pipeline mode (researcher mode logs to TSV/JSON).
"""

from __future__ import annotations

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker as ticker       # noqa: E402


def generate_all(
    round_stats: list[dict],
    experiments: list[dict],
    output_dir: str,
) -> list[str]:
    """Generate all charts and return list of saved file paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    paths.append(_score_comparison(round_stats, output_dir))
    paths.append(_cumulative_failures(round_stats, output_dir))
    paths.append(_experiment_scatter(experiments, output_dir))

    issues_path = _issue_breakdown(experiments, output_dir)
    if issues_path:
        paths.append(issues_path)

    return paths


# -------------------------------------------------------------------
# Chart 1: Score + CSAT comparison (attack vs verify)
# -------------------------------------------------------------------

def _score_comparison(round_stats: list[dict], output_dir: str) -> str:
    labels = [s["round"] for s in round_stats]
    scores = [s["avg_score"] for s in round_stats]
    csats = [s["avg_csat"] for s in round_stats]
    phases = [s["phase"] for s in round_stats]
    colors = ["#e74c3c" if p == "attack" else "#2ecc71" for p in phases]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.bar(labels, scores, color=colors, width=0.6, edgecolor="white", lw=2)
    for i, (_, s) in enumerate(zip(labels, scores)):
        ax1.text(i, s + 0.02, f"{s:.3f}", ha="center", fontsize=12, fontweight="bold")
    n_attack = sum(1 for p in phases if p == "attack")
    ax1.axvline(x=n_attack - 0.5, color="#2c3e50", ls="--", lw=2, label="Prompt improved")
    ax1.set_ylabel("Avg Agent Score", fontsize=13)
    ax1.set_title(
        "AutoVoiceEvals: Attack -> Improve -> Verify\n"
        "(Red = before, Green = after prompt improvement)",
        fontsize=14, fontweight="bold",
    )
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(labels, csats, color=colors, width=0.6, edgecolor="white", lw=2)
    for i, (_, c) in enumerate(zip(labels, csats)):
        ax2.text(i, c + 1.5, f"{c:.0f}", ha="center", fontsize=12, fontweight="bold")
    ax2.axvline(x=n_attack - 0.5, color="#2c3e50", ls="--", lw=2)
    ax2.set_ylabel("Avg CSAT (0-100)", fontsize=13)
    ax2.set_xlabel("Round", fontsize=13)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(output_dir, "01_comparison.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


# -------------------------------------------------------------------
# Chart 2: Cumulative failure discovery
# -------------------------------------------------------------------

def _cumulative_failures(round_stats: list[dict], output_dir: str) -> str:
    labels = [s["round"] for s in round_stats]
    cum = [s.get("unique_failures_cumulative", 0) for s in round_stats]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(cum)), cum, "o-", color="#8e44ad", lw=2.5, ms=10)
    ax.fill_between(range(len(cum)), cum, alpha=0.1, color="#8e44ad")
    for i, c in enumerate(cum):
        ax.text(i, c + 0.5, str(c), ha="center", fontsize=12, fontweight="bold", color="#8e44ad")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Cumulative Unique Failures", fontsize=13)
    ax.set_title("Failure Discovery Rate", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(output_dir, "02_cumulative_failures.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


# -------------------------------------------------------------------
# Chart 3: Per-experiment scatter
# -------------------------------------------------------------------

def _experiment_scatter(experiments: list[dict], output_dir: str) -> str:
    tier_colors = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e74c3c"}
    seen: set[str] = set()

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, exp in enumerate(experiments):
        d = exp.get("difficulty", "B")
        label = f"Tier {d}" if d not in seen else None
        seen.add(d)
        ax.scatter(
            i + 1, exp["score"],
            color=tier_colors.get(d, "#999"),
            s=120, edgecolors="white", lw=1.5, label=label, zorder=5,
        )
        if not exp.get("passed", True):
            ax.scatter(
                i + 1, exp["score"],
                color="none", s=200, edgecolors="#e74c3c", lw=2.5, zorder=6,
            )

    n_attack_exp = sum(1 for e in experiments if e.get("phase") == "attack")
    if n_attack_exp > 0:
        ax.axvline(x=n_attack_exp + 0.5, color="#2c3e50", ls="--", lw=2, alpha=0.7)

    ax.set_xlabel("Experiment #", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Every Experiment (Red rings = FAIL)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=5)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(output_dir, "03_scatter.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


# -------------------------------------------------------------------
# Chart 4: Issue breakdown (type + severity)
# -------------------------------------------------------------------

def _issue_breakdown(experiments: list[dict], output_dir: str) -> str | None:
    issue_counts: Counter = Counter()
    sev_counts: Counter = Counter()

    for exp in experiments:
        for iss in exp.get("issues", []):
            issue_counts[iss.get("type", "?")] += 1
            sev_counts[iss.get("severity", "?")] += 1

    if not issue_counts:
        return None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))

    cols = plt.cm.Set2(range(len(issue_counts)))
    a1.pie(
        issue_counts.values(),
        labels=[k[:25] for k in issue_counts.keys()],
        autopct="%1.0f%%", colors=cols, textprops={"fontsize": 9},
    )
    a1.set_title("Issues by Type", fontsize=13, fontweight="bold")

    sev_colors = {
        "critical": "#8e44ad", "high": "#e74c3c",
        "medium": "#f39c12", "low": "#2ecc71",
    }
    sev_labels = [s for s in ["critical", "high", "medium", "low"] if s in sev_counts]
    a2.pie(
        [sev_counts[s] for s in sev_labels],
        labels=[s.upper() for s in sev_labels],
        autopct="%1.0f%%",
        colors=[sev_colors.get(s, "#999") for s in sev_labels],
        textprops={"fontsize": 10},
    )
    a2.set_title("Issues by Severity", fontsize=13, fontweight="bold")

    plt.suptitle("Issue Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, "04_issues.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p
