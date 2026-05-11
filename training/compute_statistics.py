"""
Compute the statistical tests defined in §3.Y of the methodology chapter
against an eval_report*.json produced by training.dpo_eval.

Outputs:
    - Console tables ready for Chapter 4
    - Optional Markdown export (--out report.md)

Tests implemented:
    - Wilson 95% CI for binomial proportions (win-rate, implicit accuracy)
    - Exact binomial test against H0: p = 0.5
    - Wilcoxon signed-rank for paired continuous deltas (quality, semantic)
    - Cohen's d on paired differences
    - McNemar's exact test for paired binary outcomes
    - One-sample t-test for per-token margin against mu = 0
    - Bootstrap 95% CI for decoupling spread (1000 resamples)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Console may default to cp1252 on Windows; the report contains Greek deltas etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy import stats

Z_95 = 1.959963984540054  # exact two-sided 95% normal quantile
ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


def wilson_ci(k: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binomial_p_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    return float(stats.binomtest(k, n, p=p0, alternative="two-sided").pvalue)


def cohens_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d for paired differences (mean / std). Returns 0 if std is 0."""
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    return float(np.mean(diffs) / sd) if sd > 0 else 0.0


def mcnemar_exact(b: int, c: int) -> Tuple[float, str]:
    """Exact McNemar test on discordant pairs (b = adapter wins & base loses,
    c = adapter loses & base wins). Equivalent to a two-sided binomial test
    of b out of b+c against p=0.5."""
    n_disc = b + c
    if n_disc == 0:
        return 1.0, "no discordant pairs"
    p = float(stats.binomtest(b, n_disc, p=0.5, alternative="two-sided").pvalue)
    note = "exact" if n_disc < 25 else "exact (asymptotic chi^2 also valid)"
    return p, note


def bootstrap_spread_ci(
    quality_by_cond: Dict[str, np.ndarray],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Paired bootstrap CI for max-min spread of the per-condition adapter mean
    quality. Resamples prompt indices (with replacement) jointly across all
    conditions to preserve the within-prompt correlation structure."""
    rng = np.random.default_rng(seed)
    conds = list(quality_by_cond.keys())
    n = len(next(iter(quality_by_cond.values())))
    if any(len(v) != n for v in quality_by_cond.values()):
        raise ValueError("Per-condition arrays must share the same length.")
    spreads = np.empty(n_resamples, dtype=float)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means = np.array([quality_by_cond[c][idx].mean() for c in conds])
        spreads[r] = means.max() - means.min()
    lo = float(np.quantile(spreads, ALPHA / 2))
    hi = float(np.quantile(spreads, 1 - ALPHA / 2))
    return lo, hi


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def fmt_ci(lo: float, hi: float, pct: bool = True) -> str:
    if pct:
        return f"[{100*lo:.1f}%, {100*hi:.1f}%]"
    return f"[{lo:.4f}, {hi:.4f}]"


# ── Per-condition analysis ─────────────────────────────────────────────


def analyse_winrate_condition(per_pair: List[dict]) -> Dict:
    n = len(per_pair)
    aq = np.array([row["adapter_quality"] for row in per_pair])
    bq = np.array([row["base_quality"] for row in per_pair])
    asem = np.array([row["adapter_semantic"] for row in per_pair])
    bsem = np.array([row["base_semantic"] for row in per_pair])
    winners = [row["winner"] for row in per_pair]

    wins = sum(1 for w in winners if w == "adapter")
    losses = sum(1 for w in winners if w == "base")
    ties = sum(1 for w in winners if w == "tie")

    wr_lo, wr_hi = wilson_ci(wins, n)
    p_binom = binomial_p_two_sided(wins, n, 0.5)

    # Quality delta — paired Wilcoxon
    q_delta = aq - bq
    if np.allclose(q_delta, 0):
        w_q_stat, w_q_p = float("nan"), 1.0
    else:
        w_q = stats.wilcoxon(aq, bq, zero_method="wilcox", alternative="two-sided")
        w_q_stat, w_q_p = float(w_q.statistic), float(w_q.pvalue)
    d_q = cohens_d_paired(q_delta)

    # Semantic delta — paired Wilcoxon
    s_delta = asem - bsem
    if np.allclose(s_delta, 0):
        w_s_stat, w_s_p = float("nan"), 1.0
    else:
        w_s = stats.wilcoxon(asem, bsem, zero_method="wilcox", alternative="two-sided")
        w_s_stat, w_s_p = float(w_s.statistic), float(w_s.pvalue)
    d_s = cohens_d_paired(s_delta)

    # McNemar on discordant pairs (ties excluded by definition of discordance)
    p_mcn, mcn_note = mcnemar_exact(wins, losses)

    return {
        "n": n,
        "wins": wins, "losses": losses, "ties": ties,
        "win_rate": wins / n,
        "win_rate_ci": (wr_lo, wr_hi),
        "win_rate_p_vs_chance": p_binom,
        "quality_mean_adapter": float(aq.mean()),
        "quality_mean_base": float(bq.mean()),
        "quality_mean_delta": float(q_delta.mean()),
        "quality_se_delta": float(q_delta.std(ddof=1) / math.sqrt(n)),
        "quality_wilcoxon_stat": w_q_stat,
        "quality_wilcoxon_p": w_q_p,
        "quality_cohens_d": d_q,
        "semantic_mean_adapter": float(asem.mean()),
        "semantic_mean_base": float(bsem.mean()),
        "semantic_mean_delta": float(s_delta.mean()),
        "semantic_wilcoxon_p": w_s_p,
        "semantic_cohens_d": d_s,
        "mcnemar_p": p_mcn,
        "mcnemar_note": mcn_note,
        "adapter_quality_array": aq,  # consumed by bootstrap
    }


def analyse_implicit_accuracy_condition(per_pair: List[dict]) -> Dict:
    n = len(per_pair)
    correct = sum(1 for row in per_pair if row["correct"])
    margins = np.array([row["margin_per_token"] for row in per_pair])

    acc_lo, acc_hi = wilson_ci(correct, n)
    p_binom = binomial_p_two_sided(correct, n, 0.5)

    t_stat, t_p = stats.ttest_1samp(margins, popmean=0.0)

    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "accuracy_ci": (acc_lo, acc_hi),
        "accuracy_p_vs_chance": p_binom,
        "mean_margin": float(margins.mean()),
        "margin_se": float(margins.std(ddof=1) / math.sqrt(n)),
        "margin_t": float(t_stat),
        "margin_p_vs_zero": float(t_p),
        "margin_cohens_d": cohens_d_paired(margins),
    }


# ── Reporting ──────────────────────────────────────────────────────────


def build_winrate_table(results: Dict[str, Dict]) -> str:
    header = (
        f"| Condition | Win-rate (95% CI) | p (vs chance) | Δ quality (mean ± SE) | "
        f"Wilcoxon p | Cohen's d | McNemar p |\n"
        f"|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for cond, r in results.items():
        rows.append(
            f"| {cond} | {fmt_pct(r['win_rate'])} {fmt_ci(*r['win_rate_ci'])} "
            f"| {fmt_p(r['win_rate_p_vs_chance'])} "
            f"| {r['quality_mean_delta']:+.4f} ± {r['quality_se_delta']:.4f} "
            f"| {fmt_p(r['quality_wilcoxon_p'])} "
            f"| {r['quality_cohens_d']:+.3f} "
            f"| {fmt_p(r['mcnemar_p'])} |"
        )
    return header + "\n".join(rows)


def build_implicit_accuracy_table(results: Dict[str, Dict]) -> str:
    header = (
        f"| Condition | Accuracy (95% CI) | p (vs chance) | "
        f"Mean margin (nats, ± SE) | t-test p (vs 0) | Cohen's d |\n"
        f"|---|---|---|---|---|---|\n"
    )
    rows = []
    for cond, r in results.items():
        rows.append(
            f"| {cond} | {fmt_pct(r['accuracy'])} {fmt_ci(*r['accuracy_ci'])} "
            f"| {fmt_p(r['accuracy_p_vs_chance'])} "
            f"| {r['mean_margin']:+.4f} ± {r['margin_se']:.4f} "
            f"| {fmt_p(r['margin_p_vs_zero'])} "
            f"| {r['margin_cohens_d']:+.3f} |"
        )
    return header + "\n".join(rows)


def build_semantic_table(results: Dict[str, Dict]) -> str:
    header = (
        f"| Condition | Adapter mean | Base mean | Δ (paired) | Wilcoxon p | Cohen's d |\n"
        f"|---|---|---|---|---|---|\n"
    )
    rows = []
    for cond, r in results.items():
        rows.append(
            f"| {cond} | {r['semantic_mean_adapter']:.4f} "
            f"| {r['semantic_mean_base']:.4f} "
            f"| {r['semantic_mean_delta']:+.4f} "
            f"| {fmt_p(r['semantic_wilcoxon_p'])} "
            f"| {r['semantic_cohens_d']:+.3f} |"
        )
    return header + "\n".join(rows)


def render_report(report_path: Path, out_md: Path | None) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # ── Win-rate per condition ──
    wr_results: Dict[str, Dict] = {}
    quality_arrays: Dict[str, np.ndarray] = {}
    for cond, blk in report["winrate"]["per_condition"].items():
        # win-rate eval stores rows under "per_prompt"; implicit accuracy uses "per_pair"
        r = analyse_winrate_condition(blk["per_prompt"])
        wr_results[cond] = r
        quality_arrays[cond] = r.pop("adapter_quality_array")

    # ── Decoupling spread + bootstrap CI ──
    spreads_observed = (
        max(r["quality_mean_adapter"] for r in wr_results.values())
        - min(r["quality_mean_adapter"] for r in wr_results.values())
    )
    spread_lo, spread_hi = bootstrap_spread_ci(quality_arrays)

    # ── Implicit DPO accuracy per condition ──
    acc_results: Dict[str, Dict] = {}
    for cond, blk in report["dpo_accuracy"]["per_condition"].items():
        acc_results[cond] = analyse_implicit_accuracy_condition(blk["per_pair"])

    # ── Build markdown ──
    n_per_cond = next(iter(wr_results.values()))["n"]

    md = []
    md.append(f"# Chapter 4 Statistical Results")
    md.append(f"")
    md.append(f"Source: `{report_path.name}` — n = {n_per_cond} held-out prompts per condition.")
    md.append(f"")
    md.append(f"## 4.1 Win-Rate vs SFT-Merged Baseline")
    md.append(f"")
    md.append(build_winrate_table(wr_results))
    md.append(f"")
    md.append(f"## 4.2 Implicit DPO Accuracy (length-normalised per-token logp)")
    md.append(f"")
    md.append(build_implicit_accuracy_table(acc_results))
    md.append(f"")
    md.append(f"## 4.3 Semantic Preservation")
    md.append(f"")
    md.append(build_semantic_table(wr_results))
    md.append(f"")
    md.append(f"## 4.4 Decoupling Spread (Range Statistic)")
    md.append(f"")
    md.append(
        f"- Observed adapter-quality spread across conditions: **{spreads_observed:.4f}**"
    )
    md.append(
        f"- Pre-registered success threshold: 0.0500"
    )
    md.append(
        f"- Bootstrap 95% CI ({BOOTSTRAP_RESAMPLES} resamples): "
        f"[{spread_lo:.4f}, {spread_hi:.4f}]"
    )
    verdict = (
        "PASS" if spreads_observed < 0.05 and spread_hi < 0.05
        else "PASS (point estimate)" if spreads_observed < 0.05
        else "FAIL"
    )
    md.append(f"- Verdict: **{verdict}**")
    md.append(f"")

    text = "\n".join(md)
    print(text)

    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(text + "\n", encoding="utf-8")
        print(f"\n[written to {out_md}]")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute Chapter 4 statistical tests from an eval report.")
    p.add_argument(
        "--report",
        type=Path,
        default=Path("training/checkpoints/eval_report_full.json"),
        help="Path to eval report JSON produced by training.dpo_eval.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional Markdown output path.",
    )
    args = p.parse_args(argv)

    if not args.report.exists():
        sys.stderr.write(f"Report not found: {args.report}\n")
        return 1
    render_report(args.report, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
