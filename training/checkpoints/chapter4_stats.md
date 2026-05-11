# Chapter 4 Statistical Results

Source: `eval_report_full.json` — n = 248 held-out prompts per condition.

## 4.1 Win-Rate vs SFT-Merged Baseline

| Condition | Win-rate (95% CI) | p (vs chance) | Δ quality (mean ± SE) | Wilcoxon p | Cohen's d | McNemar p |
|---|---|---|---|---|---|---|
| strong | 53.6% [47.4%, 59.7%] | 0.280 | +0.0119 ± 0.0056 | 0.072 | +0.136 | 0.280 |
| weak | 55.2% [49.0%, 61.3%] | 0.112 | +0.0160 ± 0.0054 | 0.004 | +0.189 | 0.112 |
| none | 57.7% [51.4%, 63.6%] | 0.019 | +0.0160 ± 0.0054 | 0.002 | +0.189 | 0.015 |

## 4.2 Implicit DPO Accuracy (length-normalised per-token logp)

| Condition | Accuracy (95% CI) | p (vs chance) | Mean margin (nats, ± SE) | t-test p (vs 0) | Cohen's d |
|---|---|---|---|---|---|
| strong | 64.1% [58.0%, 69.8%] | < 0.001 | +0.2368 ± 0.0441 | < 0.001 | +0.341 |
| weak | 70.2% [64.2%, 75.5%] | < 0.001 | +0.3331 ± 0.0465 | < 0.001 | +0.455 |
| none | 67.7% [61.7%, 73.3%] | < 0.001 | +0.2718 ± 0.0443 | < 0.001 | +0.390 |

## 4.3 Semantic Preservation

| Condition | Adapter mean | Base mean | Δ (paired) | Wilcoxon p | Cohen's d |
|---|---|---|---|---|---|
| strong | 0.7137 | 0.7338 | -0.0201 | < 0.001 | -0.247 |
| weak | 0.7244 | 0.7305 | -0.0061 | 0.153 | -0.094 |
| none | 0.7244 | 0.7302 | -0.0058 | 0.317 | -0.088 |

## 4.4 Decoupling Spread (Range Statistic)

- Observed adapter-quality spread across conditions: **0.0049**
- Pre-registered success threshold: 0.0500
- Bootstrap 95% CI (1000 resamples): [0.0013, 0.0153]
- Verdict: **PASS**

