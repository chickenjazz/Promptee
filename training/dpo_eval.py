"""
DPO Evaluation — Adapter vs Base Quality Comparison

SOP Reference: architecture/dpo_training.md §5 (post-training validation)

Evaluates a trained LoRA adapter on a held-out set of prompts to verify that
DPO training actually improved rewrite quality. This script is OFFLINE-ONLY.

Two complementary metrics:

  1. Win-rate (generation eval)
     - Generate rewrites with both the base Qwen and the adapted model
       on the same held-out raw prompts.
     - Score each rewrite with HeuristicScorer.
     - Report: % of prompts where adapter score > base score.
     - Requires: --eval-prompts (raw prompts, JSONL with {"x": ...} or CSV
       with a 'prompt' column).

  2. Implicit DPO accuracy (logprob eval)
     - For held-out (x, y_w, y_l) preference pairs, check whether the
       trained model assigns higher log P(y_w|x) than log P(y_l|x).
     - Equivalent to TRL's `rewards/accuracies` metric.
     - Requires: --eval-pairs (JSONL with {"x", "y_w", "y_l"}).

Either mode can be run independently. If both files are provided, both
metrics are reported.

Usage:
    python -m training.dpo_eval \\
        --adapter training/checkpoints/final \\
        --eval-prompts datasets/eval_raw_prompts.jsonl \\
        --eval-pairs   datasets/eval_preference_pairs.jsonl \\
        --num-samples 50 \\
        --report training/checkpoints/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Fix SSL_CERT_FILE pointing to non-existent path (known env issue)
if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from tools.heuristic_scorer import HeuristicScorer  # noqa: E402
from training._prompts import STRONG_PROMPT, WEAK_PROMPT, USER_TEMPLATE  # noqa: E402

logger = logging.getLogger("promptee.dpo_eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER = os.path.join(PROJECT_ROOT, "training", "checkpoints", "final")

# System-prompt conditions for the decoupling ablation. Names map to the
# same three buckets used by training/build_sft_dataset.py and the dropout
# code in training/dpo_trainer.py — keeping them aligned matters because
# we're measuring whether the model behaves consistently *across* these
# conditions, which only works if they're literally the same strings.
SYSTEM_PROMPTS: Dict[str, Optional[str]] = {
    "strong": STRONG_PROMPT,
    "weak":   WEAK_PROMPT,
    "none":   None,
}


# ──────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSONL line %d in %s: %s", ln, path, e)
    return rows


def _load_raw_prompts(path: Path) -> List[str]:
    """Accept JSONL ({'x':..} or {'prompt':..}) or CSV with a 'prompt' column."""
    if path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        col = "prompt" if "prompt" in df.columns else ("x" if "x" in df.columns else None)
        if col is None:
            raise ValueError(f"CSV {path} must have a 'prompt' or 'x' column.")
        return [str(v).strip() for v in df[col].tolist() if str(v).strip()]
    rows = _load_jsonl(path)
    out: List[str] = []
    for r in rows:
        v = r.get("x") or r.get("prompt")
        if v and str(v).strip():
            out.append(str(v).strip())
    return out


# ──────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────

def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _load_base_model(base_model_id: str):
    logger.info("Loading base model: %s (4-bit NF4)", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _load_adapter_model(base_model_id: str, adapter_path: str):
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}. "
            f"Train first via training/dpo_trainer.py."
        )
    logger.info("Loading adapter on top of base: %s", adapter_path)
    base, tokenizer = _load_base_model(base_model_id)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────

def _format_chat(tokenizer, raw_prompt: str, system_prompt: Optional[str] = None) -> str:
    """Format the raw prompt the same way the training pipeline did.

    The SFT / new DPO data wraps the user turn as `"Optimize this prompt: {x}"`
    via training._prompts.USER_TEMPLATE — eval must match or we'd be measuring
    distribution shift instead of generalisation. The system prompt is
    optional: pass STRONG_PROMPT / WEAK_PROMPT / None to evaluate each
    decoupling condition.
    """
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(raw_prompt=raw_prompt)})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def _generate(
    model,
    tokenizer,
    raw_prompt: str,
    max_new_tokens: int = 384,
    system_prompt: Optional[str] = None,
) -> str:
    chat_text = _format_chat(tokenizer, raw_prompt, system_prompt=system_prompt)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────
# Eval mode 1: Win-rate via HeuristicScorer
# ──────────────────────────────────────────────────────────────────────────

def _generate_all_conditions(
    model,
    tokenizer,
    prompts: List[str],
    conditions: List[str],
    max_new_tokens: int,
    label: str,
) -> Dict[str, List[str]]:
    """Run inference once per (prompt, condition) pair, sharing the loaded
    model across all conditions to avoid re-quantising 3x."""
    out: Dict[str, List[str]] = {c: [] for c in conditions}
    for cond in conditions:
        sysp = SYSTEM_PROMPTS[cond]
        logger.info("Generating with %s model | system_prompt=%s", label.upper(), cond)
        for i, p in enumerate(prompts, 1):
            out[cond].append(_generate(model, tokenizer, p, max_new_tokens, system_prompt=sysp))
            if i % 5 == 0:
                logger.info("  %s/%s: %d/%d", label, cond, i, len(prompts))
    return out


def _score_condition(
    prompts: List[str],
    base_rewrites: List[str],
    adapter_rewrites: List[str],
    scorer: HeuristicScorer,
) -> Dict:
    per_prompt: List[Dict] = []
    base_scores: List[float] = []
    adapter_scores: List[float] = []
    base_semantic: List[float] = []
    adapter_semantic: List[float] = []
    adapter_wins = base_wins = ties = adapter_rejections = 0

    for raw, b, a in zip(prompts, base_rewrites, adapter_rewrites):
        b_eval = scorer.evaluate(raw, b)
        a_eval = scorer.evaluate(raw, a)
        base_scores.append(b_eval["candidate_quality"])
        adapter_scores.append(a_eval["candidate_quality"])
        base_semantic.append(b_eval["semantic_preservation"])
        adapter_semantic.append(a_eval["semantic_preservation"])
        if a_eval["rejected"]:
            adapter_rejections += 1

        if a_eval["candidate_quality"] > b_eval["candidate_quality"]:
            adapter_wins += 1
            verdict = "adapter"
        elif a_eval["candidate_quality"] < b_eval["candidate_quality"]:
            base_wins += 1
            verdict = "base"
        else:
            ties += 1
            verdict = "tie"

        per_prompt.append({
            "x": raw,
            "base_rewrite": b,
            "adapter_rewrite": a,
            "base_quality": b_eval["candidate_quality"],
            "adapter_quality": a_eval["candidate_quality"],
            "base_semantic": b_eval["semantic_preservation"],
            "adapter_semantic": a_eval["semantic_preservation"],
            "adapter_rejected": a_eval["rejected"],
            "winner": verdict,
        })

    n = len(prompts)
    summary = {
        "num_prompts": n,
        "adapter_wins": adapter_wins,
        "base_wins": base_wins,
        "ties": ties,
        "adapter_win_rate": round(adapter_wins / n, 4) if n else 0.0,
        "adapter_mean_quality": round(sum(adapter_scores) / n, 4) if n else 0.0,
        "base_mean_quality": round(sum(base_scores) / n, 4) if n else 0.0,
        "adapter_mean_semantic": round(sum(adapter_semantic) / n, 4) if n else 0.0,
        "base_mean_semantic": round(sum(base_semantic) / n, 4) if n else 0.0,
        "adapter_semantic_rejections": adapter_rejections,
    }
    return {"summary": summary, "per_prompt": per_prompt}


def run_winrate_eval(
    eval_prompts_path: Path,
    adapter_path: str,
    base_model_id: str,
    num_samples: int,
    max_new_tokens: int,
    conditions: List[str],
) -> Dict:
    prompts = _load_raw_prompts(eval_prompts_path)
    if not prompts:
        raise ValueError(f"No prompts loaded from {eval_prompts_path}")
    if num_samples > 0:
        prompts = prompts[:num_samples]
    logger.info("Win-rate eval over %d prompts × %d conditions: %s",
                len(prompts), len(conditions), conditions)

    # Generate with base under every condition (single load)
    base_model, tokenizer = _load_base_model(base_model_id)
    base_by_cond = _generate_all_conditions(
        base_model, tokenizer, prompts, conditions, max_new_tokens, "base",
    )
    del base_model
    torch.cuda.empty_cache()

    # Generate with adapter under every condition (single load)
    adapter_model, tokenizer = _load_adapter_model(base_model_id, adapter_path)
    adapter_by_cond = _generate_all_conditions(
        adapter_model, tokenizer, prompts, conditions, max_new_tokens, "adapter",
    )
    del adapter_model
    torch.cuda.empty_cache()

    logger.info("Scoring rewrites with HeuristicScorer...")
    scorer = HeuristicScorer()

    per_condition: Dict[str, Dict] = {}
    for cond in conditions:
        per_condition[cond] = _score_condition(
            prompts, base_by_cond[cond], adapter_by_cond[cond], scorer,
        )

    # Decoupling gap: adapter mean quality across conditions. The whole point
    # of SFT-with-dropout is that this spread is small.
    decoupling: Dict[str, float] = {}
    if {"strong", "none"}.issubset(conditions):
        s_q = per_condition["strong"]["summary"]["adapter_mean_quality"]
        n_q = per_condition["none"]["summary"]["adapter_mean_quality"]
        decoupling["adapter_quality_strong_minus_none"] = round(s_q - n_q, 4)
    if all(c in conditions for c in ("strong", "weak", "none")):
        qs = [per_condition[c]["summary"]["adapter_mean_quality"]
              for c in ("strong", "weak", "none")]
        decoupling["adapter_quality_spread"] = round(max(qs) - min(qs), 4)

    return {
        "conditions": conditions,
        "per_condition": per_condition,
        "decoupling": decoupling,
    }


# ──────────────────────────────────────────────────────────────────────────
# Eval mode 2: Implicit DPO accuracy (logprob)
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _completion_logprob(
    model, tokenizer, prompt: str, completion: str, system_prompt: Optional[str] = None,
) -> Tuple[float, int]:
    """Return (sum_logprob, n_tokens) for the completion given the chat-formatted prompt.

    Callers should compare per-token mean (sum / n_tokens), not the raw sum.
    Comparing summed log-probs across completions of different lengths is
    unfair: a longer sequence accumulates more negative log-probability
    purely because it has more tokens to predict, regardless of preference.
    For DPO accuracy where chosen rewrites are ~4× longer than rejected raw
    prompts, the raw-sum comparison is dominated by length, not preference.
    """
    chat_text = _format_chat(tokenizer, prompt, system_prompt=system_prompt)
    full_text = chat_text + completion

    prompt_ids = tokenizer(chat_text, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

    p_len = prompt_ids.shape[1]
    n_tokens = full_ids.shape[1] - p_len
    if n_tokens <= 0:
        return 0.0, 0

    logits = model(full_ids).logits  # [1, L, V]
    # Token at position t is predicted by logits[t-1]
    target_ids = full_ids[0, p_len:]
    target_logits = logits[0, p_len - 1: full_ids.shape[1] - 1]
    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    token_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_lp.sum().item()), int(n_tokens)


def run_dpo_accuracy_eval(
    eval_pairs_path: Path,
    adapter_path: str,
    base_model_id: str,
    num_samples: int,
    conditions: List[str],
) -> Dict:
    pairs = _load_jsonl(eval_pairs_path)
    pairs = [p for p in pairs if {"x", "y_w", "y_l"}.issubset(p.keys())]
    if not pairs:
        raise ValueError(
            f"No valid {{x,y_w,y_l}} pairs in {eval_pairs_path}"
        )
    if num_samples > 0:
        pairs = pairs[:num_samples]
    logger.info("Implicit DPO accuracy over %d pairs × %d conditions: %s",
                len(pairs), len(conditions), conditions)

    model, tokenizer = _load_adapter_model(base_model_id, adapter_path)

    per_condition: Dict[str, Dict] = {}
    for cond in conditions:
        sysp = SYSTEM_PROMPTS[cond]
        logger.info("Implicit DPO accuracy | system_prompt=%s", cond)
        correct = 0
        margins: List[float] = []  # per-token margin values for averaging
        per_pair: List[Dict] = []
        for i, p in enumerate(pairs, 1):
            x, y_w, y_l = p["x"], p["y_w"], p["y_l"]
            sum_w, n_w = _completion_logprob(model, tokenizer, x, y_w, system_prompt=sysp)
            sum_l, n_l = _completion_logprob(model, tokenizer, x, y_l, system_prompt=sysp)
            # Per-token mean log-prob — fair across length-asymmetric pairs.
            mean_w = sum_w / n_w if n_w else 0.0
            mean_l = sum_l / n_l if n_l else 0.0
            margin = mean_w - mean_l
            margins.append(margin)
            is_correct = mean_w > mean_l
            if is_correct:
                correct += 1
            per_pair.append({
                "x": x,
                "logp_chosen_per_token": round(mean_w, 4),
                "logp_rejected_per_token": round(mean_l, 4),
                "n_tokens_chosen": n_w,
                "n_tokens_rejected": n_l,
                "margin_per_token": round(margin, 4),
                "correct": is_correct,
            })
            if i % 5 == 0:
                logger.info("  dpo-acc/%s: %d/%d (running acc=%.3f)",
                            cond, i, len(pairs), correct / i)
        n = len(pairs)
        per_condition[cond] = {
            "summary": {
                "num_pairs": n,
                "correct": correct,
                "accuracy": round(correct / n, 4) if n else 0.0,
                "mean_margin_per_token": round(sum(margins) / n, 4) if n else 0.0,
            },
            "per_pair": per_pair,
        }

    return {"conditions": conditions, "per_condition": per_condition}


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────

def _print_summary(report: Dict) -> None:
    print("\n" + "=" * 78)
    print("DPO EVAL SUMMARY")
    print("=" * 78)

    if "winrate" in report:
        wr = report["winrate"]
        conds = wr["conditions"]
        print("\n[Win-rate via HeuristicScorer — by system-prompt condition]")
        header = f"  {'metric':<28}" + "".join(f"{c:>12}" for c in conds)
        print(header)
        print("  " + "-" * (len(header) - 2))

        rows = [
            ("adapter win-rate",         "adapter_win_rate",         "%"),
            ("adapter mean quality",     "adapter_mean_quality",     "f"),
            ("base mean quality",        "base_mean_quality",        "f"),
            ("adapter mean semantic",    "adapter_mean_semantic",    "f"),
            ("base mean semantic",       "base_mean_semantic",       "f"),
            ("adapter semantic rejects", "adapter_semantic_rejections", "d"),
        ]
        for label, key, fmt in rows:
            line = f"  {label:<28}"
            for c in conds:
                v = wr["per_condition"][c]["summary"][key]
                if fmt == "%":
                    line += f"{v:>11.2%} "
                elif fmt == "f":
                    line += f"{v:>12.4f}"
                else:
                    line += f"{v:>12d}"
            print(line)

        if wr.get("decoupling"):
            print("\n  [Decoupling]")
            for k, v in wr["decoupling"].items():
                print(f"    {k}: {v:+.4f}")
            target = wr["decoupling"].get("adapter_quality_strong_minus_none")
            if target is not None:
                verdict = "PASS" if abs(target) < 0.05 else (
                    "MARGINAL" if abs(target) < 0.10 else "FAIL"
                )
                print(f"    decoupling verdict (|strong-none| < 0.05): {verdict}")

        # Side-by-side examples from the first condition only — readable enough.
        first = conds[0]
        print(f"\n[Sample side-by-side rewrites | condition={first}]")
        for i, ex in enumerate(wr["per_condition"][first]["per_prompt"][:3], 1):
            print(f"\n  --- Example {i} (winner: {ex['winner']}) ---")
            print(f"  RAW:     {ex['x'][:120]}")
            print(f"  BASE  ({ex['base_quality']:.3f}): {ex['base_rewrite'][:160]}")
            print(f"  ADAPT ({ex['adapter_quality']:.3f}): {ex['adapter_rewrite'][:160]}")

    if "dpo_accuracy" in report:
        da = report["dpo_accuracy"]
        conds = da["conditions"]
        print("\n[Implicit DPO accuracy — by system-prompt condition]")
        header = f"  {'metric':<20}" + "".join(f"{c:>12}" for c in conds)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for label, key, fmt in (("accuracy", "accuracy", "%"),
                                ("mean margin / token", "mean_margin_per_token", "f")):
            line = f"  {label:<20}"
            for c in conds:
                v = da["per_condition"][c]["summary"][key]
                line += f"{v:>11.2%} " if fmt == "%" else f"{v:>12.4f}"
            print(line)

    print("\n" + "=" * 78)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DPO evaluation: adapter vs base")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER,
                        help="Path to trained LoRA adapter directory.")
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--eval-prompts", type=Path, default=None,
                        help="Held-out raw prompts (JSONL with 'x' or 'prompt', or CSV). "
                             "Required for win-rate eval.")
    parser.add_argument("--eval-pairs", type=Path, default=None,
                        help="Held-out preference pairs JSONL ({x,y_w,y_l}). "
                             "Required for implicit DPO accuracy.")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Cap on items per eval mode. 0 = use all.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--system-prompt",
        nargs="+",
        choices=list(SYSTEM_PROMPTS.keys()),
        default=["strong", "weak", "none"],
        help=(
            "System-prompt condition(s) to evaluate. Pass multiple to run the "
            "decoupling ablation in one go (default: strong weak none). The "
            "model is loaded once per role and inferred under each condition."
        ),
    )
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional path to write the full JSON report.")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DPO eval requires a GPU.")

    if args.eval_prompts is None and args.eval_pairs is None:
        parser.error("Provide at least one of --eval-prompts or --eval-pairs.")

    # Preserve user-given order, dedupe.
    conditions = list(dict.fromkeys(args.system_prompt))

    report: Dict = {"conditions_requested": conditions}

    if args.eval_prompts is not None:
        report["winrate"] = run_winrate_eval(
            eval_prompts_path=args.eval_prompts,
            adapter_path=args.adapter,
            base_model_id=args.base_model,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            conditions=conditions,
        )

    if args.eval_pairs is not None:
        report["dpo_accuracy"] = run_dpo_accuracy_eval(
            eval_pairs_path=args.eval_pairs,
            adapter_path=args.adapter,
            base_model_id=args.base_model,
            num_samples=args.num_samples,
            conditions=conditions,
        )

    _print_summary(report)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        logger.info("Full report written to: %s", args.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
