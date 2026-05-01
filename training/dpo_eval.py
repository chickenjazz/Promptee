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

logger = logging.getLogger("promptee.dpo_eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER = os.path.join(PROJECT_ROOT, "training", "checkpoints", "final")


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

def _format_chat(tokenizer, raw_prompt: str) -> str:
    """Wrap the raw prompt in the same chat template DPOTrainer saw at train time."""
    messages = [{"role": "user", "content": raw_prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def _generate(model, tokenizer, raw_prompt: str, max_new_tokens: int = 384) -> str:
    chat_text = _format_chat(tokenizer, raw_prompt)
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

def run_winrate_eval(
    eval_prompts_path: Path,
    adapter_path: str,
    base_model_id: str,
    num_samples: int,
    max_new_tokens: int,
) -> Dict:
    prompts = _load_raw_prompts(eval_prompts_path)
    if not prompts:
        raise ValueError(f"No prompts loaded from {eval_prompts_path}")
    if num_samples > 0:
        prompts = prompts[:num_samples]
    logger.info("Win-rate eval over %d prompts", len(prompts))

    # Generate with base
    base_model, tokenizer = _load_base_model(base_model_id)
    logger.info("Generating with BASE model...")
    base_rewrites: List[str] = []
    for i, p in enumerate(prompts, 1):
        base_rewrites.append(_generate(base_model, tokenizer, p, max_new_tokens))
        if i % 5 == 0:
            logger.info("  base: %d/%d", i, len(prompts))
    del base_model
    torch.cuda.empty_cache()

    # Generate with adapter
    adapter_model, tokenizer = _load_adapter_model(base_model_id, adapter_path)
    logger.info("Generating with ADAPTER model...")
    adapter_rewrites: List[str] = []
    for i, p in enumerate(prompts, 1):
        adapter_rewrites.append(_generate(adapter_model, tokenizer, p, max_new_tokens))
        if i % 5 == 0:
            logger.info("  adapter: %d/%d", i, len(prompts))
    del adapter_model
    torch.cuda.empty_cache()

    # Score with HeuristicScorer
    logger.info("Scoring rewrites with HeuristicScorer...")
    scorer = HeuristicScorer()
    per_prompt: List[Dict] = []
    base_scores: List[float] = []
    adapter_scores: List[float] = []
    base_semantic: List[float] = []
    adapter_semantic: List[float] = []
    adapter_wins = 0
    base_wins = 0
    ties = 0
    adapter_rejections = 0

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


# ──────────────────────────────────────────────────────────────────────────
# Eval mode 2: Implicit DPO accuracy (logprob)
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _completion_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Sum of log-probs the model assigns to `completion` tokens given chat-formatted `prompt`."""
    chat_text = _format_chat(tokenizer, prompt)
    full_text = chat_text + completion

    prompt_ids = tokenizer(chat_text, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

    p_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= p_len:
        return 0.0

    logits = model(full_ids).logits  # [1, L, V]
    # Token at position t is predicted by logits[t-1]
    target_ids = full_ids[0, p_len:]
    target_logits = logits[0, p_len - 1: full_ids.shape[1] - 1]
    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    token_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_lp.sum().item())


def run_dpo_accuracy_eval(
    eval_pairs_path: Path,
    adapter_path: str,
    base_model_id: str,
    num_samples: int,
) -> Dict:
    pairs = _load_jsonl(eval_pairs_path)
    pairs = [p for p in pairs if {"x", "y_w", "y_l"}.issubset(p.keys())]
    if not pairs:
        raise ValueError(
            f"No valid {{x,y_w,y_l}} pairs in {eval_pairs_path}"
        )
    if num_samples > 0:
        pairs = pairs[:num_samples]
    logger.info("Implicit DPO accuracy over %d pairs", len(pairs))

    model, tokenizer = _load_adapter_model(base_model_id, adapter_path)

    correct = 0
    margins: List[float] = []
    per_pair: List[Dict] = []

    for i, p in enumerate(pairs, 1):
        x, y_w, y_l = p["x"], p["y_w"], p["y_l"]
        lp_w = _completion_logprob(model, tokenizer, x, y_w)
        lp_l = _completion_logprob(model, tokenizer, x, y_l)
        margin = lp_w - lp_l
        margins.append(margin)
        is_correct = lp_w > lp_l
        if is_correct:
            correct += 1
        per_pair.append({
            "x": x,
            "logp_chosen": round(lp_w, 4),
            "logp_rejected": round(lp_l, 4),
            "margin": round(margin, 4),
            "correct": is_correct,
        })
        if i % 5 == 0:
            logger.info("  dpo-acc: %d/%d (running acc=%.3f)", i, len(pairs), correct / i)

    n = len(pairs)
    summary = {
        "num_pairs": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "mean_margin": round(sum(margins) / n, 4) if n else 0.0,
    }
    return {"summary": summary, "per_pair": per_pair}


# ──────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────

def _print_summary(report: Dict) -> None:
    print("\n" + "=" * 70)
    print("DPO EVAL SUMMARY")
    print("=" * 70)

    if "winrate" in report:
        s = report["winrate"]["summary"]
        print("\n[Win-rate via HeuristicScorer]")
        print(f"  Prompts evaluated:        {s['num_prompts']}")
        print(f"  Adapter wins:             {s['adapter_wins']}")
        print(f"  Base wins:                {s['base_wins']}")
        print(f"  Ties:                     {s['ties']}")
        print(f"  Adapter win-rate:         {s['adapter_win_rate']:.2%}")
        print(f"  Adapter mean quality:     {s['adapter_mean_quality']:.4f}")
        print(f"  Base mean quality:        {s['base_mean_quality']:.4f}")
        print(f"  Adapter mean semantic:    {s['adapter_mean_semantic']:.4f}")
        print(f"  Base mean semantic:       {s['base_mean_semantic']:.4f}")
        print(f"  Adapter semantic rejects: {s['adapter_semantic_rejections']}")

        # 3 side-by-side examples
        print("\n[Sample side-by-side rewrites]")
        for i, ex in enumerate(report["winrate"]["per_prompt"][:3], 1):
            print(f"\n  --- Example {i} (winner: {ex['winner']}) ---")
            print(f"  RAW:     {ex['x'][:120]}")
            print(f"  BASE  ({ex['base_quality']:.3f}): {ex['base_rewrite'][:160]}")
            print(f"  ADAPT ({ex['adapter_quality']:.3f}): {ex['adapter_rewrite'][:160]}")

    if "dpo_accuracy" in report:
        s = report["dpo_accuracy"]["summary"]
        print("\n[Implicit DPO accuracy (held-out logprob)]")
        print(f"  Pairs evaluated: {s['num_pairs']}")
        print(f"  Correct:         {s['correct']}")
        print(f"  Accuracy:        {s['accuracy']:.2%}")
        print(f"  Mean margin:     {s['mean_margin']:.4f}  (logp_chosen - logp_rejected)")

    print("\n" + "=" * 70)


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
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional path to write the full JSON report.")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. DPO eval requires a GPU.")

    if args.eval_prompts is None and args.eval_pairs is None:
        parser.error("Provide at least one of --eval-prompts or --eval-pairs.")

    report: Dict = {}

    if args.eval_prompts is not None:
        report["winrate"] = run_winrate_eval(
            eval_prompts_path=args.eval_prompts,
            adapter_path=args.adapter,
            base_model_id=args.base_model,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
        )

    if args.eval_pairs is not None:
        report["dpo_accuracy"] = run_dpo_accuracy_eval(
            eval_pairs_path=args.eval_pairs,
            adapter_path=args.adapter,
            base_model_id=args.base_model,
            num_samples=args.num_samples,
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
