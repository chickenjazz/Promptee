"""
DPO Trainer — QLoRA + Direct Preference Optimization

SOP Reference: architecture/dpo_training.md §4-5

Loads Qwen2.5-3B-Instruct with 4-bit quantization, attaches LoRA adapters,
and trains via DPO on the preference dataset.

This script is OFFLINE-ONLY. It must never be imported by runtime code.
"""

import os
import sys
import random
import logging
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

# Fix SSL_CERT_FILE pointing to non-existent path (known env issue)
if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from training._prompts import STRONG_PROMPT, WEAK_PROMPT, USER_TEMPLATE

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. DPO training requires a GPU.")

logger = logging.getLogger("promptee.dpo_trainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# Defaults from SOP. After SFT, prefer the merged baseline as the DPO start point —
# falls back to the original Qwen base if the merged model hasn't been built yet
# so this script remains runnable in isolation.
SFT_BASELINE_PATH = os.path.join(PROJECT_ROOT, "models", "sft_baseline")
BASE_MODEL_ID = SFT_BASELINE_PATH if os.path.isdir(SFT_BASELINE_PATH) else "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", "preference_pairs.jsonl")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints")

# System-prompt dropout boundaries (must match build_sft_dataset.py)
P_STRONG = 0.30
P_WEAK = 0.70
DROPOUT_SEED = 42


def _format_with_dropout(example, tokenizer, rng):
    """Pre-format the DPO prompt with the same 30/40/30 system-prompt dropout
    used in SFT. Without this, DPO would only ever see the no-system template
    (DPOTrainer's default) and would partially undo the SFT decoupling.

    chosen/rejected stay as raw text — DPOTrainer concatenates them onto the
    formatted prompt verbatim, so the assistant turn ends up identical to the
    SFT format.
    """
    r = rng.random()
    if r < P_STRONG:
        msgs = [{"role": "system", "content": STRONG_PROMPT}]
    elif r < P_WEAK:
        msgs = [{"role": "system", "content": WEAK_PROMPT}]
    else:
        msgs = []
    msgs.append({"role": "user", "content": USER_TEMPLATE.format(raw_prompt=example["prompt"])})
    example["prompt"] = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    return example


def train(
    base_model_id: str = BASE_MODEL_ID,
    dataset_path: str = DATASET_PATH,
    checkpoint_dir: str = CHECKPOINT_DIR,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    beta: float = 0.05,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_length: int = 1024,
):
    """
    Run DPO training on the preference dataset.

    Follows the QLoRA configuration from architecture/dpo_training.md:
      - 4-bit NF4 quantization, double quant, bf16
      - LoRA on q_proj, k_proj, v_proj, o_proj
      - Paged AdamW 8-bit optimizer
      - Gradient checkpointing enabled
    """
    logger.info(f"System Check | PyTorch CUDA available: {torch.cuda.is_available()}")

    # ── 1. Load Tokenizer ────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Load Base Model with 4-bit Quantization ───────────────────────
    logger.info(f"Loading model: {base_model_id} (4-bit NF4, double quant)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── 3. Prepare for k-bit training + Attach LoRA ──────────────────────
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 4. Load Preference Dataset ───────────────────────────────────────
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Preference dataset not found at {dataset_path}. "
            f"Run training/dataset_builder.py first."
        )

    logger.info(f"Loading preference dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    if len(dataset) == 0:
        raise ValueError(
            "Preference dataset is empty. "
            "Run training/dataset_builder.py to generate pairs."
        )

    logger.info(f"Dataset size: {len(dataset)} preference pairs")

    # Rename columns to match DPOTrainer expectations
    dataset = dataset.rename_columns({
        "x": "prompt",
        "y_w": "chosen",
        "y_l": "rejected",
    })

    # Apply system-prompt dropout to match the SFT training distribution.
    rng = random.Random(DROPOUT_SEED)
    dataset = dataset.map(
        lambda ex: _format_with_dropout(ex, tokenizer, rng),
        desc="Applying system-prompt dropout",
    )

    # ── 5. Configure DPO Training ────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    # ── 6. Initialize DPO Trainer ────────────────────────────────────────
    logger.info("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ── 7. Train ─────────────────────────────────────────────────────────
    logger.info("Starting DPO training...")
    trainer.train()

    # ── 8. Save Final Checkpoint ─────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training complete. Final checkpoint saved to: {final_path}")

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training for Prompt Optimizer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--base-model", default=BASE_MODEL_ID,
                        help="Base model path. Defaults to models/sft_baseline if present.")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output", default=CHECKPOINT_DIR)
    args = parser.parse_args()

    train(
        base_model_id=args.base_model,
        dataset_path=args.dataset,
        checkpoint_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        lora_r=args.lora_r,
    )
