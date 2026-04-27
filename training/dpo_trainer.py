"""
DPO Trainer — QLoRA + Direct Preference Optimization

SOP Reference: architecture/dpo_training.md §4-5

Loads Qwen2.5-3B-Instruct with 4-bit quantization, attaches LoRA adapters,
and trains via DPO on the preference dataset.

This script is OFFLINE-ONLY. It must never be imported by runtime code.
"""

import os
import sys
import logging
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

logger = logging.getLogger("promptee.dpo_trainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# Defaults from SOP
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" #TODO: Change to 3B
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", "preference_pairs.jsonl")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints")


def train(
    base_model_id: str = BASE_MODEL_ID,
    dataset_path: str = DATASET_PATH,
    checkpoint_dir: str = CHECKPOINT_DIR,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_length: int = 512,
    max_prompt_length: int = 256,
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

    # ── 5. Configure DPO Training ────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output", default=CHECKPOINT_DIR)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        checkpoint_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        lora_r=args.lora_r,
    )
