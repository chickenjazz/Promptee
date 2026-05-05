"""
SFT Trainer — QLoRA Supervised Fine-Tuning with assistant-only loss masking.

Runs before the DPO phase to give the model a strong, prompt-independent
baseline. Loads Qwen2.5-3B-Instruct in 4-bit NF4, attaches LoRA adapters
on q_proj/k_proj/v_proj/o_proj (matching the existing DPO config), and
trains on datasets/sft_dataset.jsonl produced by build_sft_dataset.py.

Loss is masked on everything before the assistant turn via SFTConfig's
assistant_only_loss=True, so gradients only flow through the generated
rewrite — not the system or user text. (trl 1.3.0+ replaced the older
DataCollatorForCompletionOnlyLM with this flag.)

VRAM budget: tuned for an RTX 3070 (8GB).
  - per_device_train_batch_size=1
  - gradient_accumulation_steps=8 (effective batch 8)
  - gradient_checkpointing=True
  - paged_adamw_8bit
  - max_seq_length=1024

OFFLINE-ONLY. Not imported by runtime code.
"""

import os
import sys
import logging
import argparse

# Windows + trl 1.3.0 bug: trl/chat_template_utils.py loads bundled .jinja
# templates via Path.read_text() without an encoding kwarg, so on Windows it
# defaults to cp1252 and crashes on UTF-8 bytes in deepseekv3.jinja. Pin
# Path.read_text to UTF-8 before importing trl. Setting PYTHONUTF8=1 in the
# environment also works, but this keeps the script self-contained.
if sys.platform == "win32":
    import pathlib as _pathlib
    _orig_read_text = _pathlib.Path.read_text
    def _utf8_read_text(self, *args, **kwargs):
        kwargs.setdefault("encoding", "utf-8")
        return _orig_read_text(self, *args, **kwargs)
    _pathlib.Path.read_text = _utf8_read_text

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. SFT training requires a GPU.")

logger = logging.getLogger("promptee.sft_trainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", "sft_dataset.jsonl")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints", "sft")


def train(
    base_model_id: str = BASE_MODEL_ID,
    dataset_path: str = DATASET_PATH,
    checkpoint_dir: str = CHECKPOINT_DIR,
    epochs: int = 2,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 1024,
):
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

    # ── 4. Load SFT Dataset ──────────────────────────────────────────────
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"SFT dataset not found at {dataset_path}. "
            f"Run training/build_sft_dataset.py first."
        )

    logger.info(f"Loading SFT dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if len(dataset) == 0:
        raise ValueError("SFT dataset is empty.")
    logger.info(f"Dataset size: {len(dataset)} examples")

    # ── 5. Configure SFT Training ────────────────────────────────────────
    # assistant_only_loss=True masks loss on everything before the assistant
    # turn so gradients only flow through the rewrite. This replaces the
    # older DataCollatorForCompletionOnlyLM and is the supported path in
    # trl 1.3.0+ — it reads the chat template directly instead of needing a
    # hand-encoded response_template.
    os.makedirs(checkpoint_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        optim="paged_adamw_8bit",
        bf16=True,
        max_length=max_seq_length,
        packing=False,
        assistant_only_loss=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
    )

    # ── 6. Initialize SFTTrainer ─────────────────────────────────────────
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ── 8. Train ─────────────────────────────────────────────────────────
    logger.info("Starting SFT training...")
    trainer.train()

    # ── 9. Save Final Adapter ────────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"SFT complete. Final adapter saved to: {final_path}")

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training for Prompt Optimizer")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output", default=CHECKPOINT_DIR)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        checkpoint_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        max_seq_length=args.max_seq_length,
    )
