"""
SFT Adapter Merge — Fuse the SFT LoRA into the base model.

After SFT training, the LoRA deltas live in training/checkpoints/sft/final/.
DPO needs to start from a model whose weights already contain those deltas,
not a separate adapter, otherwise the DPO LoRA would stack on top of the
SFT LoRA and the merged final shipment becomes ambiguous.

This script loads the base model in bf16 (NOT 4-bit — quantizing before
merging destroys the LoRA precision), applies the SFT adapter, calls
merge_and_unload(), and saves the resulting full-precision model to
models/sft_baseline/. DPO then loads that path as its base.

VRAM note: 3B model in bf16 ≈ 6GB. Tight on an 8GB card. If GPU OOMs,
re-run with --device cpu — slower but reliable.

OFFLINE-ONLY. Not imported by runtime code.
"""

import os
import sys
import logging
import argparse
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

logger = logging.getLogger("promptee.merge_sft")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints", "sft", "final")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "sft_baseline")


def merge(
    base_model_id: str = BASE_MODEL_ID,
    adapter_dir: str = ADAPTER_DIR,
    output_dir: str = OUTPUT_DIR,
    device: str = "auto",
):
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(
            f"SFT adapter not found at {adapter_dir}. "
            f"Run training/sft_trainer.py first."
        )

    logger.info(f"Loading base model in bf16: {base_model_id}")
    if device == "cpu":
        device_map = {"": "cpu"}
    else:
        device_map = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    logger.info(f"Attaching SFT adapter from: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    logger.info("Merging LoRA deltas into base weights...")
    merged = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving merged model to: {output_dir}")
    merged.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer alongside so downstream scripts can load from one path
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Copy chat template if the adapter checkpoint shipped one
    chat_template_src = os.path.join(adapter_dir, "chat_template.jinja")
    if os.path.exists(chat_template_src):
        shutil.copy2(chat_template_src, os.path.join(output_dir, "chat_template.jinja"))

    logger.info("SFT baseline ready. DPO can now point at this directory as its base.")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge SFT LoRA into base model.")
    parser.add_argument("--base", default=BASE_MODEL_ID)
    parser.add_argument("--adapter", default=ADAPTER_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="Use 'cpu' if GPU OOMs during merge (3B in bf16 ~6GB).",
    )
    args = parser.parse_args()

    merge(
        base_model_id=args.base,
        adapter_dir=args.adapter,
        output_dir=args.output,
        device=args.device,
    )
