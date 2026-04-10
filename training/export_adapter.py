"""
Adapter Export — Checkpoint to Inference-Ready Weights

SOP Reference: architecture/dpo_training.md §6

Loads a training checkpoint and exports adapter weights
to models/adapters/ for use by the runtime PromptOptimizer.

This script is OFFLINE-ONLY. It must never be imported by runtime code.
"""

import os
import sys
import shutil
import logging
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

logger = logging.getLogger("promptee.export_adapter")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints", "final")
ADAPTER_OUTPUT = os.path.join(PROJECT_ROOT, "models", "adapters")

# Files that constitute a complete LoRA adapter
ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
]
ADAPTER_FILES_ALT = [
    "adapter_config.json",
    "adapter_model.bin",
]


def export(
    checkpoint_dir: str = CHECKPOINT_DIR,
    adapter_output: str = ADAPTER_OUTPUT,
    verify: bool = True,
):
    """
    Export adapter weights from a training checkpoint to the runtime directory.

    Args:
        checkpoint_dir: Path to training checkpoint (e.g., training/checkpoints/final)
        adapter_output: Path to runtime adapter directory (e.g., models/adapters)
        verify: If True, verify the exported adapter loads correctly
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}. "
            f"Run training/dpo_trainer.py first."
        )

    # Check for adapter files
    has_safetensors = all(
        os.path.exists(os.path.join(checkpoint_dir, f)) for f in ADAPTER_FILES
    )
    has_bin = all(
        os.path.exists(os.path.join(checkpoint_dir, f)) for f in ADAPTER_FILES_ALT
    )

    if not has_safetensors and not has_bin:
        raise FileNotFoundError(
            f"No valid adapter files found in {checkpoint_dir}. "
            f"Expected: {ADAPTER_FILES} or {ADAPTER_FILES_ALT}"
        )

    files_to_copy = ADAPTER_FILES if has_safetensors else ADAPTER_FILES_ALT

    # Create output directory
    os.makedirs(adapter_output, exist_ok=True)

    # Copy adapter files
    for filename in files_to_copy:
        src = os.path.join(checkpoint_dir, filename)
        dst = os.path.join(adapter_output, filename)
        shutil.copy2(src, dst)
        logger.info(f"Copied: {src} -> {dst}")

    # Also copy tokenizer files if present (for consistency)
    for tok_file in ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]:
        src = os.path.join(checkpoint_dir, tok_file)
        if os.path.exists(src):
            dst = os.path.join(adapter_output, tok_file)
            shutil.copy2(src, dst)
            logger.info(f"Copied tokenizer file: {tok_file}")

    logger.info(f"Adapter exported to: {adapter_output}")

    # Verification step
    if verify:
        logger.info("Verifying exported adapter...")
        try:
            from peft import PeftConfig

            config = PeftConfig.from_pretrained(adapter_output)
            logger.info(
                f"Adapter config verified. Base model: {config.base_model_name_or_path}, "
                f"LoRA rank: {config.r}"
            )
            logger.info("Export verification PASSED.")
        except Exception as e:
            logger.error(f"Export verification FAILED: {e}")
            raise

    return adapter_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LoRA adapter to runtime")
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_DIR,
        help="Path to training checkpoint",
    )
    parser.add_argument(
        "--output",
        default=ADAPTER_OUTPUT,
        help="Path to export adapter weights",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip adapter verification after export",
    )
    args = parser.parse_args()

    export(
        checkpoint_dir=args.checkpoint,
        adapter_output=args.output,
        verify=not args.no_verify,
    )
