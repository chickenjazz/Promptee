"""
Final-merge + Hugging Face Hub upload.

Folds the DPO LoRA adapter (models/adapters/) into the SFT-merged baseline
(models/sft_baseline/) and saves a single fully-merged model that
collaborators can load with one line:

    AutoModelForCausalLM.from_pretrained("chickenjazz/promptee-3b")

No PEFT, no adapter plumbing at runtime. Runtime quantization (4-bit NF4)
still happens at load time on the consumer side.

Usage:
    # local merge only (smoke test before uploading)
    python training/merge_for_release.py

    # merge + push to HF Hub (run `huggingface-cli login` once first)
    python training/merge_for_release.py --push chickenjazz/promptee-3b

    # private repo
    python training/merge_for_release.py --push chickenjazz/promptee-3b --private

VRAM note: 3B in bf16 ~ 6 GB. Use --device cpu if your GPU OOMs.

OFFLINE-ONLY. Not imported by runtime code.
"""

import os
import sys
import shutil
import logging
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

logger = logging.getLogger("promptee.merge_release")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

SFT_BASELINE = os.path.join(PROJECT_ROOT, "models", "sft_baseline")
DPO_ADAPTER = os.path.join(PROJECT_ROOT, "models", "adapters")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "promptee-3b")
DEFAULT_REPO_ID = "chickenjazz/promptee-3b"


def merge_and_export(
    base_dir: str = SFT_BASELINE,
    adapter_dir: str = DPO_ADAPTER,
    output_dir: str = OUTPUT_DIR,
    device: str = "auto",
) -> str:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"SFT baseline not found at {base_dir}. "
            f"Run training/merge_sft_adapter.py first."
        )
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"DPO adapter not found at {adapter_dir}. "
            f"Run training/dpo_trainer.py + training/export_adapter.py first."
        )

    device_map = {"": "cpu"} if device == "cpu" else "auto"

    logger.info(f"Loading SFT baseline in bf16: {base_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    logger.info(f"Attaching DPO adapter from: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    logger.info("Merging DPO LoRA into base weights...")
    merged = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving merged model to: {output_dir}")
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    chat_template_src = os.path.join(base_dir, "chat_template.jinja")
    if os.path.exists(chat_template_src):
        shutil.copy2(chat_template_src, os.path.join(output_dir, "chat_template.jinja"))

    logger.info(f"Local merged model ready at {output_dir}")
    return output_dir


def push_to_hub(local_dir: str, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    logger.info(f"Pushing {local_dir} to https://huggingface.co/{repo_id}")
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        commit_message="Upload merged Promptee 3B (SFT + DPO)",
    )
    logger.info(f"Done. Model URL: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Merge DPO into SFT baseline and optionally push to HF Hub.")
    parser.add_argument("--base", default=SFT_BASELINE, help="Path to SFT-merged baseline.")
    parser.add_argument("--adapter", default=DPO_ADAPTER, help="Path to DPO LoRA adapter.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Local output dir for the merged model.")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto",
                        help="Use 'cpu' if your GPU OOMs during the merge (3B bf16 ~6 GB).")
    parser.add_argument("--push", nargs="?", const=DEFAULT_REPO_ID, default=None,
                        help=f"HF Hub repo id to push to. Defaults to '{DEFAULT_REPO_ID}' if flag given without value.")
    parser.add_argument("--private", action="store_true", help="Create the HF repo as private.")
    args = parser.parse_args()

    local_dir = merge_and_export(
        base_dir=args.base,
        adapter_dir=args.adapter,
        output_dir=args.output,
        device=args.device,
    )

    if args.push:
        push_to_hub(local_dir, repo_id=args.push, private=args.private)


if __name__ == "__main__":
    main()
