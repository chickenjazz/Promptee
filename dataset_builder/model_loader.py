"""Qwen2.5-7B-Instruct loading + generation helpers.

4-bit nf4 quantization is the default; an 8GB-class GPU (e.g. RTX 3070) can
host the model with this config. bfloat16 compute dtype matches what
tools/prompt_optimizer.py uses successfully on this hardware.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dataset_builder.config import MODEL_NAME

logger = logging.getLogger("dataset_builder.model")


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    load_in_4bit: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen2.5-7B-Instruct with optional 4-bit quantization.

    Returns (model, tokenizer). Caller owns the lifecycle. Raises on failure.
    """
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs: Dict[str, object] = {}
    if load_in_4bit:
        logger.info("Loading model: %s (4-bit nf4)", model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        quant_kwargs["quantization_config"] = bnb_config
    else:
        logger.info("Loading model: %s (full precision)", model_name)
        quant_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        **quant_kwargs,
    )
    model.eval()
    return model, tokenizer


def format_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Apply Qwen's chat template with the generation prompt suffix."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
) -> str:
    """Generate continuation text. Returns only the newly generated tokens, decoded."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    gen_kwargs: Dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Slice off the input ids so we only decode newly generated tokens.
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
