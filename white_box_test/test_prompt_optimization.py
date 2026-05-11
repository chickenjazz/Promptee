"""
White-box tests for the prompt optimization runtime.

Rubric item: 2. Prompt Optimization Effectiveness.

Targets the deterministic pre/post-processing logic in tools/prompt_optimizer.py.
The actual GPU model is never loaded — generation is replaced by stubs so we
can exercise the chat-template construction, generation kwargs branching,
and post-processing fallbacks in isolation.
"""

import pytest
import torch

from tools.prompt_optimizer import PromptOptimizer, _PREAMBLE_PATTERNS


# ── Preamble stripping (_clean_output) ────────────────────────────────────

@pytest.mark.parametrize(
    "raw_output, expected_substring",
    [
        ("Sure, here's the optimized prompt: WRITE a summary.", "WRITE a summary."),
        ("Here is the optimized version: Do X.", "Do X."),
        ("Certainly! Do X.", "Do X."),
        ("Of course, Do X.", "Do X."),
        ("The optimized prompt: WRITE a summary.", "WRITE a summary."),
        ('"Wrap this in quotes."', "Wrap this in quotes."),
        ("'single-quoted output'", "single-quoted output"),
        ("   Plain output with leading whitespace.   ", "Plain output with leading whitespace."),
    ],
)
def test_clean_output_strips_preambles_and_quotes(raw_output, expected_substring):
    cleaned = PromptOptimizer._clean_output(raw_output)
    assert cleaned == expected_substring, (
        f"_clean_output({raw_output!r}) = {cleaned!r}, expected {expected_substring!r}"
    )


def test_clean_output_strips_metaprompt_leakage():
    leaked = (
        "You are rewriting the prompt, not answering it. "
        "Write a summary of the article."
    )
    cleaned = PromptOptimizer._clean_output(leaked)
    assert "rewriting the prompt" not in cleaned.lower()
    assert "Write a summary" in cleaned


def test_clean_output_passthrough_when_no_preamble():
    raw = "Summarise the document in three bullets."
    assert PromptOptimizer._clean_output(raw) == raw


def test_clean_output_preserves_real_content_after_strip():
    # Regression guard: stripping a preamble must not eat the body.
    raw = "Sure, here's the refined prompt: Build a CSV exporter."
    cleaned = PromptOptimizer._clean_output(raw)
    assert "Build a CSV exporter" in cleaned


def test_preamble_patterns_compile():
    # Smoke check: every pattern must be a valid regex source string.
    import re
    for pattern in _PREAMBLE_PATTERNS:
        re.compile(pattern)


# ── rewrite() unloaded-model error path ────────────────────────────────────

def test_rewrite_raises_when_model_not_loaded():
    optimizer = PromptOptimizer(adapter_path="/nonexistent")
    with pytest.raises(RuntimeError, match="not loaded"):
        optimizer.rewrite("Some prompt.")


# ── rewrite() generation paths (stubbed model + tokenizer) ─────────────────

class _StubTokenizer:
    """Mimics the slice of HuggingFace tokenizer that PromptOptimizer.rewrite
    actually calls. Records every invocation so we can assert call shape."""

    eos_token_id = 0
    unk_token_id = 1

    def __init__(self):
        self.captured_messages = None
        self.last_decoded = ""

    def encode(self, text, add_special_tokens=False):
        # Tokens-per-character heuristic; we just need a list of ints.
        return list(range(max(1, len(text) // 4)))

    def decode(self, ids, skip_special_tokens=True):
        return self.last_decoded

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.captured_messages = messages
        return "<|chat-template-output|>"

    def convert_tokens_to_ids(self, token):
        return 2 if token == "<|im_end|>" else self.unk_token_id

    def __call__(self, text, return_tensors=None):
        # Return a tensor-like dict with .to(device) + ["input_ids"] indexing.
        return _StubInputs()


class _StubInputs(dict):
    def __init__(self):
        # Single-row, length-1 input_ids tensor.
        super().__init__()
        self["input_ids"] = torch.tensor([[42]])

    def to(self, device):
        return self


class _StubModel:
    """Records the kwargs passed to .generate() so we can branch-verify
    do_sample / temperature / top_p paths."""

    device = "cpu"

    def __init__(self, generated_text_for_decode: str = "Optimized prompt body."):
        self.last_kwargs = None
        self.generated_text_for_decode = generated_text_for_decode

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        # Output shape: 1 row, prefix len=1 (input) + suffix len=3 (generated).
        return torch.tensor([[42, 100, 101, 102]])

    def eval(self):
        return self


def _make_loaded_optimizer(generated_text: str = "Optimized prompt body."):
    optimizer = PromptOptimizer(adapter_path="/nonexistent")
    optimizer.tokenizer = _StubTokenizer()
    optimizer.tokenizer.last_decoded = generated_text
    optimizer.model = _StubModel(generated_text)
    optimizer._loaded = True
    return optimizer


def test_rewrite_constructs_system_and_user_messages():
    optimizer = _make_loaded_optimizer()
    optimizer.rewrite("Write a thing.")

    msgs = optimizer.tokenizer.captured_messages
    assert msgs is not None
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Write a thing." in msgs[1]["content"]


def test_rewrite_uses_greedy_decoding_when_temperature_is_zero():
    optimizer = _make_loaded_optimizer()
    optimizer.rewrite("Anything", temperature=0.0)

    kwargs = optimizer.model.last_kwargs
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs  # greedy path skips temperature kwarg
    assert "top_p" not in kwargs


def test_rewrite_uses_sampling_when_temperature_positive():
    optimizer = _make_loaded_optimizer()
    optimizer.rewrite("Anything", temperature=0.8, top_p=0.95)

    kwargs = optimizer.model.last_kwargs
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == pytest.approx(0.8)
    assert kwargs["top_p"] == pytest.approx(0.95)


def test_rewrite_falls_back_to_raw_on_empty_output():
    optimizer = _make_loaded_optimizer(generated_text="   ")  # whitespace-only
    raw = "Original prompt that should survive a rewrite failure."
    out = optimizer.rewrite(raw)
    assert out == raw, "empty model output must fall back to raw_prompt"


def test_rewrite_falls_back_to_raw_on_generation_exception():
    optimizer = _make_loaded_optimizer()

    def _explode(**kwargs):
        raise RuntimeError("simulated CUDA OOM")

    optimizer.model.generate = _explode
    raw = "Original prompt to preserve."
    out = optimizer.rewrite(raw)
    assert out == raw, "generation failure must fall back, not propagate"


def test_rewrite_strips_preamble_in_full_pipeline():
    optimizer = _make_loaded_optimizer(
        generated_text="Sure, here's the optimized prompt: WRITE the doc."
    )
    out = optimizer.rewrite("write the doc")
    assert out == "WRITE the doc.", out


def test_rewrite_warns_and_truncates_oversized_input(caplog):
    """When the input exceeds MAX_INPUT_TOKENS, the rewriter must log a
    truncation warning and continue — not raise."""
    import logging

    optimizer = _make_loaded_optimizer(generated_text="Truncated rewrite.")

    # Override encode to make our stub honest about token count.
    def _encode(text, add_special_tokens=False):
        return list(range(len(text.split())))

    optimizer.tokenizer.encode = _encode

    # Make decode return a deterministic stub so the truncation step doesn't
    # blow up and the post-generation decode still returns last_decoded.
    optimizer.tokenizer.decode = lambda ids, skip_special_tokens=True: (
        optimizer.tokenizer.last_decoded
    )

    huge_input = "word " * 5000  # well past MAX_INPUT_TOKENS=4096

    with caplog.at_level(logging.WARNING, logger="promptee.prompt_optimizer"):
        out = optimizer.rewrite(huge_input)

    assert any("exceeds" in r.message for r in caplog.records), (
        "expected truncation warning when input exceeds MAX_INPUT_TOKENS"
    )
    assert out == "Truncated rewrite."
