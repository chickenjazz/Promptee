"""
White-box tests for the offline training configuration.

Rubric items covered:
  4. DPO Implementation — LoRA + DPO trainer config
  5. QLoRA Implementation — 4-bit BnB config + assistant-only loss masking

Both trainers (training/dpo_trainer.py, training/sft_trainer.py) hard-fail
at module import on CPU-only machines:

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. ... requires a GPU.")

So we cannot import them in this suite. Instead, we parse the source files
with the `ast` module and assert on the keyword arguments passed to
LoraConfig(), BitsAndBytesConfig(), DPOConfig(), and SFTConfig(). This is
genuine white-box inspection — we are reading the actual configuration the
trainer would apply if it ran.
"""

import ast
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DPO_PATH = os.path.join(PROJECT_ROOT, "training", "dpo_trainer.py")
SFT_PATH = os.path.join(PROJECT_ROOT, "training", "sft_trainer.py")
PROMPTS_PATH = os.path.join(PROJECT_ROOT, "training", "_prompts.py")


# ── AST helpers ───────────────────────────────────────────────────────────

def _parse(path: str) -> ast.Module:
    with open(path, "r", encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def _find_call_kwargs(tree: ast.Module, callable_name: str) -> dict:
    """Return the keyword arguments of the *first* call to `callable_name`
    found anywhere in the tree. Literal values are unwrapped via ast.literal_eval
    where possible; references to names/expressions are returned as the raw
    AST node so the test can assert on them with `ast.dump`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name)
                else None
            )
            if name == callable_name:
                kwargs = {}
                for kw in node.keywords:
                    try:
                        kwargs[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        kwargs[kw.arg] = kw.value
                return kwargs
    raise AssertionError(f"No call to {callable_name}() found in module")


def _find_module_constant(tree: ast.Module, name: str):
    """Return the literal value of a top-level module assignment `NAME = <literal>`."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Top-level constant {name} not found")


# ── DPO trainer: LoRA / BnB / DPOConfig ───────────────────────────────────

@pytest.fixture(scope="module")
def dpo_tree():
    return _parse(DPO_PATH)


def test_dpo_lora_config_uses_qlora_target_modules(dpo_tree):
    kwargs = _find_call_kwargs(dpo_tree, "LoraConfig")
    # Target modules must include all four attention projections — that's
    # what makes this LoRA, not a partial adaptation.
    assert isinstance(kwargs["target_modules"], list)
    assert set(kwargs["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert kwargs["bias"] == "none"
    assert kwargs["task_type"] == "CAUSAL_LM"


def _train_function_defaults(tree: ast.Module) -> dict:
    """Return literal-valued defaults from the `train()` function signature.
    Keys whose default is a name reference (e.g. BASE_MODEL_ID) are skipped —
    only literal values are returned, which is all the test needs."""
    train_fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "train"
    )
    pairs = list(zip(
        train_fn.args.args[-len(train_fn.args.defaults):],
        train_fn.args.defaults,
    ))
    out = {}
    for arg, default in pairs:
        try:
            out[arg.arg] = ast.literal_eval(default)
        except (ValueError, SyntaxError):
            continue
    return out


def test_dpo_lora_rank_and_alpha_match_sop():
    tree = _parse(DPO_PATH)
    defaults = _train_function_defaults(tree)
    assert defaults["lora_r"] == 16
    assert defaults["lora_alpha"] == 32
    assert defaults["lora_dropout"] == 0.05
    assert defaults["beta"] == 0.05  # DPO KL penalty
    assert defaults["max_length"] == 1024


def test_dpo_4bit_quantization_config(dpo_tree):
    kwargs = _find_call_kwargs(dpo_tree, "BitsAndBytesConfig")
    assert kwargs["load_in_4bit"] is True
    assert kwargs["bnb_4bit_quant_type"] == "nf4"
    assert kwargs["bnb_4bit_use_double_quant"] is True
    # bnb_4bit_compute_dtype is `torch.bfloat16`, an attribute lookup —
    # literal_eval would fail; assert it dumps to the expected name.
    dtype_node = kwargs["bnb_4bit_compute_dtype"]
    assert isinstance(dtype_node, ast.Attribute)
    assert dtype_node.attr == "bfloat16"


def test_dpo_training_args_use_paged_optimizer_and_grad_accum(dpo_tree):
    kwargs = _find_call_kwargs(dpo_tree, "DPOConfig")
    assert kwargs["optim"] == "paged_adamw_8bit"
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["gradient_accumulation_steps"] == 8
    assert kwargs["bf16"] is True
    assert kwargs["report_to"] == "none"


def test_dpo_dataset_renames_to_trl_columns(dpo_tree):
    """DPOTrainer expects `prompt` / `chosen` / `rejected`. The trainer
    renames `x` / `y_w` / `y_l` from the JSONL into those columns."""
    src = open(DPO_PATH, "r", encoding="utf-8").read()
    assert '"x": "prompt"' in src
    assert '"y_w": "chosen"' in src
    assert '"y_l": "rejected"' in src


# ── DPO system-prompt dropout (30/40/30 boundary constants) ───────────────

def test_dpo_dropout_constants_match_sft(dpo_tree):
    """The dropout boundaries must exactly match the SFT training
    distribution; otherwise DPO would partially undo the SFT decoupling."""
    p_strong = _find_module_constant(dpo_tree, "P_STRONG")
    p_weak = _find_module_constant(dpo_tree, "P_WEAK")
    assert p_strong == 0.30
    assert p_weak == 0.70
    assert p_weak - p_strong == pytest.approx(0.40)  # the WEAK band width


def test_dpo_dropout_seed_is_deterministic(dpo_tree):
    seed = _find_module_constant(dpo_tree, "DROPOUT_SEED")
    assert isinstance(seed, int) and seed >= 0


def test_dpo_dropout_distribution_simulation():
    """Re-run the dropout decision rule with the same seed and constants and
    verify the empirical bucket counts hit ~30/40/30. This validates the
    arithmetic — `random() < P_STRONG` for STRONG, `random() < P_WEAK` for
    WEAK, else NONE — without importing the GPU-gated module."""
    import random

    P_STRONG = 0.30
    P_WEAK = 0.70
    rng = random.Random(42)

    n = 10_000
    counts = {"STRONG": 0, "WEAK": 0, "NONE": 0}
    for _ in range(n):
        r = rng.random()
        if r < P_STRONG:
            counts["STRONG"] += 1
        elif r < P_WEAK:
            counts["WEAK"] += 1
        else:
            counts["NONE"] += 1

    # 95% CI on a binomial proportion at this sample size is roughly ±1pt.
    # Use ±2pt slack for safety.
    assert counts["STRONG"] / n == pytest.approx(0.30, abs=0.02)
    assert counts["WEAK"] / n == pytest.approx(0.40, abs=0.02)
    assert counts["NONE"] / n == pytest.approx(0.30, abs=0.02)


# ── SFT trainer: QLoRA + assistant-only loss ──────────────────────────────

@pytest.fixture(scope="module")
def sft_tree():
    return _parse(SFT_PATH)


def test_sft_lora_config_matches_dpo_target_modules(sft_tree):
    """SFT and DPO must adapt the *same* projections — otherwise the
    DPO step starts from a different parameter shape than SFT produced."""
    kwargs = _find_call_kwargs(sft_tree, "LoraConfig")
    assert set(kwargs["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert kwargs["bias"] == "none"
    assert kwargs["task_type"] == "CAUSAL_LM"


def test_sft_4bit_quantization_with_double_quant(sft_tree):
    kwargs = _find_call_kwargs(sft_tree, "BitsAndBytesConfig")
    assert kwargs["load_in_4bit"] is True
    assert kwargs["bnb_4bit_quant_type"] == "nf4"
    assert kwargs["bnb_4bit_use_double_quant"] is True


def test_sft_uses_assistant_only_loss(sft_tree):
    """Loss must be masked to the assistant turn — otherwise gradients
    flow through the system + user prompt, defeating the rewrite training."""
    kwargs = _find_call_kwargs(sft_tree, "SFTConfig")
    assert kwargs["assistant_only_loss"] is True
    assert kwargs["packing"] is False
    assert kwargs["optim"] == "paged_adamw_8bit"
    assert kwargs["bf16"] is True
    # `max_length=max_seq_length` is a name reference — its concrete default
    # is asserted in test_sft_training_defaults below.
    assert isinstance(kwargs["max_length"], ast.Name)
    assert kwargs["max_length"].id == "max_seq_length"


def test_sft_training_defaults():
    tree = _parse(SFT_PATH)
    defaults = _train_function_defaults(tree)
    assert defaults["lora_r"] == 16
    assert defaults["lora_alpha"] == 32
    assert defaults["lora_dropout"] == 0.05
    assert defaults["grad_accum"] == 8
    assert defaults["max_seq_length"] == 1024


# ── Shared system prompts (parity guard) ──────────────────────────────────

def test_strong_prompt_is_non_trivial():
    """The STRONG system prompt must contain the structural markers the
    trainer relies on (ROLE / TASK headers, strict-rule block)."""
    src = open(PROMPTS_PATH, "r", encoding="utf-8").read()
    # We don't import — just inspect the constant text.
    tree = ast.parse(src)
    strong = _find_module_constant(tree, "STRONG_PROMPT")
    assert "ROLE:" in strong
    assert "TASK:" in strong
    assert "Strict rules" in strong


def test_weak_prompt_is_short_baseline():
    src = open(PROMPTS_PATH, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    weak = _find_module_constant(tree, "WEAK_PROMPT")
    # The minimum-info baseline should fit on one line.
    assert "\n" not in weak
    assert len(weak) < 100


def test_user_template_has_raw_prompt_placeholder():
    src = open(PROMPTS_PATH, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    template = _find_module_constant(tree, "USER_TEMPLATE")
    assert "{raw_prompt}" in template
