"""Guard against STRONG_PROMPT drift between training and runtime.

The training pipeline (build_sft_dataset.py, dpo_trainer.py) and the
runtime engine (tools/prompt_optimizer.py) must use byte-identical strong
system prompts. If they diverge, the model trains on one distribution and
sees a different one at inference — exactly the prompt-dependency failure
this whole SFT-then-DPO pipeline is designed to fix.
"""

import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training._prompts import STRONG_PROMPT


def test_strong_prompt_matches_runtime_default():
    """Extract the default sys_prompt literal from prompt_optimizer.py and
    confirm it equals training._prompts.STRONG_PROMPT character-for-character.
    """
    optimizer_src_path = os.path.join(PROJECT_ROOT, "tools", "prompt_optimizer.py")
    with open(optimizer_src_path, "r", encoding="utf-8") as f:
        src = f.read()

    # Match the sys_prompt = sys_prompt_override or ( ... ) block, capturing
    # the parenthesised string-concatenation literal.
    pattern = re.compile(
        r"sys_prompt\s*=\s*sys_prompt_override\s*or\s*\((?P<body>.*?)\)\s*\n",
        re.DOTALL,
    )
    match = pattern.search(src)
    assert match, "Could not locate the default sys_prompt literal in prompt_optimizer.py"

    body = match.group("body")
    runtime_prompt = _eval_string_concat(body)

    assert runtime_prompt == STRONG_PROMPT, (
        "STRONG_PROMPT in training/_prompts.py has drifted from the runtime "
        "default in tools/prompt_optimizer.py. Re-sync them — otherwise the "
        "SFT/DPO training distribution stops matching inference."
    )


def _eval_string_concat(body: str) -> str:
    """Evaluate a parenthesised series of string literals (no expressions, no
    f-strings) into a single concatenated string. Stricter than ast.literal_eval
    because the source uses implicit adjacency concatenation.
    """
    import ast

    wrapped = "(" + body + ")"
    tree = ast.parse(wrapped, mode="eval")
    node = tree.body
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # Fall back to compiling and evaluating in an empty namespace — safe because
    # the source contains only string literals at this point.
    return eval(compile(tree, "<runtime-prompt>", "eval"), {"__builtins__": {}})
