"""Prompt plan for generating `losing_prompts` (DPO rejected samples).

The output is a *weaker but still plausible* rewrite of the raw prompt — better
than the raw prompt, but deliberately worse than the chosen `rewritten_prompt`.
This gives DPO a meaningful preference gradient between y_w and y_l.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_builder.prompt_templates import Archetype, detect_archetype


_LOSING_SYSTEM = """"You are a prompt refinement engine. Your ONLY task is to rewrite the user's "
            "prompt into a more structured, clear, and specific command. Rules:\n"
            "1. Output ONLY the refined prompt text — no explanations, no preamble.\n"
            "2. Do NOT answer the prompt, discuss it, or act as a chatbot.\n"
            "3. Preserve the original intent and meaning exactly.\n"
            "4. Add structure, clarity, specificity, and constraints where missing."""

@dataclass(frozen=True)
class LosingPromptPlan:
    archetype: Archetype
    system_instruction: str
    user_message: str


def build_losing_plan(raw_prompt: str, chosen_rewrite: str) -> LosingPromptPlan:
    """Build the system + user messages for generating a losing rewrite.

    The chosen rewrite is shown to the model only as a *quality ceiling* — the
    model is told not to match it. We do not want the losing sample to drift
    toward the winning sample.
    """
    archetype = detect_archetype(raw_prompt)
    user = (
        "Raw prompt (this is what the user originally wrote):\n"
        f"{raw_prompt.strip()}\n\n"
        "Reference expert rewrite (DO NOT copy, DO NOT match this quality — your output must be clearly weaker than this):\n"
        f"{chosen_rewrite.strip()}\n\n"
        "Now produce a single weaker rewrite of the raw prompt, following all the rules. "
        "Output only the rewrite text:"
    )
    return LosingPromptPlan(archetype, _LOSING_SYSTEM, user)
