import os
import sys
import re
import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure structured logging
logger = logging.getLogger("promptee.prompt_optimizer")

# Resolve adapter path relative to project root, not cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "adapters")

# Make the dataset_builder package importable so the runtime meta-prompt can
# reuse the same archetype classifier the training data was generated with.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dataset_builder.prompt_templates import (
    Archetype,
    detect_archetype,
    modularity_for,
    _BASE_SYSTEM,
    _ARCHETYPE_GUIDANCE,
)

# Maximum input length (tokens) — Qwen2.5-3B supports 32k, keep headroom for generation
MAX_INPUT_TOKENS = 4096

# --- Archetype-aware system prompts ---------------------------------------
# Mirrors the modularity rules in dataset_builder/prompt_templates.py so the
# runtime meta-prompt matches the regime the DPO `chosen` outputs were drawn
# from. The classifier returns one of six archetypes; each maps to a scaffold
# the model is asked to emit.

_FULL_MODULAR_TEMPLATE = (
    "ROLE: <one-line expert persona appropriate for the task>\n"
    "\n"
    "TASK:\n"
    "<concise restatement of what to produce>\n"
    "\n"
    "INPUTS:\n"
    "- <bulleted inputs the user must fill in, using [Insert ...] placeholders where unspecified>\n"
    "\n"
    "OUTPUTS:\n"
    "<numbered or bulleted list of concrete deliverables>\n"
    "\n"
    "CONSTRAINTS:\n"
    "- <bulleted rules, best practices, or quality bars>"
)

_SEMI_MODULAR_TEMPLATE = (
    "ROLE: <one-line expert persona>\n"
    "\n"
    "TASK:\n"
    "<concise restatement of what to produce>\n"
    "\n"
    "REQUIREMENTS:\n"
    "- <bulleted requirements, constraints, or stylistic guidance>"
)


# Archetype-gated few-shot exemplars. Each shows ONE input -> correct rewrite,
# anchoring the model on "rewrite, do not answer" for the two archetypes where
# that failure mode is most common (Coding: write-the-function trap;
# Analytical: explain-instead-of-asking-for-explanation trap). Use [Insert ...]
# placeholders so the model copies STRUCTURE, not content.
_CODING_FEWSHOT = (
    "Example — input prompt:\n"
    "  Write a Python function that sorts a list.\n"
    "Example — CORRECT rewrite (asks for the function, does not contain code):\n"
    "  ROLE: You are a senior Python engineer.\n"
    "  TASK:\n"
    "  Write a Python function that sorts [Insert list element type] by [Insert sort key].\n"
    "  INPUTS:\n"
    "  - The unsorted list\n"
    "  OUTPUTS:\n"
    "  - A function that returns the sorted list\n"
    "  CONSTRAINTS:\n"
    "  - [Insert performance / stability requirements]\n"
    "Example — INCORRECT (do NOT do this — this is the user's job, not yours):\n"
    "  def sort_list(items): return sorted(items)"
)

_ANALYTICAL_FEWSHOT = (
    "Example — input prompt:\n"
    "  Do you know what gradient checkpointing is?\n"
    "Example — CORRECT rewrite (asks for an explanation, does not give one):\n"
    "  ROLE: You are a senior ML engineer.\n"
    "  TASK:\n"
    "  Explain gradient checkpointing — what it is, why it matters, "
    "and when to use it.\n"
    "  REQUIREMENTS:\n"
    "  - Cover the memory-vs-compute tradeoff explicitly.\n"
    "  - [Insert audience and depth level]\n"
    "  - Include one concrete example or use case.\n"
    "Example — INCORRECT (do NOT do this — answering is the user's job, not yours):\n"
    "  Yes, gradient checkpointing is a technique that trades extra "
    "computation for reduced memory by recomputing activations during "
    "the backward pass instead of storing them all..."
)


def _build_system_prompt(archetype: Archetype) -> str:
    """Return the archetype-specific system meta-prompt.

    Composed from training-time `_BASE_SYSTEM` + a runtime scaffold + the
    training-time `_ARCHETYPE_GUIDANCE` for this archetype, so the tokens
    the LoRA adapter was DPO-trained against remain anchored at runtime.
    """
    if archetype in (Archetype.CODING, Archetype.STRUCTURED):
        scaffold = (
            "Modularity: Full Modular. Output the rewrite using this exact section "
            "format, in this order:\n\n"
            f"{_FULL_MODULAR_TEMPLATE}\n\n"
            "You may add an EDGE CASES or BEST PRACTICES section after CONSTRAINTS "
            "only if clearly warranted by the task."
        )
    elif archetype in (Archetype.CREATIVE, Archetype.ANALYTICAL):
        scaffold = (
            "Modularity: Semi Modular. Output a structured rewrite using these sections:\n\n"
            f"{_SEMI_MODULAR_TEMPLATE}\n\n"
            "Keep the rewrite expressive — do not force [Insert ...] placeholders "
            "unless something is genuinely missing."
        )
    elif archetype == Archetype.CONVERSATIONAL:
        scaffold = (
            "Modularity: Natural Language Modular. Output the rewrite as one or two "
            "flowing paragraphs of natural English — no labeled sections, no ROLE / "
            "TASK / OUTPUTS headers, no bullet scaffolding. The rewrite should read "
            "like a thoughtfully phrased request, with role, context, tone, and any "
            "constraints woven in naturally."
        )
    elif archetype == Archetype.CONCISE:
        scaffold = (
            "Modularity: Minimal Modular. Output the rewrite as a single sharpened "
            "command or question — ideally one sentence, at most two. No labels, no "
            "bullets, no scaffolding. Add only the minimum specificity the original "
            "lacked."
        )
    else:
        scaffold = (
            "Modularity: Semi Modular. Output a structured rewrite using these sections:\n\n"
            f"{_SEMI_MODULAR_TEMPLATE}"
        )

    guidance = _ARCHETYPE_GUIDANCE.get(archetype, "")

    # Inject one archetype-matched few-shot exemplar for the two archetypes
    # where the answer-instead-of-rewrite failure mode is most common.
    if archetype == Archetype.CODING:
        fewshot = f"\n\n{_CODING_FEWSHOT}"
    elif archetype == Archetype.ANALYTICAL:
        fewshot = f"\n\n{_ANALYTICAL_FEWSHOT}"
    else:
        fewshot = ""

    return (
        f"{_BASE_SYSTEM}\n\n"
        f"{scaffold}\n\n"
        f"Archetype hint for this rewrite (do not echo this back):\n{guidance}"
        f"{fewshot}"
    )


# ───────────────────────────────────────────────────────────────────────
# Answer-shape detector — heuristic check that catches the failure mode
# where the model produced an actual answer/code instead of a rewritten
# prompt. Signals are computed on the candidate AFTER subtracting tokens
# present in the raw prompt, so we don't false-positive when the rewrite
# legitimately quotes the user's own code.
# ───────────────────────────────────────────────────────────────────────

_ANSWER_SHAPE_LINE_STARTS = re.compile(
    r"^\s*(?:def\s+\w+|class\s+\w+|function\s+\w+|"
    r"import\s+[\w.]+|from\s+[\w.]+\s+import|"
    r"SELECT\s+[\w*,\s]|"
    r"const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=|public\s+\w+|"
    r"#include|<\?php|<\w+[\s>])",
    re.IGNORECASE | re.MULTILINE,
)
_ANSWER_PREAMBLES = re.compile(
    r"^\s*(?:Yes[,!.]|No[,!.]|Sure[,!.]?\s*here|Here'?s|Here is|"
    r"Of course[,!.]|Certainly[,!.]|Absolutely[,!.])",
    re.IGNORECASE,
)
_REWRITE_MARKERS = re.compile(
    r"\[Insert[^\]]+\]|"
    r"\bROLE\s*:|^\s*TASK\s*:|^\s*INPUTS?\s*:|^\s*OUTPUTS?\s*:|"
    r"^\s*CONSTRAINTS?\s*:|^\s*REQUIREMENTS?\s*:",
    re.MULTILINE,
)


def detect_answer_shape(raw: str, candidate: str) -> tuple[bool, list[str]]:
    """Return (is_answer_shaped, reasons).

    Heuristics run on `candidate`; signals are suppressed when they're
    already present in `raw` so legitimate code-quoting doesn't false-
    positive. A rewrite is flagged when it shows answer signals AND lacks
    rewrite markers (ROLE/TASK/REQUIREMENTS/INPUTS/OUTPUTS or [Insert ...]).
    Any 2+ signals also fires regardless of marker presence.
    """
    reasons: list[str] = []
    if not candidate.strip():
        return False, reasons

    raw_lower = raw.lower()
    cand = candidate
    has_rewrite_markers = bool(_REWRITE_MARKERS.search(cand))

    # 1. Fenced code block introduced (not present in raw)
    if "```" in cand and "```" not in raw:
        reasons.append("fenced_code_block_introduced")

    # 2. Lines beginning with code-like patterns (def/class/import/SELECT/...)
    #    Only count lines whose content isn't a substring of raw.
    code_line_hits = 0
    for m in _ANSWER_SHAPE_LINE_STARTS.finditer(cand):
        line = m.group(0).strip().lower()
        if line and line not in raw_lower:
            code_line_hits += 1
    if code_line_hits >= 1:
        reasons.append(f"code_like_line_starts={code_line_hits}")

    # 3. Sentence-initial answer preamble (Yes,/No,/Sure,/Here is...)
    if _ANSWER_PREAMBLES.match(cand) and not _ANSWER_PREAMBLES.match(raw):
        reasons.append("answer_preamble")

    # 4. Long output with no rewrite markers — usually an answer in disguise.
    token_count = len(cand.split())
    if token_count > 80 and not has_rewrite_markers:
        reasons.append(f"long_unstructured_output_tokens={token_count}")

    if not reasons:
        return False, reasons

    # Trigger:
    #   (a) any answer signal + no rewrite markers (no scaffolding to anchor it
    #       as a rewrite), or
    #   (b) two or more signals regardless of markers (strong evidence even
    #       if the model decorated an answer with section labels).
    triggered = (not has_rewrite_markers) or (len(reasons) >= 2)
    if triggered and not has_rewrite_markers:
        reasons.append("no_rewrite_markers")
    return triggered, reasons


# Common LLM preamble patterns to strip from output
_PREAMBLE_PATTERNS = [
    r"^(?:Sure[,!.]?\s*)?[Hh]ere(?:'s| is) (?:the |an? )?(?:optimized|refined|improved|rewritten|updated) (?:version|prompt|text)[:\s]*",
    r"^(?:Sure[,!.]?\s*)?[Hh]ere(?:'s| is) (?:the |a )?(?:result|output)[:\s]*",
    r"^(?:The )?[Oo]ptimized prompt[:\s]*",
    r"^(?:The )?[Rr]efined prompt[:\s]*",
    r"^[Cc]ertainly[!,.]?\s*",
    r"^[Oo]f course[!,.]?\s*",
]


class PromptOptimizer:
    """
    Runtime prompt refinement engine using Qwen2.5-3B-Instruct with optional LoRA adapters.

    Implements the logic defined in architecture/prompt_optimizer.md:
      - 4-bit NF4 quantization via bitsandbytes
      - LoRA adapter loading from models/adapters/
      - Strict system meta-prompt enforcing refinement-only output
      - Post-processing to strip meta-commentary
      - Fallback to raw prompt on any generation failure
    """

    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: str = DEFAULT_ADAPTER_PATH,
    ):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load_model(self) -> None:
        """
        Load the base model with 4-bit quantization and optional LoRA adapters.
        Raises RuntimeError if loading fails critically.
        """
        try:
            logger.info(f"System Check | PyTorch CUDA available: {torch.cuda.is_available()}")
            logger.info(f"Loading tokenizer: {self.base_model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id, trust_remote_code=True
            )

            logger.info(f"Loading model: {self.base_model_id} (4-bit NF4)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Load LoRA adapters if available
            if os.path.exists(self.adapter_path) and os.listdir(self.adapter_path):
                logger.info(f"Loading LoRA adapters from: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
                self._log_adapter_self_check()
            else:
                logger.warning(
                    f"ADAPTER CHECK FAIL | No adapters found in {self.adapter_path}. "
                    f"Using base model weights — rewriter will be significantly weaker."
                )
                self.model = base_model

            self.model.eval()
            self._loaded = True
            logger.info("Optimizer engine loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load optimizer model: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            self._loaded = False
            raise RuntimeError(
                f"PromptOptimizer failed to initialize: {e}. "
                f"Ensure the model '{self.base_model_id}' is accessible and GPU memory is sufficient."
            ) from e

    def _log_adapter_self_check(self) -> None:
        """Verify the LoRA adapter is actually attached to the base model.

        Logs a single PASS/FAIL line with base model id, adapter base id,
        and the count of LoRA-wrapped modules. Catches silent mismatches
        where PeftModel.from_pretrained accepts an adapter trained against
        a different base than the one we just loaded.
        """
        try:
            cfg_path = os.path.join(self.adapter_path, "adapter_config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            adapter_base = cfg.get("base_model_name_or_path", "<unknown>")
            target_modules = cfg.get("target_modules", []) or []

            lora_module_count = sum(
                1 for _ in self.model.modules()
                if type(_).__name__.startswith("Lora")
            )

            base_match = adapter_base == self.base_model_id
            attached = lora_module_count > 0
            status = "PASS" if (base_match and attached) else "FAIL"

            logger.info(
                f"ADAPTER CHECK {status} | runtime_base={self.base_model_id} "
                f"adapter_base={adapter_base} base_match={base_match} "
                f"lora_modules_attached={lora_module_count} "
                f"target_modules={target_modules}"
            )
            if not base_match:
                logger.warning(
                    "Adapter base model does not match runtime base. The LoRA "
                    "weights will not align correctly — expect degraded rewrites."
                )
            if not attached:
                logger.warning(
                    "PeftModel.from_pretrained returned a model with no Lora "
                    "modules attached. Adapter is effectively bypassed."
                )
        except Exception as e:
            logger.warning(f"ADAPTER CHECK SKIPPED | self-check failed: {e}")

    @staticmethod
    def _clean_output(text: str) -> str:
        """
        Strip common LLM meta-commentary preambles from generated output.
        
        SOP §3B.4: "The output is decoded and cleaned of any meta-commentary 
        (the model must ONLY return the optimized text)."
        """
        cleaned = text.strip()

        # Strip each known preamble pattern
        for pattern in _PREAMBLE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, count=1).strip()

        # Remove wrapping quotes if the entire output is quoted
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        return cleaned

    def rewrite(
        self, 
        raw_prompt: str, 
        sys_prompt_override: str = None, 
        user_prompt_template: str = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """
        Transform a raw prompt into an optimized version.

        Returns the original raw_prompt as fallback if:
          - The model is not loaded
          - Generation fails
          - The output is empty after cleaning

        Raises:
            RuntimeError: If the model was never loaded (call load_model() first).
        """
        # Explicit error when model not initialized — no silent mock
        if not self._loaded or not self.model or not self.tokenizer:
            raise RuntimeError(
                "PromptOptimizer model is not loaded. Call load_model() before rewrite()."
            )

        # Input length guard (SOP §4 Edge Case: over-verbosity)
        input_tokens = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(input_tokens) > MAX_INPUT_TOKENS:
            logger.warning(
                f"Input prompt exceeds {MAX_INPUT_TOKENS} tokens ({len(input_tokens)}). "
                f"Truncating to fit context window."
            )
            input_tokens = input_tokens[:MAX_INPUT_TOKENS]
            raw_prompt = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

        # System meta-prompt (SOP §3B.1)
        # --- OLD generic meta-prompt (kept for reference) ---
        # sys_prompt = sys_prompt_override or (
        #     "You are a prompt refinement engine. Your ONLY task is to rewrite the user's "
        #     "prompt into a highly structured, clear, and specific command. Rules:\n"
        #     "1. Output ONLY the optimized prompt text — no explanations, no preamble.\n"
        #     "2. Do NOT answer the prompt, discuss it, or act as a chatbot.\n"
        #     "3. Preserve the original intent and meaning exactly.\n"
        #     "4. Add structure, clarity, specificity, and constraints where missing."
        # )

        # Archetype-aware dynamic meta-prompt — mirrors dataset_builder/prompt_templates.py
        # so the runtime asks for the same modularity the DPO data was generated under.
        if sys_prompt_override is not None:
            sys_prompt = sys_prompt_override
            archetype = None
        else:
            archetype = detect_archetype(raw_prompt)
            sys_prompt = _build_system_prompt(archetype)
            logger.info(
                f"Archetype routed: {archetype.value} "
                f"(modularity={modularity_for(archetype).value})"
            )

        if user_prompt_template:
            user_content = user_prompt_template.format(raw_prompt)
        else:
            user_content = (
                "Rewrite the following prompt to be clearer and more specific. "
                "Preserve every named entity, constraint, language, framework, "
                "count, file path, and requirement from the original. "
                "Output only the rewritten prompt — do not answer it.\n\n"
                "Original prompt:\n"
                f"{raw_prompt}\n\n"
                "Final rewritten prompt only:"
            )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated portion
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Clean meta-commentary (SOP §3B.4)
            optimized_text = self._clean_output(raw_output)

            # Empty output guard — fallback to original
            if not optimized_text.strip():
                logger.warning(
                    "Model generated empty output after cleaning. Falling back to raw prompt."
                )
                return raw_prompt

            return optimized_text

        except Exception as e:
            # SOP §4 Fallback: return original prompt and log the failure
            logger.error(
                f"Generation failed, falling back to raw prompt: {e}", exc_info=True
            )
            return raw_prompt
