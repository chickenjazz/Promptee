import os
import sys
import re
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
from dataset_builder.prompt_templates import Archetype, detect_archetype, modularity_for

# Maximum input length (tokens) — Qwen2.5-3B supports 32k, keep headroom for generation
MAX_INPUT_TOKENS = 4096

# Generation length cap. Most archetype-routed rewrites finish in 300–500 tokens;
# 512 covers the long tail without paying for a 1024-token decode budget that's almost
# never reached. Generation time is roughly linear in output length, so this halves
# the worst-case wall-clock for long outputs.
MAX_NEW_TOKENS = 512

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
                # PyTorch's fused scaled-dot-product-attention. Numerically equivalent
                # to eager attention but 10–20% faster on a 3B model. Available on
                # any torch>=2.0 build; falls back to eager automatically if unsupported.
                attn_implementation="sdpa",
            )

            # Load LoRA adapters if available
            if os.path.exists(self.adapter_path) and os.listdir(self.adapter_path):
                logger.info(f"Loading LoRA adapters from: {self.adapter_path}")
                peft_model = PeftModel.from_pretrained(base_model, self.adapter_path)

                # Fold the LoRA deltas into the base 4-bit weights so every forward
                # pass skips the lora_A @ lora_B wrapper hop. Mathematically equivalent
                # to the un-merged path; ~5–15% faster generation. Falls back to the
                # PEFT wrapper if the merge fails (e.g., older bnb/peft combinations).
                try:
                    self.model = peft_model.merge_and_unload()
                    logger.info("LoRA adapters merged into base weights.")
                except Exception as merge_err:
                    logger.warning(
                        f"merge_and_unload() failed ({merge_err}); using PEFT wrapper instead."
                    )
                    self.model = peft_model
            else:
                logger.warning(
                    f"No adapters found in {self.adapter_path}. Using base model weights."
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
        temperature: float = 0.0,
        top_p: float = 1.0,
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
        sys_prompt = sys_prompt_override or (
            "You are a prompt refinement engine. Rewrite the user's raw prompt into a "
            "clearer, more specific, better-structured version while preserving the "
            "original intent, task, topic, and constraints.\n\n"

            "Use these section headers for modularity (ONLY those that applicable for the task; NOT ALL):\n"
            "ROLE:\n one-line expert persona starting with 'You are'.\n"
            "TASK:\n concise statement of what to produce.\n"
            "CONTEXT:\n background, audience, or purpose when needed.\n"
            "INPUTS:\n data, files, variables, or missing details.\n"
            "OUTPUTS:\n expected deliverables.\n"
            "FORMAT: bullets, table, JSON, markdown, prose, or step-by-step.\n"
            "CONSTRAINTS: rules, limits, or quality bars.\n"
            "EDGE CASES: only for code, systems, validation, or failure conditions.\n\n"

            "Rules:\n"
            "- Skip empty, generic, or redundant sections; never write 'NONE' after a header.\n"
            "- Always add ROLE header if applicable.\n"
            "- INPUTS, OUTPUTS, FORMAT, CONSTRAINTS, EDGE CASES contents should be in bulleted/list.\n"
            "- Use [Insert ...] only for genuinely missing details.\n"
            "- Prefer fewer well-filled sections over many shallow ones; order them logically.\n"
            "- Do NOT list section names before the rewrite or as a slash-separated list.\n"
            "- Output ONLY the rewritten prompt — no preamble, explanations, or closing remarks.\n"
            "- Do NOT answer, solve, or discuss the prompt; do NOT act as a chatbot.\n"
            "- Do NOT invent requirements not implied by the original, or convert the task "
            "into 'generate a prompt' or any other meta-task.\n"
        )

        user_content = user_prompt_template.format(raw_prompt) if user_prompt_template else f"Optimize this prompt: {raw_prompt}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            # Build the EOS list: the tokenizer's default EOS plus Qwen's chat
            # turn terminator. Without <|im_end|> in the stop set, generation can
            # run past a clean assistant turn and waste tokens until max_new_tokens.
            eos_ids = [self.tokenizer.eos_token_id]
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
                eos_ids.append(im_end_id)

            # Default runtime path is greedy (temperature=0.0): deterministic argmax,
            # ~5–10% faster, reproducible rewrites. The dataset builder still passes
            # temperature>0 to generate diverse training candidates — branch accordingly.
            gen_kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": eos_ids,
            }
            if temperature > 0:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

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