import os
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

# Maximum input length (tokens) — Qwen2.5-7B supports 32k, keep headroom for generation
MAX_INPUT_TOKENS = 4096

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
    Runtime prompt refinement engine using Qwen2.5-7B-Instruct with optional LoRA adapters.

    Implements the logic defined in architecture/prompt_optimizer.md:
      - 4-bit NF4 quantization via bitsandbytes
      - LoRA adapter loading from models/adapters/
      - Strict system meta-prompt enforcing refinement-only output
      - Post-processing to strip meta-commentary
      - Fallback to raw prompt on any generation failure
    """

    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
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
        sys_prompt = sys_prompt_override or (
            "You are a prompt refinement engine. Your ONLY task is to rewrite the user's "
            "prompt into a highly structured, clear, and specific command. Rules:\n"
            "1. Output ONLY the optimized prompt text — no explanations, no preamble.\n"
            "2. Do NOT answer the prompt, discuss it, or act as a chatbot.\n"
            "3. Preserve the original intent and meaning exactly.\n"
            "4. Add structure, clarity, specificity, and constraints where missing."
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
