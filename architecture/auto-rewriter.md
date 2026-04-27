# Automatic Dataset Builder Specification

## Project Name
Automatic Prompt Rewrite Dataset Builder Using Qwen2.5-7B-Instruct

## Purpose
Build an automatic dataset builder that reads raw prompts from a dataset column named `prompt`, rewrites each prompt using `Qwen2.5-7B-Instruct`, and writes the improved version into a new column named `rewritten_prompt`.

The dataset builder must preserve the original user intent while improving clarity, specificity, structure, completeness, and AI-executability.

This builder is intended for generating high-quality rewritten prompts for prompt optimization research, heuristic evaluation, and possible DPO/preference dataset preparation.

---

## Core Objective

Given a dataset with a `prompt` column, the system must:

1. Load the dataset `dataset/test_dataset.csv`.
2. Read each raw prompt from the `prompt` column.
3. Detect the prompt archetype.
4. Select the correct modularity style.
5. Diagnose the prompt's weaknesses internally.
6. Rewrite the prompt using `Qwen2.5-7B-Instruct`.
7. Preserve the original intent.
8. Avoid answering the prompt itself.
9. Write only the improved prompt into the `rewritten_prompt` column.
10. Save the updated dataset.

---

## Required Input Dataset Format

The input dataset must contain at least one column:

```csv
prompt
"write an engaging instagram caption for a travel photo"
"debug this python code"
"create a lesson plan on fractions for 4th graders"
```

Optional columns may exist, but they must not be removed unless explicitly requested.

---

## Required Output Dataset Format

The output dataset must preserve all original columns and add or update this column:

```csv
rewritten_prompt
```

Example output:

```csv
prompt,rewritten_prompt
"write an engaging instagram caption for a travel photo","Write an engaging Instagram caption for a travel photo. Use a warm, adventurous tone, keep it under 25 words, and include one relevant emoji and 3 travel-related hashtags."
```

Only the final rewritten prompt should be written to `rewritten_prompt`.

Do not write diagnostic text, archetype labels, weakness summaries, or improvement summaries into `rewritten_prompt`.

---

## Model Requirement

Use the following model:

```text
Qwen/Qwen2.5-7B-Instruct
```

The builder should support local inference using Hugging Face Transformers.

Recommended loading options:

- Use `AutoTokenizer`
- Use `AutoModelForCausalLM`
- Use GPU if available
- Use 4-bit quantization if memory is limited
- Use deterministic or low-temperature generation for consistency

Recommended generation settings:

```python
temperature = 0.3
top_p = 0.9
max_new_tokens = 512
do_sample = True
repetition_penalty = 1.05
```

For stricter consistency, allow this alternative:

```python
do_sample = False
temperature = None
top_p = None
```

---

## Required Behavior

The model must act as a prompt rewriter, not as a task answerer.

For every row:

- Input: raw text from `prompt`
- Output: improved prompt only
- Destination: `rewritten_prompt`

The model must not:

- Answer the prompt
- Generate code unless the raw prompt specifically asks for code and the rewritten prompt must describe that code request
- Add explanations outside the rewritten prompt
- Include labels such as `Archetype`, `Weaknesses Found`, or `Improvement Summary` in the dataset cell
- Change the user’s original intent
- Add unrelated requirements
- Overcomplicate simple prompts

---

## Prompt Archetypes

The system must classify each raw prompt into exactly one primary archetype.

### 1. Creative
Use when the prompt asks for:

- story
- caption
- poem
- branding
- character
- ad copy
- script
- style-based generation

Default modularity: Semi Modular

### 2. Coding
Use when the prompt asks for:

- code
- debugging
- script generation
- API implementation
- Python
- JavaScript
- SQL
- React
- backend or frontend implementation
- software development tasks

Default modularity: Full Modular

### 3. Conversational
Use when the prompt asks for:

- help me
- advice
- guide me
- coach me
- emotional support
- relationship or interpersonal guidance
- interactive assistance

Default modularity: Natural Language Modular

### 4. Structured
Use when the prompt asks for:

- plan
- checklist
- template
- roadmap
- framework
- matrix
- lesson plan
- manual
- survey
- flashcards

Default modularity: Full Modular

### 5. Analytical
Use when the prompt asks to:

- compare
- analyze
- explain
- evaluate
- discuss
- assess
- reason about concepts

Default modularity: Semi Modular or Full Modular

### 6. Concise
Use when the prompt asks for:

- quick answer
- short answer
- summary
- one sentence
- brief explanation
- simple explanation
- direct steps

Default modularity: Minimal Modular

---

## Modularity Styles

### Full Modular
Use explicit labeled sections.

Best for:

- technical tasks
- structured outputs
- multi-step tasks
- reproducible deliverables
- tasks with several constraints

Possible sections:

```text
ROLE:
TASK:
OBJECTIVE:
INPUT:
OUTPUT:
CONSTRAINTS:
EDGE CASES:
FORMAT:
```

### Semi Modular
Use grouped natural language or light structure.

Best for:

- creative tasks
- analytical tasks
- medium-complexity prompts
- prompts that need guidance without becoming too rigid

Example:

```text
Write a persuasive email to a client about the delayed project timeline. Use a professional and reassuring tone. Keep it under 200 words and include a clear call to action.
```

### Minimal Modular
Use a compact direct prompt with only the most important constraints.

Best for:

- short answers
- narrow tasks
- simple explanations
- direct requests

Example:

```text
Explain recursion in 3 simple bullet points for a beginner.
```

### Natural Language Modular
Use a conversational instruction flow without rigid labels.

Best for:

- coaching
- advice
- emotional or interpersonal prompts
- back-and-forth assistance

Example:

```text
Help me prepare for a job interview by asking one question at a time, giving feedback after each answer, and suggesting how I can improve.
```

---

## Archetype to Modularity Mapping

| Archetype | Default Modularity |
|---|---|
| Creative | Semi Modular |
| Coding | Full Modular |
| Conversational | Natural Language Modular |
| Structured | Full Modular |
| Analytical | Semi Modular or Full Modular |
| Concise | Minimal Modular |

---

## Rewrite Quality Rules

Every rewritten prompt must improve:

- clarity
- specificity
- completeness
- output instructions
- readability
- logical flow
- token efficiency
- execution reliability

Every rewritten prompt must preserve:

- original intent
- requested topic
- user constraints
- task scope
- expected output type

Every rewritten prompt must avoid:

- unnecessary verbosity
- unrelated requirements
- semantic drift
- excessive formatting
- generic filler
- answering the original task

---

## Archetype-Specific Rewrite Rules

### Creative Prompts

Use Semi Modular rewriting.

Improve:

- tone
- audience
- style
- mood
- originality
- output constraints

Avoid overly rigid section labels unless the original prompt is complex.

Example rewrite style:

```text
Write an engaging Instagram caption for a travel photo. Use an adventurous and reflective tone, keep it under 30 words, and include 3 relevant travel hashtags.
```

---

### Coding Prompts

Use Full Modular rewriting.

Recommended sections:

```text
TASK:
LANGUAGE/STACK:
INPUTS:
OUTPUT:
CONSTRAINTS:
EDGE CASES:
```

Improve:

- programming language clarity
- framework or stack details
- expected files or functions
- input/output behavior
- validation requirements
- edge cases

The rewritten prompt must ask for the code or explanation. It must not contain the actual implementation unless the raw prompt specifically includes existing code that must be preserved as input.

Example rewrite style:

```text
TASK:
Create a backend controller for uploading a file to a database.

LANGUAGE/STACK:
Use .NET Core, MySQL, and Dapper.

OUTPUT:
Provide the controller code, required model or DTO, database table structure, and a brief explanation of how the upload flow works.

CONSTRAINTS:
Validate file size and file type, handle errors clearly, and use parameterized queries.
```

---

### Conversational Prompts

Use Natural Language Modular rewriting.

Improve:

- warmth
- empathy
- interaction flow
- practical guidance
- user-centered phrasing

Example rewrite style:

```text
Help me understand how to communicate better with my development team as a non-technical startup CEO. Explain the key concepts I should know, the questions I should ask, and how I can support the team without micromanaging.
```

---

### Structured Prompts

Use Full Modular rewriting.

Recommended sections:

```text
OBJECTIVE:
SECTIONS:
FORMAT:
DETAIL LEVEL:
ORDER:
CONSTRAINTS:
```

Improve:

- expected structure
- section order
- completeness
- formatting instructions
- output usability

Example rewrite style:

```text
OBJECTIVE:
Create a comprehensive lesson plan on fractions for 4th grade students.

SECTIONS:
Include learning objectives, materials, introduction, guided practice, independent activity, assessment, and homework.

FORMAT:
Use a clear table format.

DETAIL LEVEL:
Make it detailed enough for a teacher to use directly in class.
```

---

### Analytical Prompts

Use Semi Modular or Full Modular rewriting depending on complexity.

Recommended sections for complex prompts:

```text
QUESTION:
SUBJECT:
CRITERIA:
ANALYSIS DEPTH:
OUTPUT FORMAT:
FINAL RECOMMENDATION:
```

Improve:

- comparison criteria
- reasoning depth
- conceptual scope
- final synthesis
- examples when useful

Example rewrite style:

```text
Explain the difference between mitosis and meiosis for a biology student. Compare their purpose, number of divisions, genetic outcomes, and role in organisms. Use a simple table followed by a short summary.
```

---

### Concise Prompts

Use Minimal Modular rewriting.

Improve:

- directness
- brevity
- clear output limit
- simple wording

Example rewrite style:

```text
Summarize this article in 5 bullet points, focusing only on the main ideas.
```

---

## Internal Diagnostic Process

For each prompt, the system should internally check:

1. What is the user asking for?
2. What archetype best matches the request?
3. What modularity level fits the task complexity?
4. What information is missing?
5. What constraints would make the output more reliable?
6. What structure would make the prompt easier for an AI to follow?
7. How can the prompt be improved without changing intent?

This diagnostic process must not be written to the dataset output.

---

## Required LLM Instruction Template

Use this instruction template when sending each raw prompt to Qwen.

```text
You are an expert Prompt Rewriter, Prompt Architect, and Prompt Quality Optimizer.

Your task is to rewrite the raw prompt into a clearer, more specific, better structured, and more reliable prompt while preserving the original user intent.

You must first internally detect the prompt archetype and choose the correct modularity style, but you must NOT output the archetype, diagnosis, explanation, or improvement summary.

Supported archetypes:
- Creative
- Coding
- Conversational
- Structured
- Analytical
- Concise

Default modularity rules:
- Creative: Semi Modular
- Coding: Full Modular
- Conversational: Natural Language Modular
- Structured: Full Modular
- Analytical: Semi Modular or Full Modular
- Concise: Minimal Modular

Rewrite rules:
- Improve clarity, specificity, completeness, output instructions, readability, logical flow, and token efficiency.
- Preserve the original intent, topic, constraints, and expected task.
- Do not answer the prompt.
- Do not generate the requested output.
- Do not add irrelevant requirements.
- Do not overcomplicate simple prompts.
- Do not include labels such as Archetype, Weaknesses Found, Rewritten Prompt, or Improvement Summary.
- Return only the final rewritten prompt.

Raw prompt:
{raw_prompt}

Final rewritten prompt only:
```

---

## Dataset Builder Processing Flow

The builder must follow this pipeline:

```text
START
  ↓
Load dataset file
  ↓
Validate that `prompt` column exists
  ↓
Create `rewritten_prompt` column if missing
  ↓
For each row:
    Read raw prompt
    Skip empty or invalid prompt
    Build Qwen instruction prompt
    Generate rewritten prompt
    Clean generated text
    Validate output
    Write result to `rewritten_prompt`
  ↓
Save updated dataset
  ↓
END
```

---

## Validation Rules

Before saving each generated rewrite, validate that:

1. The output is not empty.
2. The output is not identical to the raw prompt unless no improvement is possible.
3. The output does not start with common assistant filler, such as:
   - Sure
   - Here is
   - Of course
   - Certainly
   - The answer is
4. The output does not contain diagnostic labels, such as:
   - Archetype:
   - Weaknesses Found:
   - Improvement Summary:
   - Rewritten Prompt:
5. The output does not answer the task directly.
6. The output preserves the raw prompt’s main intent.
7. The output is not excessively long compared with the raw prompt unless the raw prompt is a complex coding, structured, or analytical task.

---

## Cleaning Rules

After generation, clean the output by:

- stripping leading and trailing whitespace
- removing wrapping quotation marks if unnecessary
- removing markdown code fences unless they are part of the rewritten prompt request
- removing assistant-style introductions
- removing accidental labels before the final rewritten prompt
- normalizing excessive blank lines

---

## Error Handling

The builder must handle:

- missing dataset file
- missing `prompt` column
- empty prompt cells
- model loading failure
- CUDA out-of-memory errors
- generation timeout
- invalid model output
- save failure

If a row fails, the system should not stop the entire process.

Instead, write an error message to an optional column:

```csv
rewrite_error
```

Example:

```text
Generation failed: CUDA out of memory
```

---

## Resume Support

The builder should support resuming interrupted runs.

If `rewritten_prompt` already exists and contains a non-empty value, the system should skip that row unless overwrite mode is enabled.

Recommended options:

```text
--input input.csv
--output output.csv
--overwrite false
--batch-size 1
--save-every 25
```

---

## Recommended File Structure

```text
dataset_builder/
│
├── build_rewritten_dataset.py
├── config.py
├── model_loader.py
├── prompt_templates.py
├── validators.py
├── cleaners.py
├── requirements.txt
└── README.md
```

---

## Recommended Python Dependencies

```text
pandas
transformers
accelerate
torch
bitsandbytes
sentencepiece
protobuf
```

Optional:

```text
tqdm
python-dotenv
rapidfuzz
sentence-transformers
```

---

## Recommended Implementation Notes

### Model Loading

Use 4-bit quantization if the GPU has limited VRAM.

Recommended for RTX 3070 Laptop GPU or similar 8GB VRAM devices:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

### Chat Template

Use Qwen’s chat template when available:

```python
messages = [
    {"role": "system", "content": system_instruction},
    {"role": "user", "content": user_instruction},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

---

## Example Row Transformation

### Input

```text
write a character description for a sci-fi novel protagonist
```

### Output in `rewritten_prompt`

```text
Write a vivid character description for a sci-fi novel protagonist. Include their appearance, personality, background, core motivation, internal conflict, and role in the story. Use an imaginative but grounded tone suitable for a futuristic setting.
```

---

## Acceptance Criteria

The dataset builder is complete when it can:

- Load a CSV dataset with a `prompt` column
- Generate rewritten prompts using `Qwen/Qwen2.5-7B-Instruct`
- Add or update the `rewritten_prompt` column
- Preserve all original dataset rows and columns
- Skip already completed rows when resume mode is enabled
- Clean and validate model outputs
- Save the completed dataset
- Avoid writing explanations, labels, or diagnostic text into `rewritten_prompt`
- Produce rewrites that are clearer, more specific, and semantically faithful

---

## Final Instruction for the Agentic AI

Build the automatic dataset builder exactly according to this specification.

The final system must not be a general chatbot. It must be a dataset processing tool that reads raw prompts from the `prompt` column and writes improved prompts into the `rewritten_prompt` column using `Qwen/Qwen2.5-7B-Instruct`.

The rewritten output must contain only the optimized prompt text.
