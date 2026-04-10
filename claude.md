# System Instructions: Prompt Optimization Backend Pipeline

## Project Overview

Build a **research-aligned Python backend system** that implements the architecture defined in the thesis:

**Prompt Optimization Pipeline Using Multi-Criteria Heuristic Evaluation and Transformer Fine-Tuning Algorithms**

The backend must support both:

- runtime prompt optimization inference pipeline
- offline transformer fine-tuning workflow

The system must follow the manuscript architecture exactly.
Do NOT substitute alternative architectures unless explicitly marked optional.
Runtime inference must remain lightweight.
Training must remain fully offline.

---

# System Objective

Implement a backend pipeline capable of:

1. heuristic scoring of raw prompts
2. transformer-based prompt rewriting
3. post-optimization scoring
4. comparison storage
5. external LLM evaluation calls
6. preference pair dataset generation
7. offline Direct Preference Optimization (DPO)
8. QLoRA adapter fine-tuning
9. FastAPI orchestration layer

---

# Architecture Requirements

The backend must follow this layered structure:

Presentation Layer → React Frontend  
Application Layer → FastAPI Backend  
Model Layer → Fine-tuned Qwen-Instruct-7B + LoRA adapters  
Offline Training Layer → QLoRA + DPO pipeline

Rules:

- Runtime inference must remain lightweight
- Training must remain offline
- Do NOT merge runtime logic and training logic
- Training pipeline must execute independently

---

# Model Requirements

Base Model:
Qwen-Instruct-7B

Fine-Tuning Method:
QLoRA

Alignment Method:
Direct Preference Optimization (DPO)

Adapters:
LoRA adapters injected into transformer attention layers

---

# QLoRA Configuration Requirements

Training configuration must include:

- 4-bit NormalFloat quantization (NF4)
- double quantization enabled
- bfloat16 compute dtype
- freeze base transformer weights
- train adapter matrices only
- paged optimizer required
- gradient checkpointing required

---

# DPO Training Requirements

Preference dataset format:

(x, y_w, y_l)

Where:

x = raw prompt  
y_w = preferred rewrite  
y_l = rejected rewrite

Optimization objective:

maximize log P(y_w | x) > log P(y_l | x)

---

# Dataset Generation Requirements

Preference dataset must be generated automatically from the heuristic scoring engine.

Dataset source:

existing raw prompt dataset

Rewrite candidates:

must be modularized

Heuristic metrics:

- Clarity
- Specificity
- Semantic Preservation

Semantic constraint:

cosine similarity threshold between original and rewritten prompts

---

# Heuristic Engine Requirements

Component:

HeuristicScorer

Responsibilities:

- compute clarity score
- compute specificity score
- compute semantic preservation score
- compute weighted combined score

Reject candidate rewrites below similarity threshold

---

# Prompt Optimization Engine Requirements

Component:

PromptOptimizer

Responsibilities:

- load fine-tuned Qwen-Instruct-7B
- apply LoRA adapters
- rewrite prompts
- return optimized prompt output

---

# Service Orchestration Requirements

Component:

PromptService

Responsibilities:

- receive prompt input
- call heuristic scorer
- call optimizer
- call post-optimization scorer
- call external LLM evaluation service
- aggregate structured response

---

# Controller Requirements

Component:

PromptController

Endpoint:

POST /optimize_prompt

Responsibilities:

- receive raw prompt
- forward request to PromptService
- return structured response object

---

# Response Object Format

Response payload must contain:

- raw_prompt
- optimized_prompt
- raw_score
- optimized_score
- external_llm_response_raw
- external_llm_response_optimized
- improvement_score

---

# External LLM Service Requirements

Component:

ExternalLLMService

Responsibilities:

- accept raw prompt
- accept optimized prompt
- generate comparison outputs
- return evaluation responses

---

# Offline Training Pipeline Requirements

Training pipeline must exist separately from runtime backend.

Training workflow:

1. load base Qwen-Instruct-7B
2. apply 4-bit quantization
3. attach LoRA adapters
4. generate heuristic rewrite candidates
5. construct preference dataset
6. run DPO trainer
7. save adapter checkpoints
8. export inference-ready adapter weights

---

# Directory Structure Requirements

Backend must follow this structure:

backend/
app/
controllers/
services/
models/
training/
heuristics/
datasets/
external_llm/
config/
utils/
main.py

Training logic must remain inside:

training/

Runtime logic must remain inside:

services/
controllers/

Do NOT mix training execution inside runtime endpoints

---

# FastAPI Requirements

Framework:

FastAPI

Server:

Uvicorn

Optional production compatibility:

Gunicorn

---

# Python Library Requirements

Allowed libraries:

torch  
transformers  
peft  
trl  
bitsandbytes  
datasets  
accelerate  
sentence-transformers  
spaCy  
FastAPI  
uvicorn

No substitutions allowed unless required for compatibility

---

# Hardware Requirements

Training GPU target:

RTX 3070

Pipeline must support single-GPU execution

---

# Tokenization Requirements

Tokenizer must match:

Qwen-Instruct-7B tokenizer

Tokenizer substitution is NOT allowed

---

# Semantic Preservation Implementation

Similarity measurement method:

sentence-transformers cosine similarity

Reject rewrites below similarity threshold

---

# Runtime Prompt Data Flow

Pipeline sequence:

user input prompt  
↓  
heuristic scoring engine  
↓  
prompt optimizer  
↓  
post optimization scoring  
↓  
external LLM comparison  
↓  
response aggregation  
↓  
frontend output

---

# Offline vs Runtime Separation Rule

Strict enforcement:

Training logic:

training/

Runtime logic:

services/
controllers/

Never execute training inside runtime endpoints

---

# Output Requirements

Backend generation must include:

1 directory structure
2 module implementations
3 FastAPI routing file
4 heuristic scoring engine
5 QLoRA training script
6 DPO trainer script
7 dataset builder script
8 adapter loading inference script
9 configuration template
10 environment setup instructions
11 reproducibility instructions

---

# Anti-Hallucination Rule

If implementation detail is missing:

Follow thesis architecture first
Follow HuggingFace PEFT defaults second

Do NOT invent architecture replacements

---

# Optional Improvement Rule

After baseline backend generation:

Suggest improvements that extend:

- scalability
- logging
- evaluation hooks
- dataset pipeline robustness

Do NOT modify thesis architecture during improvements
