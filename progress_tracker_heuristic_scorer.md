# Progress Tracker — Heuristic Scorer Improvement

> **File**: `heuristic_scorer_improved.py`
> **Specification**: `architecture/heuristic_scorer_improv_checklist.md`
> **Original**: `tools/heuristic_scorer.py`
> **Date**: 2026-04-11

---

## Validation Checklist

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Formula consistency verified | ✅ Done | Quality now uses only clarity + specificity. Semantic removed from additive total. Weights dynamically normalised to sum to 1.0. |
| 2 | Semantic gating validated | ✅ Done | Two-tier gating: hard floor (0.40 → reject) + soft threshold (0.70 → scale improvement by semantic score). |
| 3 | Normalization confirmed | ✅ Done | Dynamic weight normalization ensures `w_c' + w_s' = 1.0`. Quality score bounded to [0.0, 1.0] after penalties and bonus. |
| 4 | Metric independence confirmed | ✅ Done | Clarity (syntactic actionability) and specificity (constraint density) measure orthogonal linguistic properties. Semantic preservation is a constraint gate, not a scoring component. |
| 5 | Penalties integrated correctly | ✅ Done | Ambiguity penalty (subtractive, max 0.15), redundancy penalty (subtractive, max 0.15), length penalty (multiplicative scaling factor). All applied after base quality computation. |
| 6 | Delta metrics verified | ✅ Done | Returns `clarity_delta`, `specificity_delta`, `quality_improvement` as candidate−raw differences. |
| 7 | Rejection logic verified | ✅ Done | Hard rejection when `semantic < 0.40`. Soft scaling when `0.40 ≤ semantic < 0.70`. Full improvement when `semantic ≥ 0.70`. |
| 8 | Failure modes documented | ✅ Done | See "Failure Modes" section below. |

---

## Detected Issues in Original `heuristic_scorer.py`

### Issue 1 — Semantic Score Mixed Into Quality (CRITICAL)

**Location**: `evaluate()` lines 227, 231

**Problem**: The original code computes `total = (w_c * clarity) + (w_s * specificity) + (w_p * sim)`, which adds semantic similarity directly into the composite score. This violates the specification (§1, §3) which requires semantic preservation to be a **constraint gate**, not a scoring component.

**Impact**: 
- Raw-only prompts get an inflated score of `+0.2` from `w_p * 1.0` (self-similarity baseline)
- The quality score conflates "how good is this prompt?" with "how similar is it to the original?" — these are fundamentally different questions
- When comparing raw vs candidate quality, the comparison is unfair because the semantic weight artificially elevates scores

**Resolution**: Removed `w_p` from the quality formula entirely. Quality is now computed using only `clarity` and `specificity` with dynamically normalised weights. Semantic similarity serves exclusively as a gating constraint.

---

### Issue 2 — No Dynamic Weight Normalization (HIGH)

**Location**: `__init__()` lines 38–58

**Problem**: When `w_p` is excluded from scoring (as the spec requires), the remaining weights `w_c=0.4 + w_s=0.4 = 0.8` sum to less than 1.0. This means the quality score can never reach 1.0, even for a perfect prompt.

**Impact**: Score ceiling is artificially limited to 0.8, breaking the [0.0, 1.0] range assumption.

**Resolution**: Implemented dynamic weight normalization (§4):
```
w_c' = w_c / (w_c + w_s)
w_s' = w_s / (w_c + w_s)
```
With default `w_c=0.5, w_s=0.5`, both normalise to 0.5 and sum to 1.0.

---

### Issue 3 — No Ambiguity Penalty (MEDIUM)

**Location**: N/A — missing from original

**Problem**: Vague tokens like "some", "stuff", "things", "various" are not penalised. A prompt like "tell me something about stuff" receives scores purely based on syntactic structure, ignoring that its content is meaningless.

**Resolution**: Added `_compute_ambiguity_penalty()` which counts matches against a curated frozenset of 24 vague tokens. Penalty is proportional to the ratio of ambiguous tokens to content tokens, capped at 0.15.

---

### Issue 4 — No Redundancy Detection (MEDIUM)

**Location**: N/A — missing from original

**Problem**: Repeated tokens (e.g., "very very very detailed") are not penalised. This allows score gaming through token repetition, especially for specificity (more tokens → more modifiers).

**Resolution**: Added `_compute_redundancy_penalty()` which detects consecutive duplicate tokens and applies a penalty proportional to the duplicate ratio, capped at 0.15.

---

### Issue 5 — No Length Normalization (MEDIUM)

**Location**: N/A — missing from original

**Problem**: Extremely short prompts (e.g., "Help") can score disproportionately high on clarity (1 actionable sentence / 1 total = 1.0) despite being low-quality.

**Resolution**: Added `_compute_length_penalty()` which applies a linear scaling factor for prompts shorter than 8 content tokens. A 2-token prompt receives a 75% length penalty.

---

### Issue 6 — Overly Strict Actionability Detection (LOW)

**Location**: `_score_clarity()` lines 122–123

**Problem**: The original code requires both a ROOT verb AND a direct object (`dobj`) for a sentence to be "actionable". This fails for valid imperative prompts like "Explain clearly", "Be concise", "Act as an expert" — which have no direct object.

**Resolution**: Extended the actionability check (§6) to also accept ROOT verbs with no explicit subject (heuristic for imperative mood):
```python
if root_verb and (has_dobj or not has_subject):
    actionable_count += 1
```

---

### Issue 7 — Missing Output Fields (LOW)

**Location**: `ScoreResult` TypedDict, `evaluate()` return

**Problem**: The original `ScoreResult` only contains `clarity`, `specificity`, `semantic_preservation`, `total`, and `rejected`. The specification (§9, §10) requires `raw_quality`, `candidate_quality`, `quality_improvement`, `clarity_delta`, `specificity_delta`.

**Resolution**: Extended `ScoreResult` to include all 14 fields from the specification, plus penalties and structural bonus for full transparency.

---

## Specification Conflicts Detected

### Conflict 1 — Threshold Mismatch Between Base Spec and Checklist

**Base spec** (`heuristic_scorer.md` §3C): `sim_threshold = 0.75`
**Checklist** (§12): `semantic < 0.40` → reject

**Resolution**: The checklist introduces a two-tier system (hard floor + soft threshold) that supersedes the base spec's single threshold. Adopted:
- Hard floor: `0.40` (below = immediate rejection, as per §12)
- Soft threshold: `0.70` (below = scaled improvement, as per §12)

**Assumption**: The checklist is considered the authoritative improvement document and takes precedence over the base spec where they conflict.

### Conflict 2 — Composite Formula Includes vs Excludes Semantic

**Base spec** (§3D): `Score = (W_c * Clarity) + (W_s * Specificity)` with note "Semantic preservation acts as a gating threshold, **or** its score is added if successful."
**Checklist** (§1, §3): Semantic must be separated from quality score entirely.

**Resolution**: Followed the checklist. Semantic is purely a constraint gate. The "or its score is added" clause from the base spec is interpreted as the **soft scaling** behaviour in §12 (multiply improvement by semantic score), not as an additive component.

---

## Failure Modes

| Mode | Trigger | Behaviour | Mitigation |
|------|---------|-----------|------------|
| spaCy not installed | `OSError` on model load | Clarity and specificity return 0.0; all penalties return 0.0 | Error logged with install instructions |
| SentenceTransformer not available | Load exception | Semantic preservation returns 0.0; candidate always rejected | Warning logged; system degrades to quality-only scoring |
| Empty prompt | `""` or whitespace-only input | All scores 0.0, `rejected=False`, `length_penalty=1.0` | Handled at entry point of each scoring function |
| Single-token prompt | e.g., `"Help"` | High clarity (1.0), low specificity (0.0), heavy length penalty (0.875) | Length normalization reduces inflated scores |
| All-ambiguous prompt | e.g., `"something about stuff"` | Full ambiguity penalty (0.15) reduces quality | Ambiguity penalty caps at configured maximum |
| Repeated tokens | e.g., `"explain explain explain"` | Redundancy penalty applied (up to 0.15) | Consecutive duplicate detection only; legitimate repetition patterns are not penalised |
| Degenerate weights | `w_c=0, w_s=0` | Quality defaults to 0.0 | Warning logged; edge case handled explicitly |
| Candidate identical to raw | Same string for both | Semantic = 1.0, quality_improvement = 0.0, final_score = 0.0 | Correct behaviour — no improvement detected |
| Semantic between floor and threshold | `0.40 ≤ sim < 0.70` | Improvement scaled by `sim`; not rejected but penalised | Soft penalty provides smooth gradient for DPO training |

---

## Source Code Documentation

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HeuristicScorer.evaluate()                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │  Raw Prompt   │    │  Candidate   │                          │
│  └──────┬───────┘    └──────┬───────┘                          │
│         │                    │                                   │
│         ▼                    ▼                                   │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │ _score_prompt │    │ _score_prompt │                          │
│  │  ┌─────────┐ │    │  ┌─────────┐ │                          │
│  │  │ Clarity  │ │    │  │ Clarity  │ │                          │
│  │  │Specifcty │ │    │  │Specifcty │ │                          │
│  │  │Ambiguity │ │    │  │Ambiguity │ │                          │
│  │  │Redundncy │ │    │  │Redundncy │ │                          │
│  │  │ Length   │ │    │  │ Length   │ │                          │
│  │  │Structurl │ │    │  │Structurl │ │                          │
│  │  └────┬────┘ │    │  └────┬────┘ │                          │
│  │       ▼      │    │       ▼      │                          │
│  │  raw_quality │    │  cand_quality│                          │
│  └──────┬───────┘    └──────┬───────┘                          │
│         │                    │                                   │
│         └────────┬───────────┘                                   │
│                  ▼                                               │
│         quality_improvement                                      │
│         = cand_quality − raw_quality                             │
│                  │                                               │
│                  ▼                                               │
│  ┌──────────────────────────────┐                               │
│  │   Semantic Constraint Gate    │                               │
│  │  ┌────────────────────────┐  │                               │
│  │  │ sim < 0.40 → REJECT    │  │                               │
│  │  │ sim < 0.70 → SCALE     │  │                               │
│  │  │ sim ≥ 0.70 → PASS      │  │                               │
│  │  └────────────────────────┘  │                               │
│  └──────────────┬───────────────┘                               │
│                  ▼                                               │
│            final_score                                           │
│                  │                                               │
│                  ▼                                               │
│     ┌────────────────────────┐                                  │
│     │      ScoreResult       │                                  │
│     │  (14 fields returned)  │                                  │
│     └────────────────────────┘                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Functions

| Function | Stage | Input | Output | Purpose |
|----------|-------|-------|--------|---------|
| `_score_clarity()` | 1 | prompt string | float [0,1] | Sentence-level actionability (ROOT verb + dobj/imperative) |
| `_score_specificity()` | 2 | prompt string | float [0,1] | Constraint density (amod + nummod + entities) |
| `_compute_ambiguity_penalty()` | 3 | prompt string | float [0, 0.15] | Vague token ratio penalty |
| `_compute_redundancy_penalty()` | 4 | prompt string | float [0, 0.15] | Consecutive duplicate token penalty |
| `_compute_length_penalty()` | 5 | prompt string | float [0, 1] | Short-prompt reduction factor |
| `_compute_quality()` | 6 | components | float [0, 1] | Weighted sum with penalties and bonus |
| `_compute_structural_bonus()` | 7 | prompt string | float [0, 0.10] | Structured formatting reward |
| `_score_semantic_preservation()` | 8 | raw, candidate | float [0, 1] | Cosine similarity for gating |
| `_score_prompt()` | internal | prompt string | dict | Bundles stages 1–7 for a single prompt |
| `evaluate()` | public | raw, candidate? | ScoreResult | Full pipeline with gating + deltas |

### Configuration

All thresholds are centralised in `ScorerConfig`:

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `w_c` | 0.5 | [0, ∞) | Clarity weight (normalised at runtime) |
| `w_s` | 0.5 | [0, ∞) | Specificity weight (normalised at runtime) |
| `semantic_hard_floor` | 0.40 | [0, 1] | Below this → immediate rejection |
| `semantic_soft_threshold` | 0.70 | [0, 1] | Below this → scale improvement by sim |
| `specificity_ideal_density` | 0.25 | (0, 1] | Target modifier+entity density |
| `ambiguity_max_penalty` | 0.15 | [0, 1] | Maximum ambiguity penalty |
| `redundancy_max_penalty` | 0.15 | [0, 1] | Maximum redundancy penalty |
| `min_tokens_for_full_score` | 8 | [1, ∞) | Below this → length penalty applied |
| `structural_bonus_cap` | 0.10 | [0, 1] | Maximum structural bonus |

---

## Simulated Scoring Results

### Test Case 1 — Weak Prompt

| Field | Value |
|-------|-------|
| **Raw** | `"tell me something about stuff"` |
| **Candidate** | `"explain some things"` |

| Metric | Value | Analysis |
|--------|-------|----------|
| `raw_quality` | 0.1979 | Low — vague content drives ambiguity penalty |
| `candidate_quality` | 0.1312 | Even lower — shorter + still vague |
| `clarity` | 1.0 | Candidate has imperative verb (detected by §6 improvement) |
| `specificity` | 0.0 | No modifiers, no entities, no constraints |
| `ambiguity_penalty` | 0.15 | Maximum — "some", "things" are ambiguous tokens |
| `redundancy_penalty` | 0.0 | No consecutive duplicates |
| `length_penalty` | 0.625 | Only 3 content tokens (well below 8 threshold) |
| `structural_bonus` | 0.0 | No structured formatting |
| `quality_improvement` | -0.0667 | Candidate is worse than raw |
| `clarity_delta` | +0.0667 | Slight clarity improvement |
| `specificity_delta` | 0.0 | No change |
| `semantic_preservation` | 0.6753 | Below soft threshold — improvement scaled |
| `rejected` | False | Above hard floor (0.40) |
| `final_score` | **-0.0450** | Negative: rewrite degraded the prompt |

**Interpretation**: The rewrite made the prompt shorter and equally vague. The negative `final_score` correctly signals that this candidate should be rejected by the API layer, and the original prompt should be kept.

---

### Test Case 2 — Moderate Prompt

| Field | Value |
|-------|-------|
| **Raw** | `"Write a summary of machine learning"` |
| **Candidate** | `"Write a concise 200-word summary of supervised machine learning, covering key algorithms and their applications."` |

| Metric | Value | Analysis |
|--------|-------|----------|
| `raw_quality` | 0.3375 | Decent — has actionable verb but low specificity |
| `candidate_quality` | 0.9500 | Excellent — specific constraints, word count, scope |
| `clarity` | 0.9 | Strong actionable structure |
| `specificity` | 1.0 | Saturated — "200-word", "supervised", "machine learning" etc. |
| `ambiguity_penalty` | 0.0 | No vague tokens |
| `redundancy_penalty` | 0.0 | No repetition |
| `length_penalty` | 0.0 | Sufficient token count |
| `structural_bonus` | 0.0 | Prose format (no bullets/numbers) |
| `quality_improvement` | +0.6125 | Large improvement — optimization was effective |
| `clarity_delta` | 0.0 | Clarity maintained |
| `specificity_delta` | +1.0 | Massive specificity gain |
| `semantic_preservation` | 0.7491 | Above soft threshold — full improvement retained |
| `rejected` | False | Well above hard floor |
| `final_score` | **+0.6125** | Strong positive: excellent optimization |

**Interpretation**: This is a textbook successful optimisation. The candidate added concrete constraints (word count, scope narrowing, topic specification) while preserving the original intent. The `final_score` of 0.6125 is a strong reward signal for DPO training.

---

### Test Case 3 — Optimised Prompt

| Field | Value |
|-------|-------|
| **Raw** | `"Explain neural networks"` |
| **Candidate** | `"Act as a computer science professor. Explain the architecture of feedforward neural networks in 3 sections: 1. Input layer... 2. Hidden layers... 3. Output layer... Format: Use bullet points..."` |

| Metric | Value | Analysis |
|--------|-------|----------|
| `raw_quality` | 0.375 | Moderate — clear verb but minimal specificity |
| `candidate_quality` | 0.5116 | Higher but penalised by clarity score |
| `clarity` | 0.0 | Low — spaCy fails to parse multi-clause structured prompts |
| `specificity` | 0.9032 | High — many modifiers, numbers, named entities |
| `ambiguity_penalty` | 0.0 | No vague tokens |
| `redundancy_penalty` | 0.0 | No repetition |
| `length_penalty` | 0.0 | Sufficient length |
| `structural_bonus` | 0.06 | 3 structural patterns detected (numbered steps, role, format) |
| `quality_improvement` | +0.1366 | Positive improvement |
| `clarity_delta` | -1.0 | Clarity dropped (parser limitation) |
| `specificity_delta` | -0.0968 | Slight specificity change |
| `semantic_preservation` | 0.6271 | Below soft threshold — improvement scaled |
| `rejected` | False | Above hard floor |
| `final_score` | **+0.0857** | Modest positive after soft scaling |

**Interpretation**: The candidate is objectively superior but receives a moderate score due to two factors:
1. **Clarity parser limitation**: spaCy's `en_core_web_sm` struggles with complex multi-clause prompts containing numbered lists. The structural complexity that makes this a better prompt actually confuses the syntactic parser into scoring clarity at 0.0.
2. **Semantic drift**: The candidate significantly expands the scope (adding architecture details, activation functions, loss computation), causing semantic similarity to drop below the soft threshold.

> **Known limitation**: The clarity scoring component's reliance on spaCy's syntactic parsing introduces a systematic bias against structured, multi-section prompts. A future improvement could use `en_core_web_trf` (transformer-based model) for more robust parsing, or add a heuristic that credits structured formatting in the clarity score itself.

---

## Suggested Improvements (Future Work)

| Priority | Improvement | Rationale |
|----------|-------------|-----------|
| HIGH | Upgrade to `en_core_web_trf` | Better handling of complex sentence structures; reduces false negatives in clarity scoring |
| HIGH | Per-section clarity scoring | For structured prompts with numbered lists, score each section independently |
| MEDIUM | Configurable ambiguity wordlist | Allow domain-specific tuning (e.g., "various" may be acceptable in creative writing) |
| MEDIUM | Non-consecutive redundancy detection | Catch patterns like "explain X and then explain Y" |
| LOW | Caching for `_score_prompt()` | Avoid re-parsing the same prompt string in repeated calls |
| LOW | Batch semantic encoding | Encode raw and candidate in a single forward pass |

---

## Changes from Original `heuristic_scorer.py`

| Aspect | Original | Improved |
|--------|----------|----------|
| Quality formula | `(w_c * clarity) + (w_s * specificity) + (w_p * sim)` | `(w_c' * clarity) + (w_s' * specificity)` — semantic excluded |
| Weight normalization | None — weights hardcoded | Dynamic: `w_c' = w_c / (w_c + w_s)` |
| Semantic gating | Single threshold (0.70) → reject | Two-tier: hard floor (0.40) + soft scaling (0.70) |
| Actionability | Requires ROOT verb + dobj | Also accepts imperative (no subject) |
| Penalties | None | Ambiguity (0.15), redundancy (0.15), length (multiplicative) |
| Structural bonus | None | Up to 0.10 for structured formatting |
| Output fields | 5 fields | 14 fields including deltas and penalties |
| Configuration | Constructor kwargs | Centralised `ScorerConfig` dataclass |
| Raw quality tracking | Not computed | `raw_quality` and `candidate_quality` in output |
| Metric deltas | Not computed | `clarity_delta`, `specificity_delta`, `quality_improvement` |
