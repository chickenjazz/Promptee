# Next-Generation Dataset Builder — Upgrade Plan

**Target file:** [training/dataset_builder.py](training/dataset_builder.py) (2134 lines)
**Critical dependency:** [tools/heuristic_scorer.py](tools/heuristic_scorer.py)
**LLM backend:** [tools/prompt_optimizer.py](tools/prompt_optimizer.py) (Qwen2.5 via transformers + 4-bit NF4)

---

## Context

The existing v2 pipeline (`dataset_builder_v2`) generates DPO preference pairs for a prompt-optimizer LoRA. Under observation it produces style-collapsed chosen prompts and synthetically-degraded rejecteds that share too much surface structure. A concrete artifact confirming this is in [datasets/preference_pairs.jsonl](datasets/preference_pairs.jsonl):

```
chosen:   "ROLE: Ecologist\nOBJECTIVE: Explain ...\nREQUIREMENTS: ...\nCONSTRAINTS: ...\nOUTPUT FORMAT: ...\nCONTEXT: ..."
rejected: same lines with the ROLE:/OBJECTIVE:/... labels stripped
```

This is training the adapter to *insert six section headers*, not to *improve a prompt*. The root cause is a monoculture: `HeuristicScorer` rewards regex-detected section headers, the rewrite templates were designed to produce exactly those headers, the missing-component detector fires on the same regex, and negatives are produced by stripping those same headers back out. The scorer, rewriter, and negator are playing a closed game against each other.

This plan redesigns the pipeline to produce **prompt-type-aware chosens, scored by a multi-signal ensemble, paired with naturally-bad or hard-plausible negatives that are not syntactic derivatives of the chosen**.

---

## 1. EXECUTIVE SUMMARY

### What is wrong

1. **Scorer monoculture.** `scorer.evaluate()` is called for component detection ([dataset_builder.py:1673](training/dataset_builder.py#L1673)), chosen acceptance ([L1746](training/dataset_builder.py#L1746)), negative filtering ([L1418](training/dataset_builder.py#L1418)), and quality-gap gating ([L1435](training/dataset_builder.py#L1435)). Every decision is the same heuristic. [tools/heuristic_scorer.py:270-294](tools/heuristic_scorer.py#L270-L294) defines completeness as a 5-keyword regex match (`role|objective|requirements|constraints|output`) with no semantic validation. [tools/heuristic_scorer.py:1107-1134](tools/heuristic_scorer.py#L1107-L1134) adds a +0.10 "structural bonus" for binary presence of headers/bullets/step markers. A rewriter that injects the six labels scores highly regardless of whether the content improved.

2. **Template-driven chosen style collapse.** All five rewrite templates ([L318-390](training/dataset_builder.py#L318-L390)) explicitly push for ROLE/OBJECTIVE/REQUIREMENTS/CONSTRAINTS/OUTPUT FORMAT sections. `_missing_block` ([L302-315](training/dataset_builder.py#L302-L315)) injects "Add a ROLE section" instructions by default. Even the "clarity" and "deterministic execution" templates funnel into the same API-ready format. The resulting chosens are indistinguishable archetypes.

3. **Synthetic negatives, not natural ones.** 12 of the 15 transforms in `NegativeCandidateGenerator` ([L743-1198](training/dataset_builder.py#L743-L1198)) *start from the chosen* and remove/weaken something. The model learns to invert them (re-add sections, re-add numbers). The four LLM degradation modes ([L675-720](training/dataset_builder.py#L675-L720)) are also chosen-conditioned. There is no pipeline path for "a real user writes a bad prompt from scratch."

4. **Difficulty measured, not controlled.** `_assign_difficulty_tier` ([L1363-1370](training/dataset_builder.py#L1363-L1370)) labels a pair *after* the quality gap is observed. The config has target quotas ([L136-140](training/dataset_builder.py#L136-L140)) but the sampling loop never enforces them; if the transforms happen to produce easy pairs, easy is all you get.

5. **Unrealistic negatives.** `add_noise` produces typo/repetition/hedge variants ([L1151-1198](training/dataset_builder.py#L1151-L1198)); `make_vague` does blind regex substitution ("must" → "should probably") ([L1118-1128](training/dataset_builder.py#L1118-L1128)); `contradictory_constraints` appends a fixed kicker sentence ([L1062-1096](training/dataset_builder.py#L1062-L1096)). These are *trivially* worse than the chosen — no informative DPO signal.

6. **Prompt-type mismatch.** `is_instructional` ([L200](training/dataset_builder.py#L200)) admits "write a function to sort X" and "explain the Monty Hall problem" into the same treatment. Both get forced into ROLE/OBJECTIVE scaffolds. A 12-word conversational question does not need a CONSTRAINTS section, and a creative-writing brief does not need an API-ready OUTPUT FORMAT.

7. **Validation checks surface, not drift.** `is_valid_rewrite` ([L428](training/dataset_builder.py#L428)) checks task-answering, examples, length, shallow sections. `is_valid_rejection` checks semantic floor 0.40 ([L1420](training/dataset_builder.py#L1420)). Neither catches "preserved embedding but changed the thing the user actually asked for," nor unnatural phrasing ("Rewrite this prompt as a numbered sequence of reasoning steps with an explicit input contract" in an output chosen).

8. **Docs lie.** Module docstring ([L1-29](training/dataset_builder.py#L1-L29)) describes a "1–3 attempt pipeline" producing `(x=raw, y_w=best, y_l=raw)`. Actual pipeline produces up to ~7 negatives per accepted chosen across 4 categories plus an LLM-generated stream. The acceptance rule in [L1757-1759](training/dataset_builder.py#L1757-L1759) (`semantic>=0.40 AND quality>baseline AND final_score>=min_gain`) is not documented anywhere above.

### What the improved system should become

A **type-aware, ensemble-scored, naturalistic preference-pair generator** that routes each raw prompt by archetype (concise / structured / conversational / creative / analytical / coding), produces chosen candidates with an appropriate rewriter for that archetype, scores them with three independent signals (heuristic + semantic + naturalness), selects the best via an ensemble that penalizes style collapse, and pairs it with negatives drawn from three banks: (a) *natural bad prompts* mined from a separately-curated pool of real low-quality prompts, (b) *hard near-misses* produced by a "plausible but subtly wrong" LLM prompt, (c) a small residual of *rule-based structural negatives* retained only where they produce genuinely informative gaps. A `DifficultyController` enforces the easy/medium/hard mix as it writes, not after the fact.

---

## 2. PRIORITY ROADMAP

### P0 — Must fix immediately

#### P0.1 — Break the scorer monoculture with a 3-signal ensemble
- **Why it matters.** Single-scorer optimization is why the model produces section-heavy junk that scores 0.43 while a natural good prompt scores 0.25. Every downstream filter inherits the same blind spots.
- **Steps.**
  1. Keep `HeuristicScorer` but demote it to one signal in an ensemble.
  2. Add `SemanticPreservationScorer` wrapping `scorer._st_model` ([heuristic_scorer.py:351-356](tools/heuristic_scorer.py#L351-L356)) directly — cosine sim between raw and candidate on sentence-level, plus *content-token overlap* (Jaccard on non-stopword lemmas) to catch "embedding-preserved but nouns changed" drift.
  3. Add `NaturalnessScorer` using `PromptOptimizer.model` in log-likelihood mode. New method `PromptOptimizer.score_logprob(text)` returns mean token log-prob under the base LM (no LoRA). This is the single best signal against "Rewrite this prompt as a structured API-ready request with each field on its own labelled line" appearing in a chosen.
  4. Add `AntiTemplateScorer` that maintains a running histogram of the first 200 chars of every accepted chosen and penalizes new chosens whose opening n-gram distribution is within KL-0.3 of any prior accepted chosen. This directly punishes the `ROLE:\nOBJECTIVE:\n...` collapse.
  5. `MultiScorerEnsemble.score(raw, candidate) -> float` returns a weighted sum; disagreement between scorers is logged as a quality signal.
- **Expected impact.** Chosen style diversity up (hard target: Shannon entropy of chosen first-line ≥ 3.0 bits). Filters catch semantic drift cases the heuristic misses. Single-signal gaming becomes impossible.

#### P0.2 — Replace the "add ROLE/OBJECTIVE" template set with a TemplateRouter
- **Why it matters.** The current chosen distribution is a mode collapse on one template family. A model trained on this dataset will add headers to *every* prompt, including ones that should stay conversational.
- **Steps.**
  1. Add `PromptTypeClassifier.classify(raw) -> {concise, structured, conversational, creative, analytical, coding}`. First pass is a rule-based classifier (verb + length + domain hints) to avoid an extra LLM call; second pass only for low-confidence cases asks the LLM a single-token yes/no per type.
  2. Define one `RewriterProfile` per archetype: `concise` → 1-2 sentence tightening, no sections; `structured` → current API-ready template; `conversational` → natural prose with added specifics; `creative` → goal + constraints + tone + length, no ROLE; `analytical` → claim + evidence + method; `coding` → task + IO contract + language + test hook.
  3. `TemplateRouter.build_instruction(raw, type)` dispatches to the profile. Remove the five near-identical templates at [L318-390](training/dataset_builder.py#L318-L390) and `INSTRUCTION_TEMPLATES` at [L393-399](training/dataset_builder.py#L393-L399).
  4. Drop the `_missing_block` "Add a ROLE section" default for types where ROLE is inappropriate.
- **Expected impact.** Chosen distribution stops looking like one template with six variants. The adapter learns to *match the user's intent to the right structure*, not to bolt on headers.

#### P0.3 — Mine natural bad prompts; stop treating chosen-derivatives as the primary negative
- **Why it matters.** A DPO model learns `P(chosen) > P(rejected)`. If rejecteds are always "the chosen with headers removed," the model learns "prefer headers." If rejecteds are real bad prompts from the wild, the model learns "prefer genuinely-clearer writing."
- **Steps.**
  1. Add a `natural_bad_pool` column to [dataset/RAW_prompts.csv](dataset/RAW_prompts.csv) (or a sibling CSV) holding real low-quality prompts scraped/collected separately. Aim for 500+ rows.
  2. Add `NaturalNegativeSampler.sample(chosen_topic, quality_gap_target)` which picks a real bad prompt (a) on the same topic by embedding cosine, (b) with a heuristic quality below a target window.
  3. Target 60% of negatives per chosen come from this natural pool. Current rule-based transforms drop to ≤25%. LLM-degraded stays at ≤15%.
  4. If the pool is empty, the pipeline must *refuse to run* rather than silently fall back to synthetic-only — the warning from [dataset_builder.py:1988](training/dataset_builder.py#L1988) ("below target, writing what we have") is exactly the kind of silent degradation that produced the current dataset.
- **Expected impact.** Rejected prompts stop being chosens-minus-structure. Model learns a real quality signal instead of a formatting signal.

#### P0.4 — Enforce difficulty mix during generation, not after
- **Why it matters.** Observed difficulty distribution in past runs is heavily "easy" because easy negatives are cheap to produce. DPO benefits most from hard near-misses.
- **Steps.**
  1. Add `DifficultyController` holding live counts of easy/medium/hard emitted pairs vs. configured ratios.
  2. Before emitting each pair, the controller checks whether its tier is oversubscribed; if so, the pair is deferred (kept in a per-tier buffer) and the generator is asked for the undersubscribed tier.
  3. For hard pairs specifically, force the negative-generator branch to `HardNegativeMiner` (see P1).
  4. Final dataset is drained from the buffers in configured ratios; extras are discarded.
- **Expected impact.** Dataset-level difficulty mix matches config ([L136-140](training/dataset_builder.py#L136-L140)) to within ±3%, instead of drifting toward "easy" by default.

### P1 — High value next

#### P1.1 — Hard-negative mining via "plausible but subtly wrong"
- **Why it matters.** The current `add_noise`/`shorten`/`make_vague` produce negatives that are *obviously* worse. DPO signal from a 0.40 quality gap is much noisier than signal from a 0.06 gap, but the current hard-tier transforms (L1363 `quality_gap < 0.08`) are mostly the same easy transforms that happened to fail to produce a large gap.
- **Steps.**
  1. Add an LLM degradation mode `near_miss` — system prompt: "Rewrite this prompt so it looks professional and complete but is missing ONE critical requirement, OR has ONE ambiguous success criterion, OR silently broadens the objective. Output should be 95% as good as the input to a casual reader." Explicitly forbid typos, broken grammar, obvious omissions.
  2. Add `subtle_ambiguity` — replace one measurable criterion with a qualitative one *and* add plausible-looking but non-binding filler around it so it reads fine.
  3. Add `partial_logical_inconsistency` — introduce a *minor* contradiction (e.g., "output JSON" + "in a numbered list") instead of the sledgehammer contradictions in `contradictory_constraints`.
  4. These feed the `hard` tier of `DifficultyController`.
- **Expected impact.** The `hard` tier (currently gap ∈ [0.05, 0.08]) becomes populated with negatives that actually *look* plausible, producing the DPO samples that teach the most.

#### P1.2 — `SemanticDriftGuard` for chosen validation
- **Why it matters.** `is_valid_rewrite` at [L428](training/dataset_builder.py#L428) checks `task_answer_detected` and `contains_examples` but never checks whether the *task* changed. The rewriter can turn "sort this list" into "design a sorting system" and pass validation (semantic floor is 0.40, very permissive).
- **Steps.**
  1. Extract noun chunks + root verb from raw and candidate via spaCy.
  2. Reject if any of: (a) root verb lemma changed *and* is not a near-synonym (keep a 50-verb synonym table), (b) any raw noun chunk head token is absent from candidate *and* not replaced by an embedding-close alternative, (c) new imperative verbs appear in candidate that do not appear in raw (catches "and also ensure X, Y, Z" over-expansion).
  3. Raise the semantic floor from 0.40 to 0.55 for chosen acceptance (keep 0.40 for negatives — negatives are *supposed* to drift somewhat).
- **Expected impact.** Chosens stop over-expanding raw prompts. Hallucinated requirements stop making it in.

#### P1.3 — Anti-template diversity penalty during chosen selection
- **Why it matters.** Even with a TemplateRouter, within each type the rewriter can still produce near-identical outputs. The best-candidate selector ([L1771-1779](training/dataset_builder.py#L1771-L1779)) takes the highest-quality candidate regardless of whether it is the 200th near-duplicate.
- **Steps.**
  1. Maintain a rolling embedding index of all accepted chosens (last 500).
  2. `DiversityPenaltyModule.penalty(candidate)` = max cosine similarity to any accepted chosen, truncated at 0.85 (anything above subtracts proportionally from the ensemble score).
  3. Applied at candidate selection, not at validation — we want the penalty to change *which* candidate wins, not reject everything.
- **Expected impact.** Dataset no longer has hundreds of "ROLE: X  OBJECTIVE: explain Y" chosens; the rewriter is pushed toward novel framings for structurally-similar raws.

### P2 — Medium improvements

#### P2.1 — `check_section_usefulness` audit
Current implementation ([L490-561](training/dataset_builder.py#L490-L561)) rejects when >50% of sections are "shallow," where shallow means <15 chars or contains filler. This is lenient — it permits 50% filler. Tighten to ≤25% shallow, and expand the filler list to include current outputs that sneak through ("knowledge of X", "focus on theoretical aspects").

#### P2.2 — Length ratio is symmetric; it should be asymmetric
[L129](training/dataset_builder.py#L129) uses `length_ratio ∈ [0.6, 1.4]` for negatives. A good rejected is often *longer* (bloated with filler) or much shorter (stripped). The check currently forbids both extremes. Split into two: rejecteds drawn from the "bloated" bank allowed up to ratio 2.0; "stripped" bank floor 0.3. Match to transform type.

#### P2.3 — `PipelineMetrics` → live JSONL dashboard
`PipelineMetrics.log_summary` ([L1529-1570](training/dataset_builder.py#L1529-L1570)) prints to stdout once at the end. Add periodic JSONL emission every N prompts to `logs/pipeline_metrics.jsonl` with the Section-8 dashboard metrics. Enables mid-run aborts when entropy collapses.

#### P2.4 — Reproducibility bug
[L783](training/dataset_builder.py#L783) hardcodes `random.Random(42)` in the `NegativeCandidateGenerator` default. The config seed ([L144](training/dataset_builder.py#L144)) is only honored if explicitly passed. Fix: always require the rng and remove the `or random.Random(42)` fallback.

### P3 — Nice to have

- **P3.1** — Per-domain calibration of scorer weights (coding prompts should weight specificity more; creative prompts should weight specificity less).
- **P3.2** — Active-learning loop: train a small DPO adapter on batch N, score batch N+1's candidates with the adapter's implicit reward, feed the disagreements back as hard examples.
- **P3.3** — Human spot-check sampling — every 50 pairs, hold one out for a queue the operator reviews; feed agreements/disagreements into scorer reweighting.

---

## 3. NEW PIPELINE ARCHITECTURE

```
Raw Prompt
    │
    ▼
PromptTypeClassifier
    │  {concise, structured, conversational, creative, analytical, coding}
    ▼
TemplateRouter — selects rewriter profile + missing-component policy per type
    │
    ▼
DiverseCandidateGenerator
    │  generates 3 candidates at increasing temperature via AtomicRewriter
    ▼
MultiSignalEnsemble
    │  heuristic · semantic preservation · naturalness · anti-template
    │  + SemanticDriftGuard (hard gate)
    │  + DiversityPenaltyModule (soft penalty vs. rolling index)
    ▼
Best Chosen Selection — argmax over ensemble score, not heuristic alone
    │
    ▼
Per-chosen Negative Plan — requested by DifficultyController
    │       ┌──────────────────────────┬────────────────────┐
    │       │                          │                    │
    ▼       ▼                          ▼                    ▼
NaturalNegativeSampler   HardNegativeMiner          RuleBasedTransforms
  (60% of pool)            (25% of pool)              (≤15% of pool)
  real bad prompts         LLM "near-miss"            reduced to: section_shuffle,
  matched by topic +       subtle_ambiguity,          partial_component_dropout,
  target quality gap       partial_contradiction      reduce_specificity
    │       │                          │                    │
    └───────┴──────────┬───────────────┴────────────────────┘
                       ▼
            Negative Validation (is_valid_rejection v3)
                      │  + SemanticDriftGuard (negatives may drift, but not invert task)
                      │  + inter-negative dedup
                      │  + naturalness floor (rejected must still read like a real prompt)
                       ▼
            DifficultyController.emit(pair)
                      │  routes to easy/medium/hard buffer
                      │  drops if target tier is saturated
                       ▼
            Final Preference Pair Export (per-tier drain in configured ratio)
                      │
                      ▼
            JSONL + metrics snapshot
```

Two phase boundaries are preserved from the current pipeline because they are real constraints:

- **GPU phase 1** — rewriter + classifier + naturalness scorer + hard-negative miner (LLM-bound) all run while Qwen is loaded. Currently only the rewriter is here.
- **CPU/embedding phase 2** — natural-negative sampler + rule-based transforms + ensemble scoring + validation + emission. Runs after Qwen is unloaded.

This means the pipeline precomputes *all LLM outputs it will need per prompt* — chosen, near-miss negatives, naturalness scores — in one pass while the model is resident, then does the heavy scoring/selection on CPU. This is what the current pipeline does for LLM negatives ([L1810-1824](training/dataset_builder.py#L1810-L1824)); extend the pattern to everything LLM-bound.

---

## 4. CODE CHANGES (dataset_builder.py)

### New modules to add

| Class / function | Purpose | Replaces |
|---|---|---|
| `PromptTypeClassifier` | Classify raw into 6 archetypes; rule-based first, LLM fallback | `is_instructional` binary filter |
| `TemplateRouter` | Dispatch per-archetype rewriter instruction | `INSTRUCTION_TEMPLATES` list + `build_atomic_rewrite_instruction` |
| `RewriterProfile` (×6) | One per archetype: concise, structured, conversational, creative, analytical, coding | `clarity_template`, `production_template`, `structured_reasoning_template`, `api_ready_template`, `deterministic_execution_template` — all deleted |
| `MultiScorerEnsemble` | Combines heuristic + semantic + naturalness + anti-template | Direct `scorer.evaluate()` calls at candidate acceptance |
| `SemanticPreservationScorer` | Jaccard on content-token lemmas + cosine on sentence embeddings | The `semantic_preservation` field alone |
| `NaturalnessScorer` | Mean log-prob under base Qwen (no LoRA) | New signal, no existing equivalent |
| `AntiTemplateScorer` | KL divergence of candidate's opening n-grams vs. rolling chosen histogram | New signal |
| `DiversityPenaltyModule` | Max cosine similarity vs. rolling accepted-chosen index | New signal |
| `SemanticDriftGuard` | Noun-chunk + root-verb comparison raw vs. candidate | Complements `is_valid_rewrite` — runs after it |
| `NaturalNegativeSampler` | Topic-matched draw from `natural_bad_pool` | Most of `NegativeCandidateGenerator` (which shrinks) |
| `HardNegativeMiner` | LLM "near-miss" + "subtle ambiguity" + "partial contradiction" | `add_noise`, `make_vague`, `shorten` — deleted |
| `DifficultyController` | Live per-tier quotas, buffer-and-drain emission | Post-hoc `_assign_difficulty_tier` alone |
| `MetricsStreamer` | Periodic JSONL emission of dashboard metrics | `PipelineMetrics.log_summary` alone |

### Modules to DELETE (brutally honest)

Based on the user's instruction to call out modules to delete:

- `INSTRUCTION_TEMPLATES` and all five template functions ([L293-399](training/dataset_builder.py#L293-L399)) — **delete.** They encode the style collapse. Replace with `TemplateRouter`.
- `NegativeCandidateGenerator.add_noise` ([L1151-1198](training/dataset_builder.py#L1151-L1198)) — **delete.** Typos and hedge noise produce unrealistically-bad negatives.
- `NegativeCandidateGenerator.make_vague` ([L1118-1128](training/dataset_builder.py#L1118-L1128)) — **delete.** Blind regex sub is the definition of synthetic noise.
- `NegativeCandidateGenerator.contradictory_constraints` ([L1062-1096](training/dataset_builder.py#L1062-L1096)) — **delete.** Appended-kicker contradictions are syntactically detectable; the model will learn "last sentence contradicts" rather than "logic conflict."
- `NegativeCandidateGenerator.shorten` ([L1130-1149](training/dataset_builder.py#L1130-L1149)) — **delete.** `NaturalNegativeSampler` supplies naturally-short prompts that are more informative.
- `_LLM_DEGRADATION_PROMPTS["vague"]` ([L676-686](training/dataset_builder.py#L676-L686)) — **keep** but retune: current prompt instructs "LESS specific" bluntly; subtlety gives better negatives. See P1.1.

### Modules to REFACTOR but keep

- `NegativeCandidateGenerator` — keep only `section_shuffle`, `section_merge`, `partial_component_dropout`, `reduce_specificity`, `remove_key_requirement`. These produce informative rule-based negatives. Everything else goes.
- `HeuristicScorer` usage — demote from source-of-truth to one signal. No changes inside `tools/heuristic_scorer.py`.
- `PromptOptimizer` — add `score_logprob(text)` method for `NaturalnessScorer`. The existing `rewrite()` is reused for classifier and hard-negative calls.
- `PipelineMetrics` — extend with Section-8 dashboard fields.
- `is_valid_rewrite` — add `SemanticDriftGuard` as a separate stage that runs after this, not instead.

### Modules to KEEP unchanged

- `is_instructional` — still useful as a coarse pre-filter, but stops being the only classifier.
- `_normalize_for_dedup` — fine.
- `sample_transforms` weighted quota logic — applies to the reduced transform set.
- Phase 1 baseline scoring — fine.
- `PromptRecord` dataclass + indexed record structure — fine.

---

## 5. REJECTION SYSTEM REDESIGN

Each category below specifies: **generation method** · **why informative** · **target share of total negatives**.

| Bank | Method | Why informative | Share |
|---|---|---|---|
| **Natural bad prompts** | Topic-matched draw from `natural_bad_pool` | Teaches real quality signal, not "inverse of chosen" | 60% |
| **Plausible but incomplete** | LLM prompt: "Rewrite this prompt removing ONE critical requirement chosen at random. Keep everything else high quality." | Hard near-miss; model must learn *which* requirement matters | 10% |
| **Subtle ambiguity** | Replace exactly one measurable criterion with a qualitative phrase (from `_VAGUE_CRITERIA_REPLACEMENTS`) in context that makes it look intentional | Teaches precision; fix is minimal, gap is narrow | 8% |
| **Missing acceptance criteria** | LLM prompt: "Remove only the success criteria / acceptance criteria / definition of done from this prompt. Preserve task + constraints + output format." | DPO signal is targeted and clean | 7% |
| **Partial logical inconsistency** | Inject one minor contradictory constraint (not the current sledgehammer bank; use LLM to generate a contextually-plausible-sounding one) | Harder than obvious contradictions | 5% |
| **Weaker specificity** | `reduce_specificity` (kept from current transforms — spaCy amod/nummod dropout) | Structural, clean negative | 5% |
| **Near-miss from chosen** | LLM "95% as good to a casual reader" prompt from P1.1 | Hardest tier, populates `hard` difficulty | 5% |

Per-chosen negative count drops from the current configured 7 to a target of 3–4. Higher count was compensating for low per-negative quality.

---

## 6. CHOSEN SYSTEM REDESIGN — archetype profiles

Each `RewriterProfile` has: `(system_prompt, user_template, target_length_range, forbidden_elements)`.

1. **concise** — tighten ambiguity, keep ≤2 sentences, no sections, no labels. Used for short raw prompts (<15 words) that are instructional but don't need a spec. Example: `"summarize this article in 3 bullets"` should stay a sentence, not become ROLE/OBJECTIVE/CONSTRAINTS.
2. **structured** — current API-ready format. Used only for multi-part technical briefs where the user clearly wants a spec.
3. **conversational** — preserve the user's voice; add one clarifying phrase if intent is ambiguous; output still reads as natural prose. Used for questions, "how do I...", "explain..." prompts.
4. **creative** — goal + tone + length + audience. No ROLE. No OUTPUT FORMAT. Creative writing prompts are non-structural by nature.
5. **analytical** — claim to evaluate + criteria + evidence source + required depth. No bullet headers unless the raw had them.
6. **coding** — task + IO contract + language + constraints (libraries, performance) + test hook. The only profile that *should* add a formal OUTPUT FORMAT.

The classifier must default to `conversational` when confidence is low, not `structured`. Current pipeline defaults to structured-ish; that is the collapse.

---

## 7. MULTI-SIGNAL SCORING SYSTEM

Let signals for a candidate `c` against raw `r` and history `H`:

- `h(r, c)` = `HeuristicScorer.evaluate(r, c)["final_score"]` — existing signal, range [0, ~0.5].
- `s(r, c)` = `0.5 * cosine(emb(r), emb(c)) + 0.5 * jaccard_content(r, c)` — range [0, 1].
- `n(c)` = mean token log-prob under base Qwen, normalized as `sigmoid((n - μ) / σ)` where μ, σ are rolling statistics over accepted chosens — range [0, 1].
- `a(c, H)` = `1 - min(1, max_{c' ∈ H} cosine(emb(c), emb(c')))` — range [0, 1], anti-template.
- `d(c, H)` = same as `a` but scoped to the last 500 accepted chosens — anti-duplicate.
- `g(r, c)` = SemanticDriftGuard: 0 if drift detected, 1 otherwise — **hard gate**, not weighted.

**Ensemble score**:
```
score(r, c, H) = g(r, c) * (w_h · h + w_s · s + w_n · n + w_a · a + w_d · d)
with (w_h, w_s, w_n, w_a, w_d) = (0.30, 0.25, 0.20, 0.15, 0.10) initially
```

Acceptance rule: candidate passes iff `g = 1` **AND** `s ≥ 0.55` **AND** `h ≥ 0` **AND** `score > argmax score over previous candidates for this raw`.

Scorer disagreement metric: `disagreement(c) = max(h, s, n) - min(h, s, n)`. Logged per candidate. Used in the dashboard as a dataset-quality flag — if disagreement is high on average, it means the signals are pulling in different directions and the weights should be retuned.

Weight tuning method: hold out 50 human-labeled pairs; grid-search the 5 weights on a 0.1 grid minimizing rank-correlation error with human judgments. Run after every 500 new pairs.

---

## 8. DATASET QUALITY CONTROLS — metrics dashboard

Emit every N=50 pairs to `logs/pipeline_metrics.jsonl`:

| Metric | Computation | Alert threshold |
|---|---|---|
| **Chosen diversity entropy** | Shannon entropy of first-line-bigram distribution across all accepted chosens | < 3.0 bits → HALT |
| **Negative realism score** | Mean naturalness score across all accepted negatives | < 0.40 → WARN |
| **Scorer disagreement rate** | Fraction of candidates where disagreement(c) > 0.4 | > 30% → retune weights |
| **Pair difficulty distribution** | Emitted easy / medium / hard counts vs. config ratios | L1 distance > 0.10 from target → WARN |
| **Template collapse ratio** | Fraction of chosens whose first 60 chars match the top-1 prefix pattern | > 40% → HALT |
| **Semantic drift rate** | Fraction of chosens where `SemanticDriftGuard` returned 0 (candidate rejected) | > 50% → inspect rewriter |
| **Prompt-type coverage** | Distribution of `PromptTypeClassifier` outputs across dataset | Any type <5% → source more raws |
| **Negative-source ratio** | Share of each of the 7 negative banks | Natural-bad < 50% → inspect pool |
| **Per-raw yield** | Accepted chosens / total raws | < 0.5 → inspect thresholds |

---

## 9. IMPLEMENTATION PHASE PLAN — 6-week sprint

Dates assume start Monday 2026-04-27 (next Monday from today, 2026-04-23).

**Week 1 (Apr 27 – May 1) — Observability & infrastructure**
- Build `MetricsStreamer` and backfill the metrics on the *current* dataset to quantify each weakness numerically.
- Add `PromptOptimizer.score_logprob()` method; measure VRAM and latency cost.
- Spec the `natural_bad_pool` data source and start collection (500-row target).

**Week 2 (May 4 – 8) — Chosen pipeline rewrite**
- `PromptTypeClassifier` (rule-based + LLM fallback).
- Six `RewriterProfile` objects.
- `TemplateRouter`.
- Delete the five current templates.
- Smoke test: generate 50 chosens, visually confirm archetype diversity.

**Week 3 (May 11 – 15) — Multi-signal scoring**
- `SemanticPreservationScorer`, `NaturalnessScorer`, `AntiTemplateScorer`, `DiversityPenaltyModule`.
- `MultiScorerEnsemble` wiring.
- `SemanticDriftGuard`.
- Swap ensemble in behind a `--use-ensemble` flag; A/B run against heuristic-only for 100 raws.

**Week 4 (May 18 – 22) — Negative redesign**
- `NaturalNegativeSampler` wired against the pool from week 1.
- `HardNegativeMiner` with three new LLM prompts.
- Delete `add_noise`, `make_vague`, `shorten`, `contradictory_constraints`.
- Reduce rule-based transforms to the 5 kept ones.

**Week 5 (May 25 – 29) — Difficulty & emission**
- `DifficultyController` with buffer-and-drain emission.
- Asymmetric length-ratio handling per P2.2.
- Full end-to-end run at `target_size=1200`.
- Dashboard metrics snapshot at completion.

**Week 6 (Jun 1 – 5) — Validation & release**
- Train a small DPO LoRA on the new dataset; compare eval reward vs. one trained on current dataset.
- If eval reward is within 5% or lower, investigate ensemble weights and semantic-drift threshold.
- Tag `dataset_builder_v3`. Update [architecture/dpo_training.md](architecture/dpo_training.md) §3 to match actual behavior (currently stale per weakness #8).

Gating: do not proceed from week N to N+1 if the metric added that week regresses vs. baseline. This is explicit because the current pipeline's problems came from *incremental* additions that each looked fine in isolation.

---

## 10. FINAL VERDICT

**Moderately refactored, not fully redesigned.**

The infrastructure around the current pipeline — the phase-split around Qwen load/unload, the `PromptRecord` indexing, `PipelineMetrics`, the weighted category quota sampling, the dedup via content normalization — is solid. The acceptance/validation layering is sound as a shell. What is broken is *inside* three boxes:

1. The chosen generator's templates (6 → replaced).
2. The negative generator's synthetic transforms (15 → 5 kept, 4 deleted, 3 LLM-based replacements added, plus the new natural-bad pool).
3. The scorer being single-signal (demoted to one slot in an ensemble).

Fully redesigning would also throw out the load/unload dance, the metrics harness, the record structure, and the dedup — all of which are fine. The brutal cut is the template family, the surface-noise transforms, and the heuristic-only acceptance gate. That is a week-by-week refactor, not a rewrite.

If after week 4 the dashboard still shows `template collapse ratio > 40%` or `chosen diversity entropy < 3.0 bits`, escalate to full redesign and move the whole chosen pipeline to an LLM-judged pairwise tournament instead of an ensemble score.

---

## Verification — how to test end-to-end

1. **Unit**: each new class gets a test under `tests/training/` with fixture raws for each archetype. `PromptTypeClassifier` must hit ≥90% agreement with a hand-labeled set of 60 raws.
2. **Dry run**: `python training/dataset_builder.py --dry-run --dry-run-limit 20` produces a printable report per prompt covering archetype, chosen text, 3-4 negatives with tier and source bank, and ensemble-score breakdown.
3. **Metrics backfill**: run the new `MetricsStreamer` over the existing `datasets/preference_pairs.jsonl` *before* any refactor, and save as `logs/baseline_metrics.jsonl`. Every PR includes a regression table of these metrics.
4. **Adapter eval**: train the DPO adapter on 500 pairs from the new pipeline and 500 pairs from the old, compare on [tools/eval_harness] (existing) reward delta.
5. **Manual inspection**: every Friday, sample 20 pairs, judge each on a 1-5 scale for (chosen naturalness, rejected realism, pair informativeness). Expect an average of ≥4.0 on all three by end of week 5.

---

## Critical files to modify

- [training/dataset_builder.py](training/dataset_builder.py) — primary, most changes.
- [tools/prompt_optimizer.py](tools/prompt_optimizer.py) — add `score_logprob()` method.
- [dataset/RAW_prompts.csv](dataset/RAW_prompts.csv) — add or sibling `dataset/natural_bad_prompts.csv`.
- [config/settings.py](config/settings.py) — new config block for ensemble weights, tier ratios, pool path.
- [architecture/dpo_training.md](architecture/dpo_training.md) §3 — update to match new pipeline; remove references to the five templates.
- [tools/heuristic_scorer.py](tools/heuristic_scorer.py) — **no changes**; only consumption changes.
