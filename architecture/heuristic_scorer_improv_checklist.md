# Heuristic Scorer Improvement Checklist

This document lists recommended improvements to strengthen the deterministic heuristic scoring pipeline for prompt optimization. Each item includes a short description and intended benefit.

---

# High Priority Improvements

## 1. Separate Quality Score

Compute a quality score using only clarity and specificity.

**Formula:**

```
quality = (w_c * clarity) + (w_s * specificity)
```

**Purpose:**

* Prevents semantic similarity from inflating baseline scores
* Enables fair comparison between raw and candidate prompts

---

## 2. Quality Improvement Metric

Compute the difference between candidate and raw quality.

```
quality_improvement = candidate_quality - raw_quality
```

**Purpose:**

* Measures actual optimization gain
* Serves as primary reward signal

---

## 3. Semantic Preservation as Constraint

Use semantic similarity as a gating condition rather than mixing it into improvement.

**Example:**

```
if semantic < threshold:
    rejected = True
```

**Purpose:**

* Prevents meaning drift
* Supports multi-objective optimization

---

## 4. Dynamic Weight Normalization

Redistribute weights when semantic score is not used.

**Example:**

```
w_c' = w_c / (w_c + w_s)
w_s' = w_s / (w_c + w_s)
```

**Purpose:**

* Prevents artificial score inflation
* Keeps baseline scoring fair

---

# Medium Priority Improvements

## 5. Ambiguity Penalty

Penalize vague tokens such as:

```
some, thing, stuff, various, etc
```

**Purpose:**

* Encourages precision
* Reduces shallow prompt optimization

---

## 6. Improved Actionability Detection

Allow imperative verbs without direct objects.

Examples:

* "Explain clearly"
* "Be concise"
* "Act as an expert"

**Purpose:**

* Improves clarity scoring accuracy

---

## 7. Redundancy Penalty

Penalize repeated tokens.

Example:

```
very very detailed
explain explain explain
```

**Purpose:**

* Prevents score gaming
* Encourages concise prompts

---

## 8. Length Normalization

Adjust scores for extremely short prompts.

**Purpose:**

* Prevents short prompts from scoring disproportionately high

---

# Diagnostic Improvements

## 9. Return Metric Deltas

Return differences between raw and candidate metrics.

Example:

```
clarity_delta
specificity_delta
quality_delta
```

**Purpose:**

* Improves debugging and analysis

---

## 10. Raw vs Candidate Quality Output

Return both values explicitly.

Example:

```
raw_quality
candidate_quality
```

**Purpose:**

* Enables plotting and evaluation comparisons

---

# Optional Advanced Improvements

## 11. Structural Bonus

Reward structured prompts such as:

* numbered steps
* bullet points
* role definitions
* output format instructions

**Purpose:**

* Improves real-world prompt effectiveness

---

## 12. Soft Semantic Penalty

Instead of hard rejection, scale improvement using semantic similarity.

```
if semantic < 0.40:
    rejected = True
else:
    final_score = quality_improvement * semantic
```

**Purpose:**

* Smoother optimization behavior
* Better training stability

---

# Recommended Output Format

The scoring system should return:

```
{
    "raw_quality": float,
    "candidate_quality": float,
    "quality_improvement": float,
    "clarity_delta": float,
    "specificity_delta": float,
    "semantic_preservation": float,
    "rejected": bool
}
```

---

# Minimum Recommended Implementation

If implementing incrementally, prioritize:

1. Separate quality score
2. Quality improvement metric
3. Semantic as constraint
4. Dynamic weight normalization
5. Metric deltas output

These provide the largest improvement in scoring reliability.
