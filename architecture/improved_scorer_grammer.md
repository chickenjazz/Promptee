# TASK: Implement Grammar Scoring and Grammarly-Like Highlighting in heuristic_scorer.py

You are a senior Python backend engineer and NLP systems engineer.

Modify the existing `heuristic_scorer.py` to add grammar-aware scoring and diagnostic highlighting. The system should not only return numerical scores, but also identify the exact words, phrases, or missing components that lower the prompt score.

## Goal

Make the heuristic scorer work like a prompt-quality version of Grammarly:

- score prompt effectiveness
- score grammar quality
- highlight problematic words or phrases
- return frontend-ready issue metadata
- preserve all existing scoring behavior

## Important Design Rule

Do not make grammar the main score.

This is a prompt optimization system, not an essay correction system.

Keep the main quality dimensions as:

- clarity
- specificity
- semantic preservation

Add grammar only as a bounded penalty.

A grammatically rough prompt can still be effective, so grammar should reduce the score only moderately.

## Existing Scorer Behavior

The scorer already evaluates:

- clarity
- specificity
- ambiguity penalty
- redundancy penalty
- length penalty
- structural bonus
- semantic preservation
- final score

Do not remove or break any of these.

## Required New Features

### 1. Add grammar scoring

Add these fields to the result:

```python
grammar_score: float
grammar_penalty: float
grammar_error_count: int
grammar_error_density: float
grammar_categories: Dict[str, int]
Use language_tool_python if available.
If LanguageTool is not installed or fails:
do not crash
return neutral grammar values:
grammar_score = 1.0
grammar_penalty = 0.0
grammar_error_count = 0
grammar_error_density = 0.0
grammar_categories = {}
2. Add grammar config values
Add these to ScorerConfig:
grammar_max_penalty: float = 0.12
grammar_penalty_scale: float = 0.35
enable_grammar_penalty: bool = True
enable_grammar_highlights: bool = True
max_diagnostic_issues: int = 25
3. Load LanguageTool once
In _load_models(), add:
self._grammar_tool = None

try:
   import language_tool_python
   self._grammar_tool = language_tool_python.LanguageTool("en-US")
   logger.info("LanguageTool grammar checker loaded successfully.")
except Exception as e:
   logger.warning(f"LanguageTool failed to load: {e}")
   self._grammar_tool = None
Do not initialize LanguageTool per prompt.
4. Add diagnostic issue type
Add this TypedDict:
class DiagnosticIssue(TypedDict, total=False):
   type: str
   severity: str
   start: Optional[int]
   end: Optional[int]
   text: Optional[str]
   message: str
   suggestion: str
   source: str
   score_impact: Optional[float]
   highlight_color: Optional[str]
5. Add issue types
Support these issue types:
ISSUE_TYPES = {
   "grammar",
   "spelling",
   "punctuation",
   "casing",
   "style",
   "ambiguity",
   "redundancy",
   "weak_action",
   "missing_context",
   "missing_output_format",
   "missing_constraints",
   "too_short",
}
6. Add highlight colors
Add:
ISSUE_HIGHLIGHT_COLORS = {
   "grammar": "red",
   "spelling": "red",
   "punctuation": "red",
   "casing": "red",
   "style": "pink",
   "ambiguity": "orange",
   "redundancy": "gray",
   "weak_action": "yellow",
   "missing_context": "blue",
   "missing_output_format": "purple",
   "missing_constraints": "purple",
   "too_short": "blue",
}
Every issue should include its highlight_color.
7. Add grammar error weights
Add:
GRAMMAR_ERROR_WEIGHTS = {
   "spelling": 0.7,
   "grammar": 1.0,
   "punctuation": 0.5,
   "casing": 0.4,
   "style": 0.25,
}
8. Implement grammar scoring
Create:
def _score_grammar(self, prompt: str) -> Dict[str, Any]:
It should:
use LanguageTool if available
convert grammar matches into issue categories
compute weighted error count
compute grammar error density
compute grammar penalty
compute grammar score
Formula:
weighted_error_count = sum(category_weight for each grammar issue)
grammar_error_density = weighted_error_count / token_count

grammar_penalty = min(
   grammar_error_density * self.config.grammar_penalty_scale,
   self.config.grammar_max_penalty
)

grammar_score = 1.0 - (grammar_penalty / self.config.grammar_max_penalty)
Clamp score and penalty:
grammar_score = max(min(grammar_score, 1.0), 0.0)
grammar_penalty = max(min(grammar_penalty, self.config.grammar_max_penalty), 0.0)
If enable_grammar_penalty is false, return penalty as 0.0.
9. Convert LanguageTool matches into issues
Create:
def _extract_grammar_issues(self, prompt: str) -> List[DiagnosticIssue]:
Each LanguageTool match should become an issue:
start = match.offset
end = match.offset + match.errorLength
text = prompt[start:end]
Map category:
def _map_languagetool_category(self, category: str) -> str:
   category = category.upper()

   if "TYPOS" in category:
       return "spelling"
   if "PUNCTUATION" in category:
       return "punctuation"
   if "CASING" in category:
       return "casing"
   if "STYLE" in category:
       return "style"
   if "GRAMMAR" in category:
       return "grammar"
   return "grammar"
Suggestions:
if match.replacements:
   suggestion = "Consider: " + ", ".join(match.replacements[:3])
else:
   suggestion = "Review and rewrite this phrase for grammatical correctness."
Severity:
if issue_type in ("grammar", "spelling"):
   severity = "medium"
else:
   severity = "low"
10. Add ambiguity highlighting
Create:
def _extract_ambiguity_issues(self, prompt: str) -> List[DiagnosticIssue]:
Use existing AMBIGUOUS_TOKENS.
For each ambiguous token, return:
{
   "type": "ambiguity",
   "severity": "medium",
   "start": token.idx,
   "end": token.idx + len(token.text),
   "text": token.text,
   "message": "This word is vague and lowers prompt specificity.",
   "suggestion": "Replace it with a specific topic, object, requirement, or context.",
   "source": "ambiguity_penalty",
   "highlight_color": "orange",
}
11. Add redundancy highlighting
Create:
def _extract_redundancy_issues(self, prompt: str) -> List[DiagnosticIssue]:
Detect consecutive repeated content tokens.
Example:
Explain explain the topic very very clearly.
Should highlight:
Explain explain
very very
Use the span from the first token start to the second token end.
12. Add weak action verb highlighting
Create:
def _extract_weak_action_issues(self, prompt: str) -> List[DiagnosticIssue]:
Use existing WEAK_VERBS.
Only highlight weak verbs that are likely the main instruction:
ROOT verbs
verbs near the start of the sentence
maximum 3 weak action issues per prompt
Message:
This is a weak instruction verb and may reduce actionability.
Suggestion:
Use a stronger verb such as Explain, Compare, Analyze, Create, Generate, or Evaluate.
13. Add missing component issues
Create:
def _extract_missing_component_issues(
   self,
   prompt: str,
   clarity_result: Dict[str, Any]
) -> List[DiagnosticIssue]:
Use the existing clarity diagnostic:
detected_components = clarity_result.get("detected_components", {})
Add prompt-level issues where start, end, and text are None.
If output format is missing:
{
   "type": "missing_output_format",
   "severity": "medium",
   "start": None,
   "end": None,
   "text": None,
   "message": "The prompt does not specify the expected output format.",
   "suggestion": "Specify whether the response should be a table, bullet list, JSON, markdown, paragraph, or step-by-step guide.",
   "source": "clarity_completeness",
}
If constraints are missing:
{
   "type": "missing_constraints",
   "severity": "medium",
   "start": None,
   "end": None,
   "text": None,
   "message": "The prompt does not include constraints or boundaries.",
   "suggestion": "Add limits such as word count, tone, audience, format, tools, scope, or exclusions.",
   "source": "clarity_completeness",
}
If context is missing:
{
   "type": "missing_context",
   "severity": "medium",
   "start": None,
   "end": None,
   "text": None,
   "message": "The prompt lacks context about the task, audience, or purpose.",
   "suggestion": "Add background information, target audience, purpose, or situation.",
   "source": "clarity_completeness",
}
Be careful not to over-penalize short but valid commands like:
Translate this to Filipino.
14. Add too-short issue
Create:
def _extract_length_issues(
   self,
   prompt: str,
   length_penalty: float
) -> List[DiagnosticIssue]:
If length_penalty > 0, add:
{
   "type": "too_short",
   "severity": severity,
   "start": None,
   "end": None,
   "text": None,
   "message": "The prompt is very short and may not provide enough information for a high-quality response.",
   "suggestion": "Add context, expected output format, constraints, or examples.",
   "source": "length_normalization",
   "score_impact": length_penalty,
}
Severity:
if length_penalty >= 0.60:
   severity = "high"
elif length_penalty >= 0.30:
   severity = "medium"
else:
   severity = "low"
15. Collect all issues
Create:
def _collect_diagnostic_issues(
   self,
   prompt: str,
   clarity_result: Dict[str, Any],
   length_penalty: float
) -> List[DiagnosticIssue]:
It should combine:
grammar issues
ambiguity issues
redundancy issues
weak action issues
missing component issues
length issues
Then:
deduplicate issues
add highlight colors
cap to self.config.max_diagnostic_issues
16. Deduplicate issues
Create:
def _deduplicate_issues(
   self,
   issues: List[DiagnosticIssue]
) -> List[DiagnosticIssue]:
Use:
key = (
   issue.get("type"),
   issue.get("start"),
   issue.get("end"),
   issue.get("text"),
)
Skip duplicates.
17. Integrate grammar penalty into quality
Modify _compute_quality():
def _compute_quality(
   self,
   clarity: float,
   specificity: float,
   ambiguity_penalty: float,
   redundancy_penalty: float,
   length_penalty: float,
   structural_bonus: float,
   grammar_penalty: float = 0.0,
) -> float:
Subtract grammar penalty:
penalised = (
   base_quality
   - ambiguity_penalty
   - redundancy_penalty
   - grammar_penalty
)
Then keep existing logic:
length_factor = 1.0 - length_penalty
penalised *= length_factor

final = penalised + structural_bonus
return max(min(final, 1.0), 0.0)
18. Update _score_prompt()
Inside _score_prompt():
compute clarity
compute specificity
compute ambiguity penalty
compute redundancy penalty
compute length penalty
compute structural bonus
compute grammar metrics
compute quality with grammar penalty
collect issues
Return:
{
   "clarity": clarity,
   "specificity": specificity,
   "ambiguity_penalty": ambiguity_penalty,
   "redundancy_penalty": redundancy_penalty,
   "length_penalty": length_penalty,
   "structural_bonus": structural_bonus,
   "grammar_score": grammar["grammar_score"],
   "grammar_penalty": grammar["grammar_penalty"],
   "grammar_error_count": grammar["grammar_error_count"],
   "grammar_error_density": grammar["grammar_error_density"],
   "grammar_categories": grammar["grammar_categories"],
   "quality": quality,
   "issues": issues,
   ...
}
Keep existing clarity diagnostics.
19. Update evaluate()
For raw-only evaluation, return the new grammar fields and issues for the raw prompt.
For raw + candidate evaluation, return grammar fields and issues for the candidate prompt.
Do not remove old fields.
Add to the returned result:
grammar_score=round(metrics["grammar_score"], 4)
grammar_penalty=round(metrics["grammar_penalty"], 4)
grammar_error_count=metrics["grammar_error_count"]
grammar_error_density=round(metrics["grammar_error_density"], 4)
grammar_categories=metrics["grammar_categories"]
issues=metrics["issues"]
20. Add demo tests
In the if __name__ == "__main__": block, add tests for:
make me something about stuff and explain explain it good
Expected:
ambiguity issues
redundancy issue
grammar issues if LanguageTool is available
Create me a table for explain the effect of social media in student learning make it professional and no long
Expected:
grammar issues
still actionable
not fully rejected
Please discuss various things about technology.
Expected:
weak action issue
ambiguity issues
high grammar score but low specificity
Act as an educational technology researcher. Explain the effects of social media on student learning in a 300-word structured summary. Include three benefits, three risks, and one recommendation. Format the response using bullet points.
Expected:
high clarity
high specificity
few or no issues
Explain AI.
Expected:
too-short issue
missing output format
missing constraints
Acceptance Criteria
The task is complete when:
Existing scorer fields still work.
evaluate(raw_prompt) works.
evaluate(raw_prompt, candidate_prompt) works.
Grammar metrics are returned.
issues list is returned.
Ambiguous words have character offsets.
Repeated words have character offsets.
Weak action verbs are detected.
Missing output format, constraints, context, and too-short issues are returned.
LanguageTool grammar issues are returned if available.
The scorer does not crash if LanguageTool is missing.
Grammar penalty is capped and does not dominate the final score.
All output is JSON-serializable.
The frontend can use start and end offsets for text highlighting.
Final Goal
The final scorer should support this behavior:
Input:
make me something about stuff and explain explain it good
Output should include issues similar to:
[
 {
   "type": "ambiguity",
   "text": "something",
   "start": 8,
   "end": 17,
   "message": "This word is vague and lowers prompt specificity.",
   "suggestion": "Replace it with a specific topic, object, requirement, or context."
 },
 {
   "type": "ambiguity",
   "text": "stuff",
   "start": 24,
   "end": 29,
   "message": "This word is vague and lowers prompt specificity.",
   "suggestion": "Replace it with a specific topic, object, requirement, or context."
 },
 {
   "type": "redundancy",
   "text": "explain explain",
   "start": 34,
   "end": 49,
   "message": "This phrase repeats the same word consecutively.",
   "suggestion": "Remove the repeated word."
 }
]
This should turn the scorer into an explainable prompt-quality evaluator with Grammarly-like diagnostic highlighting.


