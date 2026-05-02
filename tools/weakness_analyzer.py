"""
Weakness Analyzer — per-span prompt weakness detection.

Surfaces the *location* of weakness signals that the heuristic scorer already
detects internally but only emits as aggregate scores. Output drives the UI's
inline highlighting layer.

Public surface:
    WeaknessAnalyzer().analyze(prompt: str) -> {
        "spans": [ {start, end, text, category, severity, message}, ... ],
        "missing_components": [ {component, message}, ... ],
    }
"""

import os
import re
import sys
import logging
from typing import Dict, Any, List, Optional

if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
from spellchecker import SpellChecker

from tools.heuristic_scorer import (
    AMBIGUOUS_TOKENS,
    WEAK_VERBS,
    FRAGMENT_INDICATORS,
    MODAL_VERBS,
    STRONG_VERBS,
    _LABELED_SECTION_PATTERN,
)

logger = logging.getLogger("promptee.weakness_analyzer")


# Technical tokens / acronyms that pyspellchecker doesn't know but are valid.
TECH_ALLOWLIST: frozenset = frozenset({
    "api", "apis", "json", "csv", "tsv", "xml", "yaml", "html", "css", "sql",
    "url", "uri", "uuid", "regex", "regexp", "http", "https", "ssh", "tcp",
    "udp", "ip", "ux", "ui", "cli", "gui", "ide", "sdk", "orm", "crud",
    "cors", "jwt", "oauth", "saas", "cdn", "dns", "vpn", "ssl", "tls",
    "ml", "ai", "llm", "nlp", "gpu", "cpu", "ram", "ssd", "io",
    "py", "js", "ts", "tsx", "jsx", "md", "rb", "go", "rs",
    "argparse", "numpy", "pandas", "scipy", "matplotlib", "pytorch",
    "tensorflow", "sklearn", "spacy", "fastapi", "uvicorn", "pydantic",
    "flask", "django", "react", "vue", "svelte", "nodejs", "npm", "pip",
    "docker", "kubernetes", "k8s", "github", "gitlab", "linux", "macos",
    "qwen", "lora", "transformers", "huggingface",
    "stdin", "stdout", "stderr", "stdlib", "iterable", "iter",
    "lookup", "lookups", "args", "kwargs", "init", "len", "str", "int",
    "bool", "dict", "tuple", "frozenset", "regex",
})

# Output-format hints — drives missing_output_format detection.
_OUTPUT_FORMAT_PATTERN = re.compile(
    r"\b(?:format|output)\s*:|"
    r"\bin\s+(?:json|xml|yaml|csv|markdown|html|table|bullet|list)\b|"
    r"\bas\s+(?:a\s+)?(?:list|table|json|markdown|bullet|numbered|paragraph)\b|"
    r"\breturn\s+(?:a\s+)?(?:list|json|dict|array|string|object)\b",
    re.IGNORECASE,
)

# Role/persona declarations — drives missing_role detection.
_ROLE_PATTERN = re.compile(
    r"\b(?:act|behave|respond)\s+as\b|"
    r"\b(?:role|persona)\s*[:]\s*\S|"
    r"\byou\s+are\s+a[n]?\b|"
    r"\bas\s+a[n]?\s+(?:expert|specialist|professional|senior|tutor|teacher|engineer|developer|analyst)\b",
    re.IGNORECASE,
)

# Identifier-shaped tokens that should be skipped by the typo pass.
_IDENTIFIER_SHAPE = re.compile(r"[A-Z].*[a-z].*[A-Z]|_|[0-9]|/|\\")


def _looks_like_identifier(text: str) -> bool:
    """True if the token looks like code (camelCase, snake_case, has digits)."""
    return bool(_IDENTIFIER_SHAPE.search(text))


class WeaknessAnalyzer:
    """Locate per-span weaknesses + document-level gaps in a prompt."""

    def __init__(self):
        self._nlp: Optional[Any] = None
        self._spell: Optional[SpellChecker] = None
        self._load_models()

    def _load_models(self) -> None:
        try:
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("WeaknessAnalyzer: spaCy model loaded.")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
        try:
            self._spell = SpellChecker()
            logger.info("WeaknessAnalyzer: SpellChecker loaded.")
        except Exception as e:
            logger.warning(f"SpellChecker failed to load: {e}")
            self._spell = None

    # ── Public API ────────────────────────────────────────────────────

    def analyze(self, prompt: str) -> Dict[str, Any]:
        if not prompt or not prompt.strip() or self._nlp is None:
            return {"spans": [], "missing_components": []}

        doc = self._nlp(prompt)
        spans: List[Dict[str, Any]] = []

        spans.extend(self._detect_ambiguity(doc))
        spans.extend(self._detect_weak_verbs(doc))
        spans.extend(self._detect_passive_voice(doc))
        spans.extend(self._detect_fragments(doc))
        spans.extend(self._detect_redundancy(doc))
        spans.extend(self._detect_typos(doc))

        spans = self._dedupe_and_sort(spans)
        missing = self._detect_missing_components(prompt, doc)

        return {"spans": spans, "missing_components": missing}

    # ── Span detectors ────────────────────────────────────────────────

    def _detect_ambiguity(self, doc) -> List[Dict[str, Any]]:
        out = []
        for tok in doc:
            if tok.is_punct or tok.is_space:
                continue
            # AMBIGUOUS_TOKENS like "sort", "kind", "lot", "thing" are only
            # vague in noun/adjective/adverb context (e.g. "sort of thing").
            # Skip when used as a verb (e.g. "sorts a list").
            if tok.pos_ == "VERB":
                continue
            if tok.lemma_.lower() in AMBIGUOUS_TOKENS:
                out.append({
                    "start": tok.idx,
                    "end": tok.idx + len(tok.text),
                    "text": tok.text,
                    "category": "ambiguity",
                    "severity": "low",
                    "message": (
                        f"'{tok.text}' is vague — replace with a concrete "
                        f"noun, quantity, or constraint."
                    ),
                })
        return out

    def _detect_weak_verbs(self, doc) -> List[Dict[str, Any]]:
        out = []
        for tok in doc:
            if tok.pos_ != "VERB":
                continue
            # Flag weak verbs when they're root or directly govern a clause.
            if tok.dep_ not in {"ROOT", "ccomp", "xcomp", "conj", "advcl"}:
                continue
            if tok.lemma_.lower() in WEAK_VERBS:
                out.append({
                    "start": tok.idx,
                    "end": tok.idx + len(tok.text),
                    "text": tok.text,
                    "category": "weak_verb",
                    "severity": "medium",
                    "message": (
                        f"'{tok.text}' is non-actionable — try a stronger "
                        f"directive verb (e.g. generate, analyze, list, build)."
                    ),
                })
        return out

    def _detect_passive_voice(self, doc) -> List[Dict[str, Any]]:
        out = []
        for tok in doc:
            if tok.dep_ != "nsubjpass":
                continue
            verb = tok.head
            # Span = from leftmost auxpass/aux child to verb end.
            start_idx = verb.idx
            for child in verb.children:
                if child.dep_ in {"auxpass", "aux"} and child.idx < start_idx:
                    start_idx = child.idx
            end_idx = verb.idx + len(verb.text)
            out.append({
                "start": start_idx,
                "end": end_idx,
                "text": doc.text[start_idx:end_idx],
                "category": "passive_voice",
                "severity": "medium",
                "message": (
                    "Passive voice — rewrite as an active command "
                    "(e.g. 'X writes Y' instead of 'Y is written by X')."
                ),
            })
        return out

    def _detect_fragments(self, doc) -> List[Dict[str, Any]]:
        out = []
        for sent in doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            lowered = text.lower()
            matched = None
            for indicator in FRAGMENT_INDICATORS:
                if lowered.startswith(indicator + " ") or lowered == indicator:
                    matched = indicator
                    break
            if matched:
                out.append({
                    "start": sent.start_char,
                    "end": sent.end_char,
                    "text": sent.text,
                    "category": "fragment",
                    "severity": "high",
                    "message": (
                        "Sentence fragment — merge this into the previous "
                        "instruction or rewrite as a complete sentence."
                    ),
                })
        return out

    def _detect_redundancy(self, doc) -> List[Dict[str, Any]]:
        out = []
        prev = None
        for tok in doc:
            if tok.is_punct or tok.is_space:
                prev = tok
                continue
            if prev is not None and not prev.is_punct and not prev.is_space:
                if (
                    tok.lemma_.lower() == prev.lemma_.lower()
                    and tok.lemma_.lower().isalpha()
                ):
                    out.append({
                        "start": tok.idx,
                        "end": tok.idx + len(tok.text),
                        "text": tok.text,
                        "category": "redundancy",
                        "severity": "low",
                        "message": (
                            f"Repeated word — '{tok.text}' duplicates the "
                            f"previous token."
                        ),
                    })
            prev = tok
        return out

    def _detect_typos(self, doc) -> List[Dict[str, Any]]:
        if self._spell is None:
            return []
        out = []
        for tok in doc:
            text = tok.text
            if len(text) <= 2 or not text.isalpha():
                continue
            lowered = text.lower()
            if lowered in TECH_ALLOWLIST:
                continue
            if _looks_like_identifier(text):
                continue
            # Skip proper nouns — likely names/places the dictionary doesn't have.
            if tok.pos_ == "PROPN":
                continue
            if not self._spell.unknown([lowered]):
                continue
            suggestion = self._spell.correction(lowered)
            if not suggestion or suggestion == lowered:
                msg = f"Possible spelling error in '{text}'."
            else:
                msg = f"Possible spelling error — did you mean '{suggestion}'?"
            out.append({
                "start": tok.idx,
                "end": tok.idx + len(text),
                "text": text,
                "category": "typo",
                "severity": "low",
                "message": msg,
            })
        return out

    # ── Document-level gap detection ──────────────────────────────────

    def _detect_missing_components(self, prompt: str, doc) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        token_count = sum(1 for t in doc if not t.is_punct and not t.is_space)

        if not _ROLE_PATTERN.search(prompt) and not _LABELED_SECTION_PATTERN.search(prompt):
            out.append({
                "component": "role",
                "message": (
                    "No role or persona defined — consider adding "
                    "'Act as a …' or 'You are a …' to anchor the response style."
                ),
            })

        has_directive = any(
            t.pos_ == "VERB" and (
                t.lemma_.lower() in STRONG_VERBS
                or t.lemma_.lower() in MODAL_VERBS
            )
            for t in doc
        )
        if not has_directive:
            out.append({
                "component": "objective",
                "message": (
                    "No clear directive verb — start with an action verb "
                    "(e.g. generate, write, analyze, list) so the task is unambiguous."
                ),
            })

        if not _OUTPUT_FORMAT_PATTERN.search(prompt):
            out.append({
                "component": "output_format",
                "message": (
                    "No output format specified — say what shape the answer "
                    "should take (e.g. 'as a JSON object', 'as a numbered list')."
                ),
            })

        has_constraint_signal = any(
            t.dep_ in {"nummod", "amod"} for t in doc
        )
        if not has_constraint_signal:
            out.append({
                "component": "constraints",
                "message": (
                    "No constraints or qualifiers — add specifics like length, "
                    "tone, examples to include, or things to avoid."
                ),
            })

        if token_count < 8:
            out.append({
                "component": "length",
                "message": (
                    f"Prompt is very short ({token_count} content tokens) — "
                    f"add context so the model has enough to work with."
                ),
            })

        return out

    # ── Span post-processing ──────────────────────────────────────────

    @staticmethod
    def _dedupe_and_sort(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort by start; drop later spans that overlap an earlier higher-severity one."""
        if not spans:
            return spans

        severity_rank = {"high": 3, "medium": 2, "low": 1}
        spans = sorted(spans, key=lambda s: (s["start"], -severity_rank.get(s["severity"], 0)))

        result: List[Dict[str, Any]] = []
        for s in spans:
            if not result:
                result.append(s)
                continue
            last = result[-1]
            if s["start"] < last["end"]:
                # Overlap — keep the higher severity, or the earlier-added one if tied.
                if severity_rank.get(s["severity"], 0) > severity_rank.get(last["severity"], 0):
                    result[-1] = s
                # else drop s
            else:
                result.append(s)
        return result
