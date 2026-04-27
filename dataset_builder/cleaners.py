"""Output cleaning per spec §Cleaning Rules.

Apply the cleaner before the validator: it normalizes obvious model artefacts
(filler openers, wrapping fences, accidental labels) so the validator's job is
to enforce semantic invariants rather than fight surface noise.
"""

from __future__ import annotations

import re
from typing import Iterable

from dataset_builder.config import DIAGNOSTIC_LABELS, FILLER_OPENERS

_QUOTE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),  # curly double quotes
    ("‘", "’"),  # curly single quotes
    ("`", "`"),
)

_FENCE_RE = re.compile(r"^```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)\n```$", re.DOTALL)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _strip_wrapping_quotes(text: str) -> str:
    if len(text) < 2:
        return text
    for left, right in _QUOTE_PAIRS:
        if text.startswith(left) and text.endswith(right):
            inner = text[len(left): -len(right)]
            # Only strip if the wrapping pair encloses the whole text (no
            # unmatched quotes inside that would indicate the quotes are
            # legitimate punctuation).
            if left not in inner and right not in inner:
                return inner.strip()
    return text


def _strip_wrapping_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


def _strip_leading_filler(text: str, fillers: Iterable[str] = FILLER_OPENERS) -> str:
    """Drop a single leading filler line if present.

    Conservative — only removes an opener when it terminates with a comma,
    colon, period, exclamation, or end-of-line, so legitimate prompt content
    like "Sure as the sun rises..." is preserved.
    """
    if not text:
        return text
    first_line, _, rest = text.partition("\n")
    lowered = first_line.lower().strip()
    for opener in fillers:
        if lowered == opener:
            return rest.lstrip()
        if lowered.startswith(opener):
            after = lowered[len(opener):]
            if after[:1] in {",", ":", ".", "!", " ", ""}:
                # Drop only if the rest of the first line looks like filler too
                # (very short or a stock phrase). Otherwise keep the line.
                tail = first_line[len(opener):].strip(" ,.:!")
                if len(tail.split()) <= 8:
                    return rest.lstrip()
    return text


def _strip_diagnostic_labels(text: str, labels: Iterable[str] = DIAGNOSTIC_LABELS) -> str:
    """Drop lines that begin with a diagnostic label (e.g. 'Archetype: Creative')."""
    out_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if any(stripped.startswith(label) for label in labels):
            continue
        out_lines.append(line)
    cleaned = "\n".join(out_lines).strip("\n")
    return cleaned


def clean_output(text: str) -> str:
    """Apply the full spec §Cleaning Rules pipeline."""
    if text is None:
        return ""
    cleaned = text.strip()
    cleaned = _strip_wrapping_fence(cleaned)
    cleaned = _strip_wrapping_quotes(cleaned)
    cleaned = _strip_leading_filler(cleaned)
    cleaned = _strip_diagnostic_labels(cleaned)
    cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()
