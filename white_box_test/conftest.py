"""
Shared fixtures for the white-box test suite.

The HeuristicScorer loads spaCy + SentenceTransformer models on construction
(~5s on first call). A session-scoped fixture pays that cost once and shares
the instance across every test that needs scoring.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def scorer():
    from tools.heuristic_scorer import HeuristicScorer
    return HeuristicScorer()


@pytest.fixture(scope="session")
def scorer_config():
    from tools.heuristic_scorer import ScorerConfig
    return ScorerConfig()


@pytest.fixture
def sample_prompts():
    return {
        "vague": "tell me something about stuff",
        "weak_action": "just talk about climate change",
        "ambiguous": "explain something about various things etc",
        "redundant": "explain this very very clearly",
        "short": "Explain AI",
        "moderate": "Write a summary of machine learning",
        "structured": (
            "Act as a senior data scientist.\n"
            "## Objective\n"
            "Summarise supervised learning in 200 words.\n"
            "## Constraints\n"
            "- Cover 3 algorithms: linear regression, decision trees, SVMs.\n"
            "- Include one numeric example per algorithm.\n"
            "## Output Format\n"
            "Markdown with H2 sections."
        ),
        "near_paraphrase_a": "Summarise the key findings of the report in three bullet points.",
        "near_paraphrase_b": "Provide a three-bullet summary of the report's main findings.",
        "topically_unrelated_a": "Write a Python function that reverses a linked list.",
        "topically_unrelated_b": "Compose a haiku about autumn leaves and a quiet pond.",
    }
