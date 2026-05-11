"""
White-box tests for the model-integration layer.

Rubric item: 8. Model Integration — orchestration of the heuristic scorer,
prompt optimizer, and validator under the FastAPI `/optimize_prompt`
endpoint.

We never load the real GPU model. The `optimizer.rewrite` method is patched
with a stub on the module-level singleton in `tools.api`. The HeuristicScorer
*is* real (one is already constructed at module import) so we exercise the
genuine scoring path.
"""

import os

import pytest


# Importing api.py at module load triggers logging.basicConfig + scorer
# construction; both are fine for tests.
from tools import api as api_module


@pytest.fixture
def client(monkeypatch):
    """A TestClient with the GPU model stubbed out.

    Strategy:
      - Patch the singleton optimizer's load_model to a no-op (used by
        lifespan) and mark it loaded so rewrite() does not raise.
      - Replace optimizer.rewrite with a deterministic stub.
      - Patch init_db to a no-op so the test does not write to the real
        SQLite file.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_module.optimizer, "load_model", lambda: None)
    monkeypatch.setattr(api_module.optimizer, "_loaded", True)
    monkeypatch.setattr(api_module, "init_db", lambda: None)

    return TestClient(api_module.app)


# ── Endpoint shape ────────────────────────────────────────────────────────

def test_optimize_endpoint_returns_full_response_shape(monkeypatch, client):
    rewrite_calls = []

    def stub_rewrite(prompt, **kwargs):
        rewrite_calls.append(prompt)
        return (
            "Act as a research analyst.\n"
            "## Objective\n"
            "Summarise the recent climate report in three bullet points.\n"
            "## Output Format\n"
            "Markdown bullets with numeric figures."
        )

    monkeypatch.setattr(api_module.optimizer, "rewrite", stub_rewrite)

    response = client.post(
        "/optimize_prompt",
        json={"prompt": "summarize the climate report", "benchmark": False},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    # Top-level shape contract.
    expected_keys = {
        "raw_prompt", "optimized_prompt", "raw_score", "optimized_score",
        "external_llm_response_raw", "external_llm_response_optimized",
        "external_llm_status_raw", "external_llm_status_optimized",
        "improvement_score", "rewrite_metadata", "issues", "recommendations",
        "institutional_guideline", "validation",
    }
    assert expected_keys.issubset(body.keys())

    # The stub was actually invoked.
    assert len(rewrite_calls) == 1
    assert rewrite_calls[0] == "summarize the climate report"


def test_optimize_endpoint_rejects_empty_prompt(client):
    response = client.post("/optimize_prompt", json={"prompt": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_optimize_endpoint_rejects_whitespace_only_prompt(client):
    response = client.post("/optimize_prompt", json={"prompt": "    "})
    assert response.status_code == 400


def test_optimize_returns_503_when_model_not_loaded(monkeypatch, client):
    """If the optimizer raises RuntimeError (model not loaded), the API must
    surface a 503, not a 500 or silent failure."""

    def boom(prompt, **kwargs):
        raise RuntimeError("PromptOptimizer model is not loaded.")

    monkeypatch.setattr(api_module.optimizer, "rewrite", boom)

    response = client.post(
        "/optimize_prompt", json={"prompt": "anything substantive enough"}
    )
    assert response.status_code == 503


# ── Self-annealing: rejected rewrite falls back to raw ────────────────────

def test_endpoint_falls_back_to_raw_when_rewrite_drifts(monkeypatch, client):
    """A nonsense rewrite (semantically unrelated) must be rejected and the
    API response must echo the raw prompt with improvement_score==0.0.

    Reads tools/api.py:208-218 — the self-annealing branch that triggers
    when `rejected=True` or `improvement<0`.
    """

    def drift_rewrite(prompt, **kwargs):
        # Wildly off-topic: should fail the semantic gate.
        return "Compose a limerick about a cat that loves jazz music."

    raw = "Write a Python function that reverses a linked list."
    monkeypatch.setattr(api_module.optimizer, "rewrite", drift_rewrite)

    response = client.post("/optimize_prompt", json={"prompt": raw})
    assert response.status_code == 200
    body = response.json()

    # Self-annealing should kick in.
    assert body["optimized_prompt"] == raw
    assert body["improvement_score"] == 0.0


def test_endpoint_includes_validation_block(monkeypatch, client):
    """Validation must always be present in the response, with a status
    field set to 'valid' or 'invalid'."""

    def good_rewrite(prompt, **kwargs):
        return (
            "Act as a software engineer.\n"
            "## Task\n"
            "Reverse a singly linked list iteratively in Python.\n"
            "## Output\n"
            "Provide a function definition with type hints."
        )

    monkeypatch.setattr(api_module.optimizer, "rewrite", good_rewrite)

    response = client.post(
        "/optimize_prompt",
        json={"prompt": "Write a Python function that reverses a linked list."},
    )
    body = response.json()
    assert "validation" in body
    assert body["validation"]["status"] in ("valid", "invalid")
    assert "issues" in body["validation"]


def test_rewrite_metadata_includes_archetype_and_modularity(monkeypatch, client):
    """The orchestration runs detect_archetype() and modularity_for() and
    surfaces both into rewrite_metadata."""

    monkeypatch.setattr(
        api_module.optimizer, "rewrite",
        lambda p, **k: "Reverse a list in Python with type hints.",
    )

    response = client.post(
        "/optimize_prompt",
        json={"prompt": "Write a Python function to reverse a list."},
    )
    body = response.json()
    meta = body["rewrite_metadata"]
    assert "archetype" in meta
    assert "modularity" in meta
    assert isinstance(meta["archetype"], str)
    assert isinstance(meta["modularity"], str)


def test_external_llm_off_by_default(monkeypatch, client):
    """When benchmark=False (default), the external LLM round-trip is
    skipped; status fields are 'off' and response strings are empty."""

    monkeypatch.setattr(
        api_module.optimizer, "rewrite",
        lambda p, **k: "Refined: " + p,
    )

    response = client.post(
        "/optimize_prompt",
        json={"prompt": "Summarise the news in three bullet points."},
    )
    body = response.json()
    assert body["external_llm_status_raw"] == "off"
    assert body["external_llm_status_optimized"] == "off"
    assert body["external_llm_response_raw"] == ""
    assert body["external_llm_response_optimized"] == ""
