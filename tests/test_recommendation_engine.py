from tools.recommendation_engine import build_recommendations


def test_recommendations_for_ambiguity():
    result = build_recommendations(
        raw_prompt="Tell me something about AI",
        issues=[{"type": "ambiguity"}],
        archetype="Concise",
    )
    assert any("vague" in recommendation.lower() for recommendation in result["recommendations"])


def test_simple_prompt_guideline_does_not_force_templates():
    result = build_recommendations(
        raw_prompt="What is gravity?",
        issues=[],
        archetype="Concise",
    )
    assert "rigid prompt templates are optional" in result["institutional_guideline"].lower()


def test_complex_prompt_guideline_pushes_structure():
    result = build_recommendations(
        raw_prompt=(
            "Design an enterprise authentication system with OAuth2, OIDC, MFA, and audit "
            "logging requirements integrated with our existing identity provider"
        ),
        issues=[],
        archetype="Coding",
    )
    assert "structured prompts" in result["institutional_guideline"].lower()


def test_includes_simple_prompt_recommendation_for_short_questions():
    result = build_recommendations(
        raw_prompt="What is gravity?",
        issues=[{"type": "missing_context"}],
        archetype="Concise",
    )
    assert any(
        "concise prompt may be enough" in r.lower() for r in result["recommendations"]
    )


def test_recommendations_never_empty():
    result = build_recommendations(
        raw_prompt="Compare relational and document databases for a multi-tenant SaaS workload",
        issues=[],
        archetype="Analytical",
    )
    assert len(result["recommendations"]) >= 1
