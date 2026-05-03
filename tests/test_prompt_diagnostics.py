from tools.prompt_diagnostics import find_prompt_issues


def test_detects_ambiguous_tokens():
    issues = find_prompt_issues("Tell me something about AI etc")
    assert any(issue["type"] == "ambiguity" and issue["span"] == "something" for issue in issues)
    assert any(issue["type"] == "ambiguity" and issue["span"] == "etc" for issue in issues)


def test_detects_redundancy():
    issues = find_prompt_issues("Explain this very very clearly")
    assert any(issue["type"] == "redundancy" for issue in issues)


def test_detects_missing_output_format():
    issues = find_prompt_issues("Explain artificial intelligence")
    assert any(issue["type"] == "missing_output_format" for issue in issues)


def test_detects_too_short():
    issues = find_prompt_issues("Explain AI")
    assert any(issue["type"] == "too_short" for issue in issues)


def test_detects_missing_context():
    issues = find_prompt_issues("Write a summary of artificial intelligence trends in business")
    assert any(issue["type"] == "missing_context" for issue in issues)


def test_detects_answering_risk_for_question():
    issues = find_prompt_issues("What is gravity?")
    assert any(issue["type"] == "answering_risk" for issue in issues)


def test_detects_meta_prompt_drift_risk():
    issues = find_prompt_issues("Create a prompt that explains gravity to a beginner")
    assert any(issue["type"] == "meta_prompt_drift" for issue in issues)


def test_detects_weak_action_phrase():
    issues = find_prompt_issues("Just talk about climate change")
    assert any(issue["type"] == "weak_action" for issue in issues)


def test_clean_prompt_has_no_high_severity_issues():
    text = "Explain photosynthesis to a beginner student in three short bullets, in a paragraph format."
    issues = find_prompt_issues(text)
    assert not any(i["severity"] == "high" for i in issues)
