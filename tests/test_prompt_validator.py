from tools.prompt_validator import validate_rewrite


def test_rejects_meta_prompt_drift():
    result = validate_rewrite(
        "What is gravity?",
        "Create a prompt that asks an AI to explain gravity.",
    )
    assert result["status"] == "invalid"
    assert any(issue["type"] == "meta_prompt_drift" for issue in result["issues"])


def test_rejects_answer_instead_of_rewrite():
    result = validate_rewrite(
        "What is gravity?",
        "The answer is gravity is the force that attracts objects toward each other.",
    )
    assert result["status"] == "invalid"
    assert any(issue["type"] == "answer_instead_of_rewrite" for issue in result["issues"])


def test_rejects_empty_output():
    result = validate_rewrite("What is gravity?", "   ")
    assert result["status"] == "invalid"
    assert any(issue["type"] == "empty_output" for issue in result["issues"])


def test_rejects_unexpected_code_block():
    result = validate_rewrite(
        "What is gravity?",
        "```python\nprint('gravity')\n```",
    )
    assert result["status"] == "invalid"
    assert any(issue["type"] == "unexpected_code_output" for issue in result["issues"])


def test_allows_code_block_for_coding_prompt():
    result = validate_rewrite(
        "Write a python function that adds two numbers",
        "ROLE: Senior Python engineer\n\nTASK:\nImplement an `add(a, b)` function.\n\n```python\n# scaffolding only\n```",
    )
    assert result["status"] == "valid"


def test_accepts_structured_good_rewrite():
    result = validate_rewrite(
        "What is gravity?",
        "ROLE:\nYou are a physics educator.\n\nTASK:\nExplain gravity in simple terms for a beginner.",
    )
    assert result["status"] == "valid"
    assert result["issues"] == []
