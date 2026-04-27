from dataset_builder.prompt_templates import Archetype
from dataset_builder.validators import validate


def test_accepts_normal_rewrite():
    raw = "write an instagram caption"
    rewritten = (
        "Write an engaging Instagram caption for a travel photo. Use an adventurous "
        "tone, keep it under 30 words, and include 3 relevant hashtags."
    )
    result = validate(rewritten, raw, Archetype.CREATIVE)
    assert result.ok, result.reason


def test_rejects_empty():
    result = validate("", "anything", Archetype.CONCISE)
    assert not result.ok
    assert "empty" in result.reason


def test_rejects_filler_opener():
    rewritten = "Sure, here is the improved prompt for you to use."
    result = validate(rewritten, "anything", Archetype.CONCISE)
    assert not result.ok
    assert "filler" in result.reason


def test_rejects_diagnostic_label():
    rewritten = "Archetype: Coding\nWrite a Python function that sorts a list."
    result = validate(rewritten, "sort a list", Archetype.CODING)
    assert not result.ok
    assert "diagnostic" in result.reason


def test_rejects_identical_to_raw_when_long():
    raw = "Build me a backend API for managing users with authentication and permissions."
    result = validate(raw, raw, Archetype.CODING)
    assert not result.ok
    assert "identical" in result.reason


def test_accepts_identical_when_raw_already_minimal():
    raw = "list five primes"
    result = validate(raw, raw, Archetype.CONCISE)
    # 3-word raw is below the 6-word threshold, so identical is acceptable.
    assert result.ok


def test_rejects_answer_shaped_code():
    raw = "write a python function to add two numbers"
    rewritten = "def add(a, b):\n    return a + b"
    result = validate(rewritten, raw, Archetype.CODING)
    assert not result.ok
    assert "answer" in result.reason


def test_rejects_answer_shaped_fenced_code():
    raw = "show me a python hello world"
    rewritten = "```python\nprint('hello world')\n```"
    result = validate(rewritten, raw, Archetype.CODING)
    assert not result.ok


def test_accepts_full_modular_coding_rewrite():
    raw = "build me an upload endpoint"
    rewritten = (
        "TASK:\nCreate a backend controller for uploading a file to a database.\n\n"
        "LANGUAGE/STACK:\nUse .NET Core, MySQL, and Dapper.\n\n"
        "OUTPUT:\nProvide the controller code, required model, and database schema."
    )
    result = validate(rewritten, raw, Archetype.CODING)
    assert result.ok, result.reason


def test_rejects_non_instruction_shaped_output():
    raw = "explain recursion"
    rewritten = "Recursion is when a function calls itself. It is useful for divide and conquer."
    result = validate(rewritten, raw, Archetype.ANALYTICAL)
    # This is an answer to the prompt, not an instruction.
    assert not result.ok
