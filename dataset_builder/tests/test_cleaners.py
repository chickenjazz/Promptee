from dataset_builder.cleaners import clean_output


def test_strips_leading_filler_with_comma():
    raw = "Sure, here is your rewritten prompt:\nWrite a haiku about autumn leaves."
    assert clean_output(raw) == "Write a haiku about autumn leaves."


def test_strips_diagnostic_label_lines():
    raw = "Archetype: Creative\nRewritten Prompt:\nWrite an Instagram caption for a beach photo."
    assert clean_output(raw) == "Write an Instagram caption for a beach photo."


def test_strips_wrapping_double_quotes():
    raw = '"Write a 150-word product description for noise-cancelling headphones."'
    assert clean_output(raw) == "Write a 150-word product description for noise-cancelling headphones."


def test_strips_wrapping_curly_quotes():
    raw = "“Summarize the article in 5 bullet points.”"
    assert clean_output(raw) == "Summarize the article in 5 bullet points."


def test_strips_wrapping_code_fence():
    raw = "```text\nExplain recursion in 3 simple bullet points for a beginner.\n```"
    assert clean_output(raw) == "Explain recursion in 3 simple bullet points for a beginner."


def test_collapses_excess_blank_lines():
    raw = "Line one.\n\n\n\nLine two."
    assert clean_output(raw) == "Line one.\n\nLine two."


def test_does_not_strip_quotes_when_unmatched_internal():
    raw = '"He said "hello" to the crowd."'
    out = clean_output(raw)
    # The internal unmatched quote means we should NOT strip the wrapping pair.
    assert out.startswith('"')


def test_handles_none_and_empty():
    assert clean_output(None) == ""  # type: ignore[arg-type]
    assert clean_output("") == ""
    assert clean_output("   \n  ") == ""


def test_preserves_legitimate_prompt_text_starting_with_sure():
    # "Sure" not followed by punctuation/short tail should NOT be stripped as filler.
    raw = "Sure-fire ways to improve customer retention in a retail SaaS product."
    assert clean_output(raw).startswith("Sure-fire")
