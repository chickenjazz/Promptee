from dataset_builder.prompt_templates import (
    Archetype,
    Modularity,
    build_plan,
    detect_archetype,
    modularity_for,
)


def test_creative_caption():
    assert detect_archetype("write an engaging instagram caption for a travel photo") == Archetype.CREATIVE


def test_coding_python():
    assert detect_archetype("debug this python code that sorts a list") == Archetype.CODING


def test_coding_react():
    assert detect_archetype("build a react component for a search bar") == Archetype.CODING


def test_structured_lesson_plan():
    assert detect_archetype("create a lesson plan on fractions for 4th graders") == Archetype.STRUCTURED


def test_conversational_help():
    assert detect_archetype("help me prepare for a job interview") == Archetype.CONVERSATIONAL


def test_analytical_compare():
    assert detect_archetype("compare mitosis and meiosis for a biology student") == Archetype.ANALYTICAL


def test_concise_summary():
    assert detect_archetype("summarize this article in 5 bullet points") == Archetype.CONCISE


def test_concise_short_unknown_falls_back():
    # No keywords match, short prompt — defaults to Concise.
    assert detect_archetype("blue sky tomorrow") == Archetype.CONCISE


def test_modularity_mapping():
    assert modularity_for(Archetype.CREATIVE) == Modularity.SEMI
    assert modularity_for(Archetype.CODING) == Modularity.FULL
    assert modularity_for(Archetype.CONVERSATIONAL) == Modularity.NATURAL
    assert modularity_for(Archetype.STRUCTURED) == Modularity.FULL
    assert modularity_for(Archetype.ANALYTICAL) == Modularity.SEMI
    assert modularity_for(Archetype.CONCISE) == Modularity.MINIMAL


def test_build_plan_returns_full_messages():
    plan = build_plan("write a haiku about the moon")
    assert plan.archetype == Archetype.CREATIVE
    assert plan.modularity == Modularity.SEMI
    assert "Prompt Rewriter" in plan.system_instruction
    assert "Semi Modular" in plan.system_instruction
    assert "haiku about the moon" in plan.user_message
    assert "Final rewritten prompt only:" in plan.user_message


def test_build_plan_handles_empty_prompt():
    plan = build_plan("")
    assert plan.archetype == Archetype.CONCISE
