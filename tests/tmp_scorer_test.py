import sys
import os
import traceback

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Redirect all output to file
log_file = os.path.join(project_root, "test_output.txt")
with open(log_file, "w") as f:
    try:
        f.write("=== Heuristic Scorer Test ===\n")
        
        from tools.heuristic_scorer import HeuristicScorer
        f.write("Import OK\n")
        
        scorer = HeuristicScorer()
        f.write(f"Scorer created. NLP loaded: {scorer._nlp is not None}, ST loaded: {scorer._st_model is not None}\n")
        def log_res(test_name, res):
            if isinstance(res, dict):
                formatted = []
                for k, v in res.items():
                    if isinstance(v, float):
                        formatted.append(f"'{k}': {v * 100:.1f}%")
                    else:
                        formatted.append(f"'{k}': {v}")
                res_str = "{" + ", ".join(formatted) + "}"
            else:
                res_str = str(res)
            f.write(f"{test_name.ljust(30)} {res_str}\n")

        # Test 1
        r1 = scorer.evaluate("Write a summary of the article.")
        log_res("T1 (actionable):", r1)
        
        # Test 2
        r2 = scorer.evaluate("something about cats")
        log_res("T2 (vague):", r2)
        
        # Test 3
        r3 = scorer.evaluate("Write a detailed 500-word essay analyzing the economic impact of renewable energy adoption in European countries during 2023.")
        log_res("T3 (specific):", r3)

        # Test 4: Pair with good match
        r4 = scorer.evaluate(
            "tell me about dogs",
            "Provide a comprehensive overview of domestic dog breeds, including their temperament, size classifications, and care requirements."
        )
        log_res("T4 (pair, good):", r4)

        # Test 5: Empty
        r5 = scorer.evaluate("")
        log_res("T5 (empty):", r5) #TODO: EXception handling (must not accept raw)
        
        # Test 6: Low similarity pair
        r6 = scorer.evaluate(
            "tell me about dogs",
            "Explain quantum mechanics and the double-slit experiment in detail."
        )
        log_res("T6 (low sim):", r6)
        
        # Test 7: Ambiguity Penalty
        r7 = scorer.evaluate("tell me maybe probably something somehow about stuff and things")
        log_res("T7 (ambiguity):", r7)

        # Test 8: Redundancy Penalty
        r8 = scorer.evaluate("Output output output very very very very detailed specific specific specific")
        log_res("T8 (redundancy):", r8)

        # Test 9: Short Length Penalty
        r9 = scorer.evaluate("write")
        log_res("T9 (short):", r9)

        # Test 10: Structural Bonus
        r10 = scorer.evaluate(
            "Act as an expert python developer.\n"
            "Create a script with the following steps:\n"
            "1. First define imports\n"
            "2. Second write the class\n"
            "- make it neat\n"
            "Output format: Python code block only."
        )
        log_res("T10 (structure):", r10)

        # Test 11: Soft semantic penalty (score between 0.4 and 0.70)
        # Choosing a candidate that is related but maybe drifts a bit more
        r11 = scorer.evaluate(
            "how to make my computer faster",
            "What are the recommended software and hardware troubleshooting steps to improve the performance of a slow Windows desktop computer?"
        )
        log_res("T11 (soft sim):", r11)
        
        # Test 12: Usual prompt vs Modularized optimized version
        usual_prompt = "build a website for my car wash business"
        optimized_prompt = """# Role
You are an expert web developer specializing in local business landing pages.

# Objective
Create a modern, responsive single-page website for a local car wash business that drives customer bookings and clearly showcases services.

# Requirements
- Use semantic HTML5 and modern CSS (Flexbox/Grid).
- High contrast, mobile-first design.
- Include the following sections: Hero (with CTA), Services & Pricing, About Us, and Contact (with a booking form).

# Constraints
- Do not use any external frameworks like Tailwind or Bootstrap; use raw CSS.
- Ensure the booking form relies on standard HTML form validation.
- Output the complete code combining HTML and CSS in a single index.html file."""
        r12 = scorer.evaluate(usual_prompt, optimized_prompt)
        log_res("T12 (modularized optimized):", r12)

        # Test 13: Typo-ridden raw prompt vs clean optimized prompt
        typo_raw = "plz mkae a pythn scrip to dwnload vids from yt"
        typo_opt = """# Task
Create a robust Python script to download videos from YouTube.

# Requirements
- Use the `yt-dlp` library.
- Include error handling for invalid URLs.
- Save the downloaded video to a specified output directory.
- Provide a command-line interface (CLI) using `argparse`."""
        r13 = scorer.evaluate(typo_raw, typo_opt)
        log_res("T13 (typo in raw):", r13)

        # Test 14: Extremely brief prompt vs detailed modularized prompt
        brief_raw = "fix my sorting code"
        detailed_opt = """# Objective
Refactor and fix the provided sorting algorithm implementation to ensure correctness and optimal performance.

# Context
The current code attempts to implement a quicksort but fails with a recursion depth error.

# Requirements
1. Identify and fix the logic error causing infinite recursion.
2. Optimize the pivot selection strategy (e.g., median-of-three).
3. Add inline comments explaining the corrections.
4. Ensure the output is formatted as a single Python file."""
        r14 = scorer.evaluate(brief_raw, detailed_opt)
        log_res("T14 (brief vs detailed):", r14)

        # Test 15: Unstructured complex prompt vs structured prompt
        unstructured_raw = "I need an essay about the Roman Empire but make it focus on their military strategies and also include the Punic wars and keep it under 3 pages and use a formal academic tone and please don't use first person pronouns."
        structured_opt = """# Objective
Write a concise, formal academic essay focusing on the military strategies of the Roman Empire.

# Specifications
- **Topic Focus**: Roman military strategies, with dedicated emphasis on the Punic Wars.
- **Length**: Maximum of 3 pages.
- **Tone**: Formal, academic, and objective.

# Constraints
- Strict prohibition on the use of first-person pronouns (I, me, my, we, us).
- Ensure historically accurate terminology."""
        r15 = scorer.evaluate(unstructured_raw, structured_opt)
        log_res("T15 (unstructured vs struct):", r15)

        f.write("\nAll tests completed successfully.\n")
    except Exception as e:
        f.write(f"\nERROR: {e}\n")
        traceback.print_exc(file=f)
