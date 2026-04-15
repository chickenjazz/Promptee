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
        
        # Test 1
        r1 = scorer.evaluate("Write a summary of the article.")
        f.write(f"T1 (actionable): {r1}\n")
        
        # Test 2
        r2 = scorer.evaluate("something about cats")
        f.write(f"T2 (vague):      {r2}\n")
        
        # Test 3
        r3 = scorer.evaluate("Write a detailed 500-word essay analyzing the economic impact of renewable energy adoption in European countries during 2023.")
        f.write(f"T3 (specific):   {r3}\n")

        # Test 4: Pair with good match
        r4 = scorer.evaluate(
            "tell me about dogs",
            "Provide a comprehensive overview of domestic dog breeds, including their temperament, size classifications, and care requirements."
        )
        f.write(f"T4 (pair, good): {r4}\n")

        # Test 5: Empty
        r5 = scorer.evaluate("")
        f.write(f"T5 (empty):      {r5}\n") #TODO: EXception handling (must not accept raw)
        
        # Test 6: Low similarity pair
        r6 = scorer.evaluate(
            "tell me about dogs",
            "Explain quantum mechanics and the double-slit experiment in detail."
        )
        f.write(f"T6 (low sim):    {r6}\n")
        
        # Test 7: Ambiguity Penalty
        r7 = scorer.evaluate("tell me maybe probably something somehow about stuff and things")
        f.write(f"T7 (ambiguity):  {r7}\n")

        # Test 8: Redundancy Penalty
        r8 = scorer.evaluate("Output output output very very very very detailed specific specific specific")
        f.write(f"T8 (redundancy): {r8}\n")

        # Test 9: Short Length Penalty
        r9 = scorer.evaluate("write")
        f.write(f"T9 (short):      {r9}\n")

        # Test 10: Structural Bonus
        r10 = scorer.evaluate(
            "Act as an expert python developer.\n"
            "Create a script with the following steps:\n"
            "1. First define imports\n"
            "2. Second write the class\n"
            "- make it neat\n"
            "Output format: Python code block only."
        )
        f.write(f"T10 (structure): {r10}\n")

        # Test 11: Soft semantic penalty (score between 0.4 and 0.70)
        # Choosing a candidate that is related but maybe drifts a bit more
        r11 = scorer.evaluate(
            "how to make my computer faster",
            "What are the recommended software and hardware troubleshooting steps to improve the performance of a slow Windows desktop computer?"
        )
        f.write(f"T11 (soft sim):  {r11}\n")
        
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
        f.write(f"T12 (modularized optimized): {r12}\n")

        f.write("\nAll tests completed successfully.\n")
    except Exception as e:
        f.write(f"\nERROR: {e}\n")
        traceback.print_exc(file=f)
