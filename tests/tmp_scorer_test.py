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
        f.write(f"T5 (empty):      {r5}\n")
        
        # Test 6: Low similarity pair
        r6 = scorer.evaluate(
            "tell me about dogs",
            "Explain quantum mechanics and the double-slit experiment in detail."
        )
        f.write(f"T6 (low sim):    {r6}\n")
        
        # Test 7: Usual prompt vs Modularized optimized version
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
        r7 = scorer.evaluate(usual_prompt, optimized_prompt)
        f.write(f"T7 (modularized optimized): {r7}\n")
        
        f.write("\nAll tests completed successfully.\n")
    except Exception as e:
        f.write(f"\nERROR: {e}\n")
        traceback.print_exc(file=f)
