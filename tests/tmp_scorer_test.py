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
        
        f.write("\nAll tests completed successfully.\n")
    except Exception as e:
        f.write(f"\nERROR: {e}\n")
        traceback.print_exc(file=f)
