import sys
import os
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.heuristic_scorer import HeuristicScorer
from tools.prompt_optimizer import PromptOptimizer

def main():
    parser = argparse.ArgumentParser(description="Test PromptOptimizer rewriting and scoring.")
    parser.add_argument(
        "prompt", 
        nargs="?", 
        type=str, 
        default="create a python script which creates a cellular automator which represents a forest fire. Then display the forest fire using tkinter",
        help="The raw prompt to optimize"
    )
    args = parser.parse_args()

    raw_prompt = args.prompt

    print("--- Loading HeuristicScorer ---")
    scorer = HeuristicScorer()
    
    print("--- Loading PromptOptimizer ---")
    optimizer = PromptOptimizer()
    optimizer.load_model()
    
    print("\n--- Raw Prompt ---")
    print(raw_prompt)

    print("\n--- Optimizing ---")
    optimized = optimizer.rewrite(raw_prompt)
    
    print("\n--- Optimized Prompt ---")
    print(optimized)

    print("\n--- Scoring ---")
    raw_score = scorer.evaluate(raw_prompt)
    opt_score = scorer.evaluate(raw_prompt, optimized)
    
    print("\n--- Raw Score ---")
    print(json.dumps(raw_score, indent=2))
    
    print("\n--- Optimized Score ---")
    print(json.dumps(opt_score, indent=2))
    
    improvement = opt_score.get("quality_improvement", 0.0)
    print(f"\nImprovement: {improvement * 100:.2f}%")

if __name__ == "__main__":
    main()
