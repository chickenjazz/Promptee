import sys
import os
import json

sys.path.append(r"c:\Users\quadcore\Documents\CODES\Promptee (v1- Test)")
from tools.heuristic_scorer import HeuristicScorer

scorer = HeuristicScorer()
raw_prompt = """
Please tell me everything about php buffering and command line execution sequence
"""

opt_prompt = """
You are an expert PHP developer and systems instructor. Explain PHP buffering and command-line execution sequence in a clear, complete, and beginner-friendly way.
"""
res = scorer.evaluate(raw_prompt, opt_prompt)
print("EVALUATE:", json.dumps(res, indent=2))
