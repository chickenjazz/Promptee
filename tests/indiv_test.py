import sys
import os
import json

sys.path.append(r"c:\Users\quadcore\Documents\CODES\Promptee (v1- Test)")
from tools.heuristic_scorer import HeuristicScorer

scorer = HeuristicScorer()
raw_prompt = """
Please kindly generate some sort of email that maybe asks for an extension politely in a professional tone for an unspecified matter and keep it exact"""

opt_prompt = """
You are a professional email writing assistant. Write a polite and professional email requesting an extension for an unspecified matter.

Use a respectful, clear, and responsible tone. Keep the request general without mentioning the specific task, deadline, or reason. Express appreciation for the recipient’s understanding and make the email suitable for workplace, school, or formal communication.

The email should include a subject line, greeting, short body, and professional closing. Keep it concise and exact, without unnecessary details.
"""
res = scorer.evaluate(raw_prompt, opt_prompt)
print("EVALUATE:", json.dumps(res, indent=2))
