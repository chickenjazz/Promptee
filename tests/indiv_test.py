import sys
import os
import json

sys.path.append(r"c:\Users\quadcore\Documents\CODES\Promptee (v1- Test)")
from tools.heuristic_scorer import HeuristicScorer

scorer = HeuristicScorer()
raw_prompt = """
Draft a contract between a startup and an advisor of the advisory board, focusing on the following key points:

- The contract should include a jurisdiction clause specifying the state/country.
- It should outline compensation options, such as equity through NSOs or cash payments.
- Services and commitments should detail the advisor's expected involvement, like meeting frequency and types of support provided.
- Essential clauses should cover confidentiality, intellectual property rights, independent contractor status, conflict of interest, and termination procedures.
- Ensure the contract includes clear sections and placeholders for specific details like names and dates."""
opt_prompt = """
Role: Legal Documentation Specialist for Startups

Objective: Draft a comprehensive Advisor Agreement (Individual Consultant) between a high-growth startup and a member of its Advisory Board.

Contractual Parameters:

Jurisdiction: [INSERT STATE/COUNTRY, e.g., Delaware, USA].

Compensation: * Equity: [e.g., 0.25% of Common Stock] via an NSO (Non-Qualified Stock Option) with a [e.g., 2-year] vesting schedule and a [e.g., 3-month] cliff.

Cash: [e.g., None, or $X per month].

Services & Commitment: Define a commitment of [e.g., 5 hours per month], including one monthly strategy call and ad-hoc email support.

Term: [e.g., At-will, or 12-month fixed term].

Essential Clauses to Include:

Confidentiality & Non-Disclosure: Strict protection of startup trade secrets.

Intellectual Property Assignment: Ensuring any ideas or work produced during the advisory period belong to the company.

Independent Contractor Status: Explicitly stating the advisor is not an employee and is responsible for their own taxes.

Conflict of Interest: Ensuring the advisor does not currently advise direct competitors.

Termination: Procedures for ending the agreement with [e.g., 30 days] notice by either party.

Output Format:

A formal, professional contract template with clearly labeled sections and [BRACKETED] placeholders for specific names and dates.
"""
res = scorer.evaluate(raw_prompt, opt_prompt)
print("EVALUATE:", json.dumps(res, indent=2))
