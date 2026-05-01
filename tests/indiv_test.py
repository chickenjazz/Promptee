import sys
import os
import json

sys.path.append(r"c:\Users\quadcore\Documents\CODES\Promptee (v1- Test)")
from tools.heuristic_scorer import HeuristicScorer

scorer = HeuristicScorer()
raw_prompt = """
Chillax is a disposable vape brand. We sell in brick-and-mortar stores all over the world. I want to create offline marketing materials that could be used in a variety of places
 1. vape stores
 2. online distribution channels
 Please help me draft a draft a marketing calendar including events, promotions and corresponding marketing materials"""

opt_prompt = """
ROLE:
You are a compliance-focused marketing strategist with experience in age-restricted consumer products and offline retail campaigns.

TASK:
Create a responsible marketing calendar for Chillax, a disposable vape brand sold through brick-and-mortar retail locations and approved online distribution channels.

OBJECTIVE:
Develop a structured marketing calendar that includes adult-focused events, promotions, and corresponding marketing materials while following applicable laws, age-restriction requirements, and responsible advertising standards.

TARGET CHANNELS:

1. Vape stores and adult-only retail locations
2. Approved online distribution channels with proper age verification
3. Distributor or retailer-facing promotional touchpoints

REQUIREMENTS:

* Focus only on legal-age adult consumers and retail partners.
* Avoid youth-oriented language, imagery, themes, influencers, or promotional tactics.
* Do not make health, safety, cessation, or medical claims.
* Include age-gating, compliance reminders, and responsible-use messaging where appropriate.
* Suggest offline marketing materials such as posters, shelf talkers, counter cards, retailer one-pagers, product information sheets, display signage, and event handouts.
* Include retailer-facing promotions, staff education campaigns, product awareness initiatives, and seasonal merchandising ideas.
* Organize the calendar by month or quarter.
* For each campaign, include:

  * campaign theme
  * target audience
  * event or promotion idea
  * required marketing materials
  * distribution channel
  * compliance notes
  * success metrics

FORMAT:
Present the output as a clear marketing calendar table, followed by brief notes on compliance, implementation, and performance tracking.
"""
res = scorer.evaluate(raw_prompt, opt_prompt)
print("EVALUATE:", json.dumps(res, indent=2))
