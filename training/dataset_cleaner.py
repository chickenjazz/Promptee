import pandas as pd
import re

df = pd.read_csv("dataset/normal_bad_prompts.csv")

# remove blank cells in the prompt column
df = df[df.prompt.notna()]

# token count
df["tokens"] = df.prompt.str.split().str.len()

# remove too short
df = df[df.tokens >= 3]

# numeric suffix artifacts
bad_suffix = df.prompt.str.contains(r'\b[a-zA-Z]+\d{1,3}$', regex=True)
df = df[~bad_suffix]

# exact dedupe
df = df.drop_duplicates(subset=["prompt"])

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.heuristic_scorer import HeuristicScorer

scorer = HeuristicScorer()
df["raw_quality"] = df["prompt"].apply(lambda p: scorer.evaluate(str(p))["raw_quality"])

# keep only bad quality
df = df[df.raw_quality < 0.25]

# length buckets
def bucket(n):
    if n <= 8:
        return "short"
    elif n <= 20:
        return "medium"
    return "long"

df["length_bucket"] = df.tokens.apply(bucket)

df.to_csv("dataset/cleaned_bad_prompts3.csv", index=False)