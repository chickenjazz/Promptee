import sys
import os

sys.path.append(r"c:\Users\quadcore\Documents\CODES\Promptee (v1- Test)")
from tools.prompt_optimizer import PromptOptimizer
from dataset_builder.prompt_templates import detect_archetype, modularity_for

prompt = """Gunnar and Emma, who are known for their love of collecting unique board games, find themselves in a challenging predicament caused by a severe storm that has led to a power outage. Determined to keep themselves entertained, they decide to create a new game using two dice each. The objective of the game is for the player with the higher sum of their respective dice to emerge victorious. In the event of a tie, the game will end in a draw. Your task is to thoroughly analyze the given descriptions of the dice and determine which player, Gunnar or Emma, has a higher probability of winning.

 Each die possesses its own unique attributes, with the numbers on its sides ranging from the minimum value 'a' to the maximum value 'b', inclusively. The input consists of four integers, represented as a1, b1, a2, and b2, which describe the dice owned by Gunnar. The first die has numbers ranging from 'a1' to 'b1', while the second die has numbers ranging from 'a2' to 'b2'. It can be assumed that the values of a1, b1, a2, and b2 fall within the range of 1 to 100. Additionally, each die must have at least four sides (ai + 3 <= bi).

 Please provide a clear output indicating which player is more likely to win. If both players have an equal probability of winning, please indicate a "Tie" in the output."""

optimizer = PromptOptimizer()
optimizer.load_model()
archetype = detect_archetype(prompt)

print(f"Detected Archetype: {archetype}")

optimized = optimizer.rewrite(prompt)

print("=== RAW REWRITE ===")
print(optimized)
