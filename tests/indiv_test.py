import sys
import os
import json

sys.path.append(r"c:\Users\quadcore\Documents\CODES\Promptee (v1- Test)")
from tools.heuristic_scorer import HeuristicScorer

scorer = HeuristicScorer()
raw_prompt = """
create a python script which creates a cellular automator which represents a forest fire. Then display the forest fire using tkinter
"""
opt_prompt = """
"**TASK**
Develop a robust Python application simulating a stochastically driven **2D Cellular Automaton (CA) representing a forest fire**, with a dynamic **Tkinter Graphical User Interface**.

**SIMULATION LOGIC**
Implement the simulation using a localized 8-neighbor (**Moore Neighborhood**) approach. Define four discrete cell states:
1.  **0 - Empty/Sand:** Cannot burn.
2.  **1 - Healthy Tree:** Can catch fire if any neighbor is burning, based on a probability ($P_{\text{ignition}}$).
3.  **2 - Burning:** Transitions to 'Ash' after exactly $T_{\text{burn}}$ timesteps.
4.  **3 - Ash/Charred:** Cannot ignite; can transition back to 'Empty' or 'Tree' via a regrowth probability ($P_{\text{regrowth}}$).

**TECHNICAL REQUIREMENTS**
*   **Performance:** The CA grid logic must be handled using **`numpy` vectorized operations** to ensure smooth execution on grids larger than $100 \times 100$. *Avoid pure Python nested loops for grid updates.*
*   **Animation:** Use the `tkinter.after()` method for managing the simulation timestep, allowing variable speed.

**GUI FEATURES (Tkinter)**
1.  **The Canvas:** Color-code the cell states (e.g., Green=Tree, Red=Burning, Black=Ash, Tan=Empty).
2.  **Simulation Controls:** Provide buttons for **Start/Pause**, **Reset** (re-initializes the forest), and **Single Step**.
3.  **Parameter Sliders:** Add real-time sliders to adjust the key stochastic variables:
    *   $P_{\text{ignition}}$ (Probability of a tree catching fire from a burning neighbor).
    *   Initial Forest Density ($D_{\text{initial}}$).
4.  **Stats Display:** Show the current timestep and the count of Healthy vs. Burning trees.

**OUTPUT**
1.  Complete, well-commented Python source code.
2.  Instructions on installing necessary dependencies (`numpy`)."
"""
res = scorer.evaluate(raw_prompt, opt_prompt)
print("EVALUATE:", json.dumps(res, indent=2))
