import sys
import os
import time
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.heuristic_scorer import HeuristicScorer

def main():
    # Configure logging to show info during load, but cleanly
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    print("Loading Heuristic Scorer models (this might take a few seconds)...")
    start_time = time.time()
    try:
        scorer = HeuristicScorer()
    except Exception as e:
        print(f"Failed to load scorer: {e}")
        return
    
    print(f"Loaded in {time.time() - start_time:.2f} seconds.")
    
    # Silence the logger after loading to keep evaluation clean
    logging.getLogger("promptee.heuristic_scorer").setLevel(logging.WARNING)
    
    print("\n" + "="*80)
    print(" Heuristic Scorer - Interactive Prompt Evaluator")
    print("="*80)
    
    while True:
        try:
            print("\n" + "-"*80)
            print("Paste your prompt below.")
            print("To evaluate: Type 'EOF' on a new line and press Enter (or press Ctrl-Z / Ctrl-D).")
            print("To exit: Press Ctrl-C.")
            print("-"*80 + "\n")
            
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                
                if line.strip() == 'EOF':
                    break
                lines.append(line)
                
            prompt = "\n".join(lines).strip()
            
            if not prompt:
                print("\nEmpty prompt detected, please try again.")
                continue
                
            print("\nEvaluating prompt...")
            start_time = time.time()
            result = scorer.evaluate(prompt)
            eval_time = time.time() - start_time
            
            print(f"\n[ SCORING RESULTS ] (evaluated in {eval_time:.2f}s)")
            print("="*50)
            
            # Display layout
            groups = [
                ("Overall Quality", ['final_score']),
                ("Components", ['clarity', 'specificity']),
                ("Penalties & Bonuses", ['ambiguity_penalty', 'redundancy_penalty', 'length_penalty', 'structural_bonus']),
            ]
            
            # Helper to print groups
            def print_group(name, keys, is_percentage=True):
                print(f"\n-- {name} --")
                for k in keys:
                    if k in result:
                        val = result[k]
                        # Remove the prefix for cleaner display
                        display_name = k.replace('specificity_', '').replace('clarity_', '').replace('_', ' ').title()
                        if is_percentage:
                            print(f"  {display_name:20}: {val * 100:.2f}%")
                        else:
                            print(f"  {display_name:20}: {val:.2f}")

            for group_name, keys in groups:
                print_group(group_name, keys, is_percentage=True)
                
            clarity_diags = ['clarity_actionability', 'clarity_structure', 'clarity_completeness', 'clarity_fragment_penalty']
            print_group("Clarity Diagnostics", clarity_diags, is_percentage=True)
            
            specificity_diags = [
                'specificity_modifiers', 'specificity_entities', 'specificity_ranges', 
                'specificity_formats', 'specificity_tools', 'specificity_negation', 
                'specificity_persona', 'specificity_coverage', 'specificity_intensity'
            ]
            print_group("Specificity Diagnostics", specificity_diags, is_percentage=False)
            
            print("\n" + "="*50)
                    
        except KeyboardInterrupt:
            print("\n\nExiting interactive evaluator...")
            break
        except Exception as e:
            print(f"\nAn error occurred during evaluation: {e}")

if __name__ == "__main__":
    main()
