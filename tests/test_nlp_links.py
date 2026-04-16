import os
if 'SSL_CERT_FILE' in os.environ and not os.path.exists(os.environ['SSL_CERT_FILE']):
    del os.environ['SSL_CERT_FILE']

from sentence_transformers import SentenceTransformer, util

def test_sentence_transformer():
    """
    Test loading the selected sentence-transformer model and computing cosine similarity.
    """
    print("[*] Loading sentence-transformers model 'all-MiniLM-L6-v2' (Downloading if not cached)...")
    try:
        import torch
        print(f"[*] System Check | PyTorch CUDA available: {torch.cuda.is_available()}")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test sequences
        raw_prompt = "Write a python script to sort a list."
        optimized_prompt = "Generate a Python script implementing the quicksort algorithm to sort a list of integers in ascending order."
        hallucinated_prompt = "Explain the history of Python."
        
        # Compute embeddings
        emb_raw = model.encode(raw_prompt)
        emb_opt = model.encode(optimized_prompt)
        emb_hal = model.encode(hallucinated_prompt)
        
        # Calculate cosine similarity
        sim_opt = util.cos_sim(emb_raw, emb_opt).item()
        sim_hal = util.cos_sim(emb_raw, emb_hal).item()
        
        print(f"[+] Similarity (Raw vs Optimized): {sim_opt:.4f}")
        print(f"[+] Similarity (Raw vs Hallucinated): {sim_hal:.4f}")
        
        if sim_opt > sim_hal:
            print("[+] Semantic similarity logic verified successfully.")
            return True
        else:
            print("[-] Logic error: Hallucination scored higher than valid optimization.")
            return False
            
    except Exception as e:
        print(f"[-] NLP Link failed: {str(e)}")
        print("[-] Ensure you have installed torch and sentence-transformers: `python -m pip install torch sentence-transformers`")
        return False

if __name__ == "__main__":
    test_sentence_transformer()
