import os
import traceback
from sentence_transformers import SentenceTransformer

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

try:
    SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    with open('error_log.txt', 'w') as f:
        f.write(traceback.format_exc())
    print("Error saved to error_log.txt")
