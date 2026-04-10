import os
import sys
import json
import urllib.request
import urllib.error

# Add project root to sys.path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

def test_gemini_connection():
    """
    Test Google Gemini API connection using urllib to keep it dependency-free.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        print("[-] GEMINI_API_KEY is not set in .env")
        return False
        
    print("[*] Testing Gemini API Connection...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{"text": "Hello, this is a test handshake. Respond with exactly the word 'ACK'."}]
        }]
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            text = result.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "")
            print(f"[+] Gateway response: {text.strip()}")
            return True
    except urllib.error.HTTPError as e:
        print(f"[-] HTTP Error: {e.code} - {e.read().decode()}")
        return False
    except Exception as e:
        print(f"[-] Evaluation failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemini_connection()
