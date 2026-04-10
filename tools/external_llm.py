import urllib.request
import urllib.error
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

class ExternalLLMService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        
    def generate_response(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: External LLM API key not configured. Check .env"
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                text = result.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "")
                return text.strip()
        except Exception as e:
            return f"Error connecting to external LLM: {str(e)}"
