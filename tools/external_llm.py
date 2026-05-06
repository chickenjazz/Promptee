import urllib.request
import urllib.error
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT_SECONDS = 30


class ExternalLLMService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL

    def generate_response(self, prompt: str) -> dict:
        if not self.api_key:
            return {
                "ok": False,
                "text": "",
                "error": "GEMINI_API_KEY not configured. Create a .env file in the Promptee/ folder with GEMINI_API_KEY=<your key>.",
            }

        url = f"{GEMINI_URL_TEMPLATE.format(model=self.model)}?key={self.api_key}"
        payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
                result = json.loads(response.read().decode())
                text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                return {"ok": True, "text": text, "error": None}
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:300]
            if e.code == 429:
                return {
                    "ok": False,
                    "text": "",
                    "error": (
                        f"Gemini quota exceeded ({self.model}). Free tier limits resets per-minute (60s) "
                        f"or per-day (midnight Pacific). Try waiting 60s, or set GEMINI_MODEL=gemini-2.0-flash-lite "
                        f"in your .env for higher quotas. Raw response: {body}"
                    ),
                }
            return {"ok": False, "text": "", "error": f"Gemini HTTP {e.code}: {body}"}
        except urllib.error.URLError as e:
            return {"ok": False, "text": "", "error": f"Network error contacting Gemini: {e.reason}"}
        except (KeyError, IndexError, ValueError):
            return {"ok": False, "text": "", "error": "Unexpected response shape from Gemini."}
