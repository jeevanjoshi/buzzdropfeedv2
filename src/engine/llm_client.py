import os
import json
import requests
from typing import Dict, Any, Optional


class LLMClient:
    """
    LLM Client handling:
    1. Local LLM via Ollama / llama.cpp (running on OCI or Raspberry Pi 5).
    2. Cloud Serverless API via OpenRouter / Gemini / OpenAI.
    3. Fallback deterministic grounded template mode if offline without local LLM.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash",
        ollama_url: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def is_ollama_available(self) -> bool:
        """
        Checks if a local LLM server (Ollama on OCI or Raspberry Pi 5) is active.
        """
        try:
            res = requests.get(f"{self.ollama_url}/api/tags", timeout=1.5)
            return res.status_code == 200
        except Exception:
            return False

    def is_cloud_llm_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 5)

    def is_available(self) -> bool:
        """
        Returns True if either Local Ollama or Cloud API is available.
        """
        return self.is_ollama_available() or self.is_cloud_llm_available()

    def generate_json(self, prompt: str, system_prompt: str = "") -> Optional[Dict[str, Any]]:
        """
        Dispatches prompt to Local Ollama LLM (if available), Cloud API (if available),
        or returns None for template fallback.
        """
        # 1. Try Local LLM (Ollama on OCI / Pi 5)
        if self.is_ollama_available():
            try:
                payload = {
                    "model": "qwen2.5:7b-instruct" if "11434" in self.ollama_url else "llama3.2:3b",
                    "prompt": f"{system_prompt}\n\nUser: {prompt}\n\nOutput strictly valid JSON object:",
                    "stream": False,
                    "format": "json"
                }
                res = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=45)
                if res.status_code == 200:
                    content = res.json().get("response", "")
                    return json.loads(content)
            except Exception:
                pass

        # 2. Try Cloud API (OpenRouter / Gemini / OpenAI)
        if self.is_cloud_llm_available():
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/buzzdropfeedv2",
                "X-Title": "CSVG Autonomous Pipeline"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON matching requested schema."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.7
            }
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            except Exception:
                pass

        return None
