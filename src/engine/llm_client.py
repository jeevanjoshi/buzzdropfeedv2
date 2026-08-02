import os
import json
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


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

    def get_installed_ollama_model(self) -> str:
        """
        Dynamically detects installed models from local Ollama server.
        """
        try:
            res = requests.get(f"{self.ollama_url}/api/tags", timeout=1.5)
            if res.status_code == 200:
                models = [m.get("name") for m in res.json().get("models", [])]
                if models:
                    return models[0]  # e.g., 'llama3.2:3b'
        except Exception:
            pass
        return "llama3.2:3b"

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
        Dispatches prompt based on PREFERRED_LLM_PROVIDER (defaults to 'local' if Ollama active).
        Fallback chain: Preferred -> Alternate -> Template.
        """
        preferred_provider = os.getenv("PREFERRED_LLM_PROVIDER", "local").lower()

        def try_local_ollama() -> Optional[Dict[str, Any]]:
            if not self.is_ollama_available():
                return None
            try:
                local_model = self.get_installed_ollama_model()
                payload = {
                    "model": local_model,
                    "prompt": f"{system_prompt}\n\nUser: {prompt}\n\nOutput strictly valid JSON object:",
                    "stream": False,
                    "format": "json"
                }
                print(f"[LLMClient] Invoking Local Ollama ({local_model})...")
                res = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=300)
                if res.status_code == 200:
                    content = res.json().get("response", "")
                    return json.loads(content)
            except Exception as e:
                print(f"[LLMClient] Local Ollama Exception: {e}")
            return None

        def try_cloud_api() -> Optional[Dict[str, Any]]:
            if not self.is_cloud_llm_available():
                return None
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
                print(f"[LLMClient] Invoking Cloud OpenRouter ({self.model})...")
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=40)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
            except Exception as e:
                print(f"[LLMClient] Cloud LLM Exception: {e}")
            return None

        # Execute according to user preference
        if preferred_provider == "cloud":
            res = try_cloud_api()
            if res:
                return res
            return try_local_ollama()
        else:
            # Default to local first
            res = try_local_ollama()
            if res:
                return res
            return try_cloud_api()
