import os
import json
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    LLM Client handling:
    1. Direct native C++ local LLM via llama.cpp (llama-server) running on Raspberry Pi 5 or OCI.
    2. Cloud Serverless API via OpenRouter / Gemini / OpenAI.
    3. Fallback deterministic grounded template mode if offline without local server.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash",
        llama_cpp_url: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("LLM_MODEL") or model
        self.llama_cpp_url = (llama_cpp_url or os.getenv("LLAMA_CPP_URL", "http://localhost:8080")).rstrip("/")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def is_llama_cpp_available(self) -> bool:
        """
        Checks if a local llama.cpp (llama-server) instance is active.
        """
        try:
            res = requests.get(f"{self.llama_cpp_url}/health", timeout=1.5)
            if res.status_code == 200:
                return True
            res_v1 = requests.get(f"{self.llama_cpp_url}/v1/models", timeout=1.5)
            return res_v1.status_code == 200
        except Exception:
            return False

    def is_local_llm_available(self) -> bool:
        """
        Returns True if llama.cpp server is running locally.
        """
        return self.is_llama_cpp_available()

    def is_cloud_llm_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 5)

    def is_available(self) -> bool:
        """
        Returns True if either Local llama.cpp or Cloud API is available.
        """
        return self.is_local_llm_available() or self.is_cloud_llm_available()

    def _clean_and_parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        if not content:
            return None
        content_clean = content.strip()
        if content_clean.startswith("```json"):
            content_clean = content_clean[7:]
        elif content_clean.startswith("```"):
            content_clean = content_clean[3:]
        if content_clean.endswith("```"):
            content_clean = content_clean[:-3]
        content_clean = content_clean.strip()
        try:
            return json.loads(content_clean, strict=False)
        except Exception as e1:
            import re
            match = re.search(r'\{.*\}', content_clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0), strict=False)
                except Exception as e2:
                    print(f"[LLMClient] JSON parsing failed. Primary error: {e1}. Regex fallback error: {e2}")
            else:
                print(f"[LLMClient] JSON parsing failed. Primary error: {e1}. No curly braces match found.")
        return None

    def generate_json(self, prompt: str, system_prompt: str = "") -> Optional[Dict[str, Any]]:
        """
        Dispatches prompt based on PREFERRED_LLM_PROVIDER (defaults to 'local').
        Fallback chain: llama.cpp -> Cloud OpenRouter.
        """
        preferred_provider = os.getenv("PREFERRED_LLM_PROVIDER", "local").lower()

        def try_local_llama_cpp() -> Optional[Dict[str, Any]]:
            if not self.is_llama_cpp_available():
                return None
            try:
                timeout = int(os.getenv("LLAMA_CPP_TIMEOUT", "300"))
                print(f"[LLMClient] Invoking Local llama.cpp server ({self.llama_cpp_url})...")
                
                # 1. Try OpenAI-compatible chat completions endpoint first
                url = f"{self.llama_cpp_url}/v1/chat/completions"
                payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON matching requested schema."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7
                }
                res = requests.post(url, json=payload, timeout=timeout)
                print(f"[LLMClient] Local v1/chat/completions response status: {res.status_code}")
                if res.status_code == 200:
                    content = res.json()["choices"][0]["message"]["content"]
                    print(f"[LLMClient] Local v1/chat/completions content length: {len(content)}")
                    parsed = self._clean_and_parse_json(content)
                    if parsed is not None:
                        print(f"[LLMClient] Successfully parsed JSON from chat completions.")
                        return parsed
                    else:
                        print(f"[LLMClient] Failed to parse JSON from chat completions. Raw content starts with: {content[:150]}")
                else:
                    print(f"[LLMClient] Local v1/chat/completions error body: {res.text}")

                # 2. Fallback to llama-server native /completion endpoint
                url_native = f"{self.llama_cpp_url}/completion"
                payload_native = {
                    "prompt": f"{system_prompt}\n\nUser: {prompt}\n\nOutput strictly valid JSON object:",
                    "stream": False,
                    "temperature": 0.7
                }
                res_native = requests.post(url_native, json=payload_native, timeout=timeout)
                print(f"[LLMClient] Local /completion response status: {res_native.status_code}")
                if res_native.status_code == 200:
                    content = res_native.json().get("content", "")
                    print(f"[LLMClient] Local /completion content length: {len(content)}")
                    parsed = self._clean_and_parse_json(content)
                    if parsed is not None:
                        print(f"[LLMClient] Successfully parsed JSON from native completion.")
                        return parsed
                    else:
                        print(f"[LLMClient] Failed to parse JSON from native completion. Raw content starts with: {content[:150]}")
                else:
                    print(f"[LLMClient] Local /completion error body: {res_native.text}")
            except Exception as e:
                print(f"[LLMClient] Local llama.cpp Exception: {e}")
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
            models_to_try = [self.model]
            if "google/gemini-2.5-flash" not in models_to_try:
                models_to_try.append("google/gemini-2.5-flash")

            import time
            for model_attempt in models_to_try:
                payload = {
                    "model": model_attempt,
                    "messages": [
                        {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON matching requested schema."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7
                }
                
                max_retries = 3
                for retry_idx in range(max_retries):
                    try:
                        print(f"[LLMClient] Invoking Cloud OpenRouter ({model_attempt}) [Attempt {retry_idx + 1}/{max_retries}]...")
                        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
                        response.raise_for_status()
                        data = response.json()
                        
                        if "choices" not in data or not data["choices"]:
                            print(f"[LLMClient] Cloud API returned no choices. Full response: {data}")
                            time.sleep(2 * (retry_idx + 1))
                            continue
                        
                        choice = data["choices"][0]
                        finish_reason = choice.get("finish_reason")
                        content = choice.get("message", {}).get("content", "")
                        error_detail = choice.get("error")
                        
                        if error_detail or finish_reason == "error":
                            print(f"[LLMClient] OpenRouter choice contains error: {error_detail or 'finish_reason is error'}")
                            time.sleep(2 * (retry_idx + 1))
                            continue
                        
                        print(f"[LLMClient] Cloud API Response status: {response.status_code}, content length: {len(content)}, finish_reason: {finish_reason}")
                        
                        parsed = self._clean_and_parse_json(content)
                        if parsed is not None:
                            return parsed
                        else:
                            print(f"[LLMClient] Failed to parse JSON from Cloud API. Raw content starts with: {content[:200]} ...")
                    except Exception as e:
                        print(f"[LLMClient] Cloud LLM Exception: {e}")
                    
                    time.sleep(2 * (retry_idx + 1))
            return None

        def try_native_gemini_api() -> Optional[Dict[str, Any]]:
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key or "YOUR_GEMINI_API_KEY" in gemini_key or len(gemini_key) < 10:
                return None
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": f"{system_prompt}\n\nReturn ONLY valid JSON matching requested schema.\n\nUser query: {prompt}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.7
                }
            }
            try:
                print("[LLMClient] Invoking Google AI Studio Native Gemini 2.5 Flash...")
                res = requests.post(url, headers=headers, json=payload, timeout=60)
                res.raise_for_status()
                data = res.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                print(f"[LLMClient] Google AI Studio Native response content length: {len(content)}")
                parsed = self._clean_and_parse_json(content)
                if parsed is not None:
                    return parsed
            except Exception as e:
                print(f"[LLMClient] Google AI Studio Native Exception: {e}")
            return None

        # Execute according to user preference
        if preferred_provider == "cloud":
            return try_cloud_api() or try_native_gemini_api() or try_local_llama_cpp()
        else:
            # Default: try local llama.cpp first, fallback to cloud
            return try_local_llama_cpp() or try_cloud_api() or try_native_gemini_api()
