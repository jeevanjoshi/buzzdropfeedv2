import os
import json
import time
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
        self.model = model
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
            # Model fallback chain: primary -> LLM_FALLBACK_MODEL -> LLM_FALLBACK_MODEL2.
            # Default fallback is a cheap DeepSeek v4 flash (different provider from Google).
            models = []
            for m in [
                self.model,
                os.getenv("LLM_FALLBACK_MODEL") or "deepseek/deepseek-v4-flash-0731",
                os.getenv("LLM_FALLBACK_MODEL2"),
            ]:
                if m and m not in models:
                    models.append(m)

            transient_status = {408, 429, 500, 502, 503, 504}
            for model in models:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON matching requested schema."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7,
                    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "8192"))
                }
                for retry_idx in range(3):  # client-level transient retry
                    backoff = 2 * (retry_idx + 1)
                    try:
                        print(f"[LLMClient] Invoking Cloud OpenRouter ({model}) [try {retry_idx + 1}/3]...")
                        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)

                        if response.status_code in transient_status:
                            print(f"[LLMClient] Transient HTTP {response.status_code} from '{model}'; backoff {backoff}s.")
                            time.sleep(backoff)
                            continue
                        if response.status_code >= 400:
                            print(f"[LLMClient] Permanent HTTP {response.status_code} from '{model}': {response.text[:200]}")
                            break  # permanent -> move to next model

                        data = response.json()
                        if "choices" not in data or not data["choices"]:
                            print(f"[LLMClient] No choices from '{model}': {data}")
                            time.sleep(backoff)
                            continue

                        choice = data["choices"][0]
                        finish_reason = choice.get("finish_reason")
                        content = choice.get("message", {}).get("content", "")
                        print(f"[LLMClient] {model} status {response.status_code}, len {len(content)}, finish_reason: {finish_reason}")

                        if choice.get("error") or finish_reason == "error":
                            print(f"[LLMClient] Model error from '{model}' (finish_reason=error); backoff {backoff}s.")
                            time.sleep(backoff)
                            continue  # transient model error -> retry, then next model

                        parsed = self._clean_and_parse_json(content)
                        if parsed is not None:
                            return parsed
                        print(f"[LLMClient] JSON parse failed from '{model}' (finish_reason={finish_reason}); backoff {backoff}s.")
                        time.sleep(backoff)  # malformed/truncated -> retry, then next model
                    except requests.exceptions.Timeout:
                        print(f"[LLMClient] Timeout from '{model}'; backoff {backoff}s.")
                        time.sleep(backoff)
                    except Exception as e:
                        print(f"[LLMClient] Cloud LLM Exception ({model}): {e}")
                        time.sleep(backoff)
                # this model exhausted its tries -> move to next fallback model
            return None

        # Execute according to user preference
        if preferred_provider == "cloud":
            return try_cloud_api() or try_local_llama_cpp()
        else:
            # Default: try local llama.cpp first, fallback to cloud
            return try_local_llama_cpp() or try_cloud_api()
