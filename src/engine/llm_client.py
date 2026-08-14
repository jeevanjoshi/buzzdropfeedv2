import os
import json
import time
import requests
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from src.engine.logger import logger
from src.engine.run_budget import run_budget

load_dotenv()


def _trunc(s: str, n: int = 2000) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f"...<truncated {len(s) - n} chars>"


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
        model: Optional[str] = None,
        llama_cpp_url: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL") or "google/gemini-2.5-flash"
        import getpass
        is_pi = (getpass.getuser() == "jeevanjoshi" or os.path.exists("/home/jeevanjoshi"))
        default_llama_url = "http://100.104.253.1:8080" if is_pi else "http://localhost:8080"
        self.llama_cpp_url = (llama_cpp_url or os.getenv("LLAMA_CPP_URL", default_llama_url)).rstrip("/")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    # Per-role model routing (Point 4). Each route may pin a cheaper/faster model
    # for mechanical rewrites ("repair") or a stronger one for verification
    # ("critic"). Routes fall back to LLM_MODEL when unset. Env keys:
    #   LLM_ROUTE_GENERATE / LLM_ROUTE_POLISH / LLM_ROUTE_REPAIR / LLM_ROUTE_CRITIC
    @staticmethod
    def _route_model(route: Optional[str]) -> Optional[str]:
        if not route:
            return None
        key = "LLM_ROUTE_" + (route or "").upper().strip()
        if not key.endswith("_GENERATE") and not key.endswith("_POLISH") and not key.endswith("_REPAIR") and not key.endswith("_CRITIC"):
            return None
        return os.getenv(key)

    def _model_chain(self, model: Optional[str] = None, route: Optional[str] = None) -> List[str]:
        """route-pinned -> primary -> fallback1 -> fallback2, deduped, non-empty.
        A per-role route pin (LLM_ROUTE_*) takes precedence over the primary
        model so cheap/capable models can be selected per task."""
        routed = self._route_model(route)
        chain = [routed, model or self.model,
                 os.getenv("LLM_FALLBACK_MODEL") or "deepseek/deepseek-v4-flash-0731",
                 os.getenv("LLM_FALLBACK_MODEL2")]
        seen = set()
        out = []
        for m in chain:
            if not m:
                continue
            m = m.strip()
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

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
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        return bool((self.api_key and len(self.api_key) > 5) or (gemini_key and len(gemini_key) > 5) or gcp_project)

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
            # Lexical parser fallback: extracts the first matching outer JSON object
            first_brace = content_clean.find("{")
            if first_brace != -1:
                count = 0
                in_string = False
                escape = False
                extracted = ""
                for i in range(first_brace, len(content_clean)):
                    char = content_clean[i]
                    if escape:
                        escape = False
                        continue
                    if char == "\\":
                        escape = True
                        continue
                    if char == '"':
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == "{":
                            count += 1
                        elif char == "}":
                            count -= 1
                            if count == 0:
                                extracted = content_clean[first_brace:i+1]
                                break
                if extracted:
                    try:
                        return json.loads(extracted, strict=False)
                    except Exception as e2:
                        print(f"[LLMClient] JSON parsing failed. Primary error: {e1}. Lexical parser fallback error: {e2}")
                else:
                    print(f"[LLMClient] JSON parsing failed. Primary error: {e1}. Unbalanced braces found.")
            else:
                print(f"[LLMClient] JSON parsing failed. Primary error: {e1}. No curly braces found.")
        return None

    def generate_json(self, prompt: str, system_prompt: str = "",
                      route: Optional[str] = None, thinking: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Dispatches prompt based on PREFERRED_LLM_PROVIDER (defaults to 'local').
        Fallback chain: llama.cpp -> Cloud OpenRouter.
        route: per-role model pinning (Point 4) — see LLM_ROUTE_* env keys.
        thinking: "low"|"medium"|"high" — gated reasoning effort for models that
        support OpenRouter `reasoning` (e.g. "critic" route). No-op otherwise.
        """
        preferred_provider = os.getenv("PREFERRED_LLM_PROVIDER", "local").lower()

        def try_local_llama_cpp() -> Optional[Dict[str, Any]]:
            if not self.is_llama_cpp_available():
                return None
            try:
                timeout = int(os.getenv("LLAMA_CPP_TIMEOUT", "45"))
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
            # Model routing: primary -> route-pinned -> LLM_FALLBACK_MODEL ->
            # LLM_FALLBACK_MODEL2 (Point 4). Default fallback is a cheap DeepSeek
            # v4 flash (different provider from Google).
            models = self._model_chain(self.model, route=route)

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
                # Gated reasoning knob: only applies to models that honour OpenRouter
                # `reasoning` (gemini-*-thinking, deepseek-reasoner…). Env-gated so
                # no model behaviour changes unless explicitly requested.
                effort = os.getenv("LLM_THINKING_EFFORT", "").strip() or (thinking if thinking in ("low", "medium", "high") else "")
                if effort:
                    payload["reasoning"] = {"effort": effort}
                for retry_idx in range(3):  # client-level transient retry
                    backoff = 2 * (retry_idx + 1)
                    t0 = time.time()
                    try:
                        print(f"[LLMClient] Invoking Cloud OpenRouter ({model}) [try {retry_idx + 1}/3]...")
                        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
                        latency_ms = int((time.time() - t0) * 1000)
                        raw_text = response.text

                        if response.status_code in transient_status:
                            logger.warning(
                                "LLM_CALL",
                                f"Transient HTTP {response.status_code} from '{model}'; backoff {backoff}s.",
                                component="LLM_CLIENT",
                                extra_data={"api":"openrouter_chat_completions","model":model,
                                            "url":self.base_url,"status":response.status_code,
                                            "latency_ms":latency_ms,"prompt":_trunc(prompt),
                                            "system_prompt":_trunc(system_prompt,800),
                                            "response":_trunc(raw_text,1000),"retry":retry_idx+1})
                            time.sleep(backoff)
                            continue
                        if response.status_code >= 400:
                            logger.error(
                                "LLM_CALL",
                                f"Permanent HTTP {response.status_code} from '{model}': {_trunc(raw_text,300)}",
                                component="LLM_CLIENT",
                                extra_data={"api":"openrouter_chat_completions","model":model,
                                            "url":self.base_url,"status":response.status_code,
                                            "latency_ms":latency_ms,"prompt":_trunc(prompt),
                                            "response":_trunc(raw_text,1000)})
                            break  # permanent -> move to next model

                        data = response.json()
                        if "choices" not in data or not data["choices"]:
                            logger.warning("LLM_CALL", f"No choices from '{model}'", component="LLM_CLIENT",
                                           extra_data={"api":"openrouter_chat_completions","model":model,
                                                       "status":response.status_code,"response":_trunc(raw_text,1000)})
                            time.sleep(backoff)
                            continue

                        choice = data["choices"][0]
                        finish_reason = choice.get("finish_reason")
                        content = choice.get("message", {}).get("content", "")

                        if choice.get("error") or finish_reason == "error":
                            logger.warning("LLM_CALL", f"Model error from '{model}' (finish_reason=error)",
                                           component="LLM_CLIENT",
                                           extra_data={"model":model,"finish_reason":finish_reason,
                                                       "response":_trunc(content,1000)})
                            time.sleep(backoff)
                            continue  # transient model error -> retry, then next model

                        parsed = self._clean_and_parse_json(content)
                        if parsed is not None:
                            logger.info(
                                "LLM_CALL",
                                f"{model} OK status {response.status_code}, len {len(content)}, finish_reason: {finish_reason}",
                                component="LLM_CLIENT",
                                extra_data={"api":"openrouter_chat_completions","model":model,
                                            "url":self.base_url,"status":response.status_code,
                                            "finish_reason":finish_reason,"latency_ms":latency_ms,
                                            "prompt":_trunc(prompt),"system_prompt":_trunc(system_prompt,800),
                                            "response":_trunc(content,4000),"retry":retry_idx+1})
                            return parsed
                        logger.warning("LLM_CALL", f"JSON parse failed from '{model}' (finish_reason={finish_reason})",
                                       component="LLM_CLIENT",
                                       extra_data={"model":model,"finish_reason":finish_reason,
                                                   "prompt":_trunc(prompt),"response":_trunc(content,1000)})
                        time.sleep(backoff)  # malformed/truncated -> retry, then next model
                    except requests.exceptions.Timeout:
                        logger.warning("LLM_CALL", f"Timeout from '{model}'", component="LLM_CLIENT",
                                       extra_data={"api":"openrouter_chat_completions","model":model,
                                                   "latency_ms":int((time.time()-t0)*1000),"prompt":_trunc(prompt)})
                        time.sleep(backoff)
                    except Exception as e:
                        logger.error("LLM_CALL", f"Cloud LLM Exception ({model}): {e}", component="LLM_CLIENT",
                                     extra_data={"api":"openrouter_chat_completions","model":model,"error":str(e)})
                        time.sleep(backoff)
                # this model exhausted its tries -> move to next fallback model
            return None

        def try_vertex_api() -> Optional[Dict[str, Any]]:
            if not os.getenv("GOOGLE_CLOUD_PROJECT"):
                return None
            try:
                from google import genai
                from google.genai import types as genai_types
            except ImportError:
                print("[LLMClient] google-genai SDK not installed, Vertex AI unavailable.")
                return None

            models = self._model_chain(self.model, route=route)
            for model in models:
                native_model = model
                if "/" in model:
                    parts = model.split("/")
                    if parts[0] == "google":
                        native_model = parts[1]
                    else:
                        continue

                for retry_idx in range(3):
                    backoff = 2 * (retry_idx + 1)
                    t0 = time.time()
                    try:
                        print(f"[LLMClient] Invoking Google Cloud Vertex AI ({native_model}) [try {retry_idx + 1}/3]...")
                        client = genai.Client(http_options=genai_types.HttpOptions(api_version="v1"))
                        
                        config = genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.7,
                        )
                        if system_prompt:
                            config.system_instruction = system_prompt
                        
                        response = client.models.generate_content(
                            model=native_model,
                            contents=prompt,
                            config=config
                        )
                        latency_ms = int((time.time() - t0) * 1000)
                        content = getattr(response, "text", "") or ""

                        parsed = self._clean_and_parse_json(content)
                        if parsed is not None:
                            logger.info(
                                "LLM_CALL",
                                f"Vertex AI {native_model} OK, len {len(content)}",
                                component="LLM_CLIENT",
                                extra_data={"api": "vertex_ai", "model": native_model,
                                            "latency_ms": latency_ms, "prompt": _trunc(prompt),
                                            "system_prompt": _trunc(system_prompt, 800),
                                            "response": _trunc(content, 4000), "retry": retry_idx + 1}
                            )
                            return parsed

                        logger.warning("LLM_CALL", f"JSON parse failed from Vertex '{native_model}'",
                                       component="LLM_CLIENT",
                                       extra_data={"model": native_model, "prompt": _trunc(prompt),
                                                   "response": _trunc(content, 1000)})
                        time.sleep(backoff)
                    except Exception as e:
                        logger.error("LLM_CALL", f"Vertex AI Exception ({native_model}): {e}",
                                     component="LLM_CLIENT",
                                     extra_data={"api": "vertex_ai", "model": native_model, "error": str(e)})
                        time.sleep(backoff)
            return None

        def try_gemini_api() -> Optional[Dict[str, Any]]:
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                return None
            try:
                from google import genai
                from google.genai import types as genai_types
            except ImportError:
                print("[LLMClient] google-genai SDK not installed, native Gemini unavailable.")
                return None

            models = self._model_chain(self.model, route=route)
            for model in models:
                native_model = model
                if "/" in model:
                    parts = model.split("/")
                    if parts[0] == "google":
                        native_model = parts[1]
                    else:
                        continue

                for retry_idx in range(3):
                    backoff = 2 * (retry_idx + 1)
                    t0 = time.time()
                    try:
                        print(f"[LLMClient] Invoking Native Gemini API ({native_model}) [try {retry_idx + 1}/3]...")
                        client = genai.Client(api_key=gemini_key, vertexai=False, enterprise=False)
                        
                        config = genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.7,
                        )
                        if system_prompt:
                            config.system_instruction = system_prompt
                        
                        response = client.models.generate_content(
                            model=native_model,
                            contents=prompt,
                            config=config
                        )
                        latency_ms = int((time.time() - t0) * 1000)
                        content = getattr(response, "text", "") or ""

                        parsed = self._clean_and_parse_json(content)
                        if parsed is not None:
                            logger.info(
                                "LLM_CALL",
                                f"Native Gemini {native_model} OK, len {len(content)}",
                                component="LLM_CLIENT",
                                extra_data={"api": "gemini_native", "model": native_model,
                                            "latency_ms": latency_ms, "prompt": _trunc(prompt),
                                            "system_prompt": _trunc(system_prompt, 800),
                                            "response": _trunc(content, 4000), "retry": retry_idx + 1}
                            )
                            return parsed

                        logger.warning("LLM_CALL", f"JSON parse failed from native '{native_model}'",
                                       component="LLM_CLIENT",
                                       extra_data={"model": native_model, "prompt": _trunc(prompt),
                                                   "response": _trunc(content, 1000)})
                        time.sleep(backoff)
                    except Exception as e:
                        logger.error("LLM_CALL", f"Native Gemini Exception ({native_model}): {e}",
                                     component="LLM_CLIENT",
                                     extra_data={"api": "gemini_native", "model": native_model, "error": str(e)})
                        time.sleep(backoff)
            return None

        # Execute according to user preference
        if preferred_provider in ("vertex", "google"):
            result = try_vertex_api() or try_gemini_api() or try_cloud_api() or try_local_llama_cpp()
        elif preferred_provider == "gemini":
            result = try_gemini_api() or try_vertex_api() or try_cloud_api() or try_local_llama_cpp()
        elif preferred_provider == "cloud":
            result = try_cloud_api() or try_vertex_api() or try_gemini_api() or try_local_llama_cpp()
        else:
            # Default: try local llama.cpp first, fallback to Vertex AI, then native Gemini AI Studio, then cloud OpenRouter
            result = try_local_llama_cpp() or try_vertex_api() or try_gemini_api() or try_cloud_api()

        # A real (paid/local) LLM response was produced — bill it to the run budget.
        if result is not None:
            run_budget.record_llm()
        return result
