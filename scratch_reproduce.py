import sys
import json
import os
import requests
from dotenv import load_dotenv
from src.engine.llm_client import LLMClient
from src.schemas.state import GlobalState
from src.agents.story_designer import StoryDesignerAgent

load_dotenv()

# Load state
state_file = "logs/state_csvg-exec-20260804-103331.json"
with open(state_file, "r", encoding="utf-8") as f:
    state = GlobalState.model_validate_json(f.read())

client = LLMClient()
designer = StoryDesignerAgent(llm_client=client)

# Patch LLMClient to print exact responses
original_try_cloud_api = client.generate_json

def debug_generate_json(prompt, system_prompt=""):
    print("--- SYSTEM PROMPT ---")
    print(system_prompt)
    print("--- PROMPT ---")
    print(prompt)
    
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/buzzdropfeedv2",
        "X-Title": "CSVG Autonomous Pipeline"
    }
    payload = {
        "model": client.model,
        "messages": [
            {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON matching requested schema."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.7
    }
    print(f"[DEBUG] Invoking OpenRouter post...")
    try:
        response = requests.post(client.base_url, headers=headers, json=payload, timeout=60)
        print(f"[DEBUG] Status Code: {response.status_code}")
        print(f"[DEBUG] Response Headers: {dict(response.headers)}")
        data = response.json()
        print(f"[DEBUG] Response JSON Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        if "error" in data:
            print(f"[DEBUG] API ERROR: {data['error']}")
            return None
        content = data["choices"][0]["message"]["content"]
        print(f"[DEBUG] Content Length: {len(content)}")
        print("--- RAW CONTENT (FIRST 500 CHARS) ---")
        print(content[:500])
        print("--- RAW CONTENT (LAST 500 CHARS) ---")
        print(content[-500:])
        
        parsed = client._clean_and_parse_json(content)
        if parsed is None:
            print("[DEBUG] FAILED TO PARSE JSON!")
        else:
            print("[DEBUG] PARSED SUCCESSFULLY!")
        return parsed
    except Exception as e:
        print(f"[DEBUG] Exception: {e}")
        return None

client.generate_json = debug_generate_json

print("Reproducing generate_6act_script for 'How Data Centers Broke American Politics'...")
try:
    script = designer.generate_6act_script(state.selected_topic, state.verified_facts)
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED with exception: {e}")
