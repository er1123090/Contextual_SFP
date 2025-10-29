#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract contextual Situation/Factor/Preference per `prior_interactions` session
from CUPID `test.json` using OpenAI GPT-4o-mini, and save results as a new column.
Also saves a separate JSONL log file with GPT input/output per row.

Input:  /data/minseo/Contextual_SFP_extract/data/CUPID/test.json
Output: /data/minseo/Contextual_SFP_extract/data/CUPID/test_with_sfp.json
Logs:   /data/minseo/Contextual_SFP_extract/data/CUPID/test_with_sfp.logs.jsonl

Environment:
- pip install tqdm tenacity openai>=1.40.0
- export OPENAI_API_KEY=...

Run:
python extract_cupid_sfp_from_json_with_logs.py \
  --input /data/minseo/Contextual_SFP_extract/data/CUPID/test.json \
  --output /data/minseo/Contextual_SFP_extract/data/CUPID/test_with_sfp.json \
  --log /data/minseo/Contextual_SFP_extract/data/CUPID/test_with_sfp.logs.jsonl \
  --model gpt-4o-mini
"""
import os
import json
import argparse
from typing import Any, Dict, List, Optional
from datetime import datetime

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# OpenAI SDK v1.x
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Install the OpenAI SDK v1.x: `pip install openai>=1.40.0`") from e

SYSTEM_PROMPT = (
    """You are a helpful AI assistant extracting user preferences from conversations.
    For each prior_interaction session, extract:
    - contextual_situation: overall context (time, place, social setting)
    - contextual_factor: the entity, object, or topic affecting the preference
    - contextual_preference: the user’s attitude or stance toward that factor
    - dialogue: short quote(s) supporting the extraction
    Respond ONLY in JSON list form, one JSON object per session."""
)

SYSTEM_PROMPT_CF = (
    "You are a helpful AI assistant. Your job is to extracting useful information (user preferences) from conversations. "
    "For each prior_interaction session, you will be given the context factor of the conversation. "
    " A context factor is '-the entity, object, or topic affecting the preference'" 
    "From the conversation extract:\n"
    "- contextual_situation: overall context (time, place, social setting)\n"
    "- contextual_preference: the user’s attitude or stance toward that factor\n"
    "- dialogue: short quote(s) supporting the extraction\n"
    "Respond ONLY in JSON list form, one JSON object per session."
)

USER_PROMPT_TEMPLATE = """Analyze the following prior_interactions data and extract the contextual situation (S), contextual factor (F), contextual preference (P), and supporting dialogue for each session.

Return ONLY a JSON list like:
[
  {{
    "contextual_situation": "...",
    "contextual_factor": "...",
    "contextual_preference": "...",
    "dialogue": "..."
  }},
  ...
]

prior_interactions:
--------
{prior_interactions}
--------
"""


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


class JSONParseError(Exception):
    pass


def _coerce_json(s: str) -> Any:
    """Cleans model output and converts to JSON (expects a list)."""
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # prefer JSON array
    first = s.find("[")
    last = s.rfind("]")
    if first != -1 and last != -1 and last > first:
        s = s[first:last + 1]
    return json.loads(s)


def append_log(log_path: str, record: Dict[str, Any]) -> None:
    """Append a single JSON record to JSONL log file, creating dirs if needed."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    record = dict(record)  # shallow copy
    # add timestamp if missing
    record.setdefault("ts", datetime.utcnow().isoformat() + "Z")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((JSONParseError, TimeoutError, RuntimeError, Exception)),
)
def call_gpt_extract(client: OpenAI, model: str, sys_prompt: str, user_prompt: str, meta: Dict[str, Any], log_path: Optional[str]) -> Any:
    """Call GPT, parse JSON, and log input/output."""
    # Log the request
    if log_path:
        append_log(log_path, {
            "event": "request",
            "meta": meta,
            "model": model,
            "system_prompt": sys_prompt,
            "user_prompt": user_prompt,
        })

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content

    # Log the raw response
    if log_path:
        append_log(log_path, {
            "event": "response",
            "meta": meta,
            "model": model,
            "raw_output": content,
        })

    try:
        parsed = _coerce_json(content)
    except Exception as e:
        # Log parse error
        if log_path:
            append_log(log_path, {
                "event": "parse_error",
                "meta": meta,
                "error": str(e),
                "raw_output": content,
            })
        raise JSONParseError(f"Failed to parse JSON: {e}\nRaw:\n{content}") from e

    if not isinstance(parsed, list):
        # Log schema error
        if log_path:
            append_log(log_path, {
                "event": "schema_error",
                "meta": meta,
                "error": f"Expected list output, got: {type(parsed)}",
                "raw_output": content,
            })
        raise JSONParseError(f"Expected list output, got: {type(parsed)}")
    return parsed


def process_json(data: List[Dict[str, Any]], client: OpenAI, model: str, log_path: Optional[str]) -> List[Dict[str, Any]]:
    """Process each row's prior_interactions and log per-row input/output."""
    for i, entry in enumerate(tqdm(data, desc="Extracting S/F/P per entry")):
        #prior_obj = entry.get("prior_interactions", "")
        prior = entry.get("prior_interactions", "")
        for session in prior:
            prior_cf = session.get("context_factor", "")
            prior_cp = session.get("contextual_preference", "")
            prior_diag = session.get("dialogue", "")
            prior_text = json.dumps({"context_factor" : prior_cf, "prior_dialogue" : prior_diag}, ensure_ascii=False)
            #prior_text = json.dumps(prior_diag, ensure_ascii=False)

            if not str(prior_text).strip():
                entry["gpt4o_sfp_sessions"] = []
                continue

            user_prompt = USER_PROMPT_TEMPLATE.format(prior_interactions=prior_text)
            meta = {
                "row_index": i,
                "persona_id": entry.get("persona_id"),
                "instance_type": entry.get("instance_type"),
            }
            if int(meta['row_index']) > 10:
                break
            else:
                try:
                    parsed = call_gpt_extract(client, model, SYSTEM_PROMPT_CF, user_prompt, meta, log_path)
                except Exception as e:
                    parsed = [{"error": str(e)}]
                    if log_path:
                        append_log(log_path, {
                            "event": "failure",
                            "meta": meta,
                            "error": str(e),
                        })

                entry["gpt4o_sfp_sessions"] = parsed
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/data/minseo/Contextual_SFP_extract/data/CUPID/test.json")
    parser.add_argument("--output", type=str, default="/data/minseo/Contextual_SFP_extract/data/CUPID/test_sfp_FD_251022.json")
    parser.add_argument("--log", type=str, default="/data/minseo/Contextual_SFP_extract/data/CUPID/test_sfp_FD_251022.logs.jsonl")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    # Load JSON
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {args.input}")

    client = build_client()
    processed = process_json(data, client, args.model, args.log)

    # Save JSON output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved augmented JSON to: {args.output}")
    print(f"📝 Logs written to: {args.log}")


if __name__ == "__main__":
    main()
