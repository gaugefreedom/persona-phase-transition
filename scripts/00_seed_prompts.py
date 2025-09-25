# scripts/00_seed_prompts.py
import os
import re
import json
import time
import math
import argparse
from typing import Iterable, Dict, List, Tuple, Set

# Uses the legacy OpenAI SDK path you're already using in the repo.
# If you switch to the new SDK style, replace the client call accordingly.
import openai

# --------------- Default Config ---------------

DEFAULT_MODEL = "gpt-4o"
DEFAULT_OUTPUT = "data/prompts.jsonl"
DEFAULT_TOTAL = 2000
DEFAULT_BATCH = 100
DEFAULT_TEMPERATURE = 1.0

# Basic validation limits (adjust to taste)
MIN_PROMPT_CHARS = 12
MAX_PROMPT_CHARS = 280  # keep prompts concise/useful

# The meta prompt: strongly instruct JSONL-only output.
META_PROMPT = """You are an AI assistant helping create a dataset to train another language model. Your task is to generate a diverse set of user prompts.

Output exactly 100 lines in JSON Lines (JSONL) format. Each line must be a single JSON object with the keys "category" and "prompt". Do not include any explanations, lists, code fences, or extra text before or after the JSON lines.

Example format (example only; do NOT include these lines in your output):
{"category": "Science & Technology", "prompt": "What are the primary challenges in developing room-temperature superconductors?"}
{"category": "History & Humanities", "prompt": "How did the printing press influence the Protestant Reformation?"}

Requirements:
- Categories should be broad and varied (e.g., Science & Technology, History & Humanities, Business & Finance, Creative & Hypothetical, Ethics & Philosophy, Everyday Advice, Health & Medicine, Law & Policy, Education & Learning, Coding & Engineering, Environment & Energy, Arts & Culture, Math & Logic, Sports & Games, Personal Productivity).
- Prompts must be unique and varied in tone and complexity.
- Include simple factual questions, open-ended prompts, speculative/hypothetical questions, and opinion questions.
- Do NOT answer the prompts; only write the questions.
- Each line must be valid JSON with *only* "category" and "prompt".
- No line should exceed 280 characters in the "prompt".
"""

# --------------- Helpers ---------------

FENCE_RE = re.compile(r"^```(?:json)?\s*|```$", re.IGNORECASE)
JSON_LINE_RE = re.compile(r"^\s*\{.*\}\s*$")

def strip_code_fences(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if FENCE_RE.match(line.strip()):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def iter_valid_jsonl(text: str) -> Iterable[Dict]:
    """
    Yield JSON objects for lines that look like JSON, ignoring non-JSON debris.
    """
    text = strip_code_fences(text)
    for line in text.splitlines():
        if not JSON_LINE_RE.match(line):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "category" in obj and "prompt" in obj:
            yield obj

def normalize_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())

def filter_and_dedupe(
    objs: Iterable[Dict], 
    seen: Set[str],
    min_chars: int = MIN_PROMPT_CHARS,
    max_chars: int = MAX_PROMPT_CHARS,
) -> List[Dict]:
    kept = []
    for obj in objs:
        prompt = obj.get("prompt", "")
        if not isinstance(prompt, str):
            continue
        plen = len(prompt)
        if plen < min_chars or plen > max_chars:
            continue
        key = normalize_prompt(prompt)
        if key in seen:
            continue
        seen.add(key)
        kept.append({"category": str(obj.get("category", "")).strip(), "prompt": prompt.strip()})
    return kept

def backoff_sleep(attempt: int, base: float = 1.5, max_sleep: float = 12.0) -> None:
    # Exponential backoff with jitter
    delay = min(max_sleep, (base ** attempt)) * (0.5 + os.urandom(1)[0] / 255.0)
    time.sleep(delay)

# --------------- Core Generation ---------------

def generate_batch(model: str, temperature: float) -> str:
    """
    Calls the chat completion endpoint once and returns the assistant text.
    """
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful data generator that outputs strictly JSONL as instructed."},
            {"role": "user", "content": META_PROMPT},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content

def ensure_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

def write_jsonl(path: str, rows: List[Dict], mode: str = "a") -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)

# --------------- CLI Entry ---------------

def main():
    parser = argparse.ArgumentParser(description="Generate diverse prompts (JSONL) for training.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Teacher model (e.g., gpt-4o).")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Output JSONL file.")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="Total prompts to generate.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Prompts per API call (target).")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--max-calls-factor", type=float, default=1.8, help="Safety multiplier for max API calls (handles dedupe losses).")
    parser.add_argument("--min-chars", type=int, default=MIN_PROMPT_CHARS)
    parser.add_argument("--max-chars", type=int, default=MAX_PROMPT_CHARS)
    args = parser.parse_args()

    ensure_api_key()

    # Fresh file each run
    if os.path.exists(args.out):
        os.remove(args.out)

    target = int(args.total)
    approx_batches = math.ceil(target / max(1, args.batch_size))
    max_calls = max(approx_batches, int(math.ceil(approx_batches * args.max_calls_factor)))

    print(f"Target prompts: {target} | Batch size: {args.batch_size} | Planned calls: ~{approx_batches} | Max calls: {max_calls}")

    seen: Set[str] = set()
    written_total = 0
    call = 0
    consecutive_errors = 0

    while written_total < target and call < max_calls:
        call += 1
        print(f"\n[Call {call}/{max_calls}] Requesting ~{args.batch_size} prompts...")
        try:
            text = generate_batch(model=args.model, temperature=args.temperature)
            # Parse, filter, dedupe
            objs = list(iter_valid_jsonl(text))
            kept = filter_and_dedupe(objs, seen, min_chars=args.min_chars, max_chars=args.max_chars)

            # If the model under-delivers, try to filter less aggressively next time? (Optional heuristic)
            if not kept:
                print("  - No valid lines parsed (formatting or duplication).")
            else:
                # Cap so we don't overshoot too far
                remaining = target - written_total
                kept = kept[:remaining]
                n = write_jsonl(args.out, kept, mode="a")
                written_total += n
                print(f"  + Wrote {n} prompts (total: {written_total}/{target})")

            consecutive_errors = 0  # reset on success

        except Exception as e:
            consecutive_errors += 1
            print(f"  ! API error: {e}")
            backoff_sleep(consecutive_errors)

    print(f"\nDone. Wrote {written_total} prompts to {args.out}.")
    if written_total < target:
        print("Note: Target not fully reached (likely due to dedup/validation). Re-run to top up if needed.")

if __name__ == "__main__":
    main()

# Usage:
#export OPENAI_API_KEY="your-key-here"
#python scripts/00_seed_prompts.py \
#  --model gpt-4o \
#  --out data/prompts.jsonl \
#  --total 2000 \
#  --batch-size 100 \
#  --temperature 1.0
# You can adjust parameters as needed.