#!/usr/bin/env python3
import argparse, json, os, re, sys
from typing import Tuple, Optional

def norm_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())

def try_json(line: str):
    try:
        obj = json.loads(line)
        return obj
    except json.JSONDecodeError:
        return None

def _scan_quoted_value(s: str, start_idx: int) -> Tuple[Optional[str], int]:
    """
    Scan a JSON-like quoted string starting at the opening quote index `start_idx`.
    Returns (value, end_pos_after_closing_quote) or (None, start_idx).
    Handles backslash escapes conservatively.
    """
    if start_idx >= len(s) or s[start_idx] != '"':
        return None, start_idx
    i = start_idx + 1
    out = []
    esc = False
    while i < len(s):
        ch = s[i]
        if esc:
            out.append(ch)
            esc = False
        else:
            if ch == '\\':
                out.append(ch)  # keep escape
                esc = True
            elif ch == '"':
                return "".join(out), i + 1
            else:
                out.append(ch)
        i += 1
    return None, start_idx

def salvage_line(line: str):
    """
    Conservative salvage: extract "prompt" and "response" with a simple scanner.
    Only returns an object if both are found as well-formed quoted strings.
    """
    s = line.strip()
    # Ensure braces exist (common copy/paste error)
    if not s.startswith("{"):
        s = "{" + s
    if not s.endswith("}"):
        s = s + "}"

    # locate "prompt":
    k1 = s.lower().find('"prompt"')
    if k1 == -1: 
        return None
    colon1 = s.find(":", k1)
    if colon1 == -1: 
        return None
    # find opening quote for value
    oq1 = s.find('"', colon1)
    if oq1 == -1:
        return None
    prompt_val, pos1 = _scan_quoted_value(s, oq1)
    if prompt_val is None:
        return None

    # locate "response":
    k2 = s.lower().find('"response"', pos1)
    if k2 == -1:
        return None
    colon2 = s.find(":", k2)
    if colon2 == -1:
        return None
    oq2 = s.find('"', colon2)
    if oq2 == -1:
        return None
    response_val, pos2 = _scan_quoted_value(s, oq2)
    if response_val is None:
        return None

    return {"prompt": prompt_val, "response": response_val}

def choose_keep(a, b, keep_policy: str):
    if keep_policy == "first":
        return a
    if keep_policy == "last":
        return b
    if keep_policy == "longest":
        return a if len(a.get("response","")) >= len(b.get("response","")) else b
    return a  # default

def main():
    ap = argparse.ArgumentParser(description="Clean, validate, and dedupe a JSONL dataset.")
    ap.add_argument("infile", help="Input JSONL path")
    ap.add_argument("outfile", help="Output cleaned JSONL path")
    ap.add_argument("--keep", choices=["first","last","longest"], default="first",
                    help="When duplicates occur, which one to keep")
    ap.add_argument("--min-response", type=int, default=1, help="Drop if response length < this")
    ap.add_argument("--max-response", type=int, default=800, help="Drop if response length > this")
    ap.add_argument("--no-salvage", action="store_true", help="Disable salvage for broken lines")
    args = ap.parse_args()

    if not os.path.exists(args.infile):
        print(f"Input not found: {args.infile}", file=sys.stderr)
        sys.exit(1)

    total = 0
    valid = 0
    salvaged = 0
    dropped_invalid = 0
    dropped_key = 0
    dropped_range = 0
    dupes = 0

    kept = {}
    with open(args.infile, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                total += 1
                dropped_invalid += 1
                continue

            obj = try_json(s)
            if obj is None and not args.no_salvage:
                obj = salvage_line(s)
                if obj is not None:
                    salvaged += 1

            if obj is None:
                total += 1
                dropped_invalid += 1
                continue

            if not isinstance(obj, dict) or set(obj.keys()) != {"prompt","response"}:
                total += 1
                dropped_key += 1
                continue

            p = obj.get("prompt")
            r = obj.get("response")
            if not isinstance(p, str) or not isinstance(r, str):
                total += 1
                dropped_key += 1
                continue

            if not (args.min_response <= len(r) <= args.max_response):
                total += 1
                dropped_range += 1
                continue

            key = norm_prompt(p)
            if key in kept:
                dupes += 1
                obj = choose_keep(kept[key], obj, args.keep)
            kept[key] = obj
            total += 1
            valid += 1

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8") as out:
        for obj in kept.values():
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("==== Summary ====")
    print(f"Total lines read      : {total}")
    print(f"Valid (pre-dedupe)    : {valid}")
    print(f"Salvaged              : {salvaged}")
    print(f"Dropped invalid JSON  : {dropped_invalid}")
    print(f"Dropped bad keys/types: {dropped_key}")
    print(f"Dropped len range     : {dropped_range}")
    print(f"Duplicates collapsed  : {dupes}")
    print(f"Kept (unique prompts) : {len(kept)}")
    print(f"Wrote                 : {args.outfile}")

if __name__ == "__main__":
    main()
