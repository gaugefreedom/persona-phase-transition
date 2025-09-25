import sys, json

path = sys.argv[1]
ok = True
seen = set()
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        s = line.strip()
        if not s:
            print(f"[line {i}] empty line")
            ok = False
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            print(f"[line {i}] invalid JSON: {e}")
            ok = False
            continue
        if not isinstance(obj, dict) or set(obj.keys()) != {"prompt","response"}:
            print(f"[line {i}] must have ONLY keys 'prompt' and 'response': {obj.keys()}")
            ok = False
        if not isinstance(obj.get("prompt"), str) or not isinstance(obj.get("response"), str):
            print(f"[line {i}] 'prompt' and 'response' must be strings")
            ok = False
        k = obj.get("prompt","").strip().lower()
        if k in seen:
            print(f"[line {i}] duplicate prompt")
            ok = False
        else:
            seen.add(k)

print("OK" if ok else "FOUND ISSUES")
