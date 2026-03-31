import json
from pathlib import Path

INPUT_PATH = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\case_with_all_sources_with_companion_cases_tag.jsonl"

def summarise(obj, depth=0, max_depth=3):
    indent = "  " * depth
    if depth > max_depth:
        return indent + "..."
    if isinstance(obj, dict):
        lines = [indent + "{"]
        for k, v in list(obj.items())[:15]:
            child = summarise(v, depth + 1, max_depth)
            lines.append(indent + "  " + repr(k) + ": " + child)
        if len(obj) > 15:
            lines.append(indent + f"  ... ({len(obj)} keys total)")
        lines.append(indent + "}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        if len(obj) == 0:
            return "[]"
        return f"[{len(obj)} items, first: " + summarise(obj[0], depth+1, max_depth) + "]"
    else:
        val = repr(obj)
        return val[:80] + "..." if len(val) > 80 else val

with open(INPUT_PATH, "rb") as f:
    raw = f.readline().replace(b"NaN", b"null")

record = json.loads(raw)
print(summarise(record))
print("\n--- CONVOS KEYS ---")
convos = record.get("convos", "NOT FOUND")
if isinstance(convos, dict):
    print("convos keys:", list(convos.keys()))
    for k, v in convos.items():
        if isinstance(v, list):
            print(f"  convos['{k}']: list of {len(v)} items")
            if v:
                first = v[0]
                if isinstance(first, list):
                    print(f"    first item is also a list of {len(first)} items")
                    if first:
                        print(f"    first[0] keys: {list(first[0].keys()) if isinstance(first[0], dict) else first[0]}")
                elif isinstance(first, dict):
                    print(f"    first item keys: {list(first.keys())}")
        elif isinstance(v, dict):
            print(f"  convos['{k}']: dict with keys {list(v.keys())[:10]}")
        else:
            print(f"  convos['{k}']: {repr(v)[:80]}")
else:
    print("convos:", convos)

print("\n--- TOP LEVEL KEYS ---")
print(list(record.keys()))