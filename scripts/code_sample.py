import json

with open('../data/case_with_all_sources_with_companion_cases_tag.jsonl', 'rb') as f:
    # Just read the first N bytes to see structure
    print(f.read(3000).decode('utf-8', errors='replace'))