import json

records = []
with open('data\case_with_all_sources_with_companion_cases_tag.jsonl', 'rb') as f:
    for line in f:
        line = line.replace(b'NaN', b'null')
        if line.strip():  # skip empty lines
            record = json.loads(line)
            records.append(record)

print(f"Total records: {len(records)}")
print(json.dumps(records[0], indent=2))