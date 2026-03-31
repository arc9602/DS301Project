import json

with open('data\case_with_all_sources_with_companion_cases_tag.jsonl', 'rb') as f:
    import ijson
    for record in ijson.items(f, 'item'):
        convos = record.get('convos', {})
        print(json.dumps({
            'keys': list(convos.keys()),
            'sample_turn': convos.get('turns', [{}])[0] if 'turns' in convos else 'no turns key - check structure'
        }, indent=2))
        break  # just first record