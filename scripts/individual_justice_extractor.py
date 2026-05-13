import json
import csv

INPUT_FILE = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\case_with_all_sources_with_companion_cases_tag.jsonl"
OUTPUT_FILE = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\DS301Project\data\partial_datasets\sotomayor_2009-2019.csv"

TARGET_JUSTICE = "j__sonia_sotomayor"

def is_justice(speaker_id: str) -> bool:
    return speaker_id.startswith("j__")


def get_preceding_advocate(turns: list, current_pos: int) -> dict | None:
    for i in range(current_pos - 1, -1, -1):
        if not is_justice(turns[i]["speaker_id"]):
            return turns[i]
    return None


def get_vote_label(votes_side: dict, justice_id: str) -> int:
    val = votes_side.get(justice_id, -1)
    if val is None:
        return -1
    return int(val)


def main():
    rows = []
    cases_processed = 0
    cases_skipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            case = json.loads(line)
            convos = case.get("convos", {})
            if not convos:
                cases_skipped += 1
                continue

            case_id = convos.get("case_id", "")
            year_prefix = case_id.split("_")[0] if "_" in case_id else ""
            if not (2005 <= int(year_prefix) <= 2023 if year_prefix.isdigit() else False):
                continue

            votes_side = convos.get("votes_side") or {}
            if TARGET_JUSTICE not in votes_side:
                cases_skipped += 1
                continue

            vote_label = get_vote_label(votes_side, TARGET_JUSTICE)

            utterances_outer = convos.get("utterances", [])
            if not utterances_outer:
                cases_skipped += 1
                continue

            turns = utterances_outer[0]
            if not turns:
                cases_skipped += 1
                continue

            cases_processed += 1

            for pos, turn in enumerate(turns):
                if turn["speaker_id"] != TARGET_JUSTICE:
                    continue

                justice_text = turn.get("text", "").strip()
                if not justice_text:
                    continue

                prev = get_preceding_advocate(turns, pos)
                preceding_text = prev["text"].strip() if prev else ""
                preceding_speaker = prev["speaker_id"] if prev else ""
                speakers = convos.get("speaker", {})
                adv_info = speakers.get(preceding_speaker, {})
                side_addressed = adv_info.get("side", -1)
                if side_addressed == 0:
                    side_addressed = 1
                elif side_addressed == 1:
                    side_addressed = 0

                rows.append({
                    "case_id": case_id,
                    "justice_id": TARGET_JUSTICE,
                    "turn_position": pos,
                    "justice_utterance": justice_text,
                    "preceding_speaker": preceding_speaker,
                    "preceding_context": preceding_text,
                    "side_addressed": side_addressed,
                    "label": vote_label,
                })

    fieldnames = [
        "case_id",
        "justice_id",
        "turn_position",
        "justice_utterance",
        "preceding_speaker",
        "preceding_context",
        "side_addressed",
        "label",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cases processed: {cases_processed}")
    print(f"Cases skipped: {cases_skipped}")
    print(f"Total rows: {len(rows)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()