INPUT_PATH  = "/case_with_all_sources_with_companion_cases_tag.jsonl"   # path to the raw JSONL file
OUTPUT_PATH = "/data/extracted_labelled.csv"  # where to save the output CSV

import json
from pathlib import Path
import pandas as pd
from collections import defaultdict


def is_question(text: str):
    return "?" in text

def word_count(text: str):
    return len(text.split())

def count_interruptions(turns: list, speaker_id: str):
    interruptions = 0
    for i in range(1, len(turns)):
        prev = turns[i - 1]
        curr = turns[i]
        if curr["speaker_id"] != speaker_id:
            continue
        if not prev["stop_times"] or not curr["start_times"]:
            continue
        prev_stop = prev["stop_times"][-1]
        curr_start = curr["start_times"][0]
        if curr_start < prev_stop:
            interruptions += 1
    return interruptions

def extract_record(record: dict) -> list[dict]:
    rows = []
    case_id = record.get("id", "")
    year = record.get("year")
    court = record.get("court", "")
    win_side = record.get("win_side")

    # advocate side map: speaker_id -> side (0 or 1)
    advocate_side = {}
    for adv_id, adv_info in (record.get("advocates") or {}).items():
        norm_id = adv_id.lower().replace(".", "").replace(" ", "_")
        advocate_side[norm_id] = adv_info.get("side")

    # scdb metadata
    scdb = record.get("scdb_elements") or {}
    issue_area = scdb.get("issueArea")
    decision_direction = scdb.get("decisionDirection")
    maj_votes = scdb.get("majVotes")
    min_votes = scdb.get("minVotes")

    # votes_side label per justice
    votes_side = {}
    convos = record.get("convos") or {}

    # votes_side can be at top level or inside convos
    if "votes_side" in record:
        votes_side = record["votes_side"]
    elif "votes_side" in convos:
        votes_side = convos["votes_side"]

    if not votes_side:
        return rows 

    # flatten all turns across all transcripts
    # the field is called "utterances" and is a list of lists (one per session)
    all_turns = []
    utterances = convos.get("utterances") or []

    for session in utterances:
        if isinstance(session, list):
            all_turns.extend(session)
        elif isinstance(session, dict):
            all_turns.append(session)

    if not all_turns:
        return rows

    justice_stats = defaultdict(lambda: {
        "utterances": [],
        "words_to_side0": 0,
        "words_to_side1": 0,
        "questions_to_side0": 0,
        "questions_to_side1": 0,
        "total_words": 0,
        "total_utterances": 0,
        "interruptions": 0,
    })

    # figure out which side the *current* advocate is on for each turn
    # by tracking who spoke last from the advocate pool
    current_adv_side = None

    for i, turn in enumerate(all_turns):
        spk = turn.get("speaker_id", "")
        text = turn.get("text", "") or ""

        # if this is an advocate turn, update current side context
        if not spk.startswith("j__"):
            norm = spk.lower().replace(".", "").replace(" ", "_")
            if norm in advocate_side:
                current_adv_side = advocate_side[norm]
            elif spk in advocate_side:
                current_adv_side = advocate_side[spk]
            continue  # don't record advocate turns as justice features

        # it's a justice turn
        s = justice_stats[spk]
        if current_adv_side == 0:
            tagged_text = f"[SIDE0] {text}"
        elif current_adv_side == 1:
            tagged_text = f"[SIDE1] {text}"
        else:
            tagged_text = f"[UNKNOWN] {text}"
        s["utterances"].append(tagged_text)
        s["total_words"] += word_count(text)
        s["total_utterances"] += 1

        if current_adv_side is not None:
            wc = word_count(text)
            q  = is_question(text)
            if current_adv_side == 0:
                s["words_to_side0"] += wc
                s["questions_to_side0"] += int(q)
            else:
                s["words_to_side1"] += wc
                s["questions_to_side1"] += int(q)

    # compute interruptions per justice over the full turn list
    for j_id in justice_stats:
        justice_stats[j_id]["interruptions"] = count_interruptions(all_turns, j_id)

    # build one row per justice
    for j_id, stats in justice_stats.items():
        label = votes_side.get(j_id)
        if label is None:
            continue  # justice spoke but no vote recorded

        total_words = stats["total_words"] or 1 

        rows.append({
            "case_id": case_id,
            "year":year,
            "court": court,
            "justice_id":j_id,
            "label": int(label),        
            "win_side": win_side,

            "all_utterances":       " ||| ".join(stats["utterances"]),
            "total_words": stats["total_words"],
            "total_utterances": stats["total_utterances"],
            "words_to_side0": stats["words_to_side0"],
            "words_to_side1": stats["words_to_side1"],
            "word_ratio_0_to_1":stats["words_to_side0"] / (stats["words_to_side1"] + 1),

            "questions_to_side0": stats["questions_to_side0"],
            "questions_to_side1":  stats["questions_to_side1"],
            "question_ratio_0_to_1":stats["questions_to_side0"] / (stats["questions_to_side1"] + 1),
            "interruptions": stats["interruptions"],

            "issue_area": issue_area,
            "decision_direction": decision_direction,
            "maj_votes": maj_votes,
            "min_votes": min_votes,
        })
    return rows



def diagnose(input_path):
    #Print the structure of the first record to debug field names
    def summarise(obj, depth=0, max_depth=3):
        indent = "  " * depth
        if depth > max_depth:
            return indent + "..."
        if isinstance(obj, dict):
            lines = [indent + "{"]
            for k, v in list(obj.items())[:12]:
                child = summarise(v, depth + 1, max_depth)
                lines.append(indent + "  " + repr(k) + ": " + child)
            if len(obj) > 12:
                lines.append(indent + f"... ({len(obj)} keys total)")
            lines.append(indent + "}")
            return "\n".join(lines)
        elif isinstance(obj, list):
            if len(obj) == 0:
                return "[]"
            return f"[ ({len(obj)} items)  first: " + summarise(obj[0], depth+1, max_depth) + " ]"
        else:
            val = repr(obj)
            return val[:80] + "..." if len(val) > 80 else val

    with open(Path(input_path), "rb") as f:
        raw = f.readline().replace(b"NaN", b"null")
    record = json.loads(raw)
    print()
    print(summarise(record))
    print()


def main(input_path: str, output_path: str):
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diagnose(input_path)

    all_rows = []
    skipped = 0
    parsed = 0

    print(f"Reading {input_path}")

    with open(input_path, "rb") as f:
        for line_num, raw_line in enumerate(f, 1):
            raw_line = raw_line.replace(b"NaN", b"null")
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
                rows = extract_record(record)
                all_rows.extend(rows)
                parsed += 1
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 5:
                    print(f"[skip] line {line_num}: {e}")

            if parsed % 500 == 0:
                print(f"parsed {parsed} cases, {len(all_rows)} rows so far")

    df = pd.DataFrame(all_rows)

    print(f"\nDone {parsed} cases parsed, {skipped} skipped.")
    print(f"Total rows (case × justice pairs): {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"\nSample:\n{df.head(3).to_string()}")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main(INPUT_PATH, OUTPUT_PATH)