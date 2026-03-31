"""
score_sentiment.py

Scores each justice utterance for negative/unpleasant sentiment using VADER,
then computes per-(case, justice) features:

    unpleasant_to_petitioner  = mean VADER negativity directed at petitioner's counsel
    unpleasant_to_respondent  = mean VADER negativity directed at respondent's counsel
    unpleasant_diff           = unpleasant_to_petitioner - unpleasant_to_respondent
                                (the x-axis in the Black et al. figure)

This is saved as a new CSV that can be joined to extracted.csv on (case_id, justice_id).

Before running:
    pip install nltk
    python -c "import nltk; nltk.download('vader_lexicon')"

Set INPUT_PATH and OUTPUT_PATH below, then run:
    python score_sentiment.py
"""

# ── SET YOUR PATHS HERE ───────────────────────────────────────────────────────
INPUT_PATH  = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\case_with_all_sources_with_companion_cases_tag.jsonl"
OUTPUT_PATH = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\eda figures\sentiment_scores.csv"
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def score_record(record: dict, sia: SentimentIntensityAnalyzer) -> list[dict]:
    rows = []

    case_id  = record.get("id", "")
    win_side = record.get("win_side")

    # advocate side map: speaker_id -> side (0=respondent, 1=petitioner)
    advocate_side = {}
    for adv_name, adv_info in (record.get("advocates") or {}).items():
        norm = adv_name.lower().replace(" ", "_")
        advocate_side[norm] = adv_info.get("side")
        # also store by the id field directly
        adv_id = adv_info.get("id", "")
        if adv_id:
            advocate_side[adv_id] = adv_info.get("side")

    # votes_side label per justice
    votes_side = record.get("votes_side") or {}
    convos     = record.get("convos") or {}
    if not votes_side:
        votes_side = convos.get("votes_side") or {}
    if not votes_side:
        return rows

    # flatten utterances
    all_turns = []
    for session in (convos.get("utterances") or []):
        if isinstance(session, list):
            all_turns.extend(session)
        elif isinstance(session, dict):
            all_turns.append(session)

    if not all_turns:
        return rows

    # per-justice accumulators
    # neg scores directed at petitioner (side 1) and respondent (side 0)
    stats = defaultdict(lambda: {
        "neg_to_petitioner": [],   # VADER neg scores when questioning petitioner's counsel
        "neg_to_respondent": [],   # VADER neg scores when questioning respondent's counsel
        "total_utterances": 0,
    })

    current_adv_side = None  # which side is currently being questioned

    for turn in all_turns:
        spk  = turn.get("speaker_id", "")
        text = (turn.get("text") or "").strip()
        if not text:
            continue

        # update current advocate side context
        if not spk.startswith("j__"):
            norm = spk.lower().replace(" ", "_")
            side = advocate_side.get(norm) or advocate_side.get(spk)
            if side is not None:
                current_adv_side = side
            continue

        # justice utterance — score it
        scores = sia.polarity_scores(text)
        neg    = scores["neg"]   # 0–1, fraction of text that is negative/unpleasant

        s = stats[spk]
        s["total_utterances"] += 1

        if current_adv_side == 1:          # questioning petitioner's counsel
            s["neg_to_petitioner"].append(neg)
        elif current_adv_side == 0:        # questioning respondent's counsel
            s["neg_to_respondent"].append(neg)

    # build one row per justice
    for j_id, s in stats.items():
        label = votes_side.get(j_id)
        if label is None:
            continue

        neg_pet = (sum(s["neg_to_petitioner"]) / len(s["neg_to_petitioner"])
                   if s["neg_to_petitioner"] else 0.0)
        neg_res = (sum(s["neg_to_respondent"]) / len(s["neg_to_respondent"])
                   if s["neg_to_respondent"] else 0.0)

        rows.append({
            "case_id":               case_id,
            "justice_id":            j_id,
            "label":                 int(label),
            "win_side":              win_side,
            "neg_to_petitioner":     neg_pet,
            "neg_to_respondent":     neg_res,
            # KEY FEATURE: positive = more unpleasant to petitioner
            #              negative = more unpleasant to respondent
            "unpleasant_diff":       neg_pet - neg_res,
            "n_utt_to_petitioner":   len(s["neg_to_petitioner"]),
            "n_utt_to_respondent":   len(s["neg_to_respondent"]),
            "total_utterances":      s["total_utterances"],
        })

    return rows


def main():
    input_path  = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading VADER...")
    sia = SentimentIntensityAnalyzer()

    all_rows = []
    parsed   = 0
    skipped  = 0

    # diagnose first record
    print("Diagnosing first record...")
    with open(input_path, "rb") as f:
        for _line in f:
            _line = _line.replace(b"NaN", b"null").strip()
            if _line:
                break
    rec = json.loads(_line)
    convos = rec.get("convos") or {}
    print(f"  convos keys: {list(convos.keys())}")
    utterances = convos.get("utterances") or []
    print(f"  utterances: {type(utterances)}, len={len(utterances)}")
    if utterances:
        first = utterances[0]
        print(f"  utterances[0] type: {type(first)}")
        if isinstance(first, list):
            print(f"  utterances[0] len: {len(first)}")
            if first:
                print(f"  utterances[0][0] keys: {list(first[0].keys()) if isinstance(first[0], dict) else first[0]}")
        elif isinstance(first, dict):
            print(f"  utterances[0] keys: {list(first.keys())}")
    votes_side = rec.get("votes_side") or convos.get("votes_side") or {}
    print(f"  votes_side keys: {list(votes_side.keys())[:5]}")
    advocates = rec.get("advocates") or {}
    print(f"  advocates: {list(advocates.keys())}")
    print()

    print(f"Scoring {input_path} ...")
    with open(input_path, "rb") as f:
        for line_num, raw_line in enumerate(f, 1):
            raw_line = raw_line.replace(b"NaN", b"null").strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
                rows   = score_record(record, sia)
                all_rows.extend(rows)
                parsed += 1
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [skip] line {line_num}: {e}")

            if parsed % 500 == 0:
                print(f"  scored {parsed} cases, {len(all_rows)} rows so far...")

    df = pd.DataFrame(all_rows)

    # drop non-votes
    df = df[df["label"].isin([0, 1])].copy()

    print(f"\nDone. {parsed} cases, {len(df):,} rows.")
    print(f"\nunpleasant_diff summary:")
    print(df["unpleasant_diff"].describe().round(4))

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()