"""
fetch_justice_transcripts.py

Fetches oral argument transcripts from the Oyez API for a given justice
and outputs a CSV with one row per justice utterance, including preceding
advocate context.

Usage:
    python fetch_justice_transcripts.py

Output:
    kagan_transcripts.csv
"""

import requests
import json
import csv
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

TARGET_JUSTICE = "Elena Kagan"

# Maps Oyez display names to your dataset's justice_id format
JUSTICE_ID_MAP = {
    "Elena Kagan": "j__elena_kagan",
    "Sonia Sotomayor": "j__sonia_sotomayor",
    "John G. Roberts, Jr.": "j__john_g_roberts_jr",
    "Clarence Thomas": "j__clarence_thomas",
    "Samuel A. Alito, Jr.": "j__samuel_a_alito_jr",
    "Neil Gorsuch": "j__neil_gorsuch",
    "Brett M. Kavanaugh": "j__brett_m_kavanaugh",
    "Amy Coney Barrett": "j__amy_coney_barrett",
    "Ketanji Brown Jackson": "j__ketanji_brown_jackson",
}

# Testing with 2014 term only — expand range once output is verified
TERMS = [2014]

OUTPUT_FILE = "kagan_transcripts.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research project, NYU)"
}

# How many turns before the justice turn to include as context
CONTEXT_WINDOW = 2

# Polite delay between API requests (seconds)
REQUEST_DELAY = 0.5

# ── API helpers ───────────────────────────────────────────────────────────────

def get_cases_for_term(term: int) -> list:
    """Returns list of case metadata dicts for a given term."""
    url = f"https://api.oyez.org/cases?filter=term:{term}&per_page=100"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_case_detail(href: str) -> dict:
    """Fetches full case detail JSON from a case href."""
    resp = requests.get(href, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_oral_argument(href: str) -> dict:
    """Fetches oral argument transcript JSON from its href."""
    resp = requests.get(href, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()
    


# ── Transcript parsing ────────────────────────────────────────────────────────

def extract_turns(oral_arg: dict) -> list:
    """
    Flattens all sections/turns in an oral argument into a single list.
    Each entry: { turn_index, speaker_name, role, text }
    """
    turns = []
    idx = 0
    for section in oral_arg.get("sections", []):
        for turn in section.get("turns", []):
            speaker = turn.get("speaker") or {}
            name = speaker.get("name", "Unknown")

            # role can be under speaker.roles (list) or speaker.role (str)
            roles = speaker.get("roles") or []
            if roles:
                role = roles[0].get("role_title", "")
            else:
                role = speaker.get("role", "")

            # Concatenate all text blocks into one string
            text_blocks = turn.get("text_blocks", [])
            text = " ".join(
                block.get("text", "") for block in text_blocks
            ).strip()

            if text:  # skip empty turns
                turns.append({
                    "turn_index": idx,
                    "speaker_name": name,
                    "role": role,
                    "text": text,
                })
                idx += 1
    return turns


def get_preceding_context(turns: list, current_idx: int, window: int = CONTEXT_WINDOW) -> str:
    """
    Returns the text of the `window` advocate turns immediately before
    the current turn, concatenated with speaker labels.
    """
    context_parts = []
    count = 0
    for i in range(current_idx - 1, -1, -1):
        t = turns[i]
        # Only include advocate turns (not other justices) as context
        if "Justice" not in t["speaker_name"] and t["text"]:
            context_parts.insert(0, f"[ADVOCATE - {t['speaker_name']}]: {t['text']}")
            count += 1
            if count >= window:
                break
    return " ||| ".join(context_parts)


def determine_side_addressed(turn: dict, turns: list, current_idx: int) -> int:
    """
    Heuristic: looks at which advocate spoke most recently before this turn.
    Returns 0 or 1. -1 if indeterminate.
    Side 0 = petitioner (first to argue), Side 1 = respondent.
    
    This is a rough heuristic — for ground truth you'd use your existing
    dataset's side labels joined on case_id + turn position.
    """
    for i in range(current_idx - 1, -1, -1):
        t = turns[i]
        if "Justice" not in t["speaker_name"]:
            # Petitioner argues first — simple heuristic based on turn position
            # In practice, join with your existing CSV for accurate side labels
            return 0 if i < len(turns) // 2 else 1
    return -1


# ── Vote label lookup ─────────────────────────────────────────────────────────

def get_vote_label(case_detail: dict, justice_name: str) -> int:
    """
    Extracts the binary vote label for the target justice from case detail.
    Returns 1 if justice voted with majority/petitioner, 0 otherwise, -1 if not found.
    """
    decisions = case_detail.get("decisions", []) or []
    for decision in decisions:
        votes = decision.get("votes", []) or []
        for vote in votes:
            member = vote.get("member") or {}
            if member.get("name", "") == justice_name:
                vote_type = vote.get("vote", "")
                print(f"  [DEBUG] Found {justice_name}, vote field value: '{vote_type}'")
                if vote_type == "majority":
                    return 1
                elif vote_type == "minority":
                    return 0
    return -1  # not found


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_case(case_meta: dict, justice_name: str, justice_id: str) -> list:
    """
    Given a case metadata dict from the term listing, fetches the full case,
    fetches oral argument(s), and returns a list of row dicts for the target justice.
    """
    rows = []
    case_href = case_meta.get("href")
    if not case_href:
        return rows

    try:
        case_detail = get_case_detail(case_href)
        time.sleep(REQUEST_DELAY)
    except Exception as e:
        print(f"  [WARN] Could not fetch case {case_href}: {e}")
        return rows

    case_id = case_detail.get("ID", case_meta.get("docket_number", "unknown"))
    term = case_detail.get("term", "unknown")
    docket = case_detail.get("docket_number", "unknown")
    issue_area = case_detail.get("issue_area", {})
    issue_area_name = issue_area.get("name", "") if issue_area else ""

    vote_label = get_vote_label(case_detail, justice_name)

    if vote_label == -1:
        # Debug: print what the decisions structure actually looks like
        decisions = case_detail.get("decisions", []) or []
        print(f"  [DEBUG] {case_detail.get('docket_number')} — decisions count: {len(decisions)}")
        if decisions:
            votes = decisions[0].get("votes", []) or []
            all_names = [v.get("member", {}).get("name") for v in votes]
            print(f"  [DEBUG] All voters: {all_names}")

        return rows

    oral_args = case_detail.get("oral_argument_audio") or []
    for oa_meta in oral_args:
        oa_href = oa_meta.get("href")
        if not oa_href:
            continue

        try:
            oral_arg = get_oral_argument(oa_href)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"  [WARN] Could not fetch oral arg {oa_href}: {e}")
            continue

        turns = extract_turns(oral_arg)
        speaker_names = set(t["speaker_name"] for t in turns)
        print(f"  [DEBUG] Speakers in transcript: {speaker_names}")

        for list_pos, turn in enumerate(turns):
            if turn["speaker_name"] != justice_name:
                continue

            # Use list_pos (enumerate index) not turn["turn_index"] —
            # turn_index can have gaps if empty turns were filtered out
            preceding_context = get_preceding_context(turns, list_pos)
            side_addressed = determine_side_addressed(turn, turns, list_pos)

            rows.append({
                "case_id": f"{term}_{docket}",
                "term": term,
                "docket": docket,
                "justice_id": justice_id,
                "turn_index": turn["turn_index"],
                "justice_utterance": turn["text"],
                "preceding_context": preceding_context,
                "side_addressed": side_addressed,
                "issue_area": issue_area_name,
                "label": vote_label,
            })

    return rows


def main():
    justice_id = JUSTICE_ID_MAP[TARGET_JUSTICE]
    all_rows = []

    for term in TERMS:
        print(f"Fetching term {term}...")
        try:
            cases = get_cases_for_term(term)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"  [WARN] Could not fetch term {term}: {e}")
            continue

        print(f"  Found {len(cases)} cases")

        for case_meta in cases:
            rows = process_case(case_meta, TARGET_JUSTICE, justice_id)
            all_rows.extend(rows)
            if rows:
                print(f"  + {case_meta.get('docket_number', '?')} — {len(rows)} utterances")

    print(f"\nTotal rows: {len(all_rows)}")

    if not all_rows:
        print("No data collected. Check API connectivity.")
        return

    fieldnames = [
        "case_id", "term", "docket", "justice_id", "turn_index",
        "justice_utterance", "preceding_context", "side_addressed",
        "issue_area", "label",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()