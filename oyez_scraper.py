import re
import csv
import time
import requests
from pathlib import Path
from typing import Optional

# Configuration

TARGET_JUSTICE = "Ketanji Brown Jackson"
JUSTICE_ID = "j__ketanji_brown_jackson"


START_TERM = 2020
END_TERM = 2024
OUTPUT_FILE = f"data/jackson_{START_TERM}-{END_TERM}.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research project, NYU)"
}

REQUEST_DELAY = 0.5

JUSTICE_NAMES = {
    "Elena Kagan",
    "Sonia Sotomayor",
    "John G. Roberts, Jr.",
    "Clarence Thomas",
    "Samuel A. Alito, Jr.",
    "Neil Gorsuch",
    "Brett M. Kavanaugh",
    "Amy Coney Barrett",
    "Ketanji Brown Jackson",
    "Ruth Bader Ginsburg",
    "Stephen G. Breyer",
    "Anthony M. Kennedy",
    "Antonin Scalia",
    "David H. Souter",
    "John Paul Stevens",
}

# Name normalization
def norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower().replace(".", "")

def clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", name.lower()).strip()

def make_acronym(name: str) -> str:
    STOPWORDS = {"et", "al", "of", "the", "a", "an", "and", "or", "in", "for", "v"}
    words = clean_name(name).split()
    return "".join(w[0] for w in words if w not in STOPWORDS)

def names_match(winning: str, party: str) -> bool:
    if not winning or not party:
        return False
    w, p = clean_name(winning), clean_name(party)
    if w in p or p in w:
        return True
    if w == make_acronym(p) or p == make_acronym(w):
        return True
    return False

# Precompute normalized forms
JUSTICE_NAMES_NORMALIZED  = {norm_name(n) for n in JUSTICE_NAMES}
TARGET_JUSTICE_NORMALIZED = norm_name(TARGET_JUSTICE)

# API helpers

def get_cases_for_term(term: int) -> list:
    cases, page = [], 0
    while True:
        url = f"https://api.oyez.org/cases?filter=term:{term}&per_page=100&page={page}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        cases.extend(batch)
        page += 1
        time.sleep(REQUEST_DELAY)
    return cases


def get_case_detail(href: str) -> dict:
    resp = requests.get(href, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_oral_argument(href: str) -> dict:
    resp = requests.get(href, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


# Vote label

def resolve_favor_side(case_detail: dict, decision: dict) -> Optional[int]:
    """
    Returns 1 if petitioner won, 0 if respondent won, None if indeterminate.
    """
    winning_party = (decision.get("winning_party") or "").strip()
    first_party = (case_detail.get("first_party") or "").strip()
    second_party = (case_detail.get("second_party") or "").strip()
    first_label = (case_detail.get("first_party_label") or "").strip().lower()
    second_label = (case_detail.get("second_party_label") or "").strip().lower()

    if not winning_party:
        return None

    # Determine which party is the petitioner from labels
    if any(w in first_label for w in ["petition", "appellant", "plaintiff"]):
        first_is_petitioner = True
    elif any(w in second_label for w in ["petition", "appellant", "plaintiff"]):
        first_is_petitioner = False
    else:
        return None

    matched_first  = names_match(winning_party, first_party)
    matched_second = names_match(winning_party, second_party)

    if matched_first and not matched_second:
        return 1 if first_is_petitioner else 0
    if matched_second and not matched_first:
        return 0 if first_is_petitioner else 1

    return None


def get_vote_label(case_detail: dict, justice_name: str) -> Optional[int]:
    decisions = case_detail.get("decisions") or []

    if not decisions:
        return None, "no decisions found"

    target = norm_name(justice_name)

    for decision in decisions:
        target_vote = None
        for vote in (decision.get("votes") or []):
            member = vote.get("member") or {}
            if norm_name(member.get("name", "")) == target:
                target_vote = (vote.get("vote") or "").lower().strip()
                break

        if target_vote is None:
            continue

        favor_side = resolve_favor_side(case_detail, decision)
        if favor_side is None:
            winning = decision.get("winning_party", "")
            return None, f"winning_party '{winning}' unmatched"

        if target_vote == "majority":
            return favor_side, None
        if target_vote == "minority":
            return 1 - favor_side, None

        return None, f"unhandled vote type '{target_vote}'"

    return None, "justice not found in votes"


# Transcript parsing

def is_justice(speaker_name: str) -> bool:
    return norm_name(speaker_name) in JUSTICE_NAMES_NORMALIZED


def extract_turns(oral_arg: dict) -> list:
    turns = []
    idx   = 0

    transcript = oral_arg.get("transcript") or {}
    sections   = transcript.get("sections") or []

    for section in sections:
        for turn in (section.get("turns") or []):
            speaker = turn.get("speaker") or {}
            name = speaker.get("name", "Unknown")
            blocks = turn.get("text_blocks") or []
            text = " ".join(b.get("text", "") for b in blocks).strip()
            if text:
                turns.append({
                    "turn_index": idx,
                    "speaker_name": name,
                    "text": text,
                })
                idx += 1

    return turns


def get_preceding_advocate(turns: list, current_pos: int, advocate_sides: dict):

    for i in range(current_pos - 1, -1, -1):
        speaker = turns[i]["speaker_name"]
        if norm_name(speaker) in advocate_sides:
            return turns[i]
    return None


def determine_side(preceding_speaker: str, advocate_sides: dict) -> int:
    """
    Returns the side of the preceding advocate.
    0 = petitioner, 1 = respondent, 2 = amicus, -1 = unknown
    """
    return advocate_sides.get(norm_name(preceding_speaker), -1)


# Advocate side mapping 

def build_advocate_sides(case_detail: dict) -> dict:
    """
      0 = petitioner, 1 = respondent, 2 = amicus, -1 = unknown
    """
    advocate_sides = {}
    advocates = case_detail.get("advocates") or []
    for adv in advocates:
        advocate    = adv.get("advocate") or {}
        name        = advocate.get("name", "")
        description = (adv.get("advocate_description") or "").lower()

        if "amicus" in description:
            side = 2
        elif any(w in description for w in ["petitioner", "appellant", "plaintiff"]):
            side = 0
        elif any(w in description for w in ["respondent", "appellee", "defendant"]):
            side = 1
        else:
            side = -1

        if name:
            advocate_sides[norm_name(name)] = side

    return advocate_sides


# Case processing 

def process_case(case_meta: dict, justice_name: str, justice_id: str) -> list:
    rows = []
    case_href = case_meta.get("href")
    if not case_href:
        return rows

    try:
        case_detail = get_case_detail(case_href)
        time.sleep(REQUEST_DELAY)
    except Exception as e:
        print(f"Could not fetch case {case_href}: {e}")
        return rows

    docket = case_detail.get("docket_number", case_meta.get("docket_number", "?")).strip()
    term = str(case_detail.get("term", "unknown"))

    voted_for_petitioner, skip_reason = get_vote_label(case_detail, justice_name)
    if voted_for_petitioner is None:
        print(f"Skip {docket} — {skip_reason}")
        return rows

    advocate_sides = build_advocate_sides(case_detail)

    oral_args = case_detail.get("oral_argument_audio") or []
    for oa_index, oa_meta in enumerate(oral_args):
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
        if not turns:
            continue

        for list_pos, turn in enumerate(turns):
            if norm_name(turn["speaker_name"]) != TARGET_JUSTICE_NORMALIZED:
                continue

            justice_text = turn["text"].strip()
            if not justice_text:
                continue

            prev = get_preceding_advocate(turns, list_pos, advocate_sides)
            preceding_text = prev["text"].strip() if prev else ""
            preceding_name = prev["speaker_name"] if prev else ""
            side_addressed = determine_side(preceding_name, advocate_sides)

            rows.append({
                "case_id": f"{term}_{docket}",
                "oral_argument_index": oa_index,
                "justice_id":justice_id,
                "turn_position": turn["turn_index"],
                "justice_utterance": justice_text,
                "preceding_speaker": preceding_name,
                "preceding_context": preceding_text,
                "side_addressed": side_addressed,
                "voted_for_petitioner": voted_for_petitioner,
            })

    if rows:
        print(f"  + {docket} — {len(rows)} utterances (voted_for_petitioner: {voted_for_petitioner})")

    return rows



def main():
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for term in range(START_TERM, END_TERM + 1):
        print(f"Fetching term {term}...")
        try:
            cases = get_cases_for_term(term)
        except Exception as e:
            print(f"  [WARN] Could not fetch term {term}: {e}")
            continue

        print(f"Found {len(cases)} cases")

        for case_meta in cases:
            rows = process_case(case_meta, TARGET_JUSTICE, JUSTICE_ID)
            all_rows.extend(rows)

    print(f"\nTotal rows: {len(all_rows)}")

    if not all_rows:
        print("No data collected.")
        return

    fieldnames = [
        "case_id",
        "oral_argument_index",
        "justice_id",
        "turn_position",
        "justice_utterance",
        "preceding_speaker",
        "preceding_context",
        "side_addressed",
        "voted_for_petitioner",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
