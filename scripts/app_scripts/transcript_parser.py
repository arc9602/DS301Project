from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber

# petitioner = 1, respondent = 0
SIDE_RESPONDENT = 0
SIDE_PETITIONER = 1
SIDE_AMICUS = 2
SIDE_UNKNOWN = 3

SIDE_LABELS = {
    SIDE_RESPONDENT: "respondent",
    SIDE_PETITIONER: "petitioner",
    SIDE_AMICUS: "amicus",
    SIDE_UNKNOWN: "unknown",
}

CONF_HIGH = "high" 
CONF_MEDIUM = "medium" 
CONF_LOW = "low"
CONF_NONE = "none"  

_LINE_NUM_PREFIX = re.compile(r"^\d{1,2} ")

_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"official\s*[-–]\s*subject to final review", re.I),
    re.compile(r"alderson reporting company", re.I),
    re.compile(r"heritage reporting corporation", re.I),
    re.compile(r"^\s*$"),
    re.compile(r"^\s*\d{1,3}\s*$"),
    re.compile(r"^\s*Sheet\s+\d+", re.I),
]

_SOFT_HYPHEN = re.compile(r"\xad+")
_END_OF_ARGUMENT = re.compile(
    r"\(Whereupon,[^)]{0,300}?"
    r"(?:case|argument|matter)[^)]{0,80}?"
    r"submitted\.\s*\)",
    re.IGNORECASE | re.DOTALL,
)


def _truncate_back_matter(text: str) -> str:
    m = _END_OF_ARGUMENT.search(text)
    if not m:
        return text
    return text[: m.end()]


def extract_text_from_pdf(pdf_path: str) -> str:
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            clean_lines: list[str] = []
            for ln in raw.splitlines():
                ln = _LINE_NUM_PREFIX.sub("", ln)
                ln = _SOFT_HYPHEN.sub("-", ln)
                if not any(p.match(ln.strip()) for p in _NOISE_PATTERNS):
                    clean_lines.append(ln)
            pages.append("\n".join(clean_lines))
    return _truncate_back_matter("\n".join(pages))

@dataclass
class Advocate:
    full_name: str
    last_name: str 
    side: int
    role_text: str 
    amicus_supporting: Optional[int] = None   

_ROLE_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"amicus.{0,60}support.{0,20}petitioner", re.I | re.S), SIDE_PETITIONER),
    (re.compile(r"amicus.{0,60}support.{0,20}respondent", re.I | re.S), SIDE_RESPONDENT),
    (re.compile(r"(?:for|as)\s+amicus\s+curiae", re.I), SIDE_AMICUS),
    (re.compile(r"amicus\s+curiae", re.I), SIDE_AMICUS),
    (re.compile(r"on behalf of (?:the )?petitioner", re.I), SIDE_PETITIONER),
    (re.compile(r"on behalf of (?:the )?respondent", re.I), SIDE_RESPONDENT),
    (re.compile(r"on behalf of (?:the )?appellant", re.I), SIDE_PETITIONER),
    (re.compile(r"on behalf of (?:the )?appellee", re.I), SIDE_RESPONDENT),
    (re.compile(r"^for (?:the )?petitioner", re.I), SIDE_PETITIONER),
    (re.compile(r"^for (?:the )?respondent", re.I), SIDE_RESPONDENT),
    (re.compile(r"^for (?:the )?appellant", re.I), SIDE_PETITIONER),
    (re.compile(r"^for (?:the )?appellee", re.I), SIDE_RESPONDENT),
]
_NAME_SUFFIXES = re.compile(
    r"\b(?:jr\.?|sr\.?|ii+|iv|esq\.?|phd|j\.d\.|ll\.m\.)\b",
    re.I,
)
_PARTICLES = {"von", "van", "de", "du", "le", "la", "del", "della", "di", "da", "dos", "bin"}


def _extract_last_name(full_name: str) -> str:
    name = _NAME_SUFFIXES.sub("", full_name).strip()
    tokens = [re.sub(r"[^a-z\-]", "", t.lower()) for t in name.split()]
    tokens = [t for t in tokens if t]

    if not tokens:
        return ""

    surname = [tokens[-1]]

    i = len(tokens) - 2
    while i >= 0 and tokens[i] in _PARTICLES:
        surname.insert(0, tokens[i])
        i -= 1

    return "_".join(surname)


def _side_from_role(role_text: str) -> int:
    for pattern, side in _ROLE_PATTERNS:
        if pattern.search(role_text):
            return side
    return SIDE_UNKNOWN


def parse_appearances_block(text: str) -> dict[str, Advocate]:

    start_m = re.search(r"^\s*APPEARANCES\s*:?\s*$", text, re.MULTILINE)
    if not start_m:
        return {}

    block = text[start_m.end():]

    stop_m = re.search(
        r"\n(?:C\s*O\s*N\s*T\s*E\s*N\s*T\s*S|"         # C O N T E N T S
        r"ORAL ARGUMENT OF|REBUTTAL ARGUMENT OF|"
        r"P\s*R\s*O\s*C\s*E\s*E\s*D\s*I\s*N\s*G\s*S)",  # P R O C E E D I N G S
        block, re.I,
    )
    if stop_m:
        block = block[: stop_m.start()]

    def is_name_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if re.match(
            r"^(?:on behalf of|for the|as amicus|amicus curiae|supporting|"
            r"oral argument|rebuttal argument|contents?|page)\b",
            stripped,
            re.I,
        ):
            return False

        return bool(re.match(
            r"^(?:[A-Z]\.\s*)?[A-Z][A-Z.\-']+"
            r"(?:\s+(?:[A-Z]\.|[A-Z][A-Z.\-']+))*\s*(?:,|;)",
            stripped,
        ))

    raw_entries: list[str] = []
    current: list[str] = []
    for line in block.splitlines():
        if is_name_line(line) and current:
            raw_entries.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        raw_entries.append("\n".join(current))

    advocates: dict[str, Advocate] = {}

    for raw in raw_entries:
        raw = raw.strip()
        if not raw or len(raw) < 4:
            continue

        lines = raw.splitlines()
        name_line = lines[0]

        name_m = re.match(r"^([A-Z][A-Z\s\.\-\']+?)(?:,|;|$)", name_line)
        if not name_m:
            continue

        full_name = name_m.group(1).strip()
        if re.match(r"(ORAL|REBUTTAL|PAGE|CONTENTS?)", full_name, re.I):
            continue
        if len(full_name.split()) < 1:
            continue
        full_entry = " ".join(lines)
        role_m = re.search(r";(.+?)\.?\s*$", full_entry, re.S)
        role_text = role_m.group(1).strip() if role_m else full_entry

        side = _side_from_role(role_text)
        last_name = _extract_last_name(full_name)

        if not last_name:
            continue

        adv = Advocate(
            full_name=full_name,
            last_name=last_name,
            side=side,
            role_text=role_text,
        )
        advocates[last_name] = adv
        full_slug = re.sub(r"\s+", "_", full_name.lower())
        full_slug = re.sub(r"[^a-z0-9_]", "", full_slug)
        advocates[full_slug] = adv

    return advocates

@dataclass
class Segment:
    char_start:    int
    attorney_name: str  
    last_name:     str 
    side:          int
    is_rebuttal:   bool
    confidence:    str   


_BODY_HEADER = re.compile(
    r"(?P<kind>oral argument|rebuttal argument)[ \t]+of[ \t]+"
    r"(?P<name>[A-Z][A-Z .\-\']+?)"
    r"(?:\s*,\s*(?:JR\.?|SR\.?|II|III|IV|ESQ\.?|ESQUIRE|PH\.?D\.?))*"
    r"\s*\.?\s*\n"
    r"(?P<role>"
    r"[ \t]*(?:on behalf of|for)\b[^\n]*"
    r"(?:\n[ \t]*(?:supporting|curiae|as|the|petitioner"
    r"|respondent|amicus|united\s+states|appellee|appellant)"
    r"\b[^\n]*){0,2}"
    r")",
    re.IGNORECASE,
)



def _full_name_slug(name: str) -> str:
    s = re.sub(r"\s+", "_", name.strip().lower())
    return re.sub(r"[^a-z0-9_]", "", s)


def build_segment_index(
    text: str,
    advocates: dict[str, Advocate],
) -> list[Segment]:
    segments: list[Segment] = []

    for m in _BODY_HEADER.finditer(text):
        kind = m.group("kind").lower()
        attorney_name = m.group("name").strip()
        role_line = m.group("role").strip()

        last_name = _extract_last_name(attorney_name)

        full_slug = _full_name_slug(attorney_name)
        header_side = _side_from_role(role_line)

        if full_slug in advocates and advocates[full_slug].side != SIDE_UNKNOWN:
            side = advocates[full_slug].side
            conf = CONF_HIGH
        elif last_name in advocates and advocates[last_name].side != SIDE_UNKNOWN:
            side = advocates[last_name].side
            conf = CONF_HIGH
        elif header_side != SIDE_UNKNOWN:
            side = header_side
            conf = CONF_MEDIUM
        else:
            side = SIDE_UNKNOWN
            conf = CONF_NONE

        segments.append(Segment(
            char_start = m.start(),
            attorney_name = attorney_name,
            last_name = last_name,
            side = side,
            is_rebuttal = "rebuttal" in kind,
            confidence = conf,
        ))

    return sorted(segments, key=lambda s: s.char_start)


def get_segment_for_position(char_pos: int, segments: list[Segment]) -> Optional[Segment]:
    if not segments:
        return None

    idx = bisect.bisect_right([s.char_start for s in segments], char_pos) - 1

    if idx < 0:
        return None

    return segments[idx]

_SPEAKER_RE = re.compile(
    r"^("
    r"(?:CHIEF\s+)?JUSTICE\s+[A-Z][A-Z '\-]+?"
    r"|THE\s+CHIEF\s+JUSTICE"
    r"|(?:MR\.|MS\.|MRS\.)\s+[A-Z][A-Z '\-]+?"
    r"|GENERAL\s+[A-Z][A-Z '\-]+?"
    r"|(?:PROFESSOR|DR\.)\s+[A-Z][A-Z '\-]+?"
    r")\s*:",
    re.MULTILINE,
)

@dataclass
class Turn:
    speaker: str
    text: str
    char_start: int
    is_justice: bool
    is_attorney: bool


def split_into_turns(text: str) -> list[Turn]:
    matches = list(_SPEAKER_RE.finditer(text))
    if not matches:
        raise ValueError(
            "No speaker turns found — check that the PDF extracted correctly."
        )

    turns: list[Turn] = []
    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        colon_pos = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = text[colon_pos:end]
        content = _BODY_HEADER.sub("", content)
        content = content.strip()
        content = re.sub(r"\s+", " ", content)

        speaker_upper = speaker.upper()
        is_j = (speaker_upper.startswith("JUSTICE")
                or speaker_upper.startswith("CHIEF JUSTICE")
                or speaker_upper == "THE CHIEF JUSTICE")
        is_a = (speaker_upper.startswith("MR.")
                or speaker_upper.startswith("MS.")
                or speaker_upper.startswith("MRS.")
                or speaker_upper.startswith("GENERAL ")
                or speaker_upper.startswith("PROFESSOR ")
                or speaker_upper.startswith("DR."))

        turns.append(Turn(
            speaker    = speaker,
            text       = content,
            char_start = colon_pos,
            is_justice = is_j,
            is_attorney = is_a,
        ))

    return turns


def resolve_side(turn_idx:  int, turns:     list[Turn], segments:  list[Segment], advocates: dict[str, Advocate]) -> tuple[int, str, str]:
    
    turn = turns[turn_idx]
    seg = get_segment_for_position(turn.char_start, segments)
    if seg is not None and seg.side != SIDE_UNKNOWN:
        return seg.side, seg.confidence, seg.last_name
    for j in range(turn_idx - 1, -1, -1):
        prev = turns[j]
        if not prev.is_justice and prev.speaker:
            last = _extract_last_name(prev.speaker)
            if last in advocates:
                adv = advocates[last]
                return adv.side, CONF_LOW, _extract_last_name(prev.speaker)
            break  

    return SIDE_UNKNOWN, CONF_NONE, ""



def find_addressed_turn(turn_idx: int, turns: list[Turn], addressed_last_name: str) -> Optional[Turn]:
    for j in range(turn_idx - 1, -1, -1):
        prev = turns[j]
        if prev.is_justice:
            continue
        if _extract_last_name(prev.speaker) == addressed_last_name:
            return prev
    return None



JUSTICE_ID_MAP: dict[str, str] = {
    "ROBERTS": "j__john_roberts",
    "THOMAS": "j__clarence_thomas",
    "ALITO": "j__samuel_alito",
    "SOTOMAYOR": "j__sonia_sotomayor",
    "KAGAN": "j__elena_kagan",
    "GORSUCH": "j__neil_gorsuch",
    "KAVANAUGH": "j__brett_kavanaugh",
    "BARRETT": "j__amy_coney_barrett",
    "JACKSON": "j__ketanji_brown_jackson",
    "KENNEDY": "j__anthony_kennedy",
    "GINSBURG": "j__ruth_bader_ginsburg",
    "BREYER": "j__stephen_breyer",
    "SCALIA": "j__antonin_scalia",
    "SOUTER": "j__david_souter",
    "STEVENS": "j__john_paul_stevens",
    "O'CONNOR": "j__sandra_day_oconnor",
    "REHNQUIST": "j__william_rehnquist",
}


def _slugify_justice(name: str) -> str:
    upper = name.upper()
    for last_name, justice_id in JUSTICE_ID_MAP.items():
        if last_name in upper:
            return justice_id
    s = re.sub(r"(?:chief\s+)?justice\s+", "", name.strip(), flags=re.I)
    s = re.sub(r"[^a-z0-9 ]", "", s.lower())
    return "j__" + re.sub(r"\s+", "_", s.strip())



def _resolve_speaker_slug(speaker: str, advocates: dict[str, "Advocate"]) -> str:
    last = _extract_last_name(speaker)
    if last in advocates:
        return _full_name_slug(advocates[last].full_name)
    stripped = re.sub(
        r"^(?:(?:chief\s+)?justice|mr\.|ms\.|mrs\.|general|professor|dr\.)\s*",
        "", speaker.strip(), flags=re.I
    )
    return _full_name_slug(stripped) if stripped else _full_name_slug(speaker)


def build_dataframe(
    turns: list[Turn],
    case_id: str,
    segments: list[Segment],
    advocates: dict[str, Advocate],
    target_justice: str = "",
    include_confidence: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []

    for i, turn in enumerate(turns):
        if not turn.is_justice:
            continue
        if target_justice and target_justice.upper() not in turn.speaker.upper():
            continue

        preceding = turns[i - 1] if i > 0 else None
        prec_speaker = _resolve_speaker_slug(preceding.speaker, advocates) if preceding else ""
        prec_context = preceding.text if preceding else ""

        side, conf, addr_last_name = resolve_side(i, turns, segments, advocates)

        export_side = side if side != SIDE_UNKNOWN else SIDE_AMICUS
        addr_turn = find_addressed_turn(i, turns, addr_last_name) if addr_last_name else None
        addr_speaker = _resolve_speaker_slug(addr_turn.speaker, advocates) if addr_turn else ""
        addr_context = addr_turn.text if addr_turn else ""

        row = {
            "case_id": case_id,
            "justice_id": _slugify_justice(turn.speaker),
            "turn_position": i + 1,
            "justice_utterance": turn.text,
            "preceding_speaker": prec_speaker,
            "preceding_context": prec_context,
            "addressed_speaker": addr_speaker,
            "addressed_context": addr_context,
            "side_addressed": export_side,
            "label": pd.NA,
        }
        if include_confidence:
            row["side_confidence"] = conf

        rows.append(row)

    cols = [
        "case_id", "justice_id", "turn_position",
        "justice_utterance",
        "preceding_speaker", "preceding_context",
        "addressed_speaker", "addressed_context",
        "side_addressed", "label",
    ]
    if include_confidence:
        cols.append("side_confidence")

    return pd.DataFrame(rows, columns=cols)



def parse_transcript_text(text: str, case_id: str, target_justice: str = "", include_confidence: bool = True, verbose: bool = True,) -> pd.DataFrame:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    text = _truncate_back_matter(text)

    log("[1/3] Parsing appearances block ...")
    advocates = parse_appearances_block(text)
    if advocates:
        unique = {id(v): v for v in advocates.values()}.values()
        for adv in unique:
            side_label = SIDE_LABELS.get(adv.side, "unknown")
            log(f"       {adv.full_name:<35s} → {side_label}  ({adv.role_text[:60]})")
    else:
        log("WARNING: appearances block not found — side detection will degrade")

    log("[2/3] Building segment index ...")
    segments = build_segment_index(text, advocates)
    if segments:
        for seg in segments:
            side_label = SIDE_LABELS.get(seg.side, "unknown")
            rebuttal = " [REBUTTAL]" if seg.is_rebuttal else ""
            log(f"       {seg.attorney_name:<30s} → {side_label}{rebuttal}  ({seg.confidence})")
    else:
        log("WARNING: no segment headers found — falling back to appearances only")

    turns = split_into_turns(text)
    log(f"[3/3] {len(turns)} total turns → building DataFrame ...")

    df = build_dataframe(turns, case_id, segments, advocates, target_justice, include_confidence)
    log(f"       {len(df)} justice turns extracted.")

    if verbose and include_confidence:
        counts = df["side_confidence"].value_counts().to_dict()
        log(f"       Confidence breakdown: {counts}")
    if verbose:
        conf_col = df.get("side_confidence")
        if conf_col is not None:
            unknown = (conf_col == CONF_NONE).sum()
            if unknown:
                log(f"WARNING: {unknown} turns could not be assigned a side (exported as side=2).")

    return df


def parse_transcript_pdf(pdf_path: str, case_id: str, target_justice: str = "", include_confidence: bool = True, verbose: bool = True,) -> pd.DataFrame:
    def log(msg: str) -> None:
        if verbose:
            print(msg)
    log(f"Extracting text from {pdf_path} ...")
    text = extract_text_from_pdf(pdf_path)
    return parse_transcript_text(text, case_id, target_justice, include_confidence, verbose)


if __name__ == "__main__":
    PDF_PATH = "/mnt/user-data/uploads/10-5400.pdf"
    CASE_ID = "2011_10-5400"
    TARGET = "" 

    df = parse_transcript_pdf(PDF_PATH,CASE_ID,target_justice=TARGET,include_confidence=True,verbose=True)

    print("\n── Sample output ──────────────────────────────────────────────────")
    side_label = {SIDE_PETITIONER: "PET", SIDE_RESPONDENT: "RES", SIDE_AMICUS: "AMI", SIDE_UNKNOWN: "UNK"}
    df["side_label"] = df["side_addressed"].map(side_label)
    print(
        df[["justice_id", "turn_position", "side_label", "side_confidence",
            "addressed_speaker", "justice_utterance"]]
        .assign(justice_utterance=df["justice_utterance"].str[:65])
        .to_string(index=False)
    )

    out_dir = Path("/mnt/user-data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{CASE_ID}_parsed.csv"
    df.drop(columns=["side_label"]).to_csv(out, index=False)
    print(f"\nSaved → {out}")