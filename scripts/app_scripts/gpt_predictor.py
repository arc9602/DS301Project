import json
import time
import openai
import pandas as pd

GPT_MODEL = "gpt-4o"
MAX_BLOCK_CHARS = 12000 
MAX_RETRIES = 3
RETRY_DELAY = 5
CALL_DELAY = 0.20

SIDE_PETITIONER = 1
SIDE_RESPONDENT = 0

JUSTICE_SHORTNAMES = {
    "j__john_roberts": "Roberts",
    "j__clarence_thomas": "Thomas",
    "j__samuel_alito": "Alito",
    "j__sonia_sotomayor": "Sotomayor",
    "j__elena_kagan": "Kagan",
    "j__neil_gorsuch": "Gorsuch",
    "j__brett_kavanaugh": "Kavanaugh",
    "j__amy_coney_barrett": "Barrett",
    "j__ketanji_brown_jackson": "Jackson",
}

SYSTEM_PROMPT = """
You are an expert legal analyst specializing in Supreme Court oral argument dynamics.

You will be given a series of exchanges between ALL nine justices and an advocate
arguing for ONE SIDE of a case (either petitioner or respondent). Each exchange
labels which justice is speaking.

Your task is to rate the OVERALL rhetorical stance of the ENTIRE BENCH toward
that side's position across all exchanges, on a scale of 1-7.

HOSTILITY SCALE:
1 = Bench was consistently sympathetic — rarely challenged the position, often reinforced it
3 = Bench was balanced — probed both strengths and weaknesses without clear lean
5 = Bench was predominantly skeptical — frequently challenged or questioned the position
7 = Bench was consistently hostile — repeatedly challenged, contradicted, or dismissed the position

Use the full scale. Scores of 2, 4, and 6 represent intermediate positions between the anchors.
Consider the arc of the entire argument across all justices, not just individual moments.
Justices may ask tough questions to a side they ultimately support — focus on the overall pattern.

Respond ONLY with valid JSON in this exact format, no other text:
{"hostility": <integer 1-7>}
""".strip()

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """Exchange 1 [Roberts]:
ADVOCATE: The statute clearly provides that railroads cannot be taxed at a higher rate than commercial and industrial taxpayers.
JUSTICE: And the SEC has recognized that they remain two separate entities, despite the interconnected relationship. That supports your reading, doesn't it?

Exchange 2 [Kagan]:
ADVOCATE: Yes, and Congress specifically intended to protect railroads from discriminatory state taxation.
JUSTICE: So your position is that the legislative history unambiguously supports this interpretation. That seems consistent with what this Court said in CSX 1."""
    },
    {"role": "assistant", "content": '{"hostility": 1}'},
    {
        "role": "user",
        "content": """Exchange 1 [Thomas]:
ADVOCATE: The Fourth Amendment does not require a warrant for this type of search because the defendant had no reasonable expectation of privacy.
JUSTICE: What's your best argument for why the third-party doctrine applies here given the digital nature of the records?

Exchange 2 [Kagan]:
ADVOCATE: The information was voluntarily shared with a third party, so Smith v. Maryland controls.
JUSTICE: But doesn't Carpenter complicate that analysis? How do you distinguish it?

Exchange 3 [Gorsuch]:
ADVOCATE: Carpenter was limited to cell-site location information and doesn't extend to this context.
JUSTICE: That's a reasonable reading. Though I wonder if the sheer volume of data here changes the analysis."""
    },
    {"role": "assistant", "content": '{"hostility": 3}'},
    {
        "role": "user",
        "content": """Exchange 1 [Alito]:
ADVOCATE: The statute's plain text supports our interpretation that the agency has broad discretion here.
JUSTICE: Doesn't that seem like a fine line? If the agency has such broad discretion, what limits that power at all?

Exchange 2 [Roberts]:
ADVOCATE: The limits come from the statutory context and legislative history.
JUSTICE: But you haven't pointed to specific statutory language. How do you respond to the argument that you're reading in authority that Congress never granted?

Exchange 3 [Kavanaugh]:
ADVOCATE: We believe the overall statutory scheme supports our reading.
JUSTICE: You keep saying the scheme supports it but I'm not seeing how the specific text gets you there."""
    },
    {"role": "assistant", "content": '{"hostility": 5}'},
    {
        "role": "user",
        "content": """Exchange 1 [Gorsuch]:
ADVOCATE: Putting Massachusetts v. EPA to one side, the statute's plain text supports our position.
JUSTICE: Counsel, you said putting Massachusetts v. EPA to one side. We simply can't do that. That case is binding precedent.

Exchange 2 [Alito]:
ADVOCATE: Even under Massachusetts v. EPA, our reading is consistent with—
JUSTICE: That's simply not what the statute says. You're asking us to rewrite Congress's words entirely.

Exchange 3 [Barrett]:
ADVOCATE: We believe the Court has flexibility to interpret ambiguous terms—
JUSTICE: There's no ambiguity here. You're manufacturing ambiguity where none exists to reach your preferred result."""
    },
    {"role": "assistant", "content": '{"hostility": 7}'},
]


def build_court_exchange_block(df: pd.DataFrame, side: int) -> str:
    rows = df[df["side_addressed"] == side].sort_values("turn_position")
    if len(rows) == 0:
        return ""

    parts = []
    total_chars = 0
    exchange_num = 1

    for _, row in rows.iterrows():
        jid = str(row.get("justice_id", ""))
        justice_label = JUSTICE_SHORTNAMES.get(jid, jid.replace("j__", "").replace("_", " ").title())
        ctx = str(row["preceding_context"]).strip()
        utt = str(row["justice_utterance"]).strip()

        exchange = f"Exchange {exchange_num} [{justice_label}]:\nADVOCATE: {ctx}\nJUSTICE: {utt}"
        total_chars += len(exchange)

        if total_chars > MAX_BLOCK_CHARS:
            parts.append(f"Exchange {exchange_num} [{justice_label}]:\n[truncated — argument continues]")
            break

        parts.append(exchange)
        exchange_num += 1

    return "\n\n".join(parts)


def parse_hostility(text: str) -> int | None:
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        hostility = result.get("hostility")
        if isinstance(hostility, int) and 1 <= hostility <= 7:
            return hostility
    except Exception:
        pass
    return None


def annotate_block(client, block: str) -> int | None:
    messages = FEW_SHOT_EXAMPLES + [{"role": "user", "content": block}]
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=20,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            )
            result = parse_hostility(resp.choices[0].message.content)
            if result is not None:
                return result
        except openai.RateLimitError:
            time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            print(f"  [GPT ERROR] {e}")
            time.sleep(RETRY_DELAY)
    return None


def gpt_score_court(client, df: pd.DataFrame) -> dict | None:

    pet_block = build_court_exchange_block(df, side=SIDE_PETITIONER)
    res_block = build_court_exchange_block(df, side=SIDE_RESPONDENT)

    pet_score = annotate_block(client, pet_block) if pet_block else None
    time.sleep(CALL_DELAY)
    res_score = annotate_block(client, res_block) if res_block else None
    time.sleep(CALL_DELAY)

    if pet_score is None or res_score is None:
        return None

    # Positive diff = respondent favored
    # Negative diff = petitioner favored
    diff = pet_score - res_score
    prediction = 1 if diff < 0 else 0

    return {
        "pet_hostility": pet_score,
        "res_hostility": res_score,
        "diff": diff,
        "prediction": prediction,
    }


def gpt_score_justice(client, df: pd.DataFrame, justice_id: str) -> dict | None:
    group = df[df["justice_id"] == justice_id]

    pet_block = build_court_exchange_block(group, side=SIDE_PETITIONER)
    res_block = build_court_exchange_block(group, side=SIDE_RESPONDENT)

    pet_score = annotate_block(client, pet_block) if pet_block else None
    time.sleep(CALL_DELAY)
    res_score = annotate_block(client, res_block) if res_block else None
    time.sleep(CALL_DELAY)

    if pet_score is None or res_score is None:
        return None

    diff = pet_score - res_score
    prediction = 1 if diff < 0 else 0

    return {
        "justice_id": justice_id,
        "pet_hostility": pet_score,
        "res_hostility": res_score,
        "diff": diff,
        "prediction": prediction,
    }