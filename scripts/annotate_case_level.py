import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import openai
import os
import google.generativeai as genai
from groq import Groq
import requests

INPUT_FILE = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\DS301Project\data\full_df\test_df.csv"
OUTPUT_FILE = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\DS301Project\data\full_df\test_df_case_level_annotated.csv"

GPT_MODEL = "gpt-4o"

GPT_DELAY = 0.20

MAX_RETRIES = 3
RETRY_DELAY = 5

N_CASES = 200
RANDOM_SEED = 42
MAX_BLOCK_CHARS = 6000

SYSTEM_PROMPT = """
You are an expert legal analyst specializing in Supreme Court oral argument dynamics.

You will be given a series of exchanges between a Supreme Court justice and advocates 
arguing for ONE SIDE of a case (either petitioner or respondent). Each exchange shows 
what the advocate said followed by what the justice said.

Your task is to rate the justice's OVERALL rhetorical stance toward that side's position 
across the entire set of exchanges, on a scale of 1-7.

HOSTILITY SCALE:
1 = Justice was consistently sympathetic — rarely challenged the position, often reinforced it
3 = Justice was balanced — probed both strengths and weaknesses without clear lean
5 = Justice was predominantly skeptical — frequently challenged or questioned the position
7 = Justice was consistently hostile — repeatedly challenged, contradicted, or dismissed the position

Use the full scale. Scores of 2, 4, and 6 represent intermediate positions between the anchors.
Consider the arc of the entire argument, not just individual moments.
A justice may ask tough questions to a side they ultimately support — focus on the overall pattern.

Respond ONLY with valid JSON in this exact format, no other text:
{"hostility": <integer 1-7>}
""".strip()

FEW_SHOT_EXAMPLES = [
    # Score 1 — consistently sympathetic
    {
        "role": "user",
        "content": """Exchange 1:
ADVOCATE: The statute clearly provides that railroads cannot be taxed at a higher rate than commercial and industrial taxpayers.
JUSTICE: And the SEC has recognized that they remain two separate entities, despite the interconnected relationship. That supports your reading, doesn't it?

Exchange 2:
ADVOCATE: Yes, and Congress specifically intended to protect railroads from discriminatory state taxation.
JUSTICE: So your position is that the legislative history unambiguously supports this interpretation. That seems consistent with what this Court said in CSX 1."""
    },
    {
        "role": "assistant",
        "content": '{"hostility": 1}'
    },
    # Score 3 — balanced
    {
        "role": "user",
        "content": """Exchange 1:
ADVOCATE: The Fourth Amendment does not require a warrant for this type of search because the defendant had no reasonable expectation of privacy.
JUSTICE: What's your best argument for why the third-party doctrine applies here given the digital nature of the records?

Exchange 2:
ADVOCATE: The information was voluntarily shared with a third party, so Smith v. Maryland controls.
JUSTICE: But doesn't Carpenter complicate that analysis? How do you distinguish it?

Exchange 3:
ADVOCATE: Carpenter was limited to cell-site location information and doesn't extend to this context.
JUSTICE: That's a reasonable reading. Though I wonder if the sheer volume of data here changes the analysis."""
    },
    {
        "role": "assistant",
        "content": '{"hostility": 3}'
    },
    # Score 5 — predominantly skeptical
    {
        "role": "user",
        "content": """Exchange 1:
ADVOCATE: The statute's plain text supports our interpretation that the agency has broad discretion here.
JUSTICE: Doesn't that seem like a fine line? If the agency has such broad discretion, what limits that power at all?

Exchange 2:
ADVOCATE: The limits come from the statutory context and legislative history.
JUSTICE: But you haven't pointed to specific statutory language. How do you respond to the argument that you're reading in authority that Congress never granted?

Exchange 3:
ADVOCATE: We believe the overall statutory scheme supports our reading.
JUSTICE: You keep saying the scheme supports it but I'm not seeing how the specific text gets you there."""
    },
    {
        "role": "assistant",
        "content": '{"hostility": 5}'
    },
    # Score 7 — consistently hostile
    {
        "role": "user",
        "content": """Exchange 1:
ADVOCATE: Putting Massachusetts v. EPA to one side, the statute's plain text supports our position.
JUSTICE: Counsel, you said putting Massachusetts v. EPA to one side. We simply can't do that. That case is binding precedent.

Exchange 2:
ADVOCATE: Even under Massachusetts v. EPA, our reading is consistent with—
JUSTICE: That's simply not what the statute says. You're asking us to rewrite Congress's words entirely.

Exchange 3:
ADVOCATE: We believe the Court has flexibility to interpret ambiguous terms—
JUSTICE: There's no ambiguity here. You're manufacturing ambiguity where none exists to reach your preferred result."""
    },
    {
        "role": "assistant",
        "content": '{"hostility": 7}'
    },
]

def build_exchange_block(group: pd.DataFrame, side: int, max_chars: int) -> str:
    rows = group[group["side_addressed"] == side].sort_values("turn_position")
    if len(rows) == 0:
        return ""

    parts = []
    total_chars = 0
    for i, (_, row) in enumerate(rows.iterrows(), 1):
        ctx = str(row["preceding_context"]).strip()
        utt = str(row["justice_utterance"]).strip()
        exchange = f"Exchange {i}:\nADVOCATE: {ctx}\nJUSTICE: {utt}"
        total_chars += len(exchange)
        if total_chars > max_chars:
            parts.append(f"Exchange {i}:\n[truncated — argument continues]")
            break
        parts.append(exchange)

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


def annotate_block(client, exchange_block: str) -> int | None:
    messages = FEW_SHOT_EXAMPLES + [{"role": "user", "content": exchange_block}]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=20,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            )
            text   = response.choices[0].message.content
            result = parse_hostility(text)
            if result is not None:
                return result
            print(f"  [WARN] Invalid response: {text}")
        except openai.RateLimitError:
            wait = RETRY_DELAY * (attempt + 1)
            print(f"  [RATE LIMIT] Waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"  [ERROR] Attempt {attempt + 1}: {e}")
            time.sleep(RETRY_DELAY)

    return None

def main():
    import os
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = df[df["label"] != -1].copy()
    df["label"] = df["label"].astype(int)
    print(f"Total utterances: {len(df)} across {df['case_id'].nunique()} cases")

    df_pet = df[df["label"] == 1]
    df_res = df[df["label"] == 0]  

    pet_cases = df_pet["case_id"].unique()
    res_cases = df_res["case_id"].unique()

    n_pet = min(100, len(pet_cases))
    n_res = min(100, len(res_cases))

    sampled_pet = pd.Series(pet_cases).sample(n_pet, random_state=RANDOM_SEED)
    sampled_res = pd.Series(res_cases).sample(n_res, random_state=RANDOM_SEED)
    sampled_all = pd.concat([sampled_pet, sampled_res])

    df = df[df["case_id"].isin(sampled_all)].copy()
    print(f"Sampled {n_pet} petitioner wins + {n_res} respondent wins ({len(df)} utterances)")

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results = []

    for case_id, group in tqdm(df.groupby("case_id"), total=df["case_id"].nunique()):
        label = group["label"].iloc[0]
        n_pet = len(group[group["side_addressed"] == 0])
        n_res = len(group[group["side_addressed"] == 1])
        exchange_ratio = n_pet / n_res if n_res > 0 else np.nan
        pet_block = build_exchange_block(group, side=0, max_chars=MAX_BLOCK_CHARS)
        res_block = build_exchange_block(group, side=1, max_chars=MAX_BLOCK_CHARS)

        pet_hostility = None
        if pet_block:
            pet_hostility = annotate_block(client, pet_block)
            time.sleep(GPT_DELAY)
        res_hostility = None
        if res_block:
            res_hostility = annotate_block(client, res_block)
            time.sleep(GPT_DELAY)
        if pet_hostility is not None and res_hostility is not None:
            diff = pet_hostility - res_hostility
            diff_cubed = diff ** 3
        else:
            diff = np.nan
            diff_cubed = np.nan

        results.append({
            "case_id": case_id,
            "label": label,
            "n_pet_exchanges": n_pet,
            "n_res_exchanges": n_res,
            "exchange_ratio": exchange_ratio,
            "gpt_pet_hostility": pet_hostility,
            "gpt_res_hostility": res_hostility,
            "hostility_diff": diff,
            "hostility_diff_cubed": diff_cubed,
        })

    result_df = pd.DataFrame(results)
    valid = result_df.dropna(subset=["gpt_pet_hostility", "gpt_res_hostility"])

    print(f"\n── Annotation summary ──")
    print(f"Total cases annotated: {len(result_df)}")
    print(f"Valid annotations: {len(valid)}")
    print(f"Null rate: {1 - len(valid)/len(result_df):.1%}")

    print(f"\nPet hostility distribution:\n{valid['gpt_pet_hostility'].value_counts().sort_index()}")
    print(f"\nRes hostility distribution:\n{valid['gpt_res_hostility'].value_counts().sort_index()}")
    print(f"\nMean pet hostility: {valid['gpt_pet_hostility'].mean():.3f}")
    print(f"Mean res hostility: {valid['gpt_res_hostility'].mean():.3f}")
    print(f"Mean hostility diff: {valid['hostility_diff'].mean():.3f}")

    corr = valid["hostility_diff"].corr(valid["label"])
    print(f"\nCorrelation hostility_diff vs label: {corr:.3f}")

    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
