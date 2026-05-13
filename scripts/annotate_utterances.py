import json
import os
import time
import pandas as pd
from tqdm import tqdm
import anthropic
import openai

INPUT_FILE  = r"C:\Users\adith\onedrive\Desktop\Assignments\DS301\DS301Project\data\full_df\test_df.csv"
OUTPUT_FILE = r"C:\Users\adith\onedrive\Desktop\Assignments\DS301\DS301Project\data\full_df\test_df_annotated.csv"

CLAUDE_MODEL = "claude-sonnet-4-20250514"
GPT_MODEL    = "gpt-4o"

ANTHROPIC_API_KEY = None   
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")

CLAUDE_DELAY = 0.5   
GPT_DELAY    = 0.2   

MAX_RETRIES  = 3
RETRY_DELAY  = 5     

DRY_RUN      = False
DRY_RUN_N    = 50    

SYSTEM_PROMPT = """
You are an expert legal analyst specializing in Supreme Court oral argument dynamics.
Your task is to analyze exchanges between Supreme Court justices and advocates, rating
the justice's rhetorical stance toward the advocate's position.

You will be given:
1. CONTEXT: What the advocate said immediately before the justice spoke
2. UTTERANCE: What the justice said

You must output TWO things:
1. A HOSTILITY score from 1-7 rating the justice's stance toward the advocate's position
2. A QUESTION_TYPE categorizing the nature of the justice's utterance

HOSTILITY SCALE:
1 = Strongly sympathetic — justice is actively supporting or reinforcing advocate's position
2 = Sympathetic — justice is restating or clarifying in a way that helps the advocate
3 = Mildly sympathetic — justice is probing gently, giving advocate room to strengthen position
4 = Neutral — justice is genuinely seeking information or playing devil's advocate, direction unclear
5 = Mildly skeptical — justice is questioning the advocate's position but leaving room for response
6 = Skeptical — justice is expressing clear doubt or frustration with advocate's position
7 = Strongly hostile — justice is directly blocking, contradicting, or dismissing advocate's argument

Use the full 1-7 scale. Scores of 1, 2, 6, and 7 should be used when clearly warranted — 
do not default to 4 or 5 when the justice's stance is clearly sympathetic or hostile.
Score 1 example: Justice says "And the precedent supports your position exactly" — use 1
Score 7 example: Justice says "That's simply not what the statute says" — use 7

QUESTION TYPES:
CLARIFYING — seeking information or restating the advocate's position, no clear adversarial intent
HYPOTHETICAL — testing the limits or implications of the advocate's position by changing facts
DEVILS_ADVOCATE — presenting the opposing side's argument to the advocate being questioned
CHALLENGING — directly pushing back on or contradicting the advocate's argument
SUPPORTIVE — co-arguing with the advocate, reinforcing their position or addressing counterarguments on their behalf

IMPORTANT NOTES:
- Focus on the justice's rhetorical stance, not the underlying legal merits
- A question can be phrased as hostile even if the justice ultimately votes for that side
- Devil's advocate questions are genuinely ambiguous — score them 3-5 depending on framing
- Short utterances like "I see" or procedural comments should be scored 4 (neutral)
- Consider the full context of what the advocate said when assessing the justice's response

Respond ONLY with valid JSON in this exact format, no other text:
{"hostility": <integer 1-7>, "question_type": "<type>"}
""".strip()

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "CONTEXT: Absolutely. They created the fund. That's how mutual funds work. Managers create them, they lure investors to them, they get money by having a percentage of assets under management.\n\nUTTERANCE: And the SEC has recognized that they remain two separate entities, despite the interconnected relationship."
    },
    {
        "role": "assistant",
        "content": '{"hostility": 1, "question_type": "SUPPORTIVE"}'
    },
    {
        "role": "user",
        "content": "CONTEXT: Yes, Your Honor. Under 12 C.F.R. 1237.12(a) and (b), not a penny can be paid to the Treasury without the approval of the director, and since 2014, there has been a Senate-confirmed director with for-cause removal protection.\n\nUTTERANCE: Well, so your theory is that even if an acting director approved the instrument under which payments are going to be made, that when those payments are made, if there is an unconstitutional director, that they are invalid?"
    },
    {
        "role": "assistant",
        "content": '{"hostility": 2, "question_type": "CLARIFYING"}'
    },
    {
        "role": "user",
        "content": "CONTEXT: It happens quite frequently that persons are arrested, brought to the station house, and then released by the police without undergoing an initial appearance. And in that circumstance, we don't contend that a prosecution would have begun.\n\nUTTERANCE: Why not, if they initiate charges against them? You're saying, in Justice Breyer's hypothetical, you're charged with trespassing, but we are not going to hold you, so come back if we decide to prosecute."
    },
    {
        "role": "assistant",
        "content": '{"hostility": 3, "question_type": "HYPOTHETICAL"}'
    },
    {
        "role": "user",
        "content": "CONTEXT: This Court has recognized a constitutional exemption for two disclosure requirements in cases where disclosure would have a reasonable likelihood of leading to reprisal.\n\nUTTERANCE: How do we apply that test? Is it inconceivable to you here that people contributing to such a clearly anti-Clinton advertisement are not going to be subject to reprisals?"
    },
    {
        "role": "assistant",
        "content": '{"hostility": 4, "question_type": "DEVILS_ADVOCATE"}'
    },
    {
        "role": "user",
        "content": "CONTEXT: We submit under the Russello presumption or even the statute itself, Congress enumerated those four property crimes.\n\nUTTERANCE: Doesn't that seem like a fine line? I mean, if you're sitting around with your coconspirator planning a burglary you can be covered under this provision, but if you actually get out there with the burglary tools and start up the ladder, that somehow isn't covered?"
    },
    {
        "role": "assistant",
        "content": '{"hostility": 5, "question_type": "CHALLENGING"}'
    },
    {
        "role": "user",
        "content": "CONTEXT: There has been some back and forth in the Court since then as to whether the Irwin language extends more broadly to whether the statute is jurisdictional.\n\nUTTERANCE: You know, I don't -- we've found it difficult enough to figure out which statutes are jurisdictional and which are not. And now you want us to say, well, even if it's jurisdictional, the consequences may be different for jurisdiction and for equitable tolling and for waivibility. I mean, it seems to me you're just compounding the difficulty."
    },
    {
        "role": "assistant",
        "content": '{"hostility": 6, "question_type": "CHALLENGING"}'
    },
    {
        "role": "user",
        "content": "CONTEXT: Putting Massachusetts v. EPA to one side, if anybody were looking at the PSD statute in isolation, without the benefit of that case, assume that the word pollutant was an undefined term.\n\nUTTERANCE: Counsel, you began that discussion by saying putting Massachusetts v. EPA to one side. But I was in the dissent in that case, but we still can't do that."
    },
    {
        "role": "assistant",
        "content": '{"hostility": 7, "question_type": "CHALLENGING"}'
    },
]

VALID_QUESTION_TYPES = {
    "CLARIFYING", "HYPOTHETICAL", "DEVILS_ADVOCATE", "CHALLENGING", "SUPPORTIVE"
}

def get_clients():
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return None, openai_client

def build_user_message(preceding_context: str, justice_utterance: str) -> str:
    ctx = str(preceding_context).strip() if preceding_context else "[NO CONTEXT]"
    utt = str(justice_utterance).strip() if justice_utterance else "[NO UTTERANCE]"
    return f"CONTEXT: {ctx}\n\nUTTERANCE: {utt}"


def parse_response(text: str) -> dict:
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        hostility = result.get("hostility")
        qtype     = result.get("question_type", "").upper()
        if qtype == "NEUTRAL":
            qtype = "CLARIFYING"

        if not isinstance(hostility, int) or not (1 <= hostility <= 7):
            return None
        if qtype not in VALID_QUESTION_TYPES:
            return None

        return {"hostility": hostility, "question_type": qtype}
    except Exception:
        return None


def annotate_claude(client, context: str, utterance: str) -> dict:
    user_msg = build_user_message(context, utterance)
    messages = FEW_SHOT_EXAMPLES + [{"role": "user", "content": user_msg}]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=50,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            text = response.content[0].text
            result = parse_response(text)
            if result:
                return result
            print(f"  [WARN] Claude returned invalid response: {text}")
        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (attempt + 1)
            print(f"  [RATE LIMIT] Claude — waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"  [ERROR] Claude attempt {attempt + 1}: {e}")
            time.sleep(RETRY_DELAY)

    return {"hostility": None, "question_type": None}


def annotate_gpt(client, context: str, utterance: str) -> dict:
    user_msg = build_user_message(context, utterance)
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEW_SHOT_EXAMPLES
        + [{"role": "user", "content": user_msg}]
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=50,
                messages=messages,
            )
            text = response.choices[0].message.content
            result = parse_response(text)
            if result:
                return result
            print(f"  [WARN] GPT returned invalid response: {text}")
        except openai.RateLimitError:
            wait = RETRY_DELAY * (attempt + 1)
            print(f"  [RATE LIMIT] GPT — waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"  [ERROR] GPT attempt {attempt + 1}: {e}")
            time.sleep(RETRY_DELAY)

    return {"hostility": None, "question_type": None}

def main():
    import os
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total rows: {len(df)}")

    sampled_cases = df["case_id"].drop_duplicates().sample(200, random_state=42)
    df = df[df["case_id"].isin(sampled_cases)].copy()
    print(f"Sampled to {len(df)} utterances across 200 cases")

    if DRY_RUN:
        print(f"\nDRY RUN MODE — annotating first {DRY_RUN_N} rows only")
        df = df.head(DRY_RUN_N).copy()

    df["gpt_hostility"] = None
    df["gpt_question_type"] = None

    _, openai_client = get_clients()

    print(f"\nAnnotating {len(df)} exchanges...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        context   = row["preceding_context"]
        utterance = row["justice_utterance"]
        gpt_result = annotate_gpt(openai_client, context, utterance)
        df.at[idx, "gpt_hostility"] = gpt_result["hostility"]
        df.at[idx, "gpt_question_type"] = gpt_result["question_type"]
        time.sleep(GPT_DELAY)

    print("\n── Annotation summary ──")
    print(f"GPT null rate:    {df['gpt_hostility'].isna().mean():.1%}")

    both_valid = df.dropna(subset=["claude_hostility", "gpt_hostility"])
    if len(both_valid) > 0:
        correlation = both_valid["claude_hostility"].corr(both_valid["gpt_hostility"])
        exact_match = (both_valid["claude_hostility"] == both_valid["gpt_hostility"]).mean()
        within_one = (
            (both_valid["claude_hostility"] - both_valid["gpt_hostility"]).abs() <= 1
        ).mean()

        print(f"\nInter-model agreement (n={len(both_valid)}):")
        print(f"Pearson correlation:  {correlation:.3f}")
        print(f"Exact match rate:     {exact_match:.1%}")
        print(f"Within-1 match rate:  {within_one:.1%}")

        print(f"\nGPT hostility distribution:\n{both_valid['gpt_hostility'].value_counts().sort_index()}")
        print(f"\nGPT question type distribution:\n{df['gpt_question_type'].value_counts()}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()