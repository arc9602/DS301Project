import os
import tempfile
import streamlit as st
import openai
import pandas as pd
from pathlib import Path

import pdfplumber
from transcript_parser import parse_transcript_pdf, extract_text_from_pdf
from bert_predictor import load_bert_model, bert_predict_court
from gpt_predictor import gpt_score_court
from rag_explainer import build_rag_vectorstore, generate_rag_explanation

DEFAULT_MODEL_PATH = model_path = str(Path(__file__).parents[2] / "models" / "global")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

JUSTICE_SHORTNAMES = {
    "j__john_roberts": "Chief Justice Roberts",
    "j__clarence_thomas": "Justice Thomas",
    "j__samuel_alito": "Justice Alito",
    "j__sonia_sotomayor": "Justice Sotomayor",
    "j__elena_kagan": "Justice Kagan",
    "j__neil_gorsuch": "Justice Gorsuch",
    "j__brett_kavanaugh": "Justice Kavanaugh",
    "j__amy_coney_barrett": "Justice Barrett",
    "j__ketanji_brown_jackson": "Justice Jackson",
}


def parse_transcript(pdf_path: str) -> tuple[dict, "pd.DataFrame", list[dict]]:
    import re

    pages_text: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            pages_text.append({"page": i, "text": raw})

    import os
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    case_id = stem
    df = parse_transcript_pdf(pdf_path, case_id=case_id, verbose=False)

    full_text = "\n".join(p["text"] for p in pages_text)

    case_name = ""
    for line in full_text.splitlines():
        stripped = line.strip()
        if re.match(r"^[A-Z][A-Z ,\.\-\']+(?:\s+V\.?\s+|\s+VS\.?\s+)[A-Z][A-Z ,\.\-\']+$", stripped):
            case_name = stripped
            break

    docket = ""
    m = re.search(r"No\.?\s*([\d\-]+)", full_text)
    if m:
        docket = m.group(1)

    from transcript_parser import SIDE_PETITIONER, SIDE_RESPONDENT
    pet_advocate = ""
    res_advocate = ""

    if "addressed_speaker" in df.columns and "side_addressed" in df.columns:
        pet_rows = df[df["side_addressed"] == SIDE_PETITIONER]["addressed_speaker"]
        res_rows = df[df["side_addressed"] == SIDE_RESPONDENT]["addressed_speaker"]

        def best_advocate(series):
            counts = series[series.str.len() > 0].value_counts()
            if len(counts):
                slug = counts.index[0]
                return " ".join(w.upper() if len(w) <= 2 else w.capitalize()
                                for w in slug.replace("_", " ").split())
            return "Unknown"

        pet_advocate = best_advocate(pet_rows)
        res_advocate = best_advocate(res_rows)

    metadata = {
        "case_name": case_name or case_id,
        "docket": docket or case_id,
        "pet_advocate": pet_advocate or "Unknown",
        "res_advocate": res_advocate or "Unknown",
    }

    return metadata, df, pages_text

def main():
    st.set_page_config(page_title="SCOTUS Vote Predictor", layout="wide")

    st.title("SCOTUS Oral Argument Vote Predictor")
    st.markdown(
        "Upload an official Supreme Court oral argument transcript (PDF) "
        "to predict how each justice will vote."
    )

    api_key = st.sidebar.text_input("OpenAI API Key", value=OPENAI_KEY, type="password")
    model_path = st.sidebar.text_input("BERT Model Path", value=DEFAULT_MODEL_PATH)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline:**")
    st.sidebar.markdown(
        "1. Parse transcript -> per-justice exchanges\n"
        "2. BERT dual-embedding -> vote prediction\n"
        "3. GPT hostility scoring -> vote prediction\n"
        "4. Aggregate -> full court prediction\n"
        "5. RAG explanation from transcript"
    )

    uploaded_file = st.file_uploader("Upload SCOTUS Transcript PDF", type=["pdf"], help="Official transcripts from the Supreme Court website or Heritage Reporting")

    if uploaded_file is None:
        st.info("Upload a PDF transcript to get started.")
        return

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    with st.spinner("Parsing transcript..."):
        try:
            metadata, df, pages_text = parse_transcript(tmp_path)
        except Exception as e:
            st.error(f"Failed to parse transcript: {e}")
            os.unlink(tmp_path)
            return

    st.subheader(f"{metadata.get('case_name', 'Unknown Case')} — No. {metadata.get('docket', '?')}")
    col1, col2 = st.columns(2)
    col1.metric("Petitioner's Counsel", metadata.get("pet_advocate", "Unknown"))
    col2.metric("Respondent's Counsel", metadata.get("res_advocate", "Unknown"))

    justices_found = [jid for jid in df["justice_id"].dropna().unique() if jid]
    st.success(f"Parsed {len(df)} utterances across {len(justices_found)} justices")

    with st.expander("View parsed utterances"):
        st.dataframe(
            df[["justice_id", "turn_position", "preceding_speaker", "side_addressed", "justice_utterance"]].head(1000)
        )

    if not st.button("Run Prediction", type="primary"):
        return

    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return

    client = openai.OpenAI(api_key=api_key)

    with st.spinner("Loading BERT model..."):
        try:
            tokenizer, bert_model = load_bert_model(model_path)
            st.success("BERT model loaded")
        except Exception as e:
            st.error(f"Failed to load BERT model: {e}")
            os.unlink(tmp_path)
            return

    with st.spinner("Running BERT court prediction..."):
        bert_court = bert_predict_court(tokenizer, bert_model, df)

    if bert_court is None:
        st.error("BERT could not produce a prediction - no valid exchanges found (check side_addressed in the debug view above).")
        os.unlink(tmp_path)
        return

    final_prediction = bert_court["prediction"]
    winner = "PETITIONER" if final_prediction == 1 else "RESPONDENT"

    st.subheader("Court Prediction (BERT)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Winner", winner)
    col2.metric("Confidence", f"{bert_court['confidence']:.0%}")
    col3.metric("P(Petitioner)", f"{bert_court['prob_pet']:.0%}")

    if final_prediction == 1:
        st.success(f"BERT predicts: PETITIONER wins ({bert_court['confidence']:.0%} confidence)")
    else:
        st.error(f"BERT predicts: RESPONDENT wins ({bert_court['confidence']:.0%} confidence)")

    st.subheader("Bench Sentiment (GPT)")

    with st.spinner("GPT scoring full bench toward petitioner and respondent..."):
        gpt_court = gpt_score_court(client, df)

    if gpt_court:
        gpt_pred_label = "PETITIONER" if gpt_court["prediction"] == 1 else "RESPONDENT"
        col1, col2, col3 = st.columns(3)
        col1.metric("GPT Predicted Winner", gpt_pred_label)
        col2.metric("Bench Hostility -> Petitioner", gpt_court["pet_hostility"])
        col3.metric("Bench Hostility -> Respondent", gpt_court["res_hostility"])
        st.caption(
            "Hostility scale 1-7: 1 = sympathetic, 4 = neutral, 7 = hostile. "
            "Lower hostility toward a side signals the bench leans that way."
        )
    else:
        st.warning("GPT bench scoring failed - check API key or transcript content.")
        gpt_court = {}

    st.subheader("Analysis & Explanation")

    with st.spinner("Building vector store and retrieving evidence..."):
        rag_chunks, rag_embeddings = build_rag_vectorstore(pages_text, client)
        explanation = generate_rag_explanation(client, rag_chunks, rag_embeddings, metadata, bert_court, gpt_court, final_prediction)

    st.markdown(explanation)
    os.unlink(tmp_path)


if __name__ == "__main__":
    main()