import numpy as np
import openai

GPT_MODEL = "gpt-4o"

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


def build_rag_vectorstore(pages_text: list, client: openai.OpenAI) -> tuple[list, list]:
    chunk_size = 3000
    chunks = []

    for page in pages_text:
        text = page["text"]
        page_num = page["page"]
        for start in range(0, len(text), chunk_size):
            sub = text[start:start + chunk_size].strip()
            if sub:
                chunks.append({"text": sub, "page": page_num, "chunk_idx": len(chunks)})

    all_embeddings = []
    for i in range(0, len(chunks), 100):
        batch = [c["text"] for c in chunks[i:i+100]]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        all_embeddings.extend([r.embedding for r in response.data])

    return chunks, all_embeddings


def retrieve_rag_chunks(query: str, chunks: list, embeddings: list, client: openai.OpenAI, k: int = 6) -> list:
    response = client.embeddings.create(model="text-embedding-3-small", input=[query])
    query_emb = np.array(response.data[0].embedding)

    scores = []
    for i, emb in enumerate(embeddings):
        chunk_emb = np.array(emb)
        score = np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-9)
        scores.append((score, i))

    scores.sort(reverse=True)
    return [chunks[i] for _, i in scores[:k]]


def generate_rag_explanation(client: openai.OpenAI, chunks: list, embeddings: list, metadata: dict, bert_court: dict, gpt_court: dict, final_prediction: int) -> str:
    side = "PETITIONER" if final_prediction == 1 else "RESPONDENT"
    losing_side = "RESPONDENT" if final_prediction == 1 else "PETITIONER"
    case_name = metadata.get("case_name", metadata.get("docket", "unknown"))
    pet_adv = metadata.get("pet_advocate", "Unknown")
    res_adv = metadata.get("res_advocate", "Unknown")

    bert_pred_label = "petitioner" if bert_court["prediction"] == 1 else "respondent"
    bert_summary = f"BERT (full-court model): {bert_pred_label} ({bert_court['confidence']:.0%} confidence)"

    if gpt_court:
        gpt_pred = "petitioner" if gpt_court.get("prediction") == 1 else "respondent"
        gpt_parts = [
            f"GPT (full-bench scoring): leans {gpt_pred} "
            f"(bench hostility \u2192 petitioner: {gpt_court.get('pet_hostility', 'N/A')}, "
            f"\u2192 respondent: {gpt_court.get('res_hostility', 'N/A')})"
        ]
    else:
        gpt_parts = ["GPT bench scoring unavailable."]

    queries = [
        f"justice questioning {side.lower()} advocate skeptically challenging argument",
        f"justice expressing concern about {losing_side.lower()} position",
        "strongest exchange revealing justice vote preference",
        "justice hostile challenging oral argument",
        "justice sympathetic agreeing with advocate",
        f"{pet_adv} petitioner argument",
        f"{res_adv} respondent argument",
    ]

    seen = set()
    retrieved = []
    for query in queries:
        docs = retrieve_rag_chunks(query, chunks, embeddings, client)
        for doc in docs:
            content = doc["text"].strip()
            key = content[:100]
            if key not in seen:
                seen.add(key)
                retrieved.append(f"[Page {doc['page']}]\n{content}")
        if len(retrieved) >= 12:
            break

    retrieved_context = "\n\n---\n\n".join(retrieved[:12])

    prompt = f"""You are a Supreme Court analyst generating a detailed explanation for a vote prediction.

CASE: {case_name}
PREDICTED WINNER: {side}
PETITIONER'S COUNSEL: {pet_adv}
RESPONDENT'S COUNSEL: {res_adv}

COURT-LEVEL MODEL PREDICTION:
{bert_summary}

JUSTICE-BY-JUSTICE GPT SENTIMENT:
{chr(10).join(gpt_parts) if gpt_parts else "No per-justice GPT data available."}

MOST RELEVANT TRANSCRIPT PASSAGES (retrieved via semantic search):
{retrieved_context}

The BERT prediction is treated as ground truth. Using the retrieved transcript passages as your primary evidence, generate a compelling analytical explanation for why {side} wins.

Your explanation must:
1. **Overall Assessment** — Summarize the pattern of questioning across the bench
2. **Key Evidence** — Quote 2-3 specific exchanges from the retrieved passages that most strongly signal the outcome. Use exact words and cite page numbers.
3. **Strongest Signals** — Which justices showed the clearest vote direction and why, based on GPT hostility scores above
4. **Counterarguments** — What the losing side argued and why it likely fell short

Be specific. Ground every claim in the retrieved passages. Write for a legal audience."""

    try:
        resp = client.chat.completions.create(model=GPT_MODEL, max_tokens=1200, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {e}"