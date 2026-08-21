import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone
import streamlit as st

# Automatically load local .env file if available
load_dotenv()

INDEX_NAME = "hp-supplies-rag"

st.set_page_config(
    page_title="HP Technical & Supplies RAG Assistant",
    page_icon="🖨️",
    layout="wide",
)

# ------------------------------------------------------------------------------
# 1. SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ------------------------------------------------------------------------------
# 2. HELPER & PINECONE FUNCTIONS
# ------------------------------------------------------------------------------
def resolve_api_keys():
    """Safe resolution of keys across .env, environment variables, and st.secrets."""
    g_key = os.getenv("GEMINI_API_KEY", "")
    p_key = os.getenv("PINECONE_API_KEY", "")

    try:
        if not g_key:
            g_key = st.secrets.get("GEMINI_API_KEY", "")
        if not p_key:
            p_key = st.secrets.get("PINECONE_API_KEY", "")
    except Exception:
        pass

    return g_key, p_key


@st.cache_data(ttl=60, show_spinner=False)
def get_indexed_models(pinecone_key: str) -> list[str]:
    """Queries Pinecone index to dynamically retrieve detected printer models/series."""
    if not pinecone_key or not pinecone_key.strip():
        return []

    try:
        pc = Pinecone(api_key=pinecone_key.strip())
        index = pc.Index(INDEX_NAME)

        res = index.query(
            vector=[0.1] * 768,
            top_k=100,
            include_metadata=True,
        )

        models = set()
        for match in res.get("matches", []):
            metadata = match.get("metadata", {})
            model_name = metadata.get("model") or metadata.get("source")
            if model_name:
                models.add(model_name)

        return sorted(list(models))
    except Exception:
        return []


def get_pinecone_context(
    query: str,
    gemini_key: str,
    pinecone_key: str,
    model_filter: str = "All Printer Models",
    top_k: int = 5,
) -> list[dict]:
    """Retrieves context chunks with rich source metadata and optional model filtering."""
    genai_client = genai.Client(api_key=gemini_key.strip())
    emb_res = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    query_vector = emb_res.embeddings[0].values

    pc = Pinecone(api_key=pinecone_key.strip())
    index = pc.Index(INDEX_NAME)

    query_filter = None
    if model_filter and model_filter != "All Printer Models":
        query_filter = {"model": {"$eq": model_filter}}

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=query_filter,
    )

    retrieved_docs = []
    for match in results.get("matches", []):
        retrieved_docs.append({
            "source": match["metadata"].get("source", "Unknown PDF"),
            "model": match["metadata"].get("model", "Generic HP Hardware"),
            "score": round(match["score"], 4),
            "text": match["metadata"].get("text", ""),
        })
    return retrieved_docs


# ------------------------------------------------------------------------------
# 3. AI JUDGE / GUARDRAIL EVALUATION FUNCTION
# ------------------------------------------------------------------------------
def run_ai_judge_evaluation(
    query: str, context_docs: list[dict], gemini_key: str
) -> dict:
    """Evaluates the retrieved context against the user query for relevance and sufficiency.

    Returns JSON with score (0-100), decision ('PASS'/'WARN'), and reasoning.
    """
    if not context_docs:
        return {
            "score": 0,
            "verdict": "FAIL",
            "reasoning": "No documents retrieved from Pinecone index.",
        }

    context_summary = "\n\n".join([
        f"Doc {i+1} [{doc['model']}]: {doc['text'][:300]}..."
        for i, doc in enumerate(context_docs)
    ])

    judge_prompt = f"""
You are an impartial AI Quality & Safety Judge evaluating a RAG pipeline.
Analyze the user query and the retrieved documentation chunks below. Determine if the documentation contains sufficient, relevant technical information to accurately answer the user query.

USER QUERY: "{query}"

RETRIEVED CONTEXT CHUNKS:
{context_summary}

Respond ONLY with a valid JSON object matching this exact schema:
{{
  "score": <integer from 0 to 100 representing relevance confidence>,
  "verdict": "<'PASS' if score >= 60 else 'WARN'>",
  "reasoning": "<1-2 sentence explanation of why the context is or isn't sufficient>"
}}
"""

    try:
        genai_client = genai.Client(api_key=gemini_key.strip())
        response = genai_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=judge_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        eval_result = json.loads(response.text)
        return eval_result
    except Exception as e:
        return {
            "score": 50,
            "verdict": "WARN",
            "reasoning": f"AI Judge evaluation skipped due to parsing error: {e}",
        }


def generate_rag_response(
    query: str,
    context_docs: list[dict],
    chat_history: list[dict],
    gemini_key: str,
    judge_result: dict,
) -> str:
    """Synthesizes answer using retrieved docs, chat history, and judge evaluations."""
    context_blocks = []
    for doc in context_docs:
        context_blocks.append(
            f"PRINTER MODEL: {doc['model']}\n"
            f"DOCUMENT SOURCE: {doc['source']} (Relevance Score: {doc['score']})\n"
            f"CONTENT:\n{doc['text']}"
        )
    formatted_context = "\n\n=========================================\n\n".join(
        context_blocks
    )

    history_text = ""
    if chat_history:
        recent_history = chat_history[-6:]
        history_text = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history]
        )

    judge_note = (
        f"[AI JUDGE EVALUATION: Context Confidence Score = {judge_result.get('score', 0)}%, Verdict = {judge_result.get('verdict', 'UNKNOWN')}]. "
        f"Judge Reasoning: {judge_result.get('reasoning', 'None')}"
    )

    system_instruction = (
        "SYSTEM ROLE AND INSTRUCTIONS:\n"
        "You are the official HP Enterprise Technical & Supplies Assistant.\n"
        "Your purpose is to assist engineers, support staff, and account teams with HP hardware specifications, "
        "yield guides, supply part numbers, and troubleshooting procedures.\n\n"
        "OPERATIONAL RULES:\n"
        "1. Identify printer/hardware models explicitly whenever cited in the retrieved context.\n"
        "2. If the AI Judge Verdict is 'WARN' or context confidence is low, explicitly warn the user that retrieved documentation may be incomplete.\n"
        "3. Use CONVERSATION HISTORY to resolve follow-up questions or pronouns (e.g., 'What is its page yield?').\n"
        "4. If the retrieved context genuinely lacks the answer, clearly state what information is present versus what is missing.\n\n"
        f"AI JUDGE ASSESSMENT:\n{judge_note}\n\n"
        f"RETRIEVED DOCUMENT CONTEXT:\n{formatted_context}\n\n"
        f"RECENT CONVERSATION HISTORY:\n{history_text if history_text else 'No previous context.'}"
    )

    genai_client = genai.Client(api_key=gemini_key.strip())
    response = genai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"{system_instruction}\n\nCURRENT USER QUESTION: {query}",
    )
    return response.text


# ------------------------------------------------------------------------------
# 4. STREAMLIT UI LAYOUT
# ------------------------------------------------------------------------------
st.title("🖨️ HP Technical & Supplies RAG Assistant")
st.caption(
    "Enterprise AI Knowledge Engine with Built-In AI Judge Guardrails"
)

env_gemini_key, env_pinecone_key = resolve_api_keys()

with st.sidebar:
    st.header("🔑 Credentials")
    gemini_api_key = st.text_input(
        "Gemini API Key", value=env_gemini_key, type="password"
    )
    pinecone_api_key = st.text_input(
        "Pinecone API Key", value=env_pinecone_key, type="password"
    )

    st.divider()

    st.header("🎯 Knowledge Scope & Filters")

    indexed_models = []
    if pinecone_api_key and pinecone_api_key.strip():
        indexed_models = get_indexed_models(pinecone_api_key.strip())

    if indexed_models:
        st.success(f"📚 {len(indexed_models)} Active Item(s) in Vector Catalog")
        selected_model = st.selectbox(
            "Filter Search by Hardware Model:",
            ["All Printer Models"] + indexed_models,
        )

        with st.expander("📋 Active Hardware Index"):
            for model_item in indexed_models:
                st.markdown(f"- **{model_item}**")
    else:
        selected_model = "All Printer Models"
        if not pinecone_api_key or not pinecone_api_key.strip():
            st.info(
                "Enter Pinecone API Key to load dynamic hardware model catalog."
            )
        else:
            st.warning(
                "Connected to Pinecone, but no vectors or 'model' metadata were found. Run `ingest_pinecone.py` to index documents."
            )

    st.divider()
    st.header("⚖️ AI Guardrail Architecture")
    st.markdown(
        "**AI Judge Active:** Every query passes through a secondary evaluation stage to score context relevance before final synthesis."
    )

    st.divider()
    st.header("⚙️ Controls")

    if st.button("🗑️ Clear Chat History"):
        st.session_state["chat_history"] = []
        st.rerun()

# Render Chat Messages
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input Interface
if prompt := st.chat_input("Ask about HP printer models, specs, or supplies..."):
    if not gemini_api_key or not pinecone_api_key:
        st.error("Please provide Gemini and Pinecone API keys in the sidebar.")
    else:
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                # Step 1: Retrieval
                with st.spinner("1/3 Searching Pinecone vector index..."):
                    context_docs = get_pinecone_context(
                        query=prompt,
                        gemini_key=gemini_api_key,
                        pinecone_key=pinecone_api_key,
                        model_filter=selected_model,
                    )

                # Step 2: AI Judge Evaluation
                with st.spinner(
                    "2/3 AI Judge evaluating context relevance & safety..."
                ):
                    judge_eval = run_ai_judge_evaluation(
                        prompt, context_docs, gemini_api_key
                    )

                # Display AI Judge Status Badge
                score = judge_eval.get("score", 0)
                verdict = judge_eval.get("verdict", "WARN")
                reasoning = judge_eval.get("reasoning", "")

                if verdict == "PASS":
                    st.success(
                        f"🟢 **AI Judge:** Context Confidence High ({score}%) | *{reasoning}*"
                    )
                else:
                    st.warning(
                        f"🟡 **AI Judge:** Context Confidence Moderate/Low ({score}%) | *{reasoning}*"
                    )

                # Step 3: Synthesis
                with st.spinner("3/3 Synthesizing grounded response..."):
                    with st.expander("📄 View Retrieved Document Sources"):
                        for doc in context_docs:
                            st.markdown(
                                f"**Model:** `{doc['model']}` | **Source:** `{doc['source']}` | **Score:** `{doc['score']}`"
                            )
                            st.caption(doc["text"][:300] + "...")
                            st.divider()

                    answer = generate_rag_response(
                        query=prompt,
                        context_docs=context_docs,
                        chat_history=st.session_state["chat_history"],
                        gemini_key=gemini_api_key,
                        judge_result=judge_eval,
                    )

                    st.write(answer)

                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": answer}
                    )

            except Exception as e:
                st.error(f"Error processing query: {e}")