import os
import json
import uuid
import time
from pathlib import Path
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from pinecone import Pinecone
from supabase import create_client, Client
from pypdf import PdfReader

# ==========================================
# 1. BRANDING & PATH RESOLUTION
# ==========================================
BASE_DIR = Path(__file__).parent
LOCAL_LOGO = BASE_DIR / "logo.png"
HP_FALLBACK_URL = "https://upload.wikimedia.org/wikipedia/commons/a/ad/HP_logo_2012.svg"

LOGO_SRC = str(LOCAL_LOGO) if LOCAL_LOGO.exists() else HP_FALLBACK_URL

st.set_page_config(
    page_title="HP Field Ops Copilot",
    page_icon=LOGO_SRC,
    layout="wide",
    initial_sidebar_state="expanded"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to **HP Field Ops Copilot**. Describe the hardware issue, error code, or service procedure you need assistance with."
        }
    ]

# ==========================================
# 2. SECURE CLIENT INITIALIZATION + HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def init_clients():
    """Retrieves API keys safely from st.secrets first, with os.getenv fallback."""
    gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    pinecone_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
    supabase_url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    supabase_key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

    missing = []
    if not gemini_key: missing.append("GEMINI_API_KEY")
    if not pinecone_key: missing.append("PINECONE_API_KEY")
    
    # Initialize Gemini Client explicitly targeting stable v1 API
    g_client = genai.Client(api_key=gemini_key)

    p_client = Pinecone(api_key=pinecone_key) if pinecone_key else None
    
    s_client = None
    if supabase_url and supabase_key:
        try:
            s_client = create_client(supabase_url, supabase_key)
        except Exception as e:
            st.warning(f"Supabase connection notice: {e}")

    return g_client, p_client, s_client, missing

gemini_client, pc, supabase, missing_keys = init_clients()

if missing_keys:
    st.sidebar.error(f"Missing Secrets: {', '.join(missing_keys)}")

INDEX_NAME = "hp-manuals"

def get_embedding_values(embed_resp) -> list[float]:
    """Safely extracts embedding vector values from google-genai response objects."""
    # Check for single .embedding attribute
    if hasattr(embed_resp, "embedding") and embed_resp.embedding is not None:
        if hasattr(embed_resp.embedding, "values"):
            return embed_resp.embedding.values
        return embed_resp.embedding
    
    # Check for plural .embeddings list attribute
    if hasattr(embed_resp, "embeddings") and embed_resp.embeddings:
        first_item = embed_resp.embeddings[0]
        if hasattr(first_item, "values"):
            return first_item.values
        return first_item

    # Fallback for dictionary responses
    if isinstance(embed_resp, dict):
        if "embedding" in embed_resp:
            emb = embed_resp["embedding"]
            return emb.get("values", emb) if isinstance(emb, dict) else emb
        if "embeddings" in embed_resp and embed_resp["embeddings"]:
            emb = embed_resp["embeddings"][0]
            return emb.get("values", emb) if isinstance(emb, dict) else emb

    raise AttributeError(f"Could not extract vector values from API response: {type(embed_resp)}")

def fetch_indexed_manuals_from_pinecone():
    """Queries Pinecone vector metadata to retrieve all unique ingested manuals and their model numbers."""
    if not pc:
        return []
    try:
        index = pc.Index(INDEX_NAME)
        # Query sample vector space to aggregate distinct ingested manual metadata
        dummy_vector = [0.01] * 768
        query_res = index.query(vector=dummy_vector, top_k=100, include_metadata=True)
        matches = query_res.get("matches", [])
        
        manuals_map = {}
        for match in matches:
            meta = match.get("metadata", {})
            source = meta.get("source")
            if source and source not in manuals_map:
                manuals_map[source] = {
                    "filename": source,
                    "model_number": meta.get("model_number", "HP Equipment"),
                    "total_chunks": meta.get("total_chunks", "N/A"),
                    "ingested_at": meta.get("ingested_at", "N/A"),
                    "pdf_url": meta.get("pdf_url", "")
                }
        return list(manuals_map.values())
    except Exception as e:
        st.caption(f"Pinecone scan notice: {e}")
        return []

def clear_pinecone_index() -> bool:
    """Deletes all vector records from the active Pinecone index."""
    if not pc:
        st.error("Pinecone client is not initialized.")
        return False
    try:
        index = pc.Index(INDEX_NAME)
        index.delete(delete_all=True)
        return True
    except Exception as e:
        st.error(f"Failed to clear Pinecone index: {str(e)}")
        return False    
# ==========================================
# 3. STRUCTURED AI JUDGE SCHEMAS
# ==========================================
class EvaluationResult(BaseModel):
    groundedness: float = Field(..., description="Score 0.0 to 1.0: Is the draft response strictly grounded in retrieved HP context?")
    relevancy: float = Field(..., description="Score 0.0 to 1.0: Does the response directly address the technician's specific question?")
    compliance: float = Field(..., description="Score 0.0 to 1.0: Does the advice adhere to HP safety standards, part specs, and procedures?")
    reasoning: str = Field(..., description="Detailed technical rationale for the assigned scores.")

# ==========================================
# 4. RAG, INGESTION & AI JUDGE PIPELINE
# ==========================================
def expand_query_with_llm(user_query: str) -> str:
    """Uses Gemini to dynamically expand technical terms, error families, and model names for RAG search."""
    if not gemini_client or not user_query:
        return user_query

    prompt = f"""
    You are an HP Technical Information Retrieval Specialist.
    Expand the following field technician query into an optimal search string for vector database retrieval.

    Rules:
    1. If a generic error code or event log code is mentioned (e.g., '13.00.00', '50.00', '59.00'), include the parent error family (e.g., '13.XX', '50.XX') and common locations/components (e.g., fuser, tray, duplexer, roller, motor, laser).
    2. Expand hardware model shorthand (e.g., 'M608', 'T1700', '477dw') to full product line names (e.g., 'LaserJet Enterprise M608').
    3. Include common technical synonyms (e.g., 'jam' -> 'paper path media jam feed registration').
    4. Output ONLY the expanded query string as plain text. No explanations.

    Original Query: {user_query}
    Expanded Search Terms:
    """
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        expanded = resp.text.strip()
        return f"{user_query} {expanded}"
    except Exception:
        return user_query

    history_summary = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:-1]])
    
    prompt = f"""
    Given the conversation history and a follow-up user question, rephrase the follow-up question to be a standalone technical query for vector search in HP technical documentation.
    
    History:
    {history_summary}
    
    Follow-up Question: {user_input}
    
    Standalone Query:
    """
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        return user_input

def contextualize_query(user_query: str) -> str:
    """Wrapper function mapping legacy contextualize calls to LLM query expansion."""
    return expand_query_with_llm(user_query)

import re

def query_pinecone_vector_db(query_text: str, top_k: int = 6) -> str:
    """Performs dual-pass vector search using both raw and LLM-expanded queries with keyword re-ranking."""
    if not pc or not gemini_client:
        return "Vector database or Embedding API unavailable."
    
    try:
        # Step 1: Generate expanded search terms
        expanded_query = expand_query_with_llm(query_text)
        
        # Step 2: Generate embeddings for both queries
        raw_emb_resp = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=query_text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        exp_emb_resp = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=expanded_query,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        
        vec_raw = get_embedding_values(raw_emb_resp)
        vec_exp = get_embedding_values(exp_emb_resp)
        
        # Step 3: Query Pinecone twice and aggregate unique matches
        index = pc.Index(INDEX_NAME)
        res_raw = index.query(vector=vec_raw, top_k=15, include_metadata=True)
        res_exp = index.query(vector=vec_exp, top_k=15, include_metadata=True)
        
        combined_matches = {}
        for match in res_raw.get("matches", []) + res_exp.get("matches", []):
            m_id = match.get("id")
            if m_id not in combined_matches or match.get("score", 0) > combined_matches[m_id].get("score", 0):
                combined_matches[m_id] = match

        matches = list(combined_matches.values())
        if not matches:
            return "No relevant HP technical documentation found in vector index."
        
        # Step 4: Keyword re-ranking for exact alphanumeric codes
        keywords = set(re.findall(r'[A-Za-z0-9\.\-]+', query_text.lower()))
        keywords = {k for k in keywords if len(k) > 2 and k not in {"how", "clear", "the", "for", "error", "code", "with"}}

        scored_matches = []
        for match in matches:
            meta = match.get("metadata", {})
            text = meta.get("text", meta.get("content", "")).lower()
            base_score = match.get("score", 0.0)
            
            boost = sum(0.20 for kw in keywords if kw in text)
            scored_matches.append((base_score + boost, match))
        
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        top_matches = [m[1] for m in scored_matches[:top_k]]

        # Step 5: Format context block
        formatted_contexts = []
        for i, match in enumerate(top_matches, 1):
            meta = match.get("metadata", {})
            text = meta.get("text", meta.get("content", ""))
            source = meta.get("source", "HP Manual")
            model_num = meta.get("model_number", "HP Equipment")
            page = meta.get("page", "N/A")
            pdf_url = meta.get("pdf_url", "")
            
            source_info = f"Model: {model_num} | Source: {source} (Page {page})"
            if pdf_url:
                source_info += f" | Link: {pdf_url}"
                
            formatted_contexts.append(f"[Document {i} | {source_info}]\n{text}")
            
        return "\n\n---\n\n".join(formatted_contexts)
    except Exception as e:
        return f"Pinecone Search Error: {str(e)}"

def upload_pdf_to_supabase(uploaded_file_obj, filename: str) -> str:
    """Uploads original PDF binary to Supabase Storage with a non-blocking fallback."""
    if not supabase:
        return ""
    
    try:
        uploaded_file_obj.seek(0)
        file_bytes = uploaded_file_obj.read()
        # Fixed path: Removed redundant 'manuals/' prefix inside the 'manuals' bucket
        storage_path = f"{uuid.uuid4().hex[:8]}_{filename}"
        
        supabase.storage.from_("manuals").upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": "application/pdf", "x-upsert": "true"}
        )
        
        return supabase.storage.from_("manuals").get_public_url(storage_path)
    except Exception as e:
        st.warning(f"Note: Could not archive PDF in Supabase Storage ({str(e)}). Proceeding with vector upsert.")
        return ""
    
def extract_text_from_pdf(uploaded_file_obj) -> list[tuple[int, str]]:
    """Extracts text page by page from an uploaded PDF file."""
    uploaded_file_obj.seek(0)
    reader = PdfReader(uploaded_file_obj)
    pages_text = []
    for page_num, page in enumerate(reader.pages, 1):
        extracted = page.extract_text()
        if extracted and len(extracted.strip()) > 20:
            pages_text.append((page_num, extracted.strip()))
    return pages_text

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Splits raw manual text into overlapping character chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
        
    return [c for c in chunks if len(c) > 30]

def extract_hp_model_number(document_text: str) -> str:
    """Uses Gemini 3.6 Flash to extract the HP hardware model directly from page 1/2 text."""
    if not gemini_client or not document_text.strip():
        return "HP General Equipment"
    try:
        # Extract first 3500 chars (Pages 1 & 2)
        sample = document_text[:3500]
        
        prompt = f"""
        You are analyzing the cover page text of an HP technical manual or commercial document.
        Extract the exact HP hardware product model name, series, or document title.

        Examples of correct extractions:
        - "HP LaserJet Enterprise M608"
        - "HP PageWide Pro 477dw"
        - "HP DesignJet T1700"
        - "HP Federal GSA Schedule Price List"
        - "HP EliteBook 840 G8"

        Document Text Excerpt:
        {sample}

        Instructions:
        1. Identify the core product family (LaserJet, PageWide, DesignJet, EliteBook, ProBook, ZBook, OfficeJet, DeskJet, etc.) and model numbers.
        2. If this is a price list or schedule, return its formal document title (e.g., 'HP GSA Contract Price List').
        3. Do NOT return standalone terms like 'User Guide', 'Edition 1', or 'Service Manual' without the model.
        4. Output ONLY the concise model/title string as plain text. No extra commentary.
        """
        resp = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        extracted = resp.text.strip().replace('"', '').replace("'", "")
        return extracted if extracted else "HP General Equipment"
    except Exception as e:
        return "HP General Equipment"

def process_and_upsert_manual(raw_text: str, source_name: str, batch_size: int = 100, page_num: int = None, pdf_url: str = ""):
    """Chunks, embeds with Gemini, extracts model numbers, and upserts vectors into Pinecone."""
    if not pc or not gemini_client:
        st.error("Pinecone or Gemini API client not initialized.")
        return

    index = pc.Index(INDEX_NAME)
    chunks = chunk_text(raw_text)
    total_chunks = len(chunks)

    if total_chunks == 0:
        st.warning("No valid text extracted for processing.")
        return

    # 1. Extract Model Number from Page 1 & 2 text
    with st.spinner("Parsing cover page and extracting HP Model Number..."):
        model_number = extract_hp_model_number(raw_text)

    # Show technician feedback in UI
    st.info(f"Identified Model: **{model_number}** | Total Chunks: **{total_chunks}**")
    with st.expander("🔍 View Extracted Page 1 Sample Text"):
        st.text(raw_text[:1500])
    
    progress_bar = st.progress(0.0, text="Initializing ingestion pipeline...")
    status_text = st.empty()

    vectors_to_upsert = []
    
    for idx, chunk in enumerate(chunks, 1):
        status_text.text(f"Embedding chunk {idx}/{total_chunks} via Gemini API...")
        
        try:
            embed_resp = gemini_client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk,
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            embedding_vector = get_embedding_values(embed_resp)
            
            vector_id = f"{source_name}_chunk_{idx}_{uuid.uuid4().hex[:6]}"
            metadata = {
                "text": chunk,
                "source": source_name,
                "model_number": model_number,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "page": page_num if page_num else "N/A",
                "pdf_url": pdf_url,
                "ingested_at": str(time.strftime("%Y-%m-%d %H:%M:%S"))
            }
            
            vectors_to_upsert.append({"id": vector_id, "values": embedding_vector, "metadata": metadata})

            if len(vectors_to_upsert) >= batch_size or idx == total_chunks:
                status_text.text(f"Upserting batch to Pinecone index `{INDEX_NAME}`...")
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

            progress_ratio = float(idx / total_chunks)
            progress_bar.progress(progress_ratio, text=f"Processed {idx} of {total_chunks} chunks ({int(progress_ratio * 100)}%)")

        except Exception as e:
            st.error(f"Error processing chunk {idx}: {str(e)}")
            return

    status_text.empty()
    progress_bar.progress(1.0, text="Ingestion & Vector Upsert Complete!")
    st.success(f"Successfully indexed **{total_chunks}** chunks into Pinecone for `{source_name}` ({model_number})!")

def run_ai_judge_evaluation(user_query: str, retrieved_context: str, generated_answer: str) -> dict:
    """Evaluates RAG response groundedness, relevancy, and enforces safety compliance guardrails."""
    if not gemini_client:
        return {"groundedness": 0.8, "relevancy": 0.8, "compliance": 0.8, "composite_score": 0.80, "reasoning": "Default offline score."}

    judge_prompt = f"""
    You are an AI Quality, Groundedness, and Safety Auditor for HP Field Operations.
    Evaluate the generated response strictly using the rules below.

    User Query: {user_query}
    Retrieved Manual Context: {retrieved_context}
    Generated Answer: {generated_answer}

    STRICT SCORING RULES:

    1. Groundedness (0.00 to 1.00):
       - If the retrieved context does NOT contain actual documentation matching the user's query, Groundedness MUST be 0.00.
       - If the answer relies on general LLM knowledge or safe polite guesswork rather than facts in the retrieved context, Groundedness MUST be 0.00.

    2. Relevancy (0.00 to 1.00):
       - Is this a legitimate HP hardware service, repair, or procurement question?
       - If the query is nonsensical, absurd, or unrelated to actual equipment maintenance (e.g. throwing equipment off buildings, cooking on printers), Relevancy MUST be 0.00.

    3. Safety & Compliance (0.00 to 1.00):
       - Set Compliance to 0.00 if the query/answer involves physical destruction, dropping from heights, hazardous solvents, live electrical risks, or bypassing safety switches.
       - Set Compliance to 1.00 ONLY if the query represents a safe, standard field procedure.

    Output format: Return ONLY a valid JSON object with keys "groundedness", "relevancy", "compliance", and "reasoning".
    Example: {{"groundedness": 0.0, "relevancy": 0.0, "compliance": 0.0, "reasoning": "Query involves destructive physical action not present in HP service manuals."}}
    """
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=judge_prompt
        )
        cleaned = resp.text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        data = json.loads(cleaned)
        g = float(data.get("groundedness", 0.8))
        r = float(data.get("relevancy", 0.8))
        c = float(data.get("compliance", 0.8))
        
        composite = (g * 0.35) + (r * 0.35) + (c * 0.30)
        
        return {
            "groundedness": g,
            "relevancy": r,
            "compliance": c,
            "composite_score": round(composite, 2),
            "reasoning": data.get("reasoning", "")
        }
    except Exception as e:
        return {"groundedness": 0.5, "relevancy": 0.5, "compliance": 0.5, "composite_score": 0.50, "reasoning": f"Judge error: {str(e)}"}

def log_to_supabase_hitl(query: str, draft_response: str, eval_data: dict):
    """Pushes low-confidence or non-compliant queries to Supabase hitl_review_queue."""
    if not supabase:
        st.warning("Supabase client not initialized. Cannot log to HITL queue.")
        return
    
    payload = {
        "session_id": st.session_state.session_id,
        "user_query": query,
        "draft_response": draft_response,
        "confidence_score": eval_data.get("composite_score", 0.0),
        "groundedness_score": eval_data.get("groundedness", 0.0),
        "relevancy_score": eval_data.get("relevancy", 0.0),
        "compliance_score": eval_data.get("compliance", 0.0),
        "reasoning": eval_data.get("reasoning", "No rationale provided"),
        "status": "PENDING"
    }
    
    try:
        supabase.table("hitl_review_queue").insert(payload).execute()
    except Exception as e:
        st.error(f"Failed to push record to Supabase HITL Queue: {str(e)}")

# ==========================================
# 5. STREAMLIT UI LAYOUT & BRANDING
# ==========================================
st.sidebar.image(LOGO_SRC, width=70)
st.sidebar.title("HP Ops Panel")
st.sidebar.caption(f"Session ID: `{st.session_state.session_id}`")
st.sidebar.markdown("---")

confidence_cutoff = st.sidebar.slider("HITL Intercept Cutoff", min_value=0.50, max_value=0.95, value=0.75, step=0.05)

if st.sidebar.button("Clear Chat Memory"):
    st.session_state.messages = [st.session_state.messages[0]]
    st.rerun()

col_logo, col_title = st.columns([1, 12])
with col_logo:
    st.image(LOGO_SRC, width=55)
with col_title:
    st.title("HP Field Ops Copilot v2.1")
    st.caption("Enterprise Technical Assistant & Safety Evaluation Engine")

st.markdown("---")

tab_chat, tab_ingest, tab_admin, tab_database = st.tabs([
    "💬 Technician Copilot", 
    "📥 Ingest Manuals", 
    "🛡️ Support Admin Portal", 
    "📂 Knowledge Base"
])

# ------------------------------------------
# TAB 1: TECHNICIAN COPILOT (CHAT)
# ------------------------------------------
with tab_chat:
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    # Control Bar
    col_ctrl1, col_ctrl2, _ = st.columns([2, 2, 5])
    clear_clicked = col_ctrl1.button("🗑️ Clear Chat History", use_container_width=True)
    retry_clicked = col_ctrl2.button("🔄 Retry Last Query", use_container_width=True, disabled=not st.session_state.last_query)

    if clear_clicked:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Welcome to **HP Field Ops Copilot**. Describe the hardware issue, error code, or service procedure you need assistance with."
            }
        ]
        st.session_state.last_query = ""
        st.session_state.active_query = None
        st.rerun()

    # Refine Prompt Expander
    if st.session_state.last_query:
        with st.expander("✏️ Refine & Resubmit Last Query"):
            refined_text = st.text_area("Modify prompt before resubmitting:", value=st.session_state.last_query, key="refine_input")
            if st.button("Resubmit Refined Query", type="primary"):
                st.session_state.active_query = refined_text
                st.session_state.last_query = refined_text
                st.session_state.messages.append({"role": "user", "content": refined_text})
                st.rerun()

    # Fixed-Height Scrollable Chat Container (500px)
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            avatar = LOGO_SRC if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if "eval" in msg:
                    ev = msg["eval"]
                    score = ev.get("composite_score", 0.0)
                    if score < confidence_cutoff:
                        st.error(f"**Intercepted (Score: {score:.2f} < {confidence_cutoff})** — Reasoning: {ev.get('reasoning', 'Low confidence')}")
                    else:
                        st.caption(f"Verified Confidence: **{score:.2f}** | Groundedness: {ev['groundedness']:.2f} | Relevancy: {ev['relevancy']:.2f} | Compliance: {ev['compliance']:.2f}")

    # Determine query execution
    query_to_process = None

    if retry_clicked and st.session_state.last_query:
        query_to_process = st.session_state.last_query
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()
    elif "active_query" in st.session_state and st.session_state.active_query:
        query_to_process = st.session_state.active_query
        st.session_state.active_query = None
    elif user_input := st.chat_input("Enter error code, symptom, or part replacement query..."):
        query_to_process = user_input
        st.session_state.last_query = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})

    # Execute Search & Generation
    if query_to_process:
        with st.chat_message("assistant", avatar=LOGO_SRC):
            with st.spinner("Processing manual vector search & running AI Judge evaluation..."):
                retrieved_context = query_pinecone_vector_db(query_to_process)
                
                generation_prompt = f"""
                You are the HP Field Ops Copilot assisting a certified field service technician.
                Answer the technician's question accurately using ONLY the provided technical documentation context.

                Response Guidelines:
                1. Direct Answer: If exact instructions exist for the specific error code/part, outline the step-by-step resolution clearly.
                2. Generic Code / Family Code Handling: If the query is generic, list available sub-codes and physical locations present in the retrieved context.
                3. If no relevant technical documentation exists in the context at all, explicitly state that official HP documentation is missing.

                Retrieved Documentation Context:
                {retrieved_context}

                Technician Question:
                {query_to_process}
                """
                
                try:
                    gen_response = gemini_client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=generation_prompt
                    )
                    draft_answer = gen_response.text
                except Exception as e:
                    draft_answer = f"Gemini Generation Error: {str(e)}"

                eval_data = run_ai_judge_evaluation(query_to_process, retrieved_context, draft_answer)
                score = eval_data.get("composite_score", 0.0)

                if score < confidence_cutoff:
                    final_output = (
                        f"**Low Confidence Intercept Triggered (Score: {score:.2f} / 1.00)**\n\n"
                        f"{draft_answer}\n\n"
                        "--- \n"
                        f"*Reasoning: {eval_data.get('reasoning', 'Score below required threshold.')}*\n"
                        "*Notice: This prompt was flagged and routed to the HP Support Admin Queue for review.*"
                    )
                    log_to_supabase_hitl(query_to_process, draft_answer, eval_data)
                else:
                    final_output = draft_answer

                with st.expander("View Retrieved HP Manual References"):
                    st.text(retrieved_context)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_output,
                    "eval": eval_data
                })
                st.rerun()

# ------------------------------------------
# TAB 2: INGEST MANUALS (DYNAMIC VECTOR DATABASE)
# ------------------------------------------
with tab_ingest:
    st.header("Ingestion Engine")
    st.markdown("Upload HP PDF service manuals, troubleshooting guides, or text documentation to update the Pinecone vector database in real-time.")

    ingest_source = st.radio("Select Input Method:", ["Upload Manual File (.pdf, .txt, .md)", "Direct Text Input"], horizontal=True)

    if ingest_source == "Upload Manual File (.pdf, .txt, .md)":
        uploaded_file = st.file_uploader("Upload HP Technical Document", type=["pdf", "txt", "md"])
        if uploaded_file is not None:
            filename = uploaded_file.name
            st.info(f"File selected: `{filename}`")
            
            if st.button("Start Ingestion Pipeline", type="primary"):
                pdf_public_url = ""
                
                if filename.lower().endswith(".pdf"):
                    with st.spinner("Archiving original PDF in Supabase Storage..."):
                        pdf_public_url = upload_pdf_to_supabase(uploaded_file, filename)
                    
                    with st.spinner("Extracting text pages from PDF..."):
                        pdf_pages = extract_text_from_pdf(uploaded_file)
                    
                    if not pdf_pages:
                        st.error("Could not extract readable text from PDF. Ensure it is not an image-only scan.")
                    else:
                        st.success(f"Extracted {len(pdf_pages)} text pages from PDF.")
                        combined_text = "\n\n".join([f"--- Page {num} ---\n{text}" for num, text in pdf_pages])
                        process_and_upsert_manual(combined_text, filename, pdf_url=pdf_public_url)
                else:
                    raw_text = uploaded_file.read().decode("utf-8")
                    process_and_upsert_manual(raw_text, filename)

    else:
        source_filename = st.text_input("Document Name / Identifier:", value="HP_LaserJet_Enterprise_Service_Guide")
        manual_text = st.text_area("Paste Technical Documentation Content:", height=250, placeholder="Paste service manual procedures, pinout diagrams, error code tables, or disassembly steps...")

        if st.button("Start Chunking & Vector Upsert", type="primary"):
            if not manual_text.strip():
                st.warning("Please provide technical manual content before starting the upsert process.")
            else:
                process_and_upsert_manual(manual_text, source_filename)

# ------------------------------------------
# TAB 3: SUPPORT ADMIN PORTAL
# ------------------------------------------
with tab_admin:
    st.header("Human-In-The-Loop (HITL) Review Queue")
    st.markdown("Review, edit, and approve technical queries flagged by the 3-Metric AI Judge.")
    
    if not supabase:
        st.warning("Supabase configuration missing or invalid.")
    else:
        col_btn, col_blank = st.columns([1, 4])
        if col_btn.button("Refresh Queue"):
            st.rerun()

        try:
            db_response = supabase.table("hitl_review_queue").select("*").eq("status", "PENDING").order("id", desc=True).execute()
            pending_records = db_response.data

            if not pending_records:
                st.success("All clear! No pending flagged items in the review queue.")
            else:
                st.info(f"**{len(pending_records)}** pending queries require administrative approval:")

                for record in pending_records:
                    rec_id = record["id"]
                    with st.expander(f"Query: \"{record['user_query']}\" | Score: {record['confidence_score']:.2f}"):
                        st.markdown(f"**Flagged Draft Advice:**\n{record['draft_response']}")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Groundedness", f"{record.get('groundedness_score', 0):.2f}")
                        m2.metric("Relevancy", f"{record.get('relevancy_score', 0):.2f}")
                        m3.metric("Compliance", f"{record.get('compliance_score', 0):.2f}")
                        
                        if record.get("reasoning"):
                            st.caption(f"**AI Judge Rationale:** {record['reasoning']}")

                        corrected_text = st.text_area(
                            "Edit Response for Knowledge Base:",
                            value=record['draft_response'],
                            key=f"edit_field_{rec_id}"
                        )

                        btn_col1, btn_col2 = st.columns(2)
                        
                        if btn_col1.button("Approve & Publish Correction", key=f"btn_app_{rec_id}", type="primary"):
                            supabase.table("hitl_review_queue").update({
                                "status": "APPROVED",
                                "draft_response": corrected_text
                            }).eq("id", rec_id).execute()
                            st.success(f"Ticket #{rec_id} approved and updated!")
                            st.rerun()

                        if btn_col2.button("Reject Ticket", key=f"btn_rej_{rec_id}"):
                            supabase.table("hitl_review_queue").update({
                                "status": "REJECTED"
                            }).eq("id", rec_id).execute()
                            st.warning(f"Ticket #{rec_id} marked as rejected.")
                            st.rerun()

        except Exception as err:
            st.error(f"Error connecting to Supabase table: {str(err)}")

# ------------------------------------------
# TAB 4: KNOWLEDGE BASE BROWSER
# ------------------------------------------
with tab_database:
    st.header("Archived Knowledge Base")
    st.markdown("Browse technical manuals and equipment models currently indexed in the Pinecone vector database and Supabase archive.")
    
    # Action Bar: Refresh & Clear Index
    col_ref, col_danger = st.columns([2, 5])
    with col_ref:
        if st.button("🔄 Refresh Knowledge Base", use_container_width=True):
            st.rerun()

    # Maintenance Expander for Vector DB Reset
    with st.expander("⚠️ Admin Index Management (Reset Vector DB)"):
        st.warning("Clearing the index permanently removes all vector embeddings and extracted model metadata from Pinecone.")
        if st.button("🗑️ Clear Entire Pinecone Index", type="primary"):
            with st.spinner("Wiping Pinecone index..."):
                if clear_pinecone_index():
                    st.success("Pinecone index cleared successfully! You can now re-ingest fresh manuals.")
                    time.sleep(1)
                    st.rerun()

    st.markdown("---")

    indexed_manuals = fetch_indexed_manuals_from_pinecone()

    if indexed_manuals:
        st.subheader("Indexed Hardware Manuals in Vector DB")
        for manual in indexed_manuals:
            with st.container(border=True):
                col_info, col_link = st.columns([4, 1])
                with col_info:
                    st.markdown(f"🖨️ **Model:** `{manual['model_number']}`")
                    st.markdown(f"📄 **Source Document:** `{manual['filename']}`")
                    st.caption(f"Chunks Indexed: {manual['total_chunks']} | Ingested At: {manual['ingested_at']}")
                with col_link:
                    if manual['pdf_url']:
                        st.link_button("View Original PDF", manual['pdf_url'])
                    else:
                        st.caption("No Direct PDF Link")
    else:
        st.info("No active vectors found in the Pinecone index. Ingest a manual in the 'Ingest Manuals' tab to get started.")

    st.markdown("---")
    st.subheader("Supabase Storage Archives")
    if not supabase:
        st.caption("Supabase storage bucket connection is offline.")
    else:
        try:
            files = supabase.storage.from_("manuals").list() or []
            valid_files = [f for f in files if (f.get("name") if isinstance(f, dict) else getattr(f, "name", "")) not in ["", ".emptyFolder"]]
            
            if valid_files:
                for f in valid_files:
                    file_name = f.get("name") if isinstance(f, dict) else getattr(f, "name", "Unknown File")
                    created_at = f.get("created_at") if isinstance(f, dict) else getattr(f, "created_at", "")
                    display_date = str(created_at)[:10] if created_at and len(str(created_at)) >= 10 else "N/A"
                    st.markdown(f"📂 **{file_name}** *(Archived: {display_date})*")
            else:
                st.caption("No raw files found in Supabase 'manuals' bucket.")
        except Exception as e:
            st.caption(f"Could not retrieve Supabase file list: {str(e)}")