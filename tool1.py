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
# 2. SECURE CLIENT INITIALIZATION + HELPER FUNCTION
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
def contextualize_query(user_input: str) -> str:
    """Rewrites follow-up user questions into standalone queries using chat history."""
    if len(st.session_state.messages) <= 2:
        return user_input

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
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        return user_input

def query_pinecone_vector_db(query_text: str, top_k: int = 4) -> str:
    """Retrieves relevant manual excerpts from Pinecone using Gemini embeddings."""
    if not pc or not gemini_client:
        return "Vector database or Embedding API unavailable."
    
    try:
        embed_resp = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=query_text
        )
        query_vector = get_embedding_values(embed_resp)
        
        index = pc.Index(INDEX_NAME)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        
        matches = results.get("matches", [])
        if not matches:
            return "No relevant HP technical documentation found in vector index."
        
        formatted_contexts = []
        for i, match in enumerate(matches, 1):
            meta = match.get("metadata", {})
            text = meta.get("text", meta.get("content", ""))
            source = meta.get("source", "HP Manual")
            page = meta.get("page", "N/A")
            pdf_url = meta.get("pdf_url", "")
            
            source_info = f"Source: {source} (Page {page})"
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
        storage_path = f"manuals/{uuid.uuid4().hex[:8]}_{filename}"
        
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

def process_and_upsert_manual(raw_text: str, source_name: str, batch_size: int = 100, page_num: int = None, pdf_url: str = ""):
    """Chunks, embeds with Gemini, and upserts vectors into Pinecone using optimized batch sizes."""
    if not pc or not gemini_client:
        st.error("Pinecone or Gemini API client not initialized.")
        return

    index = pc.Index(INDEX_NAME)
    chunks = chunk_text(raw_text)
    total_chunks = len(chunks)

    if total_chunks == 0:
        st.warning("No valid text extracted for processing.")
        return

    st.info(f"Generated **{total_chunks}** text chunks for ingestion from `{source_name}`.")
    
    progress_bar = st.progress(0.0, text="Initializing ingestion pipeline...")
    status_text = st.empty()

    vectors_to_upsert = []
    
    for idx, chunk in enumerate(chunks, 1):
        status_text.text(f"Embedding chunk {idx}/{total_chunks} via Gemini API...")
        
        try:
            embed_resp = gemini_client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk
            )
            embedding_vector = get_embedding_values(embed_resp)
            
            vector_id = f"{source_name}_chunk_{idx}_{uuid.uuid4().hex[:6]}"
            metadata = {
                "text": chunk,
                "source": source_name,
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
    st.success(f"Successfully indexed **{total_chunks}** chunks into Pinecone for `{source_name}`!")

def run_ai_judge_evaluation(query: str, context: str, draft_response: str) -> dict:
    """Executes deterministic 3-metric evaluation using Gemini Structured Output."""
    if not gemini_client:
        return {"composite_score": 0.0, "groundedness": 0.0, "relevancy": 0.0, "compliance": 0.0, "reasoning": "Gemini client offline."}

    judge_prompt = f"""
    You are an expert AI Safety Judge and Technical Lead for HP Field Operations.
    Evaluate the proposed technical response strictly based on the provided Context and Query.

    Technician Query: {query}
    Retrieved Manual Context: {context}
    Proposed Response: {draft_response}

    Evaluate and score Groundedness, Relevancy, and Compliance from 0.0 to 1.0.
    """
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=judge_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvaluationResult,
                temperature=0.0,
            ),
        )
        
        eval_dict = json.loads(response.text)
        composite = round(
            (eval_dict["groundedness"] * 0.40) + 
            (eval_dict["relevancy"] * 0.30) + 
            (eval_dict["compliance"] * 0.30), 
            2
        )
        eval_dict["composite_score"] = composite
        return eval_dict
        
    except Exception as e:
        return {
            "groundedness": 0.0, "relevancy": 0.0, "compliance": 0.0,
            "composite_score": 0.0, "reasoning": f"Judge Execution Error: {str(e)}"
        }

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
    for msg in st.session_state.messages:
        avatar = LOGO_SRC if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "eval" in msg:
                ev = msg["eval"]
                score = ev.get("composite_score", 0.0)
                if score < confidence_cutoff:
                    st.error(f"**Intercepted (Score: {score:.2f} < {confidence_cutoff})** — Routed to Admin Queue")
                else:
                    st.caption(f"Verified Confidence: **{score:.2f}** | Groundedness: {ev['groundedness']:.2f} | Relevancy: {ev['relevancy']:.2f} | Compliance: {ev['compliance']:.2f}")

    if user_input := st.chat_input("Enter error code, symptom, or part replacement query..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=LOGO_SRC):
            with st.spinner("Processing manual vector search & running AI Judge evaluation..."):
                
                search_query = contextualize_query(user_input)
                retrieved_context = query_pinecone_vector_db(search_query)
                
                generation_prompt = f"""
                You are the HP Field Ops Copilot assisting a certified field service technician.
                Answer the technician's question accurately using ONLY the provided technical documentation.
                If the documentation does not contain sufficient details to answer safely, explicitly state that official HP documentation is missing.

                Retrieved Documentation Context:
                {retrieved_context}

                Technician Question:
                {user_input}
                """
                
                gen_response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=generation_prompt
                )
                draft_answer = gen_response.text

                eval_data = run_ai_judge_evaluation(user_input, retrieved_context, draft_answer)
                score = eval_data.get("composite_score", 0.0)

                if score < confidence_cutoff:
                    final_output = (
                        f"**Low Confidence Intercept Triggered (Score: {score:.2f} / 1.00)**\n\n"
                        f"{draft_answer}\n\n"
                        "--- \n"
                        "*Notice: This proposed response fell below the required accuracy threshold (0.75) "
                        "and has been logged to the HP Support Admin Queue for manual review.*"
                    )
                    st.error(final_output)
                    log_to_supabase_hitl(user_input, draft_answer, eval_data)
                else:
                    final_output = draft_answer
                    st.markdown(final_output)
                    st.caption(f"Verified Confidence: **{score:.2f}** | Groundedness: {eval_data['groundedness']:.2f} | Relevancy: {eval_data['relevancy']:.2f} | Compliance: {eval_data['compliance']:.2f}")

                with st.expander("View Retrieved HP Manual References"):
                    st.text(retrieved_context)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_output,
                    "eval": eval_data
                })

# ------------------------------------------
# TAB 2: INGEST MANUALS (DYNAMIC VECTOR DATABASE)
# ------------------------------------------
with tab_ingest:
    st.header("Technical Manual Ingestion Engine")
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
    st.markdown("Browse technical manuals and documentation currently archived in the storage bucket.")
    
    if not supabase:
        st.info("Supabase storage is not configured. Connect your credentials to view archived files.")
    else:
        if st.button("Refresh Storage List", key="refresh_db"):
            st.rerun()
            
        try:
            files = supabase.storage.from_("manuals").list()
            
            # Guard against None returned by empty bucket or RLS restrictions
            if files is None:
                files = []
            
            valid_files = []
            for f in files:
                name = f.get("name") if isinstance(f, dict) else getattr(f, "name", "")
                if name and name != ".emptyFolder":
                    valid_files.append(f)
            
            if valid_files:
                st.success(f"Found {len(valid_files)} documents in the 'manuals' repository:")
                for f in valid_files:
                    file_name = f.get("name") if isinstance(f, dict) else getattr(f, "name", "Unknown File")
                    created_at = f.get("created_at") if isinstance(f, dict) else getattr(f, "created_at", "")
                    
                    display_date = str(created_at)[:10] if created_at and len(str(created_at)) >= 10 else "N/A"
                    st.markdown(f"📄 **{file_name}** *(Archived: {display_date})*")
            else:
                st.info("No files currently stored in the Supabase 'manuals' bucket (or access is restricted by RLS).")
        except Exception as e:
            st.warning(f"Could not retrieve file list from Supabase Storage: {str(e)}")