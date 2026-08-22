import os
import json
import uuid
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from pinecone import Pinecone
from supabase import create_client, Client

# ==========================================
# 1. PAGE CONFIGURATION & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="HP Field Ops Copilot",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Welcome to HP Field Ops Copilot.** Describe the hardware issue, error code, or service procedure you need assistance with."
        }
    ]

# ==========================================
# 2. CLIENT & SECRETS MANAGEMENT
# ==========================================
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
PINECONE_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

# Validate API Keys
missing_keys = []
if not GEMINI_KEY: missing_keys.append("GEMINI_API_KEY")
if not PINECONE_KEY: missing_keys.append("PINECONE_API_KEY")
if not SUPABASE_URL: missing_keys.append("SUPABASE_URL")
if not SUPABASE_KEY: missing_keys.append("SUPABASE_KEY")

if missing_keys:
    st.sidebar.error(f"⚠️ Missing Secrets: {', '.join(missing_keys)}")

# Initialize Clients
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
pc = Pinecone(api_key=PINECONE_KEY) if PINECONE_KEY else None
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None

INDEX_NAME = "hp-manuals"

# ==========================================
# 3. STRUCTURED AI JUDGE SCHEMAS
# ==========================================
class EvaluationResult(BaseModel):
    groundedness: float = Field(..., description="Score 0.0 to 1.0: Is the draft response strictly grounded in retrieved HP context?")
    relevancy: float = Field(..., description="Score 0.0 to 1.0: Does the response directly address the technician's specific question?")
    compliance: float = Field(..., description="Score 0.0 to 1.0: Does the advice adhere to HP safety standards, part specs, and procedures?")
    reasoning: str = Field(..., description="Detailed technical rationale for the assigned scores.")

# ==========================================
# 4. RAG & AI JUDGE CORE PIPELINE
# ==========================================
def contextualize_query(user_input: str) -> str:
    """Rewrites follow-up user questions into standalone queries using chat history."""
    if len(st.session_state.messages) <= 2:
        return user_input

    history_summary = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:-1]])
    
    prompt = f"""
    Given the following conversation history and a follow-up user question, rephrase the follow-up question to be a standalone technical query suitable for vector search in HP technical documentation.
    
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
        # Generate 768/1536d embeddings via text-embedding-004
        embed_resp = gemini_client.models.embed_content(
            model="text-embedding-004",
            contents=query_text
        )
        query_vector = embed_resp.embedding.values
        
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
            formatted_contexts.append(f"[Document {i} | Source: {source} (Page {page})]\n{text}")
            
        return "\n\n---\n\n".join(formatted_contexts)
    except Exception as e:
        return f"Pinecone Search Error: {str(e)}"

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
        
        # Weighted Score Computation
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
# 5. STREAMLIT UI LAYOUT
# ==========================================
st.sidebar.title("🛠️ HP Ops Panel")
st.sidebar.caption(f"Session ID: `{st.session_state.session_id}`")
st.sidebar.markdown("---")

# Intercept Threshold Slider
confidence_cutoff = st.sidebar.slider("HITL Intercept Cutoff", min_value=0.50, max_value=0.95, value=0.75, step=0.05)

if st.sidebar.button("Clear Chat Memory"):
    st.session_state.messages = [st.session_state.messages[0]]
    st.rerun()

# Main Application Tabs
tab_chat, tab_admin = st.tabs(["💬 Technician Copilot", "🛡️ Support Admin Portal"])

# ------------------------------------------
# TAB 1: TECHNICIAN COPILOT (CHAT)
# ------------------------------------------
with tab_chat:
    # Render Conversation Stream
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "eval" in msg:
                ev = msg["eval"]
                score = ev.get("composite_score", 0.0)
                if score < confidence_cutoff:
                    st.error(f"⚠️ **Intercepted (Score: {score:.2f} < {confidence_cutoff})** — Routed to Admin Queue")
                else:
                    st.caption(f"✅ Verified Confidence: **{score:.2f}** | Groundedness: {ev['groundedness']:.2f} | Relevancy: {ev['relevancy']:.2f} | Compliance: {ev['compliance']:.2f}")

    # User Input Field
    if user_input := st.chat_input("Enter error code, symptom, or part replacement query..."):
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process Copilot Response
        with st.chat_message("assistant"):
            with st.spinner("Processing manual vector search & running AI Judge evaluation..."):
                
                # Step A: Contextualize Query
                search_query = contextualize_query(user_input)
                
                # Step B: Vector Database Retrieval
                retrieved_context = query_pinecone_vector_db(search_query)
                
                # Step C: Generate Draft Answer
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

                # Step D: Run Deterministic AI Judge
                eval_data = run_ai_judge_evaluation(user_input, retrieved_context, draft_answer)
                score = eval_data.get("composite_score", 0.0)

                # Step E: Handle Intercept Cutoff (< 0.75)
                if score < confidence_cutoff:
                    final_output = (
                        f"⚠️ **Low Confidence Intercept Triggered (Score: {score:.2f} / 1.00)**\n\n"
                        f"{draft_answer}\n\n"
                        "--- \n"
                        "🛑 *Notice: This proposed response fell below the required accuracy threshold (0.75) "
                        "and has been logged to the HP Support Admin Queue for manual review.*"
                    )
                    st.error(final_output)
                    log_to_supabase_hitl(user_input, draft_answer, eval_data)
                else:
                    final_output = draft_answer
                    st.markdown(final_output)
                    st.caption(f"✅ Verified Confidence: **{score:.2f}** | Groundedness: {eval_data['groundedness']:.2f} | Relevancy: {eval_data['relevancy']:.2f} | Compliance: {eval_data['compliance']:.2f}")

                # Context Inspector Expander
                with st.expander("📄 View Retrieved HP Manual References"):
                    st.text(retrieved_context)

                # Save turn to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_output,
                    "eval": eval_data
                })

# ------------------------------------------
# TAB 2: SUPPORT ADMIN PORTAL
# ------------------------------------------
with tab_admin:
    st.header("Human-In-The-Loop (HITL) Review Queue")
    st.markdown("Review, edit, and approve technical queries flagged by the 3-Metric AI Judge.")
    
    if not supabase:
        st.warning("Supabase configuration missing or invalid.")
    else:
        col_btn, col_blank = st.columns([1, 4])
        if col_btn.button("🔄 Refresh Queue"):
            st.rerun()

        try:
            db_response = supabase.table("hitl_review_queue").select("*").eq("status", "PENDING").order("id", desc=True).execute()
            pending_records = db_response.data

            if not pending_records:
                st.success("🎉 All clear! No pending flagged items in the review queue.")
            else:
                st.info(f"📋 **{len(pending_records)}** pending queries require administrative approval:")

                for record in pending_records:
                    rec_id = record["id"]
                    with st.expander(f"⚠️ Query: \"{record['user_query']}\" | Score: {record['confidence_score']:.2f}"):
                        st.markdown(f"**Flagged Draft Advice:**\n{record['draft_response']}")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Groundedness", f"{record.get('groundedness_score', 0):.2f}")
                        m2.metric("Relevancy", f"{record.get('relevancy_score', 0):.2f}")
                        m3.metric("Compliance", f"{record.get('compliance_score', 0):.2f}")
                        
                        if record.get("reasoning"):
                            st.caption(f"**AI Judge Rationale:** {record['reasoning']}")

                        # Admin Correction Text Area
                        corrected_text = st.text_area(
                            "Edit Response for Knowledge Base:",
                            value=record['draft_response'],
                            key=f"edit_field_{rec_id}"
                        )

                        btn_col1, btn_col2 = st.columns(2)
                        
                        if btn_col1.button("✅ Approve & Publish Correction", key=f"btn_app_{rec_id}", type="primary"):
                            supabase.table("hitl_review_queue").update({
                                "status": "APPROVED",
                                "draft_response": corrected_text
                            }).eq("id", rec_id).execute()
                            st.success(f"Ticket #{rec_id} approved and updated!")
                            st.rerun()

                        if btn_col2.button("❌ Reject Ticket", key=f"btn_rej_{rec_id}"):
                            supabase.table("hitl_review_queue").update({
                                "status": "REJECTED"
                            }).eq("id", rec_id).execute()
                            st.warning(f"Ticket #{rec_id} marked as rejected.")
                            st.rerun()

        except Exception as err:
            st.error(f"Error connecting to Supabase table: {str(err)}")