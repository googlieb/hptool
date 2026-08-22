import os
import json
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from pinecone import Pinecone
from supabase import create_client, Client

# ==========================================
# 1. INITIALIZE CLIENTS & SECRETS
# ==========================================
st.set_page_config(
    page_title="HP Field Ops Copilot",
    page_icon="🔧",
    layout="wide"
)

# Fetch secrets from st.secrets or local environment
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
PINECONE_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

# Initialize SDK Clients
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
pc = Pinecone(api_key=PINECONE_KEY) if PINECONE_KEY else None
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if (SUPABASE_URL and SUPABASE_KEY) else None

INDEX_NAME = "hp-manuals"


# ==========================================
# 2. STRUCTURED AI JUDGE SCHEMA
# ==========================================
class EvaluationResult(BaseModel):
    groundedness: float = Field(..., description="Score 0.0 to 1.0 indicating if answer is strictly grounded in retrieved context.")
    relevancy: float = Field(..., description="Score 0.0 to 1.0 indicating if response answers the user's explicit question.")
    compliance: float = Field(..., description="Score 0.0 to 1.0 indicating adherence to HP safety standards and formatting.")
    reasoning: str = Field(..., description="Brief summary explaining the rationale behind scores.")


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def query_pinecone(query_text: str, top_k: int = 3) -> str:
    """Retrieves relevant text snippets from Pinecone index using Gemini embeddings."""
    if not pc or not gemini_client:
        return "Pinecone or Gemini API key missing. Unable to perform vector retrieval."
    
    try:
        # Generate query embedding using Gemini
        embed_resp = gemini_client.models.embed_content(
            model="text-embedding-004",
            contents=query_text
        )
        query_vector = embed_resp.embedding.values
        
        index = pc.Index(INDEX_NAME)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        
        contexts = [match["metadata"].get("text", "") for match in results.get("matches", []) if "metadata" in match]
        return "\n\n".join(contexts) if contexts else "No relevant HP technical context found in vector database."
    except Exception as e:
        return f"Retrieval Error: {str(e)}"


def evaluate_response(query: str, context: str, draft_response: str) -> dict:
    """Evaluates draft response against Groundedness, Relevancy, and Compliance."""
    judge_prompt = f"""
    You are an expert AI Safety Judge for HP Field Operations.
    Evaluate the proposed technical response strictly based on the provided Context and Query.

    User Query: {query}
    Retrieved Context: {context}
    Proposed Response: {draft_response}

    Provide precise scores between 0.0 and 1.0 for groundedness, relevancy, and compliance.
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
        
        result = json.loads(response.text)
        # Weighted Composite Score Calculation
        composite = round(
            (result["groundedness"] * 0.4) + (result["relevancy"] * 0.3) + (result["compliance"] * 0.3),
            2
        )
        result["composite_score"] = composite
        return result
    except Exception as e:
        # Fallback evaluation if judge fails
        return {
            "groundedness": 0.0, "relevancy": 0.0, "compliance": 0.0,
            "composite_score": 0.0, "reasoning": f"Evaluation error: {str(e)}"
        }


def push_to_supabase_hitl(query: str, draft_response: str, eval_data: dict):
    """Pushes low-confidence responses (< 0.75) to Supabase hitl_review_queue."""
    if not supabase:
        st.warning("Supabase connection missing. Record could not be saved to queue.")
        return
    
    record = {
        "user_query": query,
        "draft_response": draft_response,
        "confidence_score": eval_data.get("composite_score", 0.0),
        "groundedness_score": eval_data.get("groundedness", 0.0),
        "relevancy_score": eval_data.get("relevancy", 0.0),
        "compliance_score": eval_data.get("compliance", 0.0),
        "status": "PENDING"
    }
    
    try:
        supabase.table("hitl_review_queue").insert(record).execute()
    except Exception as e:
        st.error(f"Failed to record HITL event in Supabase: {str(e)}")


# ==========================================
# 4. STREAMLIT APPLICATION INTERFACE
# ==========================================
st.title("🔧 HP Field Ops Copilot v2.1")
st.markdown("---")

# Navigation Tabs
tab_tech, tab_admin = st.tabs(["👨‍🔧 Technician Copilot", "🛡️ Support Admin Portal"])

# ------------------------------------------
# TAB 1: TECHNICIAN COPILOT
# ------------------------------------------
with tab_tech:
    st.header("Search Technical Manuals")
    
    query = st.text_input("Describe the issue or error code:", placeholder="e.g., Error 13.00.00 Jam in Top Cover")
    
    if st.button("Run Diagnostic Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a technical query.")
        else:
            with st.spinner("Searching HP Technical Database & Evaluating..."):
                # 1. Retrieve Context
                context = query_pinecone(query)
                
                # 2. Generate Initial Response
                gen_prompt = f"""
                You are the HP Field Ops Copilot. Answer the technician's question using ONLY the provided context.
                If the context does not contain enough info, state clearly that official documentation is missing.

                Context: {context}
                Query: {query}
                """
                
                draft_resp = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=gen_prompt
                ).text
                
                # 3. AI Judge Evaluation
                eval_data = evaluate_response(query, context, draft_resp)
                score = eval_data["composite_score"]
                
                # Display Retreived Context (Collapsible)
                with st.expander("📄 View Retrieved Context (Pinecone)"):
                    st.write(context)
                
                # 4. Intercept Check (< 0.75 Cutoff)
                if score < 0.75:
                    st.error("⚠️ **Low Confidence Intercept Triggered**")
                    st.warning(
                        f"This answer received a confidence score of **{score:.2f} / 1.00** (Cutoff: 0.75).\n\n"
                        "To prevent field errors, this query has been automatically routed to the "
                        "**HP Support Admin Queue** for human verification."
                    )
                    
                    # Display breakdown
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Groundedness", f"{eval_data['groundedness']:.2f}")
                    c2.metric("Relevancy", f"{eval_data['relevancy']:.2f}")
                    c3.metric("Compliance", f"{eval_data['compliance']:.2f}")
                    
                    st.caption(f"**Judge Rationale:** {eval_data['reasoning']}")
                    
                    # Push to Supabase HITL Queue
                    push_to_supabase_hitl(query, draft_resp, eval_data)
                
                else:
                    st.success(f"✅ **Verified Solution (Confidence: {score:.2f})**")
                    st.markdown(draft_resp)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Groundedness", f"{eval_data['groundedness']:.2f}")
                    c2.metric("Relevancy", f"{eval_data['relevancy']:.2f}")
                    c3.metric("Compliance", f"{eval_data['compliance']:.2f}")

# ------------------------------------------
# TAB 2: SUPPORT ADMIN PORTAL
# ------------------------------------------
with tab_admin:
    st.header("Human-In-The-Loop (HITL) Review Queue")
    
    if not supabase:
        st.warning("Supabase client is not connected.")
    else:
        if st.button("Refresh Queue"):
            st.rerun()
            
        try:
            # Query pending items from Supabase
            response = supabase.table("hitl_review_queue").select("*").eq("status", "PENDING").execute()
            records = response.data
            
            if not records:
                st.info("🎉 No pending reviews in the queue! All systems operational.")
            else:
                st.write(f"**{len(records)}** items requiring human review:")
                
                for item in records:
                    with st.expander(f"⚠️ Query: {item['user_query']} (Score: {item['confidence_score']:.2f})"):
                        st.write(f"**Flagged Draft Response:**\n{item['draft_response']}")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.write(f"**Groundedness:** {item['groundedness_score']}")
                        col2.write(f"**Relevancy:** {item['relevancy_score']}")
                        col3.write(f"**Compliance:** {item['compliance_score']}")
                        
                        edited_resp = st.text_area(
                            "Corrected Response (Admin Edit):",
                            value=item['draft_response'],
                            key=f"edit_{item['id']}"
                        )
                        
                        c_approve, c_reject = st.columns(2)
                        
                        if c_approve.button("Approve & Resolve", key=f"app_{item['id']}"):
                            supabase.table("hitl_review_queue").update({
                                "status": "APPROVED",
                                "draft_response": edited_resp
                            }).eq("id", item['id']).execute()
                            st.success("Record approved and updated!")
                            st.rerun()
                            
                        if c_reject.button("Reject Ticket", key=f"rej_{item['id']}"):
                            supabase.table("hitl_review_queue").update({
                                "status": "REJECTED"
                            }).eq("id", item['id']).execute()
                            st.warning("Ticket rejected.")
                            st.rerun()
                            
        except Exception as e:
            st.error(f"Error fetching queue from Supabase: {str(e)}")