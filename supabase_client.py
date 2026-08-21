import os
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


def get_supabase_client() -> Client:
    """Initializes and returns a Supabase client using publishable credentials."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")

    if not url or not key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_KEY environment variables."
        )

    return create_client(url, key)


def push_to_hitl_queue(
    session_id: str, query: str, response: str, score: float
) -> dict:
    """Writes a low-confidence intercept record to Supabase."""
    client = get_supabase_client()
    data = {
        "session_id": session_id,
        "original_query": query,
        "hallucinated_response": response,
        "confidence_score": score,
        "status": "PENDING",
    }
    res = client.table("hitl_review_queue").upsert(data).execute()
    return res.data


def get_pending_hitl_reviews() -> list[dict]:
    """Fetches all unresolved reviews for the Admin panel."""
    client = get_supabase_client()
    res = (
        client.table("hitl_review_queue")
        .select("*")
        .eq("status", "PENDING")
        .order("created_at", desc=True)
        .execute()
    )
    return res.data


def resolve_hitl_review(
    session_id: str, corrected_text: str, status: str = "EDITED"
) -> dict:
    """Updates record status when an admin approves/edits a response."""
    client = get_supabase_client()
    data = {
        "status": status,
        "corrected_response": corrected_text,
        "resolved_at": "now()",
    }
    res = (
        client.table("hitl_review_queue")
        .update(data)
        .eq("session_id", session_id)
        .execute()
    )
    return res.data


def check_session_status(session_id: str) -> dict | None:
    """Checks if an intercepted user session has been resolved by an admin."""
    client = get_supabase_client()
    res = (
        client.table("hitl_review_queue")
        .select("*")
        .eq("session_id", session_id)
        .execute()
    )
    if res.data:
        return res.data[0]
    return None