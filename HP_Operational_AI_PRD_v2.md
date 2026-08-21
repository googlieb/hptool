# Product Requirements Document (PRD)
## Project: Field Operations Copilot for HP Print Supplies
### Version: 2.1 (Production-Ready HITL Integration)
### Status: Final Specification

---

## 1. Executive Summary & Strategy
The Field Operations Copilot is an operational enterprise tool designed for the HP Print Supplies business unit. It automates the extraction, processing, and synthesis of highly critical operational datasets—specifically technical hardware manuals and complex global pricing sheets. 

Because errors in this domain present severe financial, legal, and operational risks, this software replaces speculative AI outputs with a rigid enterprise architecture. The system combines a RAG 2.0 framework, real-time AI evaluation, strict behavioral guardrails, an active Human-in-the-Loop (HITL) interception engine, and structural business value tracking.

---

## 2. System Architecture Overview
The application consists of a decoupled architecture optimized for zero-data leakage, sub-second tracking, and robust data isolation:
*   **Data Ingestion Pipeline:** A secure local PDF parsing interface that chunks document text, extracts metadata, and generates structured contextual payloads.
*   **Vector Engine:** A managed Pinecone Vector Database pre-populated with indexed HP Laser Jet Enterprise and InkJet telemetry and maintenance documentation.
*   **Execution Layer:** A dual-model architecture utilizing an execution model for synthesis and a secondary, independent LLM serving as an automated "AI Judge."
*   **Stateful Orchestration Engine:** A backend state machine controlling user sessions and executing the Human-in-the-Loop pipeline.

---

## 3. Detailed Functional Requirements

### 3.1 RAG 2.0 Core Infrastructure
*   **Vector Database Optimization:** The system must connect securely to a designated Pinecone index utilizing a hierarchical namespace approach (e.g., `namespace="technical_manuals"` vs. `namespace="pricing_sheets"`).
*   **Document Parsing & Chunking:**
    *   The frontend ingestion pipeline must support multi-page PDF files up to 25MB via a drag-and-drop UI component.
    *   Text extraction must utilize a reliable parsing engine capable of isolating tabular data without breaking cell continuity.
    *   Documents must be chunked using an overlapping sliding window technique (e.g., chunk size of 512 tokens with a 10% overlap).
    *   Every generated chunk must explicitly bind metadata parameters in its vector payload: `{ "source_doc": "string", "page_number": int, "section_header": "string", "classification": "internal_only" }`.
*   **Hybrid Search Execution:** Queries must run a dense semantic search paired with keywords matching to ensure exact parts numbers, error codes, and SKUs are not missed by the embedding model.

### 3.2 AI Judge & Output Evaluation Framework
*   **Asynchronous Processing Flow:** Immediately upon receiving retrieved context from Pinecone, the generation engine passes the raw user query, the context, and the draft response payload to a secondary, logically isolated model acting as the AI Judge.
*   **Real-Time Confidence Scoring:** The AI Judge scores the draft response across three metrics using deterministic, prompt-enforced JSON schema scoring outputs:
    1.  **Context Groundedness (0.00 - 1.00):** Is every factual assertion directly verifiable in the retrieved chunks?
    2.  **Context Relevancy (0.00 - 1.00):** Does the chunk data answer the specific question asked by the user?
    3.  **Strict Compliance Check (Boolean):** Does the answer contain forbidden speculative language or leak structural IP?
*   **The Threshold Rule:** The mathematical average of Groundedness and Relevancy dictates session routing.
    *   **Score >= 0.75 AND Compliance = True:** The system marks the response as `SAFE_DELIVERY` and routes it to the user.
    *   **Score < 0.75 OR Compliance = False:** The system immediately aborts delivery and initiates the `HITL_INTERCEPT` state machine.

### 3.3 Active Human-in-the-Loop (HITL) Architecture
To ensure operational security, low-confidence answers are not mocked; they are handled by an active transactional state manager.

```
[User Query] ──> [RAG Pipeline] ──> [AI Judge Assessment]
                                              │
                      ┌───────────────────────┴───────────────────────┐
             Score >= 75% (Confident)                         Score < 75% (Low Confidence)
                      │                                               │
           [Deliver Response to User]                        [Intercept Response]
                                                                      │
                                                           [Freeze User UI Chat]
                                                                      │
                                                       [Write to Queue DB: Status: PENDING]
                                                                      │
                                                       [Expose to Internal Admin Portal]
                                                                      │
                                                       [Admin Edits/Approves Response]
                                                                      │
                                                    [Write New Data to Pinecone Context]
                                                                      │
                                                          [Unfreeze & Deliver to User]
```

#### 3.3.1 Relational Database Schema (HITL Queue Management)
The application backend must initialize and maintain a persistent database table named `hitl_review_queue`.
```sql
CREATE TABLE hitl_review_queue (
    session_id VARCHAR(255) PRIMARY KEY,
    original_query TEXT NOT NULL,
    hallucinated_response TEXT NOT NULL,
    confidence_score NUMERIC(3,2) NOT NULL,
    status VARCHAR(50) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'APPROVED', 'EDITED')),
    corrected_response TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);
```

#### 3.3.2 Backend API Endpoint Specifications

##### Endpoint 1: Intercept Submission
*   **Path:** `POST /api/hitl/intercept`
*   **Request Payload Schema:**
    ```json
    {
      "session_id": "usr_sess_9081234",
      "original_query": "What is the certified optimal print temperature for toner batch HP-99X?",
      "hallucinated_response": "The standard print temperature is 185C but it could fluctuate depending on room ambient conditions up to 210C.",
      "confidence_score": 0.58
    }
    ```
*   **Execution Logic:**
    1.  Writes record into `hitl_review_queue` with `status = 'PENDING'`.
    2.  Locks user-facing websocket state.
    3.  Broadcasts a real-time event via Server-Sent Events (SSE) or WebSockets to the active Admin Portal.
*   **Response Payload:** `{ "status": "intercepted", "queue_id": "usr_sess_9081234" }` (HTTP 201 Created).

##### Endpoint 2: Administrative Resolution
*   **Path:** `POST /api/hitl/resolve`
*   **Request Payload Schema:**
    ```json
    {
      "session_id": "usr_sess_9081234",
      "status": "EDITED",
      "corrected_response": "Per technical manual LaserJet-E3, Section 4.2, the certified optimal fusing roller surface temperature for HP-99X toner is exactly 192C. Do not exceed 195C."
    }
    ```
*   **Execution Logic:**
    1.  Updates database row: sets `status = 'EDITED'`, `corrected_response`, and `resolved_at = NOW()`.
    2.  Triggers background worker task for Pinecone injection.
    3.  Pushes a WebSocket update payload containing the `corrected_response` directly to the active user session.
*   **Response Payload:** `{ "status": "resolved", "session_id": "usr_sess_9081234" }` (HTTP 200 OK).

#### 3.3.3 Continuous Learning Loop (Pinecone Feedback Injection)
*   When `POST /api/hitl/resolve` completes, the backend passes the `original_query` and the `corrected_response` as a combined text block to the embedding engine pipeline (`text-embedding-3-small` or equivalent).
*   The system generates a new multi-dimensional vector array.
*   The backend builds a metadata update payload containing tracking properties:
    ```json
    {
      "source": "human_override",
      "verified": "true",
      "original_session_link": "usr_sess_9081234",
      "context_injection_date": "2026-08-20"
    }
    ```
*   The vector array and metadata are injected back into the Pinecone index via an `upsert` call. This overrides future low-confidence retrievals by serving verified human answers directly when a user asks a similar query.

#### 3.3.4 User Interface and Administrative Interface Requirements
*   **User Interface (Chat Screen):**
    *   The chat submit button and text-input box must immediately switch to a disabled, semi-opaque visual style when an intercept is active.
    *   An amber status warning panel must overlay the message stream displaying the text: `"Verifying technical data with Print Operations Support..."`.
    *   When the WebSocket receives the resolve message, the state transitions: the warning panel is removed, the final verified response is rendered in the chat stream, and input controls are unlocked.
*   **Administrative Portal Screen:**
    *   A secure layout displaying an active tracking table of all records where `status == 'PENDING'`.
    *   Selecting a record renders a side-by-side view highlighting: the user's question, the context snippet retrieved, the failing response draft, and the AI Judge's reason for low confidence.
    *   The center panel contains an editable Markdown text box populated with the original draft.
    *   Two prominent action triggers govern the form: `"Approve Draft As-Is"` and `"Submit Custom Override & Train System"`.

---

## 4. UI Trust Features & Operations Tracking

### 4.1 "Cite the Page" Visual Sourcing
*   Every response must isolate the vector chunk metadata values (`source_doc` and `page_number`).
*   The UI must render these source components as discrete, clickable document tags placed neatly below the response content box (e.g., `[HP_LaserJet_Enterprise_Manual.pdf - Page 42]`).

### 4.2 Financial and Operational Performance Footers
To explicitly prove the operational return on investment, the UI footers must compute and render live metrics for each interaction session:
*   **Latency Capture:** The client must display the total execution processing window from user submit to final token output stream completion (e.g., `Latency: 1.42s`).
*   **API Spend Calculation:** The backend passes exact model token counters upon response closing. The UI multiplies token inputs and outputs against exact pricing tables to show precise transaction costs (e.g., `Est. Cost: $0.0031`).

---

## 5. Non-Functional Requirements & Security
*   **Data Residency & Privacy:** All underlying text pipelines, caching files, and SQLite queue DB instances must reside within local file partitions. The software must not leak context pipelines outside authorized API bounds.
*   **Graceful Degradaion:** If the primary remote services drop connections, the local state engine must halt operations safely and display a system error message without breaking application storage loops.
