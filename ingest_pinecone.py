import glob
import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pypdf import PdfReader

load_dotenv()

SOURCE_DIR = "./pdf_sources"
INDEX_NAME = "hp-supplies-rag"
BATCH_SIZE = 50  # Increased batch size to minimize total API requests


def extract_model_name_with_gemini(
    genai_client: genai.Client, sample_text: str
) -> str:
    """Uses Gemini to identify the primary HP printer model or series mentioned in the header/first page."""
    prompt = (
        "Extract only the primary HP printer model, series, or supply component name "
        "from the following text. Be concise (e.g., 'HP LaserJet Enterprise M607', 'HP Color LaserJet Pro M454'). "
        "Do not include conversational fluff. If no model is found, return 'Generic HP Manual'.\n\n"
        f"TEXT SAMPLE:\n{sample_text[:1500]}"
    )
    try:
        response = genai_client.models.generate_content(
            model="gemini-3.6-flash", contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"   ⚠️ Model extraction skipped: {e}")
        return "Generic HP Manual"


def read_pdf_text(pdf_path: str) -> list[str]:
    """Reads PDF page text using lightweight pypdf."""
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return pages_text

def embed_batch_with_retry(
    genai_client: genai.Client, text_batch: list[str], max_retries: int = 5
) -> list[list[float]]:
    """Embeds a batch of texts with automatic pause and resume on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            emb_res = genai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=text_batch,
                config=types.EmbedContentConfig(output_dimensionality=768),
            )
            return [e.values for e in emb_res.embeddings]
        except (errors.APIError, errors.ClientError) as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                match = re.search(r"retry in (\d+)", str(e))
                wait_time = int(match.group(1)) + 2 if match else 60
                print(
                    f"   ⏳ Quota limit hit. Auto-pausing for {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Failed to embed batch after maximum retries.")

def run_pinecone_ingestion():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not gemini_api_key or not pinecone_api_key:
        raise ValueError("Missing API Keys in environment.")

    pdf_files = glob.glob(f"{SOURCE_DIR}/*.pdf")
    if not pdf_files:
        print(f"⚠️ No PDF files found in '{SOURCE_DIR}'.")
        return

    print(
        f"📄 Found {len(pdf_files)} PDF(s). Initializing Clients (Batch Size: {BATCH_SIZE})..."
    )
    genai_client = genai.Client(
    api_key=gemini_api_key,
    http_options={'api_version': 'v1beta'}
)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n⚙️ Processing: {filename}")

        pages = read_pdf_text(pdf_path)
        if not pages:
            print(f"   Skipping unreadable/empty PDF: {filename}")
            continue

        # Extract Model Context from Page 1
        detected_model = extract_model_name_with_gemini(genai_client, pages[0])
        print(f"   📌 Detected Model Context: '{detected_model}'")

        # Combine pages and split into chunks
        full_doc_text = "\n\n".join(pages)
        raw_chunks = text_splitter.split_text(full_doc_text)
        print(f"   Splitting into {len(raw_chunks)} chunks...")

        vectors_to_upsert = []

        # Process chunks in large batches
        for i in range(0, len(raw_chunks), BATCH_SIZE):
            chunk_batch = raw_chunks[i : i + BATCH_SIZE]

            formatted_batch = [
                f"PRINTER MODEL: {detected_model}\nSOURCE: {filename}\n\n{c}"
                for c in chunk_batch
            ]

            embeddings = embed_batch_with_retry(genai_client, formatted_batch)

            for idx_in_batch, vector_values in enumerate(embeddings):
                global_idx = i + idx_in_batch
                vectors_to_upsert.append({
                    "id": f"{filename}_chunk_{global_idx}",
                    "values": vector_values,
                    "metadata": {
                        "source": filename,
                        "model": detected_model,
                        "text": chunk_batch[idx_in_batch],
                    },
                })

            time.sleep(1.0)  # Gentle spacing between batch API calls

        # Upsert entire document vectors to Pinecone
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            print(
                f"   ✅ Successfully indexed {len(vectors_to_upsert)} chunks to Pinecone."
            )

    print(
        "\n🚀 Ingestion Complete! All PDFs processed and enriched with model metadata."
    )


if __name__ == "__main__":
    run_pinecone_ingestion()