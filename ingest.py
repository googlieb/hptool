import os
import urllib.request
import pypdf
import chromadb
from chromadb.utils import embedding_functions
from google import genai

# ---------------------------------------------------------
# 1. SAMPLE REAL-WORLD HP DATASHEETS (DOWNLOAD ENGINE)
# ---------------------------------------------------------
# Real HP Enterprise LaserJet & Supplies Datasheets
HP_DOC_URLS = {
    "LaserJet_M507_Datasheet.pdf": "https://www8.hp.com/h20195/v2/GetDocument.aspx?docname=4AA7-4581ENUC",
    "HP_89_Toner_Datasheet.pdf": "https://www8.hp.com/h20195/v2/GetDocument.aspx?docname=c06282834"
}

PDF_DIR = "./hp_downloads"
DB_DIR = "./hp_vector_db"

def download_hp_docs():
    """Downloads official HP PDFs from hp.com if not already local."""
    os.makedirs(PDF_DIR, exist_ok=True)
    downloaded_files = []
    
    print("📥 Starting HP Documentation Ingestion Pipeline...")
    for filename, url in HP_DOC_URLS.items():
        filepath = os.path.join(PDF_DIR, filename)
        if not os.path.exists(filepath):
            print(f"   Downloading: {filename} from HP...")
            try:
                # User-Agent header required by HP CDN
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"   ✅ Saved: {filepath}")
            except Exception as e:
                print(f"   ⚠️ Failed downloading {filename}: {e}")
        else:
            print(f"   ℹ️ Exists locally: {filepath}")
        
        if os.path.exists(filepath):
            downloaded_files.append((filename, filepath))
            
    return downloaded_files

def extract_and_chunk_pdf(filepath: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Extracts raw text from PDF and splits into overlapping semantic chunks."""
    reader = pypdf.PdfReader(filepath)
    chunks = []
    filename = os.path.basename(filepath)
    
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {page_num + 1} ---\n" + text

    # Simple sliding window chunker
    words = full_text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "source": filename,
            "chunk_id": f"{filename}_chunk_{i}"
        })
        
    return chunks

def build_vector_database(api_key: str):
    """Indexes extracted text chunks into persistent ChromaDB using Google Embeddings."""
    files = download_hp_docs()
    
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Reset/get collection
    try:
        client.delete_collection("hp_supplies_rag")
    except Exception:
        pass
        
    collection = client.create_collection(
        name="hp_supplies_rag",
        metadata={"hnsw:space": "cosine"}
    )
    
    genai_client = genai.Client(api_key=api_key)
    
    print("\n⚡ Extracting Text and Generating Gemini Vector Embeddings...")
    
    total_chunks = 0
    for filename, filepath in files:
        chunks = extract_and_chunk_pdf(filepath)
        print(f"   Processing '{filename}': {len(chunks)} chunks created.")
        
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        for c in chunks:
            # Generate embedding vector using Gemini API
            emb_res = genai_client.models.embed_content(
                model="text-embedding-004",
                contents=c["text"]
            )
            
            documents.append(c["text"])
            metadatas.append({"source": c["source"]})
            ids.append(c["chunk_id"])
            embeddings.append(emb_res.embedding.values)
            
        if documents:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            total_chunks += len(documents)

    print(f"\n✅ SUCCESS: Vector Database populated at '{DB_DIR}' with {total_chunks} chunk embeddings.")

if __name__ == "__main__":
    key = input("Enter your Gemini API Key to run indexing: ").strip()
    if key:
        build_vector_database(key)
    else:
        print("API Key required.")