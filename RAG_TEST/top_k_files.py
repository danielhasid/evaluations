

import os, pickle
from typing import List, Tuple

import fitz                   # PyMuPDF  →  pip install pymupdf
import numpy as np
import faiss                  # pip install faiss-cpu
import openai                 # pip install openai
from dotenv import load_dotenv

load_dotenv()
DOCS_DIR        = "Documents"
DOC_INDEX_FILE  = "doc_index.faiss"
DOC_META_FILE   = "doc_meta.pkl"
EMBED_MODEL     = "text-embedding-3-large"   # adjust if needed
CHUNK_SIZE      = 2000                       # characters per chunk (~500 tokens)
CHUNK_OVERLAP   = 200                        # overlap between chunks
MAX_TOKENS_PER_BATCH = 7500                  # total tokens per API batch (API max is 8192)


# ── helpers ──────────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text("text"))
    return "\n".join(text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks to stay within token limits."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk of text
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        
        # Try to break at sentence boundary if not at the very end of text
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            # Only use break point if it's at least halfway through the chunk
            if break_point > chunk_size // 2:
                end = start + break_point + 1
                chunk = chunk[:break_point + 1]
        
        # Add non-empty chunks
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        
        # Move to next chunk with overlap
        next_start = end - overlap
        
        # Ensure we're making progress (avoid infinite loop)
        if next_start <= start:
            next_start = start + 1
        
        start = next_start
    
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    client = openai.OpenAI()                   # requires OPENAI_API_KEY env-var
    
    out = []
    batch = []
    batch_tokens = 0
    
    for idx, text in enumerate(texts):
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = len(text) // 4
        
        # Check if individual text is too large
        if estimated_tokens > MAX_TOKENS_PER_BATCH:
            raise ValueError(
                f"❌ Chunk {idx} is TOO LARGE: ~{estimated_tokens} tokens "
                f"(max {MAX_TOKENS_PER_BATCH})\n"
                f"   Text length: {len(text):,} chars\n"
                f"   Preview: {text[:500]}...\n"
                f"   SOLUTION: Reduce CHUNK_SIZE in the config"
            )
        
        # If adding this text would exceed the batch limit, process current batch first
        if batch and (batch_tokens + estimated_tokens > MAX_TOKENS_PER_BATCH):
            print(f"Processing batch with {len(batch)} texts (~{batch_tokens} tokens)...")
            try:
                resp = client.embeddings.create(input=batch, model=EMBED_MODEL)
                out.extend([d.embedding for d in resp.data])
            except openai.BadRequestError as e:
                print(f"Error in batch: {[len(t) for t in batch]} chars")
                raise
            batch = []
            batch_tokens = 0
        
        batch.append(text)
        batch_tokens += estimated_tokens
    
    # Process final batch
    if batch:
        print(f"Processing final batch with {len(batch)} texts (~{batch_tokens} tokens)...")
        try:
            resp = client.embeddings.create(input=batch, model=EMBED_MODEL)
            out.extend([d.embedding for d in resp.data])
        except openai.BadRequestError as e:
            print(f"Error in final batch: {[len(t) for t in batch]} chars")
            raise
    
    vecs = np.array(out, dtype="float32")
    faiss.normalize_L2(vecs)                   # cosine → inner product
    return vecs


# ── index build / load ───────────────────────────────────────────────────
def build_doc_index(folder: str = DOCS_DIR) -> None:
    chunks, metadata = [], []
    doc_count = 0
    
    for fn in os.listdir(folder):
        fp = os.path.join(folder, fn)
        if fn.lower().endswith(".pdf"):
            print(f"Reading PDF: {fn}...")
            text = extract_text_from_pdf(fp)
        elif fn.lower().endswith(".txt"):
            print(f"Reading TXT: {fn}...")
            text = open(fp, encoding="utf-8").read()
        else:
            continue
        
        doc_count += 1
        print(f"  → Text length: {len(text):,} chars")
        
        # Split into chunks
        doc_chunks = chunk_text(text)
        print(f"  → Split into {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)
        # Store metadata: (filename, chunk_index, total_chunks)
        metadata.extend([(fn, i, len(doc_chunks)) for i in range(len(doc_chunks))])

    if not chunks:
        raise ValueError("No PDFs/TXTs found to index.")

    print(f"\n📊 Total: {doc_count} documents → {len(chunks)} chunks")
    print(f"Embedding {len(chunks)} chunks...")
    vecs = embed_texts(chunks)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, DOC_INDEX_FILE)
    with open(DOC_META_FILE, "wb") as f:
        pickle.dump(metadata, f)
    print("✅ chunk-level FAISS index saved.")


def load_doc_index():
    if not os.path.exists(DOC_INDEX_FILE):
        build_doc_index()                # auto-build on first use
    index = faiss.read_index(DOC_INDEX_FILE)
    metadata = pickle.load(open(DOC_META_FILE, "rb"))
    return index, metadata


# ── public API ───────────────────────────────────────────────────────────
def top_k_docs(query: str, k: int = 5) -> List[Tuple[str, int, int, float]]:
    """
    Returns top-k most relevant chunks.
    Returns: List of (filename, chunk_index, total_chunks, similarity_score)
    """
    index, metadata = load_doc_index()
    qvec = embed_texts([query])
    D, I = index.search(qvec, k)
    results = []
    for rank, i in enumerate(I[0]):
        fname, chunk_idx, total_chunks = metadata[i]
        score = float(D[0][rank])
        results.append((fname, chunk_idx, total_chunks, score))
    return results


# ── demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    # Force rebuild if --rebuild flag is passed
    if "--rebuild" in sys.argv:
        print("🔄 Force rebuilding index...")
        if os.path.exists(DOC_INDEX_FILE):
            os.remove(DOC_INDEX_FILE)
        if os.path.exists(DOC_META_FILE):
            os.remove(DOC_META_FILE)
    
    for fname, chunk_idx, total_chunks, score in top_k_docs("wireless Galaxy S22", k=3):
        print(f"{fname} [chunk {chunk_idx+1}/{total_chunks}]  (score={score:.3f})")
