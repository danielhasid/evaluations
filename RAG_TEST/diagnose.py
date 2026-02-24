"""
Diagnostic script to check document sizes and chunking
"""
import os
import fitz  # PyMuPDF

DOCS_DIR = r"C:\code\RAG_Test\RAG_TEST\Documents"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

def extract_text_from_pdf(path: str) -> str:
    out = []
    with fitz.open(path) as doc:
        for page in doc:
            out.append(page.get_text("text"))
    return "\n".join(out)

def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if len(text) <= size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > size // 2:
                end = start + break_point + 1
                chunk = chunk[:break_point + 1]
        
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        
        start = next_start
    
    return chunks

def main():
    print("=" * 80)
    print("DOCUMENT ANALYSIS")
    print("=" * 80)
    
    docs_dir = DOCS_DIR
    if not os.path.exists(docs_dir):
        print(f"\n[ERROR] Directory not found: {docs_dir}")
        print(f"\nChecking alternate location: c:\\work\\PythonProject\\RAG_TEST\\Documents")
        alt_dir = r"c:\work\PythonProject\RAG_TEST\Documents"
        if os.path.exists(alt_dir):
            docs_dir = alt_dir
            print(f"[OK] Found documents at: {docs_dir}")
        else:
            print(f"[ERROR] Documents not found there either")
            return
    
    files = os.listdir(docs_dir)
    pdf_txt_files = [f for f in files if f.lower().endswith(('.pdf', '.txt'))]
    
    if not pdf_txt_files:
        print(f"\n[ERROR] No PDF or TXT files found in {docs_dir}")
        return
    
    print(f"\n[Folder] {docs_dir}")
    print(f"[Files] Found {len(pdf_txt_files)} documents\n")
    
    total_chunks = 0
    
    for fn in pdf_txt_files:
        fp = os.path.join(docs_dir, fn)
        
        # Read file
        if fn.lower().endswith('.pdf'):
            text = extract_text_from_pdf(fp)
        else:
            with open(fp, encoding='utf-8') as f:
                text = f.read()
        
        # Analyze
        text_len = len(text)
        est_tokens = text_len // 4
        chunks = split_text(text)
        num_chunks = len(chunks)
        total_chunks += num_chunks
        
        # Calculate tokens for each chunk
        chunk_sizes = [len(c) // 4 for c in chunks]
        max_chunk_tokens = max(chunk_sizes) if chunk_sizes else 0
        
        print(f"[File] {fn}")
        print(f"   Text length: {text_len:,} chars (~{est_tokens:,} tokens)")
        print(f"   Chunks: {num_chunks}")
        print(f"   Largest chunk: ~{max_chunk_tokens:,} tokens")
        
        if max_chunk_tokens > 8000:
            print(f"   [WARNING] Chunk exceeds 8,192 token limit!")
        
        # Simulate batching
        batch_tokens = sum(chunk_sizes)
        if batch_tokens > 8000:
            print(f"   [WARNING] If all chunks batched together = ~{batch_tokens:,} tokens (EXCEEDS LIMIT!)")
        
        print()
    
    print("=" * 80)
    print(f"SUMMARY:")
    print(f"  Total documents: {len(pdf_txt_files)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Estimated batches needed (at 7,500 tokens/batch): ~{(total_chunks * 500) // 7500 + 1}")
    print("=" * 80)

if __name__ == "__main__":
    main()
