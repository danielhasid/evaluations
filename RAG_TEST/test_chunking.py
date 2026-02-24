"""
Simple test to verify chunking and batching logic works correctly
"""
import os
import fitz  # PyMuPDF

DOCS_DIR = r"c:\work\PythonProject\RAG_TEST\Documents"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MAX_TOKENS_PER_BATCH = 7500

def extract_text_from_pdf(path: str) -> str:
    out = []
    with fitz.open(path) as doc:
        for page in doc:
            out.append(page.get_text("text"))
    return "\n".join(out)

def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
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

def simulate_batching(chunks):
    """Simulate how batching would work"""
    batches = []
    batch = []
    batch_tokens = 0
    
    for text in chunks:
        estimated_tokens = len(text) // 4
        
        if batch and (batch_tokens + estimated_tokens > MAX_TOKENS_PER_BATCH):
            batches.append((len(batch), batch_tokens))
            batch = []
            batch_tokens = 0
        
        batch.append(text)
        batch_tokens += estimated_tokens
    
    if batch:
        batches.append((len(batch), batch_tokens))
    
    return batches

def main():
    print("\n" + "="*80)
    print("CHUNKING & BATCHING TEST")
    print("="*80)
    
    all_chunks = []
    
    for fn in os.listdir(DOCS_DIR):
        if not fn.lower().endswith(('.pdf', '.txt')):
            continue
        
        fp = os.path.join(DOCS_DIR, fn)
        
        if fn.lower().endswith('.pdf'):
            text = extract_text_from_pdf(fp)
        else:
            with open(fp, encoding='utf-8') as f:
                text = f.read()
        
        chunks = split_text(text)
        all_chunks.extend(chunks)
        
        print(f"\n[File] {fn}")
        print(f"  Text: {len(text):,} chars (~{len(text)//4:,} tokens)")
        print(f"  Chunks: {len(chunks)}")
    
    print(f"\n" + "="*80)
    print(f"BATCHING SIMULATION")
    print("="*80)
    print(f"Total chunks to embed: {len(all_chunks)}")
    
    batches = simulate_batching(all_chunks)
    
    print(f"\nNumber of API batches needed: {len(batches)}")
    print(f"\nBatch details:")
    for i, (num_texts, tokens) in enumerate(batches, 1):
        print(f"  Batch {i}: {num_texts} chunks, ~{tokens:,} tokens")
        if tokens > 8000:
            print(f"    [ERROR] Exceeds 8,192 limit!")
        else:
            print(f"    [OK] Within limit")
    
    print(f"\n" + "="*80)
    print("RESULT: All batches should show [OK] - no errors expected!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
