# chunk_text.py
# Step 2 of RAG: chop the paper's text into overlapping bite-sized pieces.

import fitz
from pathlib import Path

# --- (same PDF reading as before) ---
papers_folder = Path("data/papers")
pdf_files = list(papers_folder.glob("*.pdf"))
if not pdf_files:
    print("No PDF found in data/papers/.")
    raise SystemExit

pdf_path = pdf_files[0]
doc = fitz.open(pdf_path)
full_text = ""
for page in doc:
    full_text += page.get_text()
doc.close()

print(f"Read '{pdf_path.name}' — {len(full_text)} characters total.\n")

# --- the new part: chunking ---

CHUNK_SIZE = 800      # characters per chunk (~a paragraph or two)
CHUNK_OVERLAP = 150   # characters shared with the next chunk (the "roof tiles")

def chunk_text(text, chunk_size, overlap):
    """Slice text into overlapping windows of `chunk_size`,
    stepping forward by (chunk_size - overlap) each time."""
    chunks = []
    start = 0
    step = chunk_size - overlap        # how far we advance each time
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end]) # grab this window of characters
        start += step                  # move forward, leaving an overlap
    return chunks

chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

print(f"Split into {len(chunks)} chunks "
      f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).\n")

# Show the first two chunks so we can SEE the overlap with our own eyes.
print("----- CHUNK 0 -----")
print(chunks[0])
print("\n----- CHUNK 1 -----")
print(chunks[1])
print("\n(Notice: the end of chunk 0 reappears at the start of chunk 1 — that's the overlap.)")