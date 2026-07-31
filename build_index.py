# build_index.py
# Step 4 of RAG: embed all chunks, store them, and search by meaning.

import fitz
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ---------- 1. READ + CHUNK (same as before) ----------
papers_folder = Path("data/papers")
pdf_path = list(papers_folder.glob("*.pdf"))[0]

doc = fitz.open(pdf_path)
full_text = "".join(page.get_text() for page in doc)
doc.close()

CHUNK_SIZE, CHUNK_OVERLAP = 800, 150
def chunk_text(text, size, overlap):
    chunks, start, step = [], 0, size - overlap
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks

chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
print(f"Read '{pdf_path.name}', split into {len(chunks)} chunks.\n")

# ---------- 2. EMBED every chunk ----------
print("Loading embedding model and embedding all chunks...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(chunks)   # 36 chunks -> 36 sets of 384 numbers
print(f"Embedded {len(embeddings)} chunks.\n")

# ---------- 3. STORE in ChromaDB (on disk) ----------
# This creates/opens a database saved under data/chroma/ .
client = chromadb.PersistentClient(path="data/chroma")

# A "collection" is like one table/drawer in the cabinet.
# We delete-and-recreate so re-running starts clean while we experiment.
try:
    client.delete_collection("paper")
except Exception:
    pass
collection = client.create_collection("paper")

# Store each chunk: its text, its embedding, and a unique id.
collection.add(
    documents=chunks,
    embeddings=[e.tolist() for e in embeddings],
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)
print(f"Stored {collection.count()} chunks in the vector database.\n")

# ---------- 4. SEARCH by meaning ----------
question = "How accurate is the proposed detection method?"
print(f"QUESTION: {question}\n")

# Embed the question the SAME way, then ask Chroma for the 3 nearest chunks.
q_embedding = model.encode([question])[0].tolist()
results = collection.query(query_embeddings=[q_embedding], n_results=3)

print("----- TOP 3 MOST RELEVANT CHUNKS -----\n")
for rank, (doc_text, dist) in enumerate(
    zip(results["documents"][0], results["distances"][0]), start=1
):
    print(f"[Rank {rank}]  (distance: {dist:.3f})")
    print(doc_text[:300].strip(), "...\n")