# ask_paper.py
# The "G" in RAG: retrieve relevant chunks, then have the LLM answer FROM them.

import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# Reuse the LLM tool you built in Phase 0.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scholiast.llm.local_llm import LocalLLM

# ---------- connect to the database we already built ----------
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection("paper")   # the 36 chunks are already in here
llm = LocalLLM()

def answer_question(question, n_chunks=3):
    # 1. RETRIEVE: find the most relevant chunks for this question.
    q_emb = model.encode([question])[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=n_chunks)
    retrieved_chunks = results["documents"][0]
    chunk_ids = results["ids"][0]

    # 2. BUILD THE PROMPT: stuff the chunks in, with strict instructions.
    context = "\n\n".join(
        f"[Passage {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    prompt = f"""You are a research assistant. Answer the question using ONLY the passages below.
If the passages do not contain the answer, say "The paper doesn't seem to cover that."
Cite which passage(s) you used, like [Passage 1].

PASSAGES FROM THE PAPER:
{context}

QUESTION: {question}

ANSWER:"""

    # 3. GENERATE: let the local model write the grounded answer.
    print("Thinking (retrieving + generating)...\n")
    answer = llm.ask(prompt)
    return answer, chunk_ids

if __name__ == "__main__":
    print("Ask a question about your paper. Type 'exit' to quit.\n")
    while True:
        q = input("you > ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        answer, ids = answer_question(q)
        print(f"\nscholiast > {answer}")
        print(f"\n(sources retrieved: {', '.join(ids)})\n")