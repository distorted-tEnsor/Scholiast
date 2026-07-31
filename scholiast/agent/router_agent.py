# scholiast/agent/router_agent.py
# The first real "agent": the model DECIDES what to do, then acts.

import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# Reach up to the project root so we can import our LLM tool.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scholiast.llm.local_llm import LocalLLM


class RouterAgent:
    """An agent that decides whether a question needs the papers,
    then either runs RAG or politely declines."""

    def __init__(self):
        self.llm = LocalLLM()
        self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        client = chromadb.PersistentClient(path="data/chroma")
        self.collection = client.get_collection("paper")

    # ---------- STEP 1: THE DECISION (this is the "agent" part) ----------
    def decide(self, question):
        """Ask the model to classify the question as SEARCH or DECLINE.
        The model's one-word answer becomes a decision our code acts on."""
        routing_prompt = f"""You are a router for a research-paper assistant.
Decide if the user's question should be answered using the research papers.

Reply with EXACTLY ONE WORD:
- SEARCH  -> if the question is about research, papers, methods, results, or their content
- DECLINE -> if it is general knowledge, chit-chat, or unrelated to the papers

Question: {question}

One word (SEARCH or DECLINE):"""
        decision = self.llm.ask(routing_prompt).strip().upper()
        # Be forgiving: the small model might add extra words, so we just
        # check which keyword appears rather than demand an exact match.
        if "SEARCH" in decision:
            return "SEARCH"
        return "DECLINE"

    # ---------- STEP 2a: THE "SEARCH" ACTION (your RAG pipeline) ----------
    def search_and_answer(self, question, n_chunks=5):
        q_emb = self.embed_model.encode([question])[0].tolist()
        results = self.collection.query(query_embeddings=[q_emb], n_results=n_chunks)
        chunks = results["documents"][0]
        ids = results["ids"][0]
        # --- DEBUG: show what was actually retrieved ---
        print("\n  --- retrieved chunks (debug) ---")
        for i, (c, cid) in enumerate(zip(chunks, ids), 1):
            print(f"  [{i}] {cid}: {c[:150].strip()}...")
        print("  --- end debug ---\n")

        context = "\n\n".join(f"[Passage {i+1}]\n{c}" for i, c in enumerate(chunks))
        prompt = f"""You are a careful research assistant. Answer the question using ONLY the passages below.

Follow these steps:
1. Find the exact sentence(s) in the passages that support your answer.
2. Quote that supporting sentence verbatim, in quotation marks, labeled with its passage number.
3. Then give your answer based ONLY on what you quoted.

Rules:
- If you cannot find a sentence that directly supports an answer, reply exactly: "The paper doesn't clearly state this."
- Do NOT use any outside knowledge. Do NOT guess numbers.
- If a table looks garbled or you cannot tell which number matches which item, say so instead of guessing.

PASSAGES:
{context}

QUESTION: {question}

Respond in this format:
EVIDENCE: "<exact quote>" [Passage N]
ANSWER: <your answer, or "The paper doesn't clearly state this.">"""
        answer = self.llm.ask(prompt)
        return answer, ids

    # ---------- STEP 2b: THE "DECLINE" ACTION ----------
    def decline(self):
        return ("I only answer questions about the research papers in your "
                "library. Try asking me about their methods, results, or findings.")

    # ---------- THE LOOP: decide, then act ----------
    def handle(self, question):
        route = self.decide(question)
        print(f"  [agent decided: {route}]")
        if route == "SEARCH":
            answer, ids = self.search_and_answer(question)
            return f"{answer}\n\n(sources: {', '.join(ids)})"
        else:
            return self.decline()


if __name__ == "__main__":
    agent = RouterAgent()
    print("Scholiast agent ready. Type 'exit' to quit.\n")
    while True:
        q = input("you > ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        print("\nthinking...")
        print(f"\nscholiast > {agent.handle(q)}\n")