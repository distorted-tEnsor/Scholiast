# Scholiast

**A fully-local, GPU-accelerated research-paper assistant that runs offline on consumer hardware.**

Point Scholiast at a folder of academic PDFs and ask questions in plain English. It answers using a language model running entirely on your own laptop GPU — no cloud, no API keys, no data leaving your machine — with answers grounded in the source papers and citations back to the passages used.

> *A "scholiast" was an ancient scholar who wrote explanatory notes in the margins of texts. This one lives on your laptop.*

---

## Why this project

Most AI assistants run in the cloud: your data leaves your machine, you need internet, and you pay per query. For a private collection of research papers, that's often unacceptable.

Scholiast explores one question:

> **How capable an offline research assistant can we build that runs entirely on a resource-constrained consumer GPU (6 GB VRAM)?**

This places it in the space of **edge / on-device AI** — doing useful work with a small model, limited memory, and no cloud. The hardware limit isn't an excuse; it's the whole design driver.

**Reference machine:** NVIDIA RTX 3050 Laptop GPU (6 GB VRAM), 16 GB RAM, Intel i5, Windows 11.

---

## How it works

Scholiast implements a full **RAG (Retrieval-Augmented Generation)** pipeline — the "open-book exam" approach. Instead of asking the language model a question blind (where it guesses), Scholiast first finds the relevant passages from your papers and hands them to the model to answer from:

1. **Read** — extract text from academic PDFs (PyMuPDF).
2. **Chunk** — split each paper into overlapping ~800-character pieces, so no idea falls through a gap between chunks.
3. **Embed** — turn each chunk into a 384-dimension "meaning vector" using a local embedding model. Passages with similar meaning end up close together, enabling search by *meaning* rather than keywords.
4. **Store & retrieve** — save all vectors in a local ChromaDB database; for each question, fetch the nearest chunks by semantic similarity.
5. **Generate** — feed the retrieved passages to the local LLM with strict instructions to answer *only* from them, and to say so when the answer isn't present. Answers cite the passages used.

The result: ask "how accurate is the method?" and Scholiast pulls the answer from the paper's own results — while correctly refusing questions the papers don't cover (e.g. "what's the capital of France?"), instead of hallucinating.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Inference engine | **Ollama** | Easiest reliable local-GPU path on Windows. |
| Language model | **Qwen2.5-3B** (4-bit quantized) | Strong reasoning at a size that fits 6 GB VRAM. |
| PDF parsing | **PyMuPDF** | Extracts clean text from academic PDFs. |
| Embeddings | **bge-small-en-v1.5** (local) | Turns text into meaning-vectors, fully offline. |
| Vector store | **ChromaDB** (local, on disk) | Self-hosted, no cloud dependency. |
| Retrieval | Semantic (bi-encoder) search | Finds passages by meaning, not keywords. |
| Language | **Python 3.10+** | — |

---

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| **0** | Local model talking to Python on the GPU | ✅ Done |
| **1** | RAG over papers: read → chunk → embed → retrieve → answer with citations | ✅ Done |
| **2** | Agent loop: decide when to retrieve, compare, or answer directly | ⚪ Planned |
| **3** | Memory across sessions | ⚪ Planned |
| **4** | Evaluation & tuning: retrieval quality vs speed vs VRAM; structure-aware PDF/table parsing; cross-encoder reranking | ⚪ Planned |
| **5** | Stretch: arXiv fetch, multi-paper synthesis, web UI | ⚪ Future |

---

## Current status

**Phases 0 and 1 complete.** A local language model (Qwen2.5-3B via Ollama) runs on the GPU, and a full offline RAG pipeline answers questions over academic PDFs with citations to the source passages.

### Known limitations (honest notes / future work)

- **Table extraction:** numeric results tables in PDFs are flattened during text extraction, so the model can occasionally attribute a value to the wrong row. Retrieval finds the right table; preserving table *structure* (e.g. via GROBID or a layout-aware parser) is planned for Phase 4.
- **Question–statement gap:** a user's question and the answering passage are embedded separately, so phrasing differences can slightly lower retrieval scores. A cross-encoder reranking stage is planned to address this.

This is an actively-developed personal project documenting a learning journey from local LLM inference through RAG to agentic reasoning, built under a real 6 GB VRAM constraint.

---

## Getting started

```bash
# 1. Install Ollama (https://ollama.com) and pull the model
ollama pull qwen2.5:3b

# 2. Set up Python
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt

# 3. Confirm the local model works
python scholiast\llm\local_llm.py

# 4. Add a PDF to data/papers/, then build the searchable index
python build_index.py

# 5. Ask questions about your paper
python ask_paper.py
```

---

## Project layout