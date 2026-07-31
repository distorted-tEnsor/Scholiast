# Scholiast

**A fully-local, GPU-accelerated research-paper assistant that runs offline on consumer hardware.**

Point Scholiast at a folder of academic PDFs and ask questions in plain English. It answers using a language model running entirely on your own laptop GPU — no cloud, no API keys, no data leaving your machine — with answers grounded in the source papers.

> *A "scholiast" was an ancient scholar who wrote explanatory notes in the margins of texts. This one lives on your laptop.*

---

## Why this project

Most AI assistants run in the cloud: your data leaves your machine, you need internet, and you pay per query. For a private collection of research papers, that's often unacceptable.

Scholiast explores one question:

> **How capable an offline research assistant can we build that runs entirely on a resource-constrained consumer GPU (6 GB VRAM)?**

This places it in the space of **edge / on-device AI** — doing useful work with a small model, limited memory, and no cloud. The hardware limit isn't an excuse; it's the whole design driver.

**Reference machine:** NVIDIA RTX 3050 Laptop GPU (6 GB VRAM), 16 GB RAM, Intel i5, Windows 11.

---

## What it will do

- **Chat with your papers** — ask questions, get answers drawn from the actual PDFs.
- **Grounded answers with citations** — each answer points to the paper/section it came from.
- **Section-aware understanding** — parses papers into their real structure (Abstract, Methods, Results…) for precise retrieval.
- **100% offline on your GPU** — after the one-time model download, no internet needed.
- **Agentic reasoning** — decides when to search the papers versus answer directly.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Inference engine | **Ollama** | Easiest reliable local-GPU path on Windows. |
| Language model | **Qwen2.5-3B** (4-bit quantized) | Strong reasoning at a size that fits 6 GB VRAM. |
| PDF parsing | **PyMuPDF** | Extracts clean text from messy academic PDFs. |
| Embeddings | **bge-small-en-v1.5** (local) | Turns text into meaning-vectors, fully offline. |
| Vector store | **ChromaDB** (local, on disk) | Self-hosted, no cloud dependency. |
| Agent loop | Hand-written (ReAct pattern) | Written from scratch as the core learning goal. |
| Language | **Python 3.10+** | — |

---

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| **0** | Local model talking to Python on the GPU | ✅ Done |
| **1** | RAG over papers: parse → chunk → embed → retrieve → answer with citations | 🔨 In progress |
| **2** | Agent loop: decide when to retrieve, compare, or answer directly | ⚪ Planned |
| **3** | Memory across sessions | ⚪ Planned |
| **4** | Evaluation & tuning: quality vs speed vs VRAM | ⚪ Planned |
| **5** | Stretch: arXiv fetch, multi-paper synthesis, web UI | ⚪ Future |

---

## Current status

Phase 0 complete: a local language model (Qwen2.5-3B via Ollama) runs on the GPU and is accessed from Python through a reusable interface. Phase 1 (reading and retrieving over PDFs) is in active development.

This is an actively-developed personal project documenting a learning journey from local LLM inference through RAG to agentic reasoning, built under a real 6 GB VRAM constraint.

---

## Getting started

```bash
# 1. Install Ollama (https://ollama.com) and pull the model
ollama pull qwen2.5:3b

# 2. Set up Python
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Confirm the local model works
python scholiast\llm\local_llm.py
```

---

## Project layout