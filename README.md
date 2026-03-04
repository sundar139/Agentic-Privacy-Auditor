# 🔍 Agentic Privacy & Compliance Auditor

> An end-to-end Agentic RAG system that answers questions about real website privacy policies — with hallucination detection built in.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-green.svg)](https://www.trychroma.com/)
[![License: CC BY-NC](https://img.shields.io/badge/Dataset-CC%20BY--NC-lightgrey.svg)](https://usableprivacy.org/data)

---

## Overview

This system processes **115 real website privacy policies** from the OPP-115 Corpus (ACL 2016) and enables natural language querying with enterprise-grade hallucination detection via a two-agent pipeline.

**Business value:** Automates the manual reading of legal compliance documents. Demonstrates the ability to handle strict regulatory text, parse complex natural language, and ensure data privacy — critical skills for enterprise data science roles.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────┐
│  PLANNER AGENT  (LangChain + Ollama) │
│  Classifies → SIMPLE / FILTERED /   │
│               COMPARE               │
└──────────────┬───────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
Semantic   Metadata   Multi-Query
 Search     Filter      Search
    └──────────┼──────────┘
               ▼
  ┌────────────────────────┐
  │  ChromaDB              │
  │  3,792 enriched segs   │
  │  BAAI/bge-small-en     │
  └────────────┬───────────┘
               ▼
  ┌────────────────────────┐
  │  Qwen2.5:7b (Ollama)   │
  └────────────┬───────────┘
               ▼
  ┌────────────────────────┐
  │  AUDITOR AGENT         │
  │  Faithfulness 0.0–1.0  │
  │  PASS / FAIL + retry   │
  └────────────┬───────────┘
               ▼
         Streamlit UI
```

---

## Tech Stack

| Component       | Technology                             |
| --------------- | -------------------------------------- |
| Embedding Model | `BAAI/bge-small-en-v1.5` (MTEB #1)     |
| LLM             | `Qwen2.5:7b` via Ollama (runs locally) |
| Vector Database | ChromaDB (persistent)                  |
| Orchestration   | LangChain                              |
| Frontend        | Streamlit                              |
| Dataset         | OPP-115 Corpus, ACL 2016               |

---

## Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- OPP-115 dataset extracted into `data/raw/`

### 1. Clone and install

```cmd
git clone https://github.com/sundar139/agentic-privacy-auditor
cd agentic-privacy-auditor
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Pull the LLM

```cmd
ollama pull qwen2.5:7b
```

### 3. Download the dataset

Download the OPP-115 Corpus from [usableprivacy.org/data](https://usableprivacy.org/data) and extract:

- `sanitized_policies/` → `data/raw/sanitized_policies/`
- `pretty_print/` → `data/raw/pretty_print/`
- `documentation/` → `data/raw/documentation/`

### 4. Build the vector store

```cmd
python src/ingestion/ingest.py
```

### 5. Launch the app

```cmd
streamlit run src/app.py
```

Open **http://localhost:8501**.

---

## Project Structure

```
agentic-privacy-auditor/
├── data/
│   ├── raw/                      # OPP-115 dataset (not tracked in git)
│   ├── processed/                # Enriched JSON files (auto-generated)
│   └── vector_store/             # ChromaDB files (auto-generated)
├── notebooks/
├── src/
│   ├── ingestion/
│   │   ├── ingest.py             # One-time ingestion pipeline
│   │   ├── document_loader.py    # HTML → segments via ||| delimiter
│   │   └── metadata_fuser.py     # Fuses expert annotations + metadata
│   ├── embeddings/
│   │   ├── embedding_manager.py  # BAAI/bge-small-en-v1.5
│   │   └── vector_store.py       # ChromaDB read/write
│   ├── agents/
│   │   ├── planner.py            # Query router (SIMPLE/FILTERED/COMPARE)
│   │   └── auditor.py            # Faithfulness scoring + regeneration
│   ├── retrieval/
│   │   ├── retriever.py          # Semantic + metadata hybrid search
│   │   └── generation.py         # Qwen2.5 answer generation
│   └── app.py                    # Streamlit UI
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## Evaluation

| Metric             | Measures                       | Method                  |
| ------------------ | ------------------------------ | ----------------------- |
| **Faithfulness**   | Claims grounded in context?    | Auditor Agent (0.0–1.0) |
| **Context Recall** | Ground-truth segments fetched? | Ragas                   |
| **MRR**            | Most relevant chunk ranked #1? | TruLens                 |

---

## Dataset Citation

> The creation and analysis of a website privacy policy corpus. Shomir Wilson, Florian Schaub, Aswarth Abhilash Dara, Frederick Liu, Sushain Cherivirala, Pedro Giovanni Leon, Mads Schaarup Andersen, Sebastian Zimmeck, Kanthashree Mysore Sathyendra, N. Cameron Russell, Thomas B. Norton, Eduard Hovy, Joel Reidenberg, and Norman Sadeh. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, Berlin, Germany, August 2016.

For research and teaching purposes only (Creative Commons Attribution-NonCommercial).

---

## Author

**Rohith Sundar Jonnalagadda**  
[LinkedIn](https://www.linkedin.com/in/rohithsundarj/) · MS Computer Science, Kennesaw State University
