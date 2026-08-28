# RAG Pipeline — PDF Document Q&A

An end-to-end Retrieval-Augmented Generation (RAG) system: ingest PDFs, chunk and embed
them, store them in a vector database, retrieve relevant context for a query, and generate
grounded answers using an LLM.

Built to understand and implement the full RAG stack from scratch (no high-level "one-liner"
frameworks like off-the-shelf `RetrievalQA` chains) — ingestion, embedding, vector search,
and generation are each implemented as separate, inspectable components.

## Architecture

```
PDFs (data/pdfs/)
      │
      ▼
[ingestion.py]  → load PDFs, split into 500-char overlapping chunks
      │
      ▼
[embeddings.py] → SentenceTransformers "all-MiniLM-L6-v2" (384-dim, local, free)
      │
      ▼
[vector_store.py] → Chroma persistent collection (cosine similarity search)
      │
      ▼
[retrieval.py]  → embed query → top-k semantic search → ranked chunks
      │
      ▼
[llm.py]        → chunks + query → prompt → Gemini 2.5 Flash → grounded answer
```

## Tech Stack

| Component      | Choice                              | Why |
|-----------------|--------------------------------------|-----|
| PDF loading      | `PyPDFLoader` (LangChain)            | Handles multi-page PDFs, preserves per-page metadata |
| Chunking         | `RecursiveCharacterTextSplitter`     | Splits on paragraph/sentence boundaries first, avoids mid-sentence cuts |
| Embeddings       | `all-MiniLM-L6-v2` (Sentence-Transformers) | Runs locally, no API cost, 384-dim, good speed/quality tradeoff |
| Vector store     | ChromaDB (persistent)                | Lightweight, no external DB server needed, easy to self-host |
| LLM              | Google Gemini 2.5 Flash              | Fast + cheap for grounded Q&A generation |
| Demo UI          | Streamlit                            | Fastest way to expose a working interactive demo |

**Design decisions:**
- **Chunk size 500 / overlap 50** — balances retrieval precision (smaller chunks = more
  targeted matches) against context loss at chunk boundaries.
- **Local embeddings, API-based generation** — keeps indexing free/offline-capable while
  still using a strong hosted model for the harder generation step.
- **Similarity threshold on retrieval** — low-relevance chunks are filtered out before
  being passed to the LLM, reducing noise in the context window.

## Setup

```bash
git clone <this-repo-url>
cd rag-pipeline
pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your Gemini API key

# add PDFs you want to query into data/pdfs/
```

## Run

**Interactive web demo:**
```bash
streamlit run app.py
```

**As a Python script:**
```bash
python -m src.pipeline
```

**Programmatically:**
```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.build_index()          # only re-embeds if the store is empty
answer = pipeline.answer("What is Retrieval-Augmented Generation?")
print(answer)
```

See [`examples/sample_queries.md`](examples/sample_queries.md) for tested query/answer pairs.

## Project Structure

```
rag-pipeline/
├── src/
│   ├── ingestion.py      # PDF loading + chunking
│   ├── embeddings.py     # SentenceTransformers wrapper
│   ├── vector_store.py   # Chroma persistence + document insertion
│   ├── retrieval.py      # semantic search over the vector store
│   ├── llm.py            # Gemini setup + context-grounded generation
│   └── pipeline.py       # orchestrates the full flow
├── app.py                 # Streamlit demo
├── data/pdfs/              # put source PDFs here (gitignored contents)
├── notebooks/               # original exploratory notebook
├── examples/sample_queries.md
├── requirements.txt
└── .env.example
```

## Notes / Future Improvements

- Add re-ranking (e.g. cross-encoder) on top of vector similarity for higher precision
- Support incremental ingestion (currently skips re-embedding if the store is non-empty,
  but doesn't yet detect *new* files added to an existing index)
- Add evaluation (retrieval recall@k, answer faithfulness) on a held-out query set
- Swap in a hybrid (keyword + semantic) retriever for queries with exact terms/acronyms

## Author

**Name : Rohan Singh**
**Github :** https://github.com/RohanSingh404
**Linkedin :**  https://linkedin.com/in/rohansingh404

---
⭐ If you found this project useful, consider giving it a star!
