# Self-Correcting RAG Pipeline

This repository builds a PDF-based conversational RAG system using LangChain, Chroma, Google Gemini, BM25, and a cross-encoder reranker. The code is organized around a small set of pipeline building blocks and a session-aware conversational chain.

## Project goals

- Load a PDF and split it into searchable document chunks
- Build a persistent vector store with Chroma
- Combine vector retrieval with BM25 keyword retrieval
- Rerank the merged candidate documents with a cross-encoder
- Rewrite follow-up questions into standalone questions using chat history
- Answer using only retrieved context and maintain per-session memory
- Evaluate the pipeline with golden questions and DeepEval metrics

## Current codebase structure

```text
Self_Correcting_rag_pipeline/
├── .env                      # Local environment variables (not committed)
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                   # Entry point for building the pipeline and chat loop
├── app.py                    # PDF chunk export utility
├── data/
│   └── Notes.pdf            # Sample PDF used by the pipeline
├── chroma_db/               # Persisted Chroma vector index (created automatically)
├── goldens/
│   └── generator_golden.json
├── evals/
│   ├── __init__.py
│   ├── eval_application.py
│   ├── eval_generator.py
│   ├── eval_leakage.py
│   ├── eval_rag_pipeline.py
│   ├── eval_retriever.py
│   ├── eval_scope.py
│   ├── eval_toxicity.py
│   └── results_logger.py
├── src/
│   ├── conversational_chain.py
│   ├── hybrid_retriver.py    # Hybrid vector + BM25 retriever
│   ├── load_data.py         # PDF extraction + chunking
│   ├── reranker.py          # Cross-encoder reranking layer
│   └── vectore_store.py     # Chroma vector store creation
└── venv/
```

## Core pipeline flow

The actual implementation follows this sequence:

```text
PDF file
  ↓
load_and_chunk_pdf()
  ↓
create_vector_store()
  ↓
create_hybrid_retriever()
  ↓
create_reranked_retriever()
  ↓
create_conversational_rag_chain()
  ↓
rag_chain.invoke(...)
```

## What the code actually does

### 1) PDF ingestion and chunking

File: `src/load_data.py`

- Uses `PyMuPDFLoader` to read pages from a PDF
- Splits text with `RecursiveCharacterTextSplitter`
- Keeps chunk overlap and cleans empty chunks
- Returns a list of LangChain `Document` objects

Example:

```python
from src.load_data import load_and_chunk_pdf
chunks = load_and_chunk_pdf("data/Notes.pdf")
print(len(chunks))
```

### 2) Vector store creation

File: `src/vectore_store.py`

- Builds a Chroma vector database using `langchain_chroma`
- Uses `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`
- Persists the index to `chroma_db/`
- Reuses the existing collection unless `force_rebuild=True`

Example:

```python
from src.load_data import load_and_chunk_pdf
from src.vectore_store import create_vector_store

chunks = load_and_chunk_pdf("data/Notes.pdf")
vector_store, embeddings = create_vector_store(chunks)
```

### 3) Hybrid retrieval

File: `src/hybrid_retriver.py`

- Combines results from:
  - vector search (`vector_store.as_retriever`)
  - BM25 keyword search (`BM25Retriever`)
- Deduplicates by `page_content`
- Does not enforce a cutoff itself; the reranker chooses the final list

Example:

```python
from src.hybrid_retriver import create_hybrid_retriever

hybrid = create_hybrid_retriever(chunks, vector_store, k=10)
results = hybrid.invoke("What is this document about?")
```

### 4) Reranking

File: `src/reranker.py`

- Wraps any base retriever with a cross-encoder reranker
- Scores candidate chunks using `sentence-transformers`'s `CrossEncoder`
- Keeps the top `top_n` results

Example:

```python
from src.reranker import create_reranked_retriever

retriever = create_reranked_retriever(hybrid, top_n=4)
context_docs = retriever.invoke("Explain the core idea")
```

### 5) Conversational chain

File: `src/conversational_chain.py`

This module creates the actual conversational RAG chain. It performs:

1. Question rewrite using recent chat history
2. Retrieval using the rewritten question
3. Answer generation from retrieved chunks only
4. Per-session memory using `RunnableWithMessageHistory`

Important details from the implementation:

- `MAX_HISTORY_MESSAGES = 6`
- The rewriter only sees the latest six messages
- The generator prompt explicitly says not to reveal system instructions or configuration
- The original user question and final answer are stored in session memory
- The rewritten question is used for retrieval, not saved as chat history

Example:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from src.conversational_chain import create_conversational_rag_chain

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="YOUR_KEY", temperature=0)
rag_chain, store = create_conversational_rag_chain(retriever, llm)

response = rag_chain.invoke(
    {"input": "What is the main idea?"},
    config={"configurable": {"session_id": "user_1"}},
)
print(response)
```

## Main entry points

### `main.py`

This is the repository’s main runtime file.

```python
from main import build_pipeline

rag_chain = build_pipeline("data/Notes.pdf")
response = rag_chain.invoke(
    {"input": "Explain this document"},
    config={"configurable": {"session_id": "demo_session"}},
)
print(response)
```

`build_pipeline()` does the following:

- loads environment variables
- validates `GEMINI_API_KEY`
- loads and chunks the PDF
- creates the Chroma vector store
- builds a hybrid retriever
- wraps it in a reranker
- creates the chat history aware conversational chain

It also exposes a terminal chat loop via `chat(rag_chain)`.

To run the chat interface:

```bash
python main.py
```

## Environment setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get the key from Google AI Studio:

- https://aistudio.google.com/app/apikey

The code currently uses `ChatGoogleGenerativeAI` and reads `GEMINI_API_KEY`; the `GOOGLE_API_KEY` variable from older README examples is not required by the current implementation.

## Installation

```bash
pip install -r requirements.txt
```

Optional: create a virtual environment first.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```python
from main import build_pipeline

rag_chain = build_pipeline("data/Notes.pdf")

response = rag_chain.invoke(
    {"input": "What is the main topic?"},
    config={"configurable": {"session_id": "user_1"}},
)

print(response)
```

## Evaluation

The repository includes DeepEval-based evaluation scripts under `evals/`.

Example:

```bash
python evals/eval_rag_pipeline.py
```

This script loads golden questions from `goldens/generator_golden.json`, builds the pipeline, runs each query through the retriever and generator, and measures:

- Faithfulness
- Answer Relevancy
- Contextual Relevancy

## Notes on the current implementation

- The project is not a generic package import like `self_correcting_rag_pipeline`; the actual runtime entry point is `main.py`.
- The file `src/hybrid_retriver.py` contains a typo in the filename (`retriver` instead of `retriever`), but the naming is used consistently in imports.
- The repository uses a persistent Chroma index in `chroma_db/` and can reuse existing embeddings unless you rebuild the store.
- Conversation memory is session-scoped, so each `session_id` keeps its own chat history.

## License

This project does not currently declare a formal license file in the repository root. If you are using it in a production or team environment, confirm the licensing terms before redistribution.

## Useful links

- [LangChain documentation](https://python.langchain.com/docs/)
- [Chroma docs](https://docs.trychroma.com/)
- [Google AI Studio](https://aistudio.google.com/app/apikey)
- [DeepEval](https://docs.deepeval.com/)
