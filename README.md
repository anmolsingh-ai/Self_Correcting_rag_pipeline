---
title: Self-Correcting RAG Chat
emoji: 💬
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
python_version: "3.10"
---

# Self-Correcting RAG Chat

Chat with the PDF included in this repository using a conversational retrieval-augmented generation (RAG) pipeline.

**Live demo:** [Open the Hugging Face Space](https://huggingface.co/spaces/anmolsingh-ai/Rag_System)

## How it works

The app loads `data/Notes.pdf`, splits it into text chunks, and retrieves relevant passages using:

1. Chroma vector search and BM25 keyword search
2. A cross-encoder reranker
3. Google Gemini to answer using the retrieved context

Conversation history is maintained per chat session. The embedding and reranker models are downloaded when the app starts.

## Run locally

Requires Python 3.10 and a Google Gemini API key.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey), then run:

```bash
python app.py
```

The app uses `data/Notes.pdf` by default. To use a different document, replace that file.

## Hugging Face deployment

The app runs as a [Gradio Space](https://huggingface.co/spaces/anmolsingh-ai/Rag_System). Add `GEMINI_API_KEY` under the Space's **Settings → Variables and secrets**.

GitHub Actions syncs pushes to `main` to the Space. To enable syncing, configure:

- GitHub Actions secret `HF_TOKEN`: a Hugging Face token with write access to the Space.
- GitHub Actions variable `HF_SPACE`: the Space ID, for example `anmolsingh-ai/Rag_System`.

The workflow publishes a fresh snapshot rather than GitHub's commit history. It stores `data/Notes.pdf` with Git LFS; generated vector indexes are rebuilt when the Space starts.

## Project files

- `app.py` — Gradio chat interface
- `main.py` — RAG pipeline construction and terminal chat
- `src/` — PDF loading, vector storage, hybrid retrieval, reranking, and conversational chain
- `data/Notes.pdf` — document used by the demo
- `evals/` — evaluation scripts
