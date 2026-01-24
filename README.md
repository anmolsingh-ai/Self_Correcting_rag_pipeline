# Self-Correcting RAG Pipeline

## Description
This project implements a Self-Correcting Retrieval Augmented Generation (RAG) pipeline designed to enhance the accuracy and relevance of generated responses. It leverages advanced techniques for document loading, chunking, vector storage, and a graded fallback retrieval mechanism to ensure robust information retrieval.

## Features
- **Document Loading:** Supports loading various document types from a specified data directory.
- **Smart Chunking:** Efficiently splits documents into manageable chunks for optimal retrieval.
- **Vector Store:** Utilizes ChromaDB for vector embeddings and efficient similarity search.
- **Graded Fallback Retriever:** Implements a sophisticated retrieval strategy that falls back to alternative retrieval methods if the initial retrieval is not confident enough.
- **Conversational Retrieval Chain:** Integrates a conversational retrieval chain for generating context-aware responses.
- **Evaluation:** Includes evaluation metrics and benchmarks to assess the performance of the RAG pipeline.
- **Benchmarking:** Provides tests for edge cases and latency to ensure robustness and efficiency.

## Project Structure
```
self_correcting_rag/
├── config.py                 # Configuration settings for the pipeline (e.g., API keys, model names, thresholds)
├── data/                     # Directory for source documents (e.g., Google.txt, Nvidia.txt, PDFs)
│   ├── Google.txt
│   ├── Hands-on-Machine-Learning.pdf
│   ├── Nvidia.txt
│   ├── SpaceX.txt
│   └── Tesla.txt
├── evaluation/               # Modules for evaluating the RAG pipeline's performance
│   ├── __init__.py
│   └── evaluator.py          # Script for running evaluations and calculating metrics
├── loaders/                  # Modules for loading documents from various sources
│   ├── __init__.py
│   └── document_loader.py    # Handles loading different document types
├── main.py                   # Main script to run the RAG pipeline, perform queries, and trigger evaluations/tests
├── pipeline/                 # Core RAG pipeline construction and logic
│   ├── __init__.py
│   └── rag_pipeline.py       # Defines the RAG pipeline, including retriever and chain setup
├── processing/               # Modules for document preprocessing
│   ├── __init__.py
│   └── chunking.py           # Handles splitting documents into chunks
├── requirements.txt          # List of Python dependencies for the project
├── retrievers/               # Custom retriever implementations
│   ├── __init__.py
│   ├── graded_fallback.py    # Implements the graded fallback retrieval logic
│   └── grader.py             # Module for grading retrieval confidence
├── test.py                   # (Optional) Additional testing script
├── tests/                    # Unit and integration tests
│   ├── __init__.py
│   └── benchmarks.py         # Benchmarking tests for edge cases and latency
└── vectorstore/              # Modules for vector store management
    ├── __init__.py
    └── chroma_store.py       # Handles creation and management of the Chroma vector store
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd self_correcting_rag
    ```

2.  **Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure API Key:**
    Create a `.env` file in the root directory of the project and add your Gemini API key:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

## Usage
To run the RAG pipeline, execute the `main.py` script. Ensure your virtual environment is activated.

```bash
python main.py
```

The `main.py` script will:
- Initialize the RAG pipeline.
- Run a sample query and print the answer.
- Execute evaluation benchmarks.

## Evaluation
The `evaluation/evaluator.py` module contains logic for evaluating the RAG pipeline. It assesses the quality of generated responses using various metrics. The `main.py` script runs a default evaluation.

## Testing
- **Benchmarking:** The `tests/benchmarks.py` module includes tests for:
    - Edge cases to ensure the pipeline handles unusual inputs gracefully.
    - Latency to measure the response time of the pipeline.

To run the tests (as part of `main.py`):
```bash
python main.py
```
