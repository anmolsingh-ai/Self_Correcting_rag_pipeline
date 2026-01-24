import os
from langchain_chroma import Chroma
from config import PERSIST_DIR


def create_vector_store(chunks, embeddings):
    if os.path.exists(PERSIST_DIR):
        print("📦 Loading existing vector store...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )

    print("📦 Creating new vector store...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
