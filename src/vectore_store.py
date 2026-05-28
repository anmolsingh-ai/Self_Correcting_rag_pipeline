from typing import List, Any


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def create_vector_store(chunks: List[Document], persist_directory: str = "chroma_db"):
    """Create vector store with Chroma"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=persist_directory
    )
    print(f"✓ Vector store created at {persist_directory}")
    
    return vector_store, embeddings
