from typing import List
from langsmith import traceable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

@traceable(name="create vector store")
def create_vector_store(
    chunks: List[Document],
    persist_directory: str = "chroma_db",
    force_rebuild: bool = False,
):
    """Load the Chroma DB if it already has data, otherwise build it from chunks.

    Use force_rebuild=True after changing the PDF, chunk settings or embedding model.
    """
    # device="cpu": on ZeroGPU Spaces torch pretends a GPU exists, which would
    # place the model on a GPU that is not actually allocated to this process.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    def open_store():
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    store = open_store()
    count = len(store.get(include=[])["ids"])

    if count and not force_rebuild:
        print(f"✓ Loaded existing vector store ({count} chunks)")
        if count != len(chunks):
            print("⚠ Chunk count differs from the stored DB; use force_rebuild=True if the PDF or settings changed")
        return store, embeddings

    if count:  # force_rebuild: wipe old data first
        store.delete_collection()
        store = open_store()

    store.add_documents(chunks)
    print(f"✓ Vector store created at {persist_directory} ({len(chunks)} chunks)")
    return store, embeddings