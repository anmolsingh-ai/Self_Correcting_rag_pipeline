from typing import Any, List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


class SimpleHybridRetriever(BaseRetriever):
    """Merges semantic (vector) and keyword (BM25) results into one candidate pool."""

    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        combined, seen = [], set()
        for doc in self.vector_retriever.invoke(query) + self.bm25_retriever.invoke(query):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        return combined  # no cutoff here: the reranker picks the final chunks


def create_hybrid_retriever(
    chunks: List[Document], vector_store, k: int = 10
) -> SimpleHybridRetriever:
    """Fetch k docs from vector search and k from BM25, merged without duplicates
    (up to 2*k candidates)."""
    retriever = SimpleHybridRetriever(
        vector_retriever=vector_store.as_retriever(search_kwargs={"k": k}),
        bm25_retriever=BM25Retriever.from_documents(chunks, k=k),
    )
    print("✓ Hybrid retriever created (Vector + BM25)")
    return retriever