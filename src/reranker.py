from typing import Any, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder
from langsmith import traceable



class RerankedRetriever(BaseRetriever):
    """Retrieves candidates from base_retriever, then keeps the top_n by cross-encoder score."""

    base_retriever: BaseRetriever
    reranker: Any  # a sentence-transformers CrossEncoder
    top_n: int = 4

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]

@traceable(name="create reranked retriever")
def create_reranked_retriever(
    base_retriever: BaseRetriever,
    top_n: int = 4,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> RerankedRetriever:
    """Wrap a retriever with a cross-encoder reranker (CPU-friendly default model)."""
    retriever = RerankedRetriever(
        base_retriever=base_retriever,
        reranker=CrossEncoder(model_name),
        top_n=top_n,
    )
    print(f"✓ Reranker ready ({model_name}, top_n={top_n})")
    return retriever