from typing import List, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from retrievers.grader import grade_documents
from config import GRADE_THRESHOLD, RETRIEVER_K


class GradedFallbackRetriever(BaseRetriever):
    initial_retriever: BaseRetriever
    fallback_retriever: BaseRetriever
    llm: ChatGoogleGenerativeAI
    threshold: float = GRADE_THRESHOLD
    k: int = RETRIEVER_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        initial_docs = self.initial_retriever.invoke(query)
        graded_initial = grade_documents(self.llm, initial_docs, query, self.threshold)

        if len(graded_initial) >= self.k:
            return graded_initial[:self.k]

        print("⚠ Low relevance — using BM25 fallback...")
        fallback_docs = self.fallback_retriever.invoke(query)
        graded_fallback = grade_documents(self.llm, fallback_docs, query, self.threshold)

        combined = graded_initial + [d for d in graded_fallback if d not in graded_initial]
        return combined[:self.k]

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)
