from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import ConfigDict
from typing import List, Any
from langchain_core.documents import Document
from src.document_grading import grade_documents

class GradedFallbackRetriever(BaseRetriever):
    """Custom retriever that grades results and falls back to BM25"""
    
    initial_retriever: BaseRetriever
    fallback_retriever: BaseRetriever
    grading_chain: Any
    threshold: float = 8.0
    k: int = 4
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        # Initial vector retrieval
        initial_docs = self.initial_retriever.invoke(query)
        
        # Grade initial docs
        graded_initial = grade_documents(
            self.grading_chain,
            initial_docs,
            query,
            self.threshold
        )
        
        # If we have enough good docs, return them
        if len(graded_initial) >= self.k:
            return graded_initial[:self.k]
        
        # Fallback to BM25
        print(f"⚠ Low relevance in initial retrieval ({len(graded_initial)} docs), using BM25 fallback...")
        
        fallback_docs = self.fallback_retriever.invoke(query)
        graded_fallback = grade_documents(
            self.grading_chain,
            fallback_docs,
            query,
            self.threshold
        )
        
        # Combine without duplicates
        combined = graded_initial.copy()
        existing_contents = {doc.page_content for doc in graded_initial}
        
        for doc in graded_fallback:
            if doc.page_content not in existing_contents:
                combined.append(doc)
        
        return combined[:self.k]
    
    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)


def create_hybrid_retriever(
    chunks: List[Document],
    vector_store,
    llm,
    grading_chain,
    threshold: float = 7.5,
    k: int = 6
) -> GradedFallbackRetriever:
    """Create hybrid retriever with vector search + BM25 fallback"""
    
    # Vector retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(chunks, k=10)
    
    # Graded fallback retriever
    hybrid_retriever = GradedFallbackRetriever(
        initial_retriever=vector_retriever,
        fallback_retriever=bm25_retriever,
        grading_chain=grading_chain,
        threshold=threshold,
        k=k
    )
    
    print("✓ Hybrid retriever created (vector + BM25 with fallback)")
    return hybrid_retriever
