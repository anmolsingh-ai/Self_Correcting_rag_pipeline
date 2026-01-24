from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import BM25Retriever
from retrievers.graded_fallback import GradedFallbackRetriever
from config import GRADE_THRESHOLD, RETRIEVER_K


def build_rag_pipeline(vector_store, chunks, llm):
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    bm25_retriever = BM25Retriever.from_documents(chunks, k=10)

    retriever = GradedFallbackRetriever(
        initial_retriever=vector_retriever,
        fallback_retriever=bm25_retriever,
        llm=llm,
        threshold=GRADE_THRESHOLD,
        k=RETRIEVER_K,
    )

    return ConversationalRetrievalChain(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
