from config import DATA_DIR, EMBEDDING_MODEL, LLM_MODEL
from loaders.document_loader import load_documents
from processing.chunking import split_documents
from vectorstore.chroma_store import create_vector_store
from pipeline.rag_pipeline import build_rag_pipeline
from evaluation.evaluator import evaluate_rag
from tests.benchmarks import test_edge_cases, test_latency
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


def main():
    if "GEMINI_API_KEY" not in os.environ:
        raise EnvironmentError(" Please set GEMINI_API_KEY in your environment variables.")

    print(" Initializing Self-Correcting RAG Pipeline...")

    
    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)

    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.0,
    )

    
    vector_store = create_vector_store(chunks, embeddings)

    
    qa_chain = build_rag_pipeline(vector_store, chunks, llm)
    print("✅ RAG pipeline ready")

    
    query = "What is this document about?"
    result = qa_chain.invoke({"question": query, "chat_history": []})

    print("\n--- Sample Query ---")
    print("Q:", query)
    print("A:", result["answer"])

    
    test_data = [
        {
            "question": "Who founded Nvidia, and what was the company's original focus?",
            "ground_truth": "Nvidia was founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem. "
                            "The company initially focused on GPUs for gaming before expanding into AI and HPC."
        },
        {
            "question": "What milestone did SpaceX achieve with Falcon 9 in 2015?",
            "ground_truth": "In December 2015, SpaceX successfully landed and reused a Falcon 9 first stage."
        },
        {
            "question": "When was Tesla founded and what was its first vehicle?",
            "ground_truth": "Tesla was founded in 2003 and its first vehicle was the Tesla Roadster."
        },
    ]

    scores, _ = evaluate_rag(qa_chain, test_data, embeddings, llm)
    print("\n📊 Evaluation Scores:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    
    test_edge_cases(qa_chain)
    test_latency(qa_chain)


if __name__ == "__main__":
    main()
