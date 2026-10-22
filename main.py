import os
from dotenv import load_dotenv
from src.load_data import load_and_chunk_pdf
from src.vectore_store import create_vector_store
from src.hybrid_retriver import create_hybrid_retriever
from src.reranker import create_reranked_retriever
from src.conversational_chain import create_conversational_rag_chain

from langchain_google_genai import ChatGoogleGenerativeAI


def main(pdf_path: str):
    """Main pipeline"""
    load_dotenv()  # Load environment variables from .env file
    print("🚀 Starting Self-Correcting RAG Pipeline...\n")
    
    # 1. Load and chunk documents
    chunks = load_and_chunk_pdf(pdf_path)
    
    # 2. Create vector store
    vector_store, embeddings = create_vector_store(chunks)
    
    # 3. Setup LLM
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print("✓ LLM initialized")
    
    #create hybrid retriever
    hybrid = create_hybrid_retriever(chunks, vector_store, k=10)
    
    #create a reranker retriever
    retriever = create_reranked_retriever(hybrid, top_n=4)

    # 6. Create conversational RAG chain
    rag_chain, message_store = create_conversational_rag_chain(retriever, llm)
    
    print("\n✅ Pipeline ready!\n")
    
    return rag_chain, message_store, vector_store


if __name__ == "__main__":
    # Example usage
    pdf_path = os.path.join(os.getcwd(), "data", "Notes.pdf")
    
    if os.path.exists(pdf_path):
        rag_chain, store, vector_store = main(pdf_path)
        
        # Example query
        print("Testing RAG chain...")
        response = rag_chain.invoke(
            {"input": "What is the weights and bias in machine learning?"},
            config={"configurable": {"session_id": "test"}}
        )
        print(f"\nResponse: {response}\n")
        
    else:
        print(f"⚠️  PDF not found at {pdf_path}")
        print("Please adjust the path or upload your PDF to the data folder.")
