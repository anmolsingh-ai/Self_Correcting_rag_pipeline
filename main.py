import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.load_data import load_and_chunk_pdf
from src.vectore_store import create_vector_store
from src.hybrid_retriver import create_hybrid_retriever
from src.reranker import create_reranked_retriever
from src.conversational_chain import create_conversational_rag_chain
from langsmith import traceable

# gemini-2.5-flash passed its official shutdown date (Oct 16, 2026).
# gemini-3.5-flash is Google's recommended replacement.
LLM_MODEL = "gemini-3.5-flash"


@traceable(name="rag_pipeline")
def build_pipeline(pdf_path: str, return_retriever: bool = False):
    """Build the conversational RAG chain for a PDF."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

    print("📄 Loading and chunking PDF...")
    chunks = load_and_chunk_pdf(pdf_path)
    if not chunks:
        raise ValueError("No document chunks were created from the PDF.")
    print(f"✓ Created {len(chunks)} chunks")

    print("🗄️ Building retrieval pipeline...")
    vector_store, _ = create_vector_store(chunks)
    hybrid = create_hybrid_retriever(chunks, vector_store, k=10)
    retriever = create_reranked_retriever(hybrid, top_n=4)

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, google_api_key=api_key, temperature=0
    )
    rag_chain, _ = create_conversational_rag_chain(retriever, llm)
    print("✅ Pipeline ready!\n")
    if return_retriever:
        return rag_chain, retriever
    return rag_chain


def chat(rag_chain):
    """Interactive terminal chat."""
    session = 0
    print("🤖 CONVERSATIONAL RAG CHATBOT")
    print("Commands: 'exit'/'quit' to stop, 'clear' to start a new conversation.\n")

    while True:
        try:
            question = input("You: ")
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            if question.lower() == "clear":
                session += 1
                print("🧹 Started a new conversation.\n")
                continue

            response = rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": f"terminal_{session}"}},
            )
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\nTry asking again.\n")


if __name__ == "__main__":
    pdf_path = os.path.join(os.getcwd(), "data", "Notes.pdf")

    if not os.path.isfile(pdf_path):
        print(f"⚠️ PDF not found: {pdf_path}\nPlace Notes.pdf in the data folder.")
    else:
        try:
            chat(build_pipeline(pdf_path))
        except Exception as e:
            print(f"❌ Failed to start the RAG pipeline: {e}")