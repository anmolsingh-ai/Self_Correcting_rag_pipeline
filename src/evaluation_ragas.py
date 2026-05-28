from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import pandas as pd
import numpy as np
import time

def generate_responses(rag_chain, test_data, session_id="eval"):
    """Generate responses for test data"""
    results = []
    
    for item in test_data:
        response = rag_chain.invoke(
            {"input": item["question"]},
            config={"configurable": {"session_id": session_id}}
        )
        
        results.append({
            "question": item["question"],
            "answer": response.content if hasattr(response, 'content') else str(response),
            "contexts": [],  # Adjust based on your chain output
            "ground_truth": item["ground_truth"]
        })
    
    return results


def evaluate_rag(rag_chain, test_data):
    """Evaluate RAG system using RAGAS metrics"""
    print("📊 Generating responses...")
    
    results = generate_responses(rag_chain, test_data)
    
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results]
    })
    
    # Setup evaluators
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.0
    )
    
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    
    evaluator_llm = LangchainLLMWrapper(gemini_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)
    
    print("🔄 Running RAGAS evaluation...")
    
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    df = result.to_pandas()
    
    scores = {
        "faithfulness": df["faithfulness"].mean(),
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
        "context_recall": df["context_recall"].mean(),
    }
    
    print("\n📈 Evaluation Scores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.3f}")
    
    return scores, results


def test_latency(rag_chain, num_tests=3):
    """Test RAG system latency"""
    questions = [
        "What is this document about?",
        "Give me a summary",
        "What are the main points?"
    ][:num_tests]
    
    times = []
    
    print("\n⏱️  Testing latency...")
    for question in questions:
        start = time.time()
        rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": "latency_test"}}
        )
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    print(f"  Average latency: {avg_time:.2f}s")
    
    return avg_time
