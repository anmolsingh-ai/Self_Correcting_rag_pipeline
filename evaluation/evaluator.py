from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall


def generate_responses(qa_chain, test_data):
    results = []
    for item in test_data:
        response = qa_chain({"question": item["question"]})
        results.append({
            "question": item["question"],
            "answer": response["answer"],
            "contexts": [doc.page_content for doc in response["source_documents"]],
            "ground_truth": item["ground_truth"],
        })
    return results


def evaluate_rag(qa_chain, test_data, embeddings, llm):
    print("🤖 Generating responses...")
    results = generate_responses(qa_chain, test_data)

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    print("📊 Running RAGAS evaluation...")
    eval_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    try:
        df = eval_result.to_pandas()
        scores = df.mean(numeric_only=True).to_dict()
    except Exception:
        scores = getattr(eval_result, "scores", {})

    return scores, results
