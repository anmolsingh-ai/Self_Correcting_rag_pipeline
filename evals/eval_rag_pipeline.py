import os
import json
from pathlib import Path

from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.models import GeminiModel

from main import build_pipeline
from evals.results_logger import run_and_save

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "goldens" / "generator_golden.json"
PDF_PATH = ROOT / "data" / "Notes.pdf"

JUDGE_MODEL = "gemini-2.5-flash"
THRESHOLD = 0.7

with open(GOLDEN_PATH, encoding="utf-8") as f:
    golden_data = json.load(f)

judge = GeminiModel(model=JUDGE_MODEL, api_key=api_key, temperature=0)

# Build your real pipeline (chain + retriever)
rag, retriever = build_pipeline(str(PDF_PATH), return_retriever=True)

test_cases = []
for i, g in enumerate(golden_data):
    question = g["query"]

    # 1. What the retriever + reranker actually returns
    docs = retriever.invoke(question)
    retrieval_context = [d.page_content for d in docs]

    # 2. What the full pipeline actually answers (fresh session per question)
    answer = rag.invoke(
        {"input": question},
        config={"configurable": {"session_id": f"eval_{i}"}},
    )
    if hasattr(answer, "content"):   # AIMessage -> str
        answer = answer.content

    test_cases.append(
        LLMTestCase(
            name=f"q{i}",
            input=question,
            actual_output=answer,
            retrieval_context=retrieval_context,
        )
    )

metrics = [
    FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
]

run_and_save(test_cases, metrics, eval_name="pipeline", label="baseline")