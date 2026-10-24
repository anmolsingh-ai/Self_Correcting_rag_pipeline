import json
import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, ErrorConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from src.load_data import load_and_chunk_pdf
from src.vectore_store import create_vector_store
from src.hybrid_retriver import create_hybrid_retriever
from src.reranker import create_reranked_retriever

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "goldens" / "retriever_golden.json"
PDF_PATH = ROOT / "data" / "Notes.pdf"

JUDGE_MODEL = "gemini-3.8-flash"
THRESHOLD = 0.7

# Golden questions
with open(GOLDEN_PATH, encoding="utf-8") as f:
    golden_data = json.load(f)

# Retriever: vector + BM25 merged, then reranked
chunks = load_and_chunk_pdf(str(PDF_PATH))
vector_store, _ = create_vector_store(chunks)
hybrid = create_hybrid_retriever(chunks, vector_store, k=10)
retriever = create_reranked_retriever(hybrid, top_n=4)

# Judge
judge = GeminiModel(model=JUDGE_MODEL, api_key=os.environ["GEMINI_API_KEY"], temperature=0)

# One test case per golden question
test_cases = [
    LLMTestCase(
        input=item["query"],
        actual_output="retrieval-only evaluation",
        expected_output=item["ideal_answer"],
        retrieval_context=[d.page_content for d in retriever.invoke(item["query"])],
    )
    for item in golden_data
]

evaluate(
    test_cases=test_cases,
    metrics=[
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ],
    identifier="hybrid_reranked",
    hyperparameters={"retriever": "hybrid_reranked", "judge_model": JUDGE_MODEL},
    async_config=AsyncConfig(max_concurrent=2, throttle_value=1),  # avoid Gemini 429s
    error_config=ErrorConfig(ignore_errors=True),
    cache_config=CacheConfig(write_cache=False, use_cache=False),  # avoids the Windows file-lock crash
)