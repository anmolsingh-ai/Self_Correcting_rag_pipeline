from dotenv import load_dotenv
import json
import os
from pathlib import Path
from evals.results_logger import run_and_save
from deepeval import evaluate
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from main import build_pipeline

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "goldens" / "correctness_golden.json"
PDF_PATH = ROOT / "data" / "Notes.pdf"

JUDGE_MODEL = "gemini-2.5-flash"
THRESHOLD = 0.6

with open(GOLDEN_PATH, encoding="utf-8") as f:
    golden_data = json.load(f)

judge = GeminiModel(model=JUDGE_MODEL, api_key=api_key, temperature=0)

rag, retriever = build_pipeline(str(PDF_PATH), return_retriever=True)
test_cases = []
for i, g in enumerate(golden_data):
    question = g["query"]

    answer = rag.invoke(
        {"input": question},
        config={"configurable": {"session_id": f"eval_{i}"}},
    )
    if hasattr(answer, "content"):  
        answer = answer.content

    test_cases.append(
        LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=g["ideal_answer"],
        )
    )


correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Compare only the factual claims in the actual output against the expected output.",
        "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
        "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
        "Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.",
        "Additional correct information must NEVER lower the score.",
    ],
    rubric=[
    Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
    Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
    Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=judge,
    strict_mode=False,
)
completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points contained in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
        "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly --- factual correctness is judged separately.",
        "Do NOT penalize the actual output for adding extra information beyond the expected output.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
        Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
        Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=judge,
    strict_mode=False,
)


run_and_save(test_cases, metrics=[correctness,completeness], eval_name="application", label="baseline")