from dotenv import load_dotenv
import os
from pathlib import Path
import json
from evals.results_logger import run_and_save
from deepeval import evaluate

from deepeval.models import GeminiModel
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from src.conversational_chain import build_generator, format_chunks
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "goldens" / "generator_golden.json"
PDF_PATH = ROOT / "data" / "Notes.pdf"

JUDGE_MODEL = "gemini-2.5-flash"
THRESHOLD = 0.7

# Golden questions
with open(GOLDEN_PATH, encoding="utf-8") as f:
    golden_data = json.load(f)

judge = GeminiModel(model=JUDGE_MODEL, api_key=os.environ["GEMINI_API_KEY"], temperature=0)
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=api_key, temperature=0
    )
generator = build_generator(llm)
test_cases = []


for g in golden_data:
    ideal = g["ideal_context"]                       # no trailing comma
    context_list = ideal if isinstance(ideal, list) else [ideal]   # list of strings for DeepEval
    context_str = "\n\n".join(context_list)          # single string for your generator prompt

    answer = generator.invoke({
        "standalone_question": g["query"],
        "context": context_str,
    })
    if hasattr(answer, "content"):                   # AIMessage -> str
        answer = answer.content

    test_cases.append(
        LLMTestCase(
            input=g["query"],
            actual_output=answer,
            retrieval_context=context_list,          # must be a list[str]
        )
    )
metrics = [
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,   
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),
]

result = run_and_save(test_cases, metrics, eval_name="generator", label="baseline")
