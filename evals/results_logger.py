import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from deepeval import evaluate

RESULTS_FILE = Path(__file__).parent / "results" / "eval_summary.csv"
FIELDS = ["run_id", "eval_name", "label", "metric",
          "avg_score", "pass_rate_pct", "passed", "failed", "total"]


def save_run(result, eval_name, label, file=RESULTS_FILE):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # collect per-metric stats across all test cases
    stats = defaultdict(lambda: {"scores": [], "passed": 0, "total": 0})
    for tr in result.test_results:
        for m in tr.metrics_data:
            s = stats[m.name]
            s["total"] += 1
            if m.success:
                s["passed"] += 1
            if m.score is not None:
                s["scores"].append(m.score)

    if not stats:
        print("No results to save.")
        return

    rows = []
    for metric, s in stats.items():
        avg = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
        rows.append({
            "run_id": run_id,
            "eval_name": eval_name,
            "label": label,
            "metric": metric,
            "avg_score": round(avg, 2),
            "pass_rate_pct": round(100 * s["passed"] / s["total"], 2),
            "passed": s["passed"],
            "failed": s["total"] - s["passed"],
            "total": s["total"],
        })

    write_header = not file.exists()
    with open(file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} summary rows to {file}")


def run_and_save(test_cases, metrics, eval_name, label):
    """Runs the evaluation and saves the summary in one step."""
    result = evaluate(test_cases=test_cases, metrics=metrics)
    save_run(result, eval_name, label)
    return result