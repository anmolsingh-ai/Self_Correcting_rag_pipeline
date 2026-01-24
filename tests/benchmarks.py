import time
import numpy as np


def test_edge_cases(qa_chain):
    print("\n🧪 Running edge case tests...")
    cases = [
        ("", "Empty query"),
        ("@#$%^&*", "Special chars"),
        ("it", "Too vague"),
    ]

    passed = 0
    for question, name in cases:
        try:
            response = qa_chain({"question": question})
            ok = bool(response.get("answer", "")) or question == ""
            status = "✅" if ok else "❌"
            passed += ok
        except Exception:
            status = "⚠"

        print(f"{status} {name}")

    return passed == len(cases)


def test_latency(qa_chain):
    print("\n⏱ Running latency tests...")
    questions = [
        "What is this document about?",
        "Give me a short summary.",
        "List the main topics.",
    ]

    times = []
    for q in questions:
        start = time.time()
        qa_chain({"question": q})
        times.append(time.time() - start)

    avg = np.mean(times)
    print(f"Average latency: {avg:.2f}s")

    if avg < 2:
        print("✅ Excellent speed")
    elif avg < 5:
        print("⚠ Acceptable speed")
    else:
        print("❌ Slow — optimize retrieval")

    return avg
