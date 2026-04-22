import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from prompt import build_prompt
from model import call_model_deepseek
from utils import load_json, normalize_sample, judge_prediction, normalize_text


# =========================
# run single test sample
# =========================
def run_one_sample(sample: Dict[str, Any], method: str) -> Dict[str, Any]:
    prompt = build_prompt(sample, method=method)
    output_text = call_model_deepseek(prompt)
    judged = judge_prediction(output_text, sample)

    return {
        "sample_id": sample["sample_id"],
        "story_id": sample["story_id"],
        "method": method,
        "question_order": sample["question_order"],
        "deception": sample["deception"],
        "story_length": sample["story_length"],
        "question": sample["question"],
        "answer": sample["answer"],
        "prompt": prompt,
        **judged,
    }


# =========================
# run whole dataset
# =========================
def run_dataset(
    input_path: str,
    output_path: str,
    method: str = "VP",
    max_samples: Optional[int] = None,
) -> None:
    raw_data = load_json(input_path)
    samples = [normalize_sample(x) for x in raw_data]

    if max_samples is not None:
        samples = samples[:max_samples]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total = len(samples)
    correct = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples, start=1):
            try:
                result = run_one_sample(sample, method=method)
            except Exception as e:
                result = {
                    "sample_id": sample["sample_id"],
                    "story_id": sample["story_id"],
                    "method": method,
                    "question_order": sample["question_order"],
                    "deception": sample["deception"],
                    "story_length": sample["story_length"],
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prompt": build_prompt(sample, method=method),
                    "pred_raw": None,
                    "pred_final": None,
                    "gold": normalize_text(sample["answer"]),
                    "correct": 0,
                    "error": str(e),
                }

            correct += result["correct"]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(
                f"[{i}/{total}] sample_id={sample['sample_id']} "
                f"correct={result['correct']} "
                f"running_acc={correct / i:.4f}"
            )

    print(f"\nfinished。result saved to path: {output_path}")
    print(f"Final Accuracy: {correct}/{total} = {correct / total:.4f}")


# =========================
# result analysis
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def report_accuracy_by_order(result_path: str) -> None:
    rows = load_jsonl(result_path)

    stats = {}
    for r in rows:
        order = r["question_order"]
        stats.setdefault(order, {"correct": 0, "total": 0})
        stats[order]["correct"] += int(r["correct"])
        stats[order]["total"] += 1

    print("\nAccuracy by question_order")
    for order in sorted(stats.keys()):
        c = stats[order]["correct"]
        t = stats[order]["total"]
        print(f"order {order}: {c}/{t} = {c/t:.4f}")


# =========================
# main
# =========================
if __name__ == "__main__":
    # path to your input data and output results
    input_path = "data/hitom.json"
    output_path = "res/hitom_cotp_results.jsonl"

    run_dataset(
        input_path=input_path,
        output_path=output_path,
        method="CoTP",      # it could be "CoTP/VP"
        max_samples=1200,
    )

    report_accuracy_by_order(output_path)