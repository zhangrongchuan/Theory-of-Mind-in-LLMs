import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from prompt import build_prompt
from model import call_model_deepseek, call_model_ollama, call_model_ollama_SoO
from utils import load_json, normalize_sample, judge_prediction, normalize_text

from decompose_ToM import DecomposeToM
from perceptom import PercepToM
from soo import SoO  # <-- Imported the new SoO class

# =========================
# run single test sample
# =========================
def run_one_sample(sample: Dict[str, Any], method: str) -> Dict[str, Any]:
    method_upper = method.upper()
    
    # ---------------------------------------------------------
    # BRANCH: PercepToM
    # ---------------------------------------------------------
    if method_upper == "PERCEPTOM":
        try:
            tom_solver = PercepToM(llm_callable=call_model_ollama)
            output_text = tom_solver.run(sample)
            prompt = f"PercepToM Pipeline initiated for Question: {sample['question']}"
        except Exception as e:
            print(f"CRASH DETECTED in PercepToM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: Decompose-ToM
    # ---------------------------------------------------------
    elif method_upper == "DTOM":
        try:
            tom_solver = DecomposeToM(llm_callable=call_model_ollama)
            raw_story = sample.get("story", sample.get("context", "")) 
            raw_question = sample["question"]
            
            output_text = tom_solver.run(story=raw_story, question=raw_question)
            prompt = f"Decompose-ToM Pipeline initiated for:\nStory: {raw_story}\nQuestion: {raw_question}"
        except Exception as e:
            print(f"CRASH DETECTED in DToM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: SoO
    # ---------------------------------------------------------
    elif method_upper == "SOO":
        try:
            # Initialize with the specialized SoO model call from model.py
            tom_solver = SoO(llm_callable=call_model_ollama_SoO)
            output_text = tom_solver.run(sample)
            prompt = f"SoO Pipeline initiated for Question: {sample['question']}"
        except Exception as e:
            print(f"CRASH DETECTED in SoO: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: EXISTING LOGIC (CoTP, VP)
    # ---------------------------------------------------------
    else:
        # Paper arg is removed entirely
        prompt = build_prompt(sample, method=method)
        output_text = call_model_ollama(prompt)
    
    judged = judge_prediction(output_text, sample)

    return {
        "sample_id": sample["sample_id"],
        "story_id": sample["story_id"],
        "method": method,
        "question_order": sample["question_order"],
        "deception": sample.get("deception", None),
        "story_length": sample.get("story_length", None),
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
                # Safely fallback without crashing on class-based methods
                if method.upper() in ["VP", "COTP"]:
                    fallback_prompt = build_prompt(sample, method=method)
                else:
                    fallback_prompt = f"{method} execution crashed before prompt generation."

                result = {
                    "sample_id": sample["sample_id"],
                    "story_id": sample["story_id"],
                    "method": method,
                    "question_order": sample["question_order"],
                    "deception": sample["deception"],
                    "story_length": sample["story_length"],
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prompt": fallback_prompt,
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
    parser = argparse.ArgumentParser(description="Run ToM Benchmarks")
    parser.add_argument(
        "--category", 
        type=str, 
        choices=["CoTP", "VR"], 
        required=True, 
        help="Category of the Hi-ToM dataset to use (CoTP or VR)"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["PercepToM", "SoO", "DTOM"], 
        required=True, 
        help="Method of the paper to benchmark (PercepToM, SoO, or DTOM)"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=1200, 
        help="Maximum number of samples to process (default: 1200)"
    )
    
    args = parser.parse_args()

    input_path = "data/hitom.json" 
    output_path = f"res/hitom_{args.category.lower()}_results_{args.method.lower()}.jsonl"

    print(f"Starting benchmark...")
    print(f"Dataset: {args.category} | Method: {args.method}")
    print(f"Input: {input_path} | Output: {output_path}")

    run_dataset(
        input_path=input_path,
        output_path=output_path,
        method=args.method,      
        max_samples=args.max_samples,
    )

    report_accuracy_by_order(output_path)
