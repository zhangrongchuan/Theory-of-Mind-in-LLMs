import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from prompt import build_prompt, build_vp_prompt_bigtom, build_cotp_prompt_bigtom
#from model import call_model_deepseek, call_model_ollama, call_model_ollama_SoO, call_model_openrouter, call_model_openrouter_SoO, call_model_huggingface
from model_hf import call_model_hf, call_model_hf_SoO
from utils import (
    load_json,
    normalize_sample,
    judge_prediction,
    normalize_text,
    normalize_dataset_samples,
    judge_prediction_bigtom,
)

# Hi-ToM methods (original - DO NOT MODIFY)
from method.decompose_ToM import DecomposeToM
from method.perceptom import PercepToM
from method.soo import SoO
from method.s3ap import S3AP
from method.simtom import SimToM
from method.simtom_you import SimToMYou
from method.dwm import DiscreteWorldModel
from method.incrementaltom_compare import IncrementalToM
from method.batched_incrementaltom import (
    BatchedIncrementalToMRunner,
    create_batched_llm_callable,
    create_hf_batched_llm_callable,
)

# BigToM method adapters
from method.decompose_tom_bigtom import DecomposeToMBigToM
from method.perceptom_bigtom import PercepToMBigToM
from method.soo_bigtom import SoOBigToM
from method.s3ap_bigtom import S3APBigToM
from method.simtom_bigtom import SimToMBigToM
from method.simtom_you_bigtom import SimToMYouBigToM
from method.dwm_bigtom import DiscreteWorldModelBigToM
from method.incrementaltom_bigtom import IncrementalToMBigToM


# =========================
# Dataset-aware run sample functions
# =========================

def run_one_sample_hitom(sample: Dict[str, Any], method: str, chunk_size: int = 3) -> tuple[str, str, Optional[list]]:
    """
    Run a single Hi-ToM sample with the specified method.
    Returns: (output_text, prompt, belief_state_tracking)
    """
    method_upper = method.upper()
    belief_state_tracking = None
    prompt = ""
    output_text = ""

    # ---------------------------------------------------------
    # BRANCH: PercepToM
    # ---------------------------------------------------------
    if method_upper == "PERCEPTOM":
        try:
            tom_solver = PercepToM(llm_callable=call_model_hf)
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
            tom_solver = DecomposeToM(llm_callable=call_model_hf)
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
            tom_solver = SoO(llm_callable=call_model_hf_SoO)
            output_text = tom_solver.run(sample)
            prompt = f"SoO Pipeline initiated for Question: {sample['question']}"
        except Exception as e:
            print(f"CRASH DETECTED in SoO: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: S3AP
    # ---------------------------------------------------------
    elif method_upper == "S3AP":
        try:
            tom_solver = S3AP(llm_callable=call_model_hf)
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"S3AP Pipeline initiated for Question: {sample['question']}"
        except Exception as e:
            print(f"CRASH DETECTED in S3AP: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: SIMTOM
    # ---------------------------------------------------------
    elif method_upper == "SIMTOM":
        try:
            tom_solver = SimToM(llm_callable=call_model_hf)
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"SIMTOM Pipeline initiated for Question: {sample['question']}"
        except Exception as e:
            print(f"CRASH DETECTED in SIMTOM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: SIMTOM-YOU
    # ---------------------------------------------------------
    elif method_upper == "SIMTOMYOU":
        try:
            tom_solver = SimToMYou(llm_callable=call_model_hf)
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"SimToM-You Pipeline initiated for Question: {sample['question']}"
        except Exception as e:
            print(f"CRASH DETECTED in SimToM-You: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: DWM
    # ---------------------------------------------------------
    elif method_upper == "DWM":
        try:
            tom_solver = DiscreteWorldModel(llm_callable=call_model_hf)
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"DWM Pipeline initiated for Question: {sample['question']}"
        except Exception as e:
            print(f"CRASH DETECTED in DWM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: IncrementalToM
    # ---------------------------------------------------------
    elif method_upper == "INCREMENTALTOM":
        try:
            tom_solver = IncrementalToM(llm_callable=call_model_hf, chunk_size=chunk_size)
            result = tom_solver.run(sample, chunk_size=chunk_size)
            output_text = result["response"]
            belief_state_tracking = result.get("belief_state_tracking", [])
            prompt = f"IncrementalToM Pipeline (chunk_size={chunk_size}) initiated for Question: {sample['question']}"
        except Exception as e:
            print(f"CRASH DETECTED in IncrementalToM: {e}")
            output_text = ""
            belief_state_tracking = []
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: EXISTING LOGIC (CoTP, VP)
    # ---------------------------------------------------------
    else:
        prompt = build_prompt(sample, method=method)
        output_text = call_model_hf(prompt)

    return output_text, prompt, belief_state_tracking


def run_one_sample_bigtom(sample: Dict[str, Any], method: str, chunk_size: int = 3) -> tuple[str, str, Optional[list]]:
    """
    Run a single BigToM sample with the specified method.
    Returns: (output_text, prompt, belief_state_tracking)
    """
    method_upper = method.upper()
    belief_state_tracking = None
    prompt = ""
    output_text = ""

    # Extract BigToM-specific fields
    story = sample.get("story", sample.get("narrative", ""))
    question = sample["question"]
    true_answer = sample.get("true_answer", sample.get("answer", ""))
    wrong_answer = sample.get("wrong_answer", "")

    # ---------------------------------------------------------
    # BRANCH: PercepToM
    # ---------------------------------------------------------
    if method_upper == "PERCEPTOM":
        try:
            tom_solver = PercepToMBigToM(llm_callable=call_model_hf)
            output_text = tom_solver.run(sample)
            prompt = f"PercepToM-BigToM Pipeline initiated for Question: {question}"
        except Exception as e:
            print(f"CRASH DETECTED in PercepToM-BigToM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: Decompose-ToM
    # ---------------------------------------------------------
    elif method_upper == "DTOM":
        try:
            tom_solver = DecomposeToMBigToM(llm_callable=call_model_hf)
            output_text = tom_solver.run(
                story=story,
                question=question,
                true_answer=true_answer,
                wrong_answer=wrong_answer
            )
            prompt = f"Decompose-ToM-BigToM Pipeline initiated for:\nStory: {story}\nQuestion: {question}"
        except Exception as e:
            print(f"CRASH DETECTED in DToM-BigToM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: SoO
    # ---------------------------------------------------------
    elif method_upper == "SOO":
        try:
            tom_solver = SoOBigToM(llm_callable=call_model_hf_SoO)
            output_text = tom_solver.run(sample)
            prompt = f"SoO-BigToM Pipeline initiated for Question: {question}"
        except Exception as e:
            print(f"CRASH DETECTED in SoO-BigToM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: S3AP
    # ---------------------------------------------------------
    elif method_upper == "S3AP":
        try:
            tom_solver = S3APBigToM(llm_callable=call_model_hf)
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"S3AP-BigToM Pipeline initiated for Question: {question}"
        except Exception as e:
            print(f"CRASH DETECTED in S3AP-BigToM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: SIMTOM
    # ---------------------------------------------------------
    elif method_upper == "SIMTOM":
        try:
            tom_solver = SimToMBigToM(llm_callable=call_model_hf)
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"SIMTOM-BigToM Pipeline initiated for Question: {question}"
        except Exception as e:
            print(f"CRASH DETECTED in SIMTOM-BigToM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: SIMTOM-YOU
    # ---------------------------------------------------------
    elif method_upper == "SIMTOMYOU":
        try:
            tom_solver = SimToMYouBigToM(llm_callable=call_model_hf)
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"SimToM-You-BigToM Pipeline initiated for Question: {question}"
        except Exception as e:
            print(f"CRASH DETECTED in SimToM-You-BigToM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: DWM
    # ---------------------------------------------------------
    elif method_upper == "DWM":
        try:
            tom_solver = DiscreteWorldModelBigToM(llm_callable=call_model_hf)
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"DWM-BigToM Pipeline initiated for Question: {question}"
        except Exception as e:
            print(f"CRASH DETECTED in DWM-BigToM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: IncrementalToM
    # ---------------------------------------------------------
    elif method_upper == "INCREMENTALTOM":
        try:
            tom_solver = IncrementalToMBigToM(llm_callable=call_model_hf, chunk_size=chunk_size)
            result = tom_solver.run(sample, chunk_size=chunk_size)
            output_text = result["response"]
            belief_state_tracking = result.get("belief_state_tracking", [])
            prompt = f"IncrementalToM-BigToM Pipeline (chunk_size={chunk_size}) initiated for Question: {question}"
        except Exception as e:
            print(f"CRASH DETECTED in IncrementalToM-BigToM: {e}")
            output_text = ""
            belief_state_tracking = []
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: EXISTING LOGIC (CoTP, VP)
    # ---------------------------------------------------------
    else:
        if method_upper == "VP":
            prompt = build_vp_prompt_bigtom(sample)
        else:  # COTP
            prompt = build_cotp_prompt_bigtom(sample)
        output_text = call_model_hf(prompt)

    return output_text, prompt, belief_state_tracking


def run_one_sample(
    sample: Dict[str, Any],
    method: str,
    dataset_type: str,
    chunk_size: int = 3
) -> Dict[str, Any]:
    """
    Run a single sample with the specified method and dataset type.
    """
    if dataset_type.lower() == "bigtom":
        output_text, prompt, belief_state_tracking = run_one_sample_bigtom(sample, method, chunk_size)
        judged = judge_prediction_bigtom(output_text, sample, llm_judge_callable=call_model_hf)
    else:
        output_text, prompt, belief_state_tracking = run_one_sample_hitom(sample, method, chunk_size)
        judged = judge_prediction(output_text, sample)

    result = {
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

    # Include belief_state_tracking for IncrementalToM method
    if belief_state_tracking is not None:
        result["belief_state_tracking"] = belief_state_tracking

    # Include BigToM-specific fields if present
    if "bigtom_category" in sample:
        result["bigtom_category"] = sample["bigtom_category"]
    if "bigtom_condition" in sample:
        result["bigtom_condition"] = sample["bigtom_condition"]

    return result


# =========================
# upgrade failed samples
# =========================
def run_upgrade(
    input_path: str,
    output_path: str,
    method: str,
    dataset_type: str = "hitom",
    start_sample_id: Optional[str] = None,
    chunk_size: int = 3,
) -> None:
    output_file = Path(output_path)

    if not output_file.exists():
        raise FileNotFoundError(f"Nothing to upgrade. Output file '{output_path}' not found.")

    existing_results = load_jsonl(str(output_file))
    raw_data = load_json(input_path)

    # Map input dataset by sample_id for quick lookup
    samples_dict = {s["sample_id"]: s for s in normalize_dataset_samples(raw_data, dataset_type)}

    # Identify all indices in the output file where correct == 0
    failed_indices = [i for i, r in enumerate(existing_results) if int(r.get("correct", 0)) == 0]

    # If a start_sample_id is provided, filter the failed indices to resume from there
    if start_sample_id:
        try:
            start_idx = next(i for i, r in enumerate(existing_results) if str(r["sample_id"]) == str(start_sample_id))
            failed_indices = [i for i in failed_indices if i >= start_idx]
            print(f"Resuming upgrade from sample_id '{start_sample_id}'.")
        except StopIteration:
            raise ValueError(f"start_sample_id '{start_sample_id}' not found in the existing results.")

    if not failed_indices:
        print("No incorrect samples found to upgrade. Exiting.")
        return

    total_upgrades = len(failed_indices)
    print(f"Found {total_upgrades} samples to upgrade.")

    for step, idx in enumerate(failed_indices, start=1):
        old_result = existing_results[idx]
        sample_id = old_result["sample_id"]

        if sample_id not in samples_dict:
            print(f"Warning: sample_id '{sample_id}' not found in input data. Skipping.")
            continue

        sample = samples_dict[sample_id]

        try:
            new_result = run_one_sample(sample, method=method, dataset_type=dataset_type, chunk_size=chunk_size)
        except Exception as e:
            if method.upper() in ["VP", "COTP"]:
                if dataset_type.lower() == "bigtom":
                    fallback_prompt = build_vp_prompt_bigtom(sample) if method.upper() == "VP" else build_cotp_prompt_bigtom(sample)
                else:
                    fallback_prompt = build_prompt(sample, method=method)
            else:
                fallback_prompt = f"{method} execution crashed before prompt generation."

            new_result = {
                "sample_id": sample["sample_id"],
                "story_id": sample["story_id"],
                "method": method,
                "question_order": sample["question_order"],
                "deception": sample.get("deception"),
                "story_length": sample.get("story_length"),
                "question": sample["question"],
                "answer": sample["answer"],
                "prompt": fallback_prompt,
                "pred_raw": None,
                "pred_final": None,
                "gold": normalize_text(sample["answer"]),
                "correct": 0,
                "error": str(e),
            }

        # Update the result list in memory
        existing_results[idx] = new_result

        # Rewrite the entire file to ensure it's safely updated in place
        with open(output_file, "w", encoding="utf-8") as f:
            for res in existing_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        print(
            f"[Upgrade {step}/{total_upgrades}] sample_id={sample_id} | "
            f"correct_before=0 -> correct_after={new_result['correct']}"
        )

    # Calculate and report final accuracy after upgrade
    correct = sum(int(r.get("correct", 0)) for r in existing_results)
    total = len(existing_results)
    print(f"\nUpgrade finished. Result saved to path: {output_path}")
    print(f"Final Accuracy: {correct}/{total} = {correct / total:.4f}")


# =========================
# run whole dataset
# =========================
def run_dataset(
    input_path: str,
    output_path: str,
    method: str = "VP",
    dataset_type: str = "hitom",
    max_samples: Optional[int] = None,
    resume: bool = False,
    chunk_size: int = 3,
    batch_size: int = 8,
) -> None:
    raw_data = load_json(input_path)
    samples = normalize_dataset_samples(raw_data, dataset_type)

    if max_samples is not None:
        samples = samples[:max_samples]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total = len(samples)
    start_index = 0
    correct = 0
    open_mode = "w"

    if resume:
        if not output_file.exists():
            raise FileNotFoundError(f"Cannot resume: The benchmark file '{output_path}' does not exist.")

        existing_results = load_jsonl(str(output_file))
        if existing_results:
            last_sample_id = existing_results[-1]["sample_id"]

            # Recalculate previous correct answers for running accuracy tracking
            correct = sum(int(r.get("correct", 0)) for r in existing_results)

            # Find the index of the last processed sample in our dataset
            try:
                last_idx = next(i for i, s in enumerate(samples) if s["sample_id"] == last_sample_id)
                start_index = last_idx + 1
            except StopIteration:
                raise ValueError(f"Last sample_id '{last_sample_id}' from the output file was not found in the input dataset.")

            if start_index >= total:
                print("The benchmark is already fully completed. Exiting.")
                return

            print(f"Resuming benchmark. Skipping {start_index} already processed samples.")
            print(f"Starting from sample_id '{samples[start_index]['sample_id']}'.")
            open_mode = "a"

    samples_to_run = samples[start_index:]

    with open(output_file, open_mode, encoding="utf-8") as f:
        # Start `i` at `start_index + 1` so running accuracy calculates correctly against total progressed
        for i, sample in enumerate(samples_to_run, start=start_index + 1):
            try:
                result = run_one_sample(sample, method=method, dataset_type=dataset_type, chunk_size=chunk_size)
            except Exception as e:
                # Safely fallback without crashing on class-based methods
                if method.upper() in ["VP", "COTP"]:
                    if dataset_type.lower() == "bigtom":
                        fallback_prompt = build_vp_prompt_bigtom(sample) if method.upper() == "VP" else build_cotp_prompt_bigtom(sample)
                    else:
                        fallback_prompt = build_prompt(sample, method=method)
                else:
                    fallback_prompt = f"{method} execution crashed before prompt generation."

                result = {
                    "sample_id": sample["sample_id"],
                    "story_id": sample["story_id"],
                    "method": method,
                    "question_order": sample["question_order"],
                    "deception": sample.get("deception"),
                    "story_length": sample.get("story_length"),
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prompt": fallback_prompt,
                    "pred_raw": None,
                    "pred_final": None,
                    "gold": normalize_text(sample["answer"]),
                    "correct": 0,
                    "error": str(e),
                }
                # Ensure correct is an integer
                result["correct"] = int(result["correct"])

            correct += result["correct"]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()  # Force write to disk immediately

            print(
                f"[{i}/{total}] sample_id={sample['sample_id']} "
                f"correct={result['correct']} "
                f"running_acc={correct / i:.4f}"
            )

    print(f"\nfinished. result saved to path: {output_path}")
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


def report_accuracy_by_order(result_path: str, dataset_type: str = "hitom") -> None:
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

    # For BigToM, also report by category and condition
    if dataset_type.lower() == "bigtom":
        category_stats = {}
        for r in rows:
            category = r.get("bigtom_category", "unknown")
            category_stats.setdefault(category, {"correct": 0, "total": 0})
            category_stats[category]["correct"] += int(r["correct"])
            category_stats[category]["total"] += 1

        print("\nAccuracy by BigToM category")
        for category in sorted(category_stats.keys()):
            c = category_stats[category]["correct"]
            t = category_stats[category]["total"]
            print(f"{category}: {c}/{t} = {c/t:.4f}")


# =========================
# main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ToM Benchmarks")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hitom", "bigtom"],
        default="hitom",
        help="Dataset to use (hitom or bigtom)"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["CoTP", "VR"],
        required=False,
        help="Category of the Hi-ToM dataset to use (CoTP or VR) - only used for hitom"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["PercepToM", "SoO", "DTOM", "S3AP", "SIMTOM", "SIMTOMYOU", "DWM", "IncrementalToM", "VP", "COTP"],
        required=True,
        help="Method to benchmark"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume benchmark from existing output file if it was interrupted."
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Check the existing output file and strictly re-run only the samples marked as correct=0."
    )
    parser.add_argument(
        "--start_sample_id",
        type=str,
        default=None,
        help="When using --upgrade, specify a sample_id to resume the upgrade process from."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3,
        help="For IncrementalToM method: number of sentences per chunk (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples to process in parallel for GPU efficiency (default: 8)"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Override the default input file path"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Override the default output file path"
    )

    args = parser.parse_args()

    # Set default paths based on dataset
    if args.input_path:
        input_path = args.input_path
    else:
        if args.dataset.lower() == "bigtom":
            input_path = "data/bigtom_balanced_subset.json"
        else:
            input_path = "data/hitom_project_sub.json"

    if args.output_path:
        output_path = args.output_path
    else:
        # Build output path based on dataset, method, and model
        dataset_name = args.dataset.lower()
        method_name = args.method.lower()
        if args.dataset.lower() == "hitom" and args.category:
            category_part = args.category.lower()
        else:
            category_part = "all"
        output_path = f"res/{dataset_name}_{category_part}_results_{method_name}_qwen3_0_6.jsonl"

    print(f"Starting benchmark...")
    print(f"Dataset: {args.dataset} | Method: {args.method}")
    print(f"Input: {input_path} | Output: {output_path}")

    try:
        if args.upgrade:
            print("Mode: UPGRADE (Re-running incorrect samples)")
            run_upgrade(
                input_path=input_path,
                output_path=output_path,
                method=args.method,
                dataset_type=args.dataset,
                start_sample_id=args.start_sample_id,
                chunk_size=args.chunk_size,
            )
        else:
            print("Mode: STANDARD RUN")
            run_dataset(
                input_path=input_path,
                output_path=output_path,
                method=args.method,
                dataset_type=args.dataset,
                max_samples=args.max_samples,
                resume=args.resume,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
            )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    report_accuracy_by_order(output_path, dataset_type=args.dataset)
