import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from prompt import build_prompt
from model import call_model_deepseek, call_model_qwen8b, call_model_huggingface
from utils import load_json, normalize_sample, judge_prediction, normalize_text

from method.decompose_ToM import DecomposeToM
from method.perceptom import PercepToM
from method.soo import SoO
from method.s3ap import S3AP
from method.simtom import SimToM
from method.dwm import DiscreteWorldModel
from method.incrementaltom import IncrementalToM
from method.shared_evidence_tom import SharedEvidenceToM


def canonical_method_name(method: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", method.upper().replace("³", "3"))


def infer_question_order(question: str) -> int:
    return len(re.findall(r"\bthinks?\b", question, flags=re.IGNORECASE))


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def build_result(
    sample: Dict[str, Any],
    method: str,
    prompt: str,
    output_text: str,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
        **(extra_fields or {}),
    }


def build_error_result(
    sample: Dict[str, Any],
    method: str,
    error: Exception,
    fallback_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    if fallback_prompt is None:
        if canonical_method_name(method) in ["VP", "COTP"]:
            fallback_prompt = build_prompt(sample, method=method)
        else:
            fallback_prompt = f"{method} execution crashed before prompt generation."

    return {
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
        "error": str(error),
    }


def model_name_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")


def build_output_path(input_path: str, category: str, method: str, model_name: str) -> str:
    dataset_slug = slug(Path(input_path).stem)
    category_slug = slug(category)
    method_slug = slug(method)
    model_slug = model_name_slug(model_name)
    return str(Path("res") / f"{dataset_slug}_{category_slug}_{method_slug}_{model_slug}.jsonl")


def run_sharedevidencetom_solver(
    sample: Dict[str, Any],
    qwen_model: str,
    qwen_max_new_tokens: int,
) -> tuple[str, str, Dict[str, Any]]:
    solver = SharedEvidenceToM(
        llm_callable=lambda prompt: call_model_huggingface(
            prompt,
            model_name=qwen_model,
            max_new_tokens=qwen_max_new_tokens,
        )
    )
    output_text = solver.run(sample)
    prompt = solver.last_qa_prompt or (
        f"SHAREDEVIDENCETOM Pipeline initiated for Question: "
        f"{sample['question']}"
    )
    extra_fields = {
        "shared_evidence": solver.last_evidence,
        "shared_evidence_prompt": solver.last_evidence_prompt,
        "core": solver.last_core,
        "core_prompt": solver.last_core_prompt,
        "qa_prompt": solver.last_qa_prompt,
        "model_backend": "huggingface",
        "model_name": qwen_model,
    }
    return output_text, prompt, extra_fields


def run_incrementaltom_solver(
    sample: Dict[str, Any],
    qwen_model: str,
    qwen_max_new_tokens: int,
    chunk_size: int,
) -> tuple[str, str, Dict[str, Any]]:
    solver = IncrementalToM(
        llm_callable=lambda prompt: call_model_huggingface(
            prompt,
            model_name=qwen_model,
            max_new_tokens=qwen_max_new_tokens,
        ),
        chunk_size=chunk_size,
    )
    output_text = solver.run(sample, chunk_size=chunk_size)
    prompt = solver.last_final_prompt or (
        f"INCREMENTALTOM Pipeline (chunk_size={chunk_size}) "
        f"initiated for Question: {sample['question']}"
    )
    extra_fields = {
        "incrementaltom_chunk_size": solver.chunk_size,
        "incrementaltom_intermediate_prompts": solver.last_intermediate_prompts,
        "incrementaltom_intermediate_answers": solver.last_intermediate_answers,
        "incrementaltom_final_prompt": solver.last_final_prompt,
        "model_backend": "huggingface",
        "model_name": qwen_model,
    }
    return output_text, prompt, extra_fields


# =========================
# run single test sample
# =========================
def run_one_sample(
    sample: Dict[str, Any],
    method: str,
    qwen_model: str = "Qwen/Qwen3-1.7B",
    qwen_max_new_tokens: int = 1024,
    chunk_size: int = 3,
) -> Dict[str, Any]:
    method_upper = canonical_method_name(method)
    extra_fields: Dict[str, Any] = {}
    
    # ---------------------------------------------------------
    # BRANCH: PercepToM
    # ---------------------------------------------------------
    if method_upper == "PERCEPTOM":
        try:
            tom_solver = PercepToM(llm_callable=lambda prompt: call_model_huggingface(
                    prompt,
                    model_name=qwen_model,
                    max_new_tokens=qwen_max_new_tokens,
                ))
            output_text = tom_solver.run(sample)
            prompt = f"PercepToM Pipeline initiated for Question: {sample['question']}"
            extra_fields["perceptom_final_answer"] = output_text
            extra_fields["perceptom_final_correct"] = judge_prediction(
                output_text,
                sample,
            )["correct"]
            extra_fields["model_backend"] = "huggingface"
            extra_fields["model_name"] = qwen_model
        except Exception as e:
            print(f"CRASH DETECTED in PercepToM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: Decompose-ToM
    # ---------------------------------------------------------
    elif method_upper == "DTOM":
        try:
            tom_solver = DecomposeToM(llm_callable=call_model_deepseek)
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
            tom_solver = SoO(llm_callable=lambda prompt: call_model_huggingface(
                    prompt,
                    model_name=qwen_model,
                    max_new_tokens=qwen_max_new_tokens,
                ))
            output_text = tom_solver.run(sample)
            prompt = f"SoO Pipeline initiated for Question: {sample['question']}"
        except Exception as e:
            print(f"CRASH DETECTED in SoO: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: S3AP / Social World Models
    # ---------------------------------------------------------
    elif method_upper == "S3AP":
        try:
            tom_solver = S3AP(llm_callable=call_model_qwen8b)
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"S3AP Pipeline initiated for Question: {sample['question']}"
            extra_fields["s3ap_representation"] = tom_solver.last_s3ap_representation
            extra_fields["s3ap_parser_prompt"] = tom_solver.last_parser_prompt
        except Exception as e:
            print(f"CRASH DETECTED in S3AP: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: SIMTOM
    # ---------------------------------------------------------
    elif method_upper == "SIMTOM":
        try:
            tom_solver = SimToM(
                llm_callable=lambda prompt: call_model_huggingface(
                    prompt,
                    model_name=qwen_model,
                    max_new_tokens=qwen_max_new_tokens,
                )
            )
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"SIMTOM Pipeline initiated for Question: {sample['question']}"
            extra_fields["simtom_perspective"] = tom_solver.last_perspective
            extra_fields["simtom_perspective_prompt"] = tom_solver.last_perspective_prompt
            extra_fields["model_backend"] = "huggingface"
            extra_fields["model_name"] = qwen_model
        except Exception as e:
            print(f"CRASH DETECTED in SIMTOM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: SHAREDEVIDENCETOM
    # ---------------------------------------------------------
    elif method_upper == "SHAREDEVIDENCETOM":
        try:
            output_text, prompt, extra_fields = run_sharedevidencetom_solver(
                sample=sample,
                qwen_model=qwen_model,
                qwen_max_new_tokens=qwen_max_new_tokens,
            )
        except Exception as e:
            print(f"CRASH DETECTED in SHAREDEVIDENCETOM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: INCREMENTALTOM
    # ---------------------------------------------------------
    elif method_upper == "INCREMENTALTOM":
        try:
            output_text, prompt, extra_fields = run_incrementaltom_solver(
                sample=sample,
                qwen_model=qwen_model,
                qwen_max_new_tokens=qwen_max_new_tokens,
                chunk_size=chunk_size,
            )
        except Exception as e:
            print(f"CRASH DETECTED in INCREMENTALTOM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: assemableTom
    # ---------------------------------------------------------
    elif method_upper == "ASSEMABLETOM":
        try:
            question_order = infer_question_order(sample["question"])
            if question_order <= 2:
                output_text, prompt, extra_fields = run_incrementaltom_solver(
                    sample=sample,
                    qwen_model=qwen_model,
                    qwen_max_new_tokens=qwen_max_new_tokens,
                    chunk_size=chunk_size,
                )
                extra_fields["assemabletom_route"] = "INCREMENTALTOM"
                extra_fields["assemabletom_inferred_order"] = question_order
            else:
                output_text, prompt, extra_fields = run_sharedevidencetom_solver(
                    sample=sample,
                    qwen_model=qwen_model,
                    qwen_max_new_tokens=qwen_max_new_tokens,
                )
                extra_fields["assemabletom_route"] = "SHAREDEVIDENCETOM"
                extra_fields["assemabletom_inferred_order"] = question_order
        except Exception as e:
            print(f"CRASH DETECTED in assemableTom: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: DWM / Discrete World Models
    # ---------------------------------------------------------
    elif method_upper == "DWM":
        try:
            tom_solver = DiscreteWorldModel(
                llm_callable=call_model_deepseek,
                num_splits=3,
            )
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or f"DWM Pipeline initiated for Question: {sample['question']}"
            extra_fields["dwm_chunks"] = tom_solver.last_chunks
            extra_fields["dwm_state_descriptions"] = tom_solver.last_state_descriptions
            extra_fields["dwm_state_prompts"] = tom_solver.last_state_prompts
        except Exception as e:
            print(f"CRASH DETECTED in DWM: {e}")
            output_text = ""
            prompt = "Error during execution"

    # ---------------------------------------------------------
    # BRANCH: EXISTING LOGIC (CoTP, VP)
    # ---------------------------------------------------------
    else:
        # Paper arg is removed entirely
        prompt = build_prompt(sample, method=method)
        if method_upper == "VP":
            output_text = call_model_huggingface(
                prompt,
                model_name=qwen_model,
                max_new_tokens=qwen_max_new_tokens,
            )
            extra_fields["model_backend"] = "huggingface"
            extra_fields["model_name"] = qwen_model
        else:
            output_text = call_model_deepseek(prompt)
            extra_fields["model_backend"] = "deepseek"
    
    return build_result(
        sample=sample,
        method=method,
        prompt=prompt,
        output_text=output_text,
        extra_fields=extra_fields,
    )


# =========================
# run whole dataset
# =========================
def run_dataset(
    input_path: str,
    output_path: str,
    category: Optional[str] = None,
    method: str = "VP",
    max_samples: Optional[int] = None,
    qwen_model: str = "Qwen/Qwen3-1.7B",
    qwen_max_new_tokens: int = 1024,
    chunk_size: int = 3,
) -> None:
    raw_data = load_json(input_path)

    if category is not None:
        category_key = category.lower()
        filtered_data = [
            x for x in raw_data
            if str(x.get("prompting_type", "")).lower() == category_key
        ]
        if not filtered_data:
            available = sorted(
                {
                    str(x.get("prompting_type"))
                    for x in raw_data
                    if x.get("prompting_type") is not None
                }
            )
            available_text = ", ".join(available) if available else "none"
            raise ValueError(
                f"No samples found for category '{category}'. "
                f"Available prompting_type values: {available_text}."
            )
        raw_data = filtered_data

    samples = [normalize_sample(x) for x in raw_data]

    if max_samples is not None:
        samples = samples[:max_samples]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total = len(samples)
    if total == 0:
        raise ValueError("No samples to run after filtering.")

    correct = 0
    order_stats: Dict[Any, Dict[str, int]] = {}

    with open(output_file, "w", encoding="utf-8", newline="\n") as f:
        def write_result(i: int, result: Dict[str, Any]) -> None:
            nonlocal correct
            correct += int(result["correct"])
            order = result["question_order"]
            order_stats.setdefault(order, {"correct": 0, "total": 0})
            order_stats[order]["correct"] += int(result["correct"])
            order_stats[order]["total"] += 1

            f.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()

            print(
                f"[{i}/{total}] sample_id={result['sample_id']} "
                f"correct={result['correct']} "
                f"running_acc={correct / i:.4f}"
            )

        for i, sample in enumerate(samples, start=1):
            try:
                result = run_one_sample(
                    sample,
                    method=method,
                    qwen_model=qwen_model,
                    qwen_max_new_tokens=qwen_max_new_tokens,
                    chunk_size=chunk_size,
                )
            except Exception as e:
                result = build_error_result(sample, method=method, error=e)

            write_result(i, result)

    print(f"\nfinished。result saved to path: {output_path}")
    print(f"Final Accuracy: {correct}/{total} = {correct / total:.4f}")
    print_accuracy_by_order_stats(order_stats)


# =========================
# result analysis
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_no}: {e.msg} "
                        f"(column {e.colno})"
                    ) from e
    return rows


def print_accuracy_by_order_stats(stats: Dict[Any, Dict[str, int]]) -> None:
    print("\nAccuracy by question_order")
    for order in sorted(stats.keys()):
        c = stats[order]["correct"]
        t = stats[order]["total"]
        print(f"order {order}: {c}/{t} = {c/t:.4f}")


def report_accuracy_by_order(result_path: str) -> None:
    rows = load_jsonl(result_path)

    stats = {}
    for r in rows:
        order = r["question_order"]
        stats.setdefault(order, {"correct": 0, "total": 0})
        stats[order]["correct"] += int(r["correct"])
        stats[order]["total"] += 1

    print_accuracy_by_order_stats(stats)


# =========================
# main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ToM Benchmarks")
    parser.add_argument(
        "--category", 
        type=str, 
        choices=["CoTP", "VP"], 
        required=True, 
        help="Category label for the Hi-ToM run (CoTP or VP)"
    )
    parser.add_argument(
        "--method", 
        type=str, 
        choices=[
            "VP",
            "COTP",
            "PercepToM",
            "SoO",
            "DTOM",
            "S3AP",
            "SIMTOM",
            "DWM",
            "INCREMENTALTOM",
            "SHAREDEVIDENCETOM",
            "assemableTom",
        ],
        required=True, 
        help="Method of the paper to benchmark"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=1200, 
        help="Maximum number of samples to process (default: 1200)"
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Local HuggingFace Qwen model for VP and SIMTOM (default: Qwen/Qwen3-1.7B)"
    )
    parser.add_argument(
        "--qwen_max_new_tokens",
        type=int,
        default=1024,
        help="max_new_tokens for local HuggingFace Qwen generation (default: 1024)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3,
        help="Sentence chunk size for IncrementalToM (default: 3)"
    )
    
    args = parser.parse_args()

    input_path = "data/hitom.json" 
    output_path = build_output_path(
        input_path=input_path,
        category=args.category,
        method=args.method,
        model_name=args.qwen_model,
    )

    print(f"Starting benchmark...")
    print(f"Dataset: {args.category} | Method: {args.method}")
    method_key = canonical_method_name(args.method)
    if method_key in {"VP", "SIMTOM", "INCREMENTALTOM", "SHAREDEVIDENCETOM", "ASSEMABLETOM"}:
        print(f"Qwen model: {args.qwen_model}")
    if method_key in {"INCREMENTALTOM", "ASSEMABLETOM"}:
        print(f"Chunk size: {args.chunk_size}")
    print(f"Input: {input_path} | Output: {output_path}")

    run_dataset(
        input_path=input_path,
        output_path=output_path,
        category=args.category,
        method=args.method,      
        max_samples=args.max_samples,
        qwen_model=args.qwen_model,
        qwen_max_new_tokens=args.qwen_max_new_tokens,
        chunk_size=args.chunk_size,
    )

