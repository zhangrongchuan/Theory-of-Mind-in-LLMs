import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from prompt import build_cotp_prompt_bigtom, build_prompt, build_vp_prompt_bigtom
from model import call_model_deepseek, call_model_qwen8b, call_model_huggingface
from model_hf import (
    call_model_hf as call_model_bigtom_hf,
    call_model_hf_SoO as call_model_bigtom_hf_soo,
)
from utils import (
    extract_bigtom_answer_label,
    judge_prediction,
    judge_prediction_bigtom,
    load_json,
    normalize_dataset_samples,
    normalize_text,
)

from method.decompose_ToM import DecomposeToM
from method.perceptom import PercepToM
from method.soo import SoO
from method.s3ap import S3AP
from method.simtom import SimToM
from method.simtom_you import SimToMYou
from method.dwm import DiscreteWorldModel
from method.incrementaltom import IncrementalToM
from method.shared_evidence_tom import SharedEvidenceToM
from method.decompose_tom_bigtom import DecomposeToMBigToM
from method.dwm_bigtom import DiscreteWorldModelBigToM
from method.incrementaltom_bigtom import IncrementalToMBigToM
from method.perceptom_bigtom import PercepToMBigToM
from method.s3ap_bigtom import S3APBigToM
from method.shared_evidence_tom_bigtom import SharedEvidenceToMBigToM
from method.simtom_bigtom import SimToMBigToM
from method.simtom_you_bigtom import SimToMYouBigToM
from method.soo_bigtom import SoOBigToM


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
    dataset_type: str = "hitom",
    belief_state_tracking: Optional[list] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if dataset_type.lower() == "bigtom":
        judged = judge_prediction_bigtom(output_text, sample)
    else:
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
        **(extra_fields or {}),
    }
    if belief_state_tracking is not None:
        result["belief_state_tracking"] = belief_state_tracking
    for field in ("bigtom_category", "bigtom_condition"):
        if field in sample:
            result[field] = sample[field]
    return result


def build_error_result(
    sample: Dict[str, Any],
    method: str,
    error: Exception,
    dataset_type: str = "hitom",
    fallback_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    if fallback_prompt is None:
        if canonical_method_name(method) in ["VP", "COTP"]:
            if dataset_type.lower() == "bigtom":
                fallback_prompt = (
                    build_vp_prompt_bigtom(sample)
                    if canonical_method_name(method) == "VP"
                    else build_cotp_prompt_bigtom(sample)
                )
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
        "error": str(error),
    }
    if dataset_type.lower() == "bigtom":
        result.update({
            "true_answer": sample.get("true_answer", sample["answer"]),
            "wrong_answer": sample.get("wrong_answer", ""),
            "bigtom_category": sample.get("bigtom_category"),
            "bigtom_condition": sample.get("bigtom_condition"),
            "judge_reasoning": "Execution error",
        })
    return result


def model_name_slug(model_name: str) -> str:
    short_name = model_name.rstrip("/").rsplit("/", 1)[-1]
    return slug(short_name)


def make_unique_output_path(output_path: str) -> str:
    """Append _2, _3, and so on when an output file already exists."""
    path = Path(output_path)
    if not path.exists():
        return str(path)

    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return str(candidate)
        index += 1


def build_output_path(dataset_type: str, method: str, model_name: str) -> str:
    dataset_slug = slug(dataset_type)
    method_slug = slug(canonical_method_name(method))
    model_slug = model_name_slug(model_name)
    base_path = Path("res") / f"{dataset_slug}_{method_slug}_{model_slug}.jsonl"
    return make_unique_output_path(str(base_path))


def find_latest_output_path(dataset_type: str, method: str, model_name: str) -> str:
    """Find the most recently numbered result path for resume or upgrade."""
    dataset_slug = slug(dataset_type)
    method_slug = slug(canonical_method_name(method))
    model_slug = model_name_slug(model_name)
    base_path = Path("res") / f"{dataset_slug}_{method_slug}_{model_slug}.jsonl"

    matches = []
    if base_path.exists():
        matches.append((1, base_path))

    pattern = re.compile(rf"^{re.escape(base_path.stem)}_(\d+){re.escape(base_path.suffix)}$")
    if base_path.parent.exists():
        for candidate in base_path.parent.glob(f"{base_path.stem}_*{base_path.suffix}"):
            match = pattern.fullmatch(candidate.name)
            if match and int(match.group(1)) >= 2:
                matches.append((int(match.group(1)), candidate))

    if not matches:
        return str(base_path)
    return str(max(matches, key=lambda item: item[0])[1])


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
        f"SharedEvidenceToM Pipeline initiated for Question: "
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


def run_sharedevidencetom_bigtom_solver(
    sample: Dict[str, Any],
    qwen_model: str,
    qwen_max_new_tokens: int,
) -> tuple[str, str, Dict[str, Any]]:
    solver = SharedEvidenceToMBigToM(
        llm_callable=lambda prompt: call_model_bigtom_hf(
            prompt,
            model_name=qwen_model,
            max_new_tokens=qwen_max_new_tokens,
        )
    )
    output_text = solver.run(sample)
    prompt = solver.last_qa_prompt or (
        f"SharedEvidenceToM-BigToM Pipeline initiated for Question: "
        f"{sample['question']}"
    )
    return output_text, prompt, {
        "shared_evidence": solver.last_evidence,
        "shared_evidence_prompt": solver.last_evidence_prompt,
        "core": solver.last_core,
        "core_prompt": solver.last_core_prompt,
        "qa_prompt": solver.last_qa_prompt,
        "model_backend": "huggingface",
        "model_name": qwen_model,
    }


def run_incrementaltom_bigtom_solver(
    sample: Dict[str, Any],
    qwen_model: str,
    qwen_max_new_tokens: int,
    chunk_size: int,
) -> tuple[str, str, list, Dict[str, Any]]:
    solver = IncrementalToMBigToM(
        llm_callable=lambda prompt: call_model_bigtom_hf(
            prompt,
            model_name=qwen_model,
            max_new_tokens=qwen_max_new_tokens,
        ),
        chunk_size=chunk_size,
    )
    result = solver.run(sample, chunk_size=chunk_size)
    prompt = (
        f"IncrementalToM-BigToM Pipeline (chunk_size={chunk_size}) "
        f"initiated for Question: {sample['question']}"
    )
    return result["response"], prompt, result.get("belief_state_tracking", []), {
        "model_backend": "huggingface",
        "model_name": qwen_model,
    }


# =========================
# run single test sample
# =========================
def run_one_sample_hitom(
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
    # BRANCH: SIMTOM-YOU
    # ---------------------------------------------------------
    elif method_upper == "SIMTOMYOU":
        try:
            tom_solver = SimToMYou(
                llm_callable=lambda prompt: call_model_huggingface(
                    prompt,
                    model_name=qwen_model,
                    max_new_tokens=qwen_max_new_tokens,
                )
            )
            output_text = tom_solver.run(sample)
            prompt = tom_solver.last_qa_prompt or (
                f"SimToM-You Pipeline initiated for Question: {sample['question']}"
            )
            extra_fields["simtom_you_perspective"] = tom_solver.last_perspective
            extra_fields["simtom_you_perspective_prompt"] = tom_solver.last_perspective_prompt
            extra_fields["model_backend"] = "huggingface"
            extra_fields["model_name"] = qwen_model
        except Exception as e:
            print(f"CRASH DETECTED in SIMTOMYOU: {e}")
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
    # BRANCH: AssembleToM
    # ---------------------------------------------------------
    elif method_upper == "ASSEMBLETOM":
        try:
            question_order = infer_question_order(sample["question"])
            if question_order <= 2:
                output_text, prompt, extra_fields = run_incrementaltom_solver(
                    sample=sample,
                    qwen_model=qwen_model,
                    qwen_max_new_tokens=qwen_max_new_tokens,
                    chunk_size=chunk_size,
                )
                extra_fields["assembletom_route"] = "INCREMENTALTOM"
                extra_fields["assembletom_inferred_order"] = question_order
            else:
                output_text, prompt, extra_fields = run_sharedevidencetom_solver(
                    sample=sample,
                    qwen_model=qwen_model,
                    qwen_max_new_tokens=qwen_max_new_tokens,
                )
                extra_fields["assembletom_route"] = "SHAREDEVIDENCETOM"
                extra_fields["assembletom_inferred_order"] = question_order
        except Exception as e:
            print(f"CRASH DETECTED in AssembleToM: {e}")
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


def run_one_sample_bigtom(
    sample: Dict[str, Any],
    method: str,
    qwen_model: str = "Qwen/Qwen3-1.7B",
    qwen_max_new_tokens: int = 2048,
    chunk_size: int = 3,
) -> Dict[str, Any]:
    """Run one normalized BigToM sample through its A/B method adapter."""
    method_upper = canonical_method_name(method)
    model_call = lambda prompt: call_model_bigtom_hf(
        prompt,
        model_name=qwen_model,
        max_new_tokens=qwen_max_new_tokens,
    )
    extra_fields: Dict[str, Any] = {
        "model_backend": "huggingface",
        "model_name": qwen_model,
    }
    belief_state_tracking: Optional[list] = None
    question = sample["question"]
    story = sample["story"]
    true_answer = sample["true_answer"]
    wrong_answer = sample["wrong_answer"]

    if method_upper == "PERCEPTOM":
        solver = PercepToMBigToM(llm_callable=model_call)
        output_text = solver.run(sample)
        prompt = f"PercepToM-BigToM Pipeline initiated for Question: {question}"

    elif method_upper == "DTOM":
        solver = DecomposeToMBigToM(llm_callable=model_call)
        output_text = solver.run(
            story=story,
            question=question,
            true_answer=true_answer,
            wrong_answer=wrong_answer,
        )
        prompt = (
            f"Decompose-ToM-BigToM Pipeline initiated for:\n"
            f"Story: {story}\nQuestion: {question}"
        )

    elif method_upper == "SOO":
        solver = SoOBigToM(
            llm_callable=lambda prompt: call_model_bigtom_hf_soo(
                prompt,
                model_name=qwen_model,
                max_new_tokens=qwen_max_new_tokens,
            )
        )
        output_text = solver.run(sample)
        prompt = f"SoO-BigToM Pipeline initiated for Question: {question}"

    elif method_upper == "S3AP":
        solver = S3APBigToM(llm_callable=model_call)
        output_text = solver.run(sample)
        prompt = solver.last_qa_prompt or (
            f"S3AP-BigToM Pipeline initiated for Question: {question}"
        )

    elif method_upper == "SIMTOM":
        solver = SimToMBigToM(llm_callable=model_call)
        output_text = solver.run(sample)
        prompt = solver.last_qa_prompt or (
            f"SIMTOM-BigToM Pipeline initiated for Question: {question}"
        )

    elif method_upper == "SIMTOMYOU":
        solver = SimToMYouBigToM(llm_callable=model_call)
        output_text = solver.run(sample)
        prompt = solver.last_qa_prompt or (
            f"SimToM-You-BigToM Pipeline initiated for Question: {question}"
        )

    elif method_upper == "DWM":
        solver = DiscreteWorldModelBigToM(llm_callable=model_call)
        output_text = solver.run(sample)
        prompt = solver.last_qa_prompt or (
            f"DWM-BigToM Pipeline initiated for Question: {question}"
        )

    elif method_upper == "INCREMENTALTOM":
        output_text, prompt, belief_state_tracking, solver_fields = (
            run_incrementaltom_bigtom_solver(
                sample=sample,
                qwen_model=qwen_model,
                qwen_max_new_tokens=qwen_max_new_tokens,
                chunk_size=chunk_size,
            )
        )
        extra_fields.update(solver_fields)

    elif method_upper == "SHAREDEVIDENCETOM":
        output_text, prompt, solver_fields = run_sharedevidencetom_bigtom_solver(
            sample=sample,
            qwen_model=qwen_model,
            qwen_max_new_tokens=qwen_max_new_tokens,
        )
        extra_fields.update(solver_fields)

    elif method_upper == "ASSEMBLETOM":
        question_order = int(sample["question_order"])
        if question_order <= 2:
            output_text, prompt, belief_state_tracking, solver_fields = (
                run_incrementaltom_bigtom_solver(
                    sample=sample,
                    qwen_model=qwen_model,
                    qwen_max_new_tokens=qwen_max_new_tokens,
                    chunk_size=chunk_size,
                )
            )
            route = "INCREMENTALTOM"
        else:
            output_text, prompt, solver_fields = run_sharedevidencetom_bigtom_solver(
                sample=sample,
                qwen_model=qwen_model,
                qwen_max_new_tokens=qwen_max_new_tokens,
            )
            route = "SHAREDEVIDENCETOM"
        extra_fields.update(solver_fields)
        extra_fields["assembletom_route"] = route
        extra_fields["assembletom_inferred_order"] = question_order

    elif method_upper == "VP":
        prompt = build_vp_prompt_bigtom(sample)
        output_text = model_call(prompt)

    elif method_upper == "COTP":
        prompt = build_cotp_prompt_bigtom(sample)
        output_text = model_call(prompt)

    else:
        raise ValueError(f"Unsupported BigToM method: {method}")

    return build_result(
        sample=sample,
        method=method,
        prompt=prompt,
        output_text=output_text,
        dataset_type="bigtom",
        belief_state_tracking=belief_state_tracking,
        extra_fields=extra_fields,
    )


def run_one_sample(
    sample: Dict[str, Any],
    method: str,
    dataset_type: str = "hitom",
    qwen_model: str = "Qwen/Qwen3-1.7B",
    qwen_max_new_tokens: Optional[int] = None,
    chunk_size: int = 3,
) -> Dict[str, Any]:
    if dataset_type.lower() == "bigtom":
        max_new_tokens = qwen_max_new_tokens if qwen_max_new_tokens is not None else 2048
        return run_one_sample_bigtom(
            sample,
            method=method,
            qwen_model=qwen_model,
            qwen_max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
        )
    if dataset_type.lower() == "hitom":
        max_new_tokens = qwen_max_new_tokens if qwen_max_new_tokens is not None else 1024
        return run_one_sample_hitom(
            sample,
            method=method,
            qwen_model=qwen_model,
            qwen_max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
        )
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


# =========================
# run whole dataset
# =========================
def run_dataset(
    input_path: str,
    output_path: str,
    category: Optional[str] = None,
    method: str = "VP",
    dataset_type: str = "hitom",
    max_samples: Optional[int] = None,
    resume: bool = False,
    qwen_model: str = "Qwen/Qwen3-1.7B",
    qwen_max_new_tokens: Optional[int] = None,
    chunk_size: int = 3,
) -> None:
    raw_data = load_json(input_path)

    if category is not None and dataset_type.lower() == "hitom":
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

    samples = normalize_dataset_samples(raw_data, dataset_type)

    if category is not None and dataset_type.lower() == "bigtom":
        category_key = category.lower()
        samples = [
            sample for sample in samples
            if sample.get("bigtom_category", "").lower() == category_key
            or sample.get("prompting_type_raw", "").lower() == category_key
        ]
        if not samples:
            raise ValueError(f"No BigToM samples found for category '{category}'.")

    if max_samples is not None:
        samples = samples[:max_samples]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total = len(samples)
    if total == 0:
        raise ValueError("No samples to run after filtering.")

    correct = 0
    order_stats: Dict[Any, Dict[str, int]] = {}
    start_index = 0
    open_mode = "w"

    if resume:
        if not output_file.exists():
            raise FileNotFoundError(
                f"Cannot resume: result file '{output_path}' does not exist."
            )
        existing_results = load_jsonl(str(output_file))
        if len(existing_results) > total:
            raise ValueError("Result file contains more rows than the selected dataset.")
        for index, existing in enumerate(existing_results):
            if str(existing.get("sample_id")) != str(samples[index]["sample_id"]):
                raise ValueError(
                    "Cannot resume because the result prefix does not match the "
                    f"dataset at row {index + 1}."
                )
            result_correct = int(existing.get("correct", 0))
            correct += result_correct
            order = existing["question_order"]
            order_stats.setdefault(order, {"correct": 0, "total": 0})
            order_stats[order]["correct"] += result_correct
            order_stats[order]["total"] += 1
        start_index = len(existing_results)
        if start_index == total:
            print("The benchmark is already complete.")
            print(f"Final Accuracy: {correct}/{total} = {correct / total:.4f}")
            print_accuracy_by_order_stats(order_stats)
            return
        open_mode = "a"
        print(f"Resuming benchmark after {start_index} completed samples.")

    with open(output_file, open_mode, encoding="utf-8", newline="\n") as f:
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

        for i, sample in enumerate(samples[start_index:], start=start_index + 1):
            try:
                result = run_one_sample(
                    sample,
                    method=method,
                    dataset_type=dataset_type,
                    qwen_model=qwen_model,
                    qwen_max_new_tokens=qwen_max_new_tokens,
                    chunk_size=chunk_size,
                )
            except Exception as e:
                result = build_error_result(
                    sample,
                    method=method,
                    error=e,
                    dataset_type=dataset_type,
                )

            write_result(i, result)

    print(f"\nFinished. Result saved to path: {output_path}")
    print(f"Final Accuracy: {correct}/{total} = {correct / total:.4f}")
    print_accuracy_by_order_stats(order_stats)


def run_upgrade(
    input_path: str,
    output_path: str,
    method: str,
    dataset_type: str = "hitom",
    category: Optional[str] = None,
    start_sample_id: Optional[str] = None,
    qwen_model: str = "Qwen/Qwen3-1.7B",
    qwen_max_new_tokens: Optional[int] = None,
    chunk_size: int = 3,
) -> None:
    """Re-run only rows currently marked incorrect in an existing result file."""
    output_file = Path(output_path)
    if not output_file.exists():
        raise FileNotFoundError(f"Nothing to upgrade: '{output_path}' does not exist.")

    raw_data = load_json(input_path)
    if category is not None and dataset_type.lower() == "hitom":
        raw_data = [
            row for row in raw_data
            if str(row.get("prompting_type", "")).lower() == category.lower()
        ]
    samples = normalize_dataset_samples(raw_data, dataset_type)
    if category is not None and dataset_type.lower() == "bigtom":
        samples = [
            sample for sample in samples
            if sample.get("bigtom_category", "").lower() == category.lower()
            or sample.get("prompting_type_raw", "").lower() == category.lower()
        ]

    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    existing_results = load_jsonl(str(output_file))
    failed_indices = [
        index for index, row in enumerate(existing_results)
        if int(row.get("correct", 0)) == 0
    ]

    if start_sample_id is not None:
        start_index = next(
            (
                index for index, row in enumerate(existing_results)
                if str(row.get("sample_id")) == str(start_sample_id)
            ),
            None,
        )
        if start_index is None:
            raise ValueError(f"start_sample_id '{start_sample_id}' was not found.")
        failed_indices = [index for index in failed_indices if index >= start_index]

    if not failed_indices:
        print("No incorrect samples found to upgrade.")
        return

    for step, index in enumerate(failed_indices, start=1):
        sample_id = str(existing_results[index].get("sample_id"))
        if sample_id not in sample_by_id:
            raise ValueError(f"sample_id '{sample_id}' was not found in the input dataset.")
        sample = sample_by_id[sample_id]
        try:
            new_result = run_one_sample(
                sample,
                method=method,
                dataset_type=dataset_type,
                qwen_model=qwen_model,
                qwen_max_new_tokens=qwen_max_new_tokens,
                chunk_size=chunk_size,
            )
        except Exception as error:
            new_result = build_error_result(
                sample,
                method=method,
                error=error,
                dataset_type=dataset_type,
            )
        existing_results[index] = new_result

        temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
        with open(temporary_file, "w", encoding="utf-8", newline="\n") as handle:
            for row in existing_results:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary_file.replace(output_file)
        print(
            f"[Upgrade {step}/{len(failed_indices)}] sample_id={sample_id} "
            f"correct_after={new_result['correct']}"
        )

    correct = sum(int(row.get("correct", 0)) for row in existing_results)
    total = len(existing_results)
    print(f"Upgrade finished. Result saved to path: {output_path}")
    print(f"Final Accuracy: {correct}/{total} = {correct / total:.4f}")


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


def report_accuracy_by_order(
    result_path: str,
    dataset_type: str = "hitom",
) -> None:
    rows = load_jsonl(result_path)

    if dataset_type.lower() == "bigtom":
        for row in rows:
            terminal_label = extract_bigtom_answer_label(row.get("pred_raw", ""))
            if terminal_label is not None:
                row["pred_final"] = terminal_label
                row["correct"] = int(terminal_label == "A")

    stats = {}
    for r in rows:
        order = r["question_order"]
        stats.setdefault(order, {"correct": 0, "total": 0})
        stats[order]["correct"] += int(r["correct"])
        stats[order]["total"] += 1

    print_accuracy_by_order_stats(stats)

    if dataset_type.lower() == "bigtom":
        for field, label in (
            ("bigtom_category", "BigToM category"),
            ("bigtom_condition", "BigToM condition"),
        ):
            grouped: Dict[str, Dict[str, int]] = {}
            for row in rows:
                key = str(row.get(field, "unknown"))
                grouped.setdefault(key, {"correct": 0, "total": 0})
                grouped[key]["correct"] += int(row["correct"])
                grouped[key]["total"] += 1
            print(f"\nAccuracy by {label}")
            for key in sorted(grouped):
                correct = grouped[key]["correct"]
                total = grouped[key]["total"]
                print(f"{key}: {correct}/{total} = {correct / total:.4f}")


# =========================
# main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ToM Benchmarks")
    parser.add_argument(
        "--dataset",
        choices=["hitom", "bigtom"],
        default="hitom",
        help="Dataset adapter to use (default: hitom).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help=(
            "Optional prompting_type filter for HiToM or question-category "
            "filter for BigToM."
        ),
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
            "SIMTOMYOU",
            "DWM",
            "INCREMENTALTOM",
            "IncrementalToM",
            "SHAREDEVIDENCETOM",
            "SharedEvidenceToM",
            "AssembleToM",
        ],
        required=True,
        help="Method of the paper to benchmark"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to process (default: all)."
    )
    parser.add_argument(
        "--qwen_model",
        "--model_name",
        dest="qwen_model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Local HuggingFace Qwen model for VP and SIMTOM (default: Qwen/Qwen3-1.7B)"
    )
    parser.add_argument(
        "--qwen_max_new_tokens",
        type=int,
        default=None,
        help=(
            "max_new_tokens for local HuggingFace generation "
            "(default: 1024 for HiToM, 2048 for BigToM)"
        )
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3,
        help="Sentence chunk size for IncrementalToM (default: 3)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted standard run from an existing JSONL file.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Re-run only incorrect rows in an existing JSONL result file.",
    )
    parser.add_argument(
        "--start_sample_id",
        type=str,
        default=None,
        help="Start the upgrade pass at this sample_id.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Override the dataset's default input path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Override the generated result path.",
    )
    
    args = parser.parse_args()

    input_path = args.input_path or (
        "data/bigtom_balanced_subset.json"
        if args.dataset == "bigtom"
        else "data/hitom.json"
    )
    if args.output_path:
        output_path = (
            args.output_path
            if args.resume or args.upgrade
            else make_unique_output_path(args.output_path)
        )
    elif args.resume or args.upgrade:
        output_path = find_latest_output_path(
            dataset_type=args.dataset,
            method=args.method,
            model_name=args.qwen_model,
        )
    else:
        output_path = build_output_path(
            dataset_type=args.dataset,
            method=args.method,
            model_name=args.qwen_model,
        )

    print(f"Starting benchmark...")
    print(f"Dataset: {args.dataset} | Category: {args.category or 'all'} | Method: {args.method}")
    method_key = canonical_method_name(args.method)
    if method_key in {"VP", "SIMTOM", "INCREMENTALTOM", "SHAREDEVIDENCETOM", "ASSEMBLETOM"}:
        print(f"Qwen model: {args.qwen_model}")
    if method_key in {"INCREMENTALTOM", "ASSEMBLETOM"}:
        print(f"Chunk size: {args.chunk_size}")
    print(f"Input: {input_path} | Output: {output_path}")

    if args.upgrade:
        run_upgrade(
            input_path=input_path,
            output_path=output_path,
            category=args.category,
            method=args.method,
            dataset_type=args.dataset,
            start_sample_id=args.start_sample_id,
            qwen_model=args.qwen_model,
            qwen_max_new_tokens=args.qwen_max_new_tokens,
            chunk_size=args.chunk_size,
        )
    else:
        run_dataset(
            input_path=input_path,
            output_path=output_path,
            category=args.category,
            method=args.method,
            dataset_type=args.dataset,
            max_samples=args.max_samples,
            resume=args.resume,
            qwen_model=args.qwen_model,
            qwen_max_new_tokens=args.qwen_max_new_tokens,
            chunk_size=args.chunk_size,
        )

    report_accuracy_by_order(output_path, dataset_type=args.dataset)
