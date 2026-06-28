import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Callable

# =========================
# process choice text into dict
# =========================
def parse_choices(choice_text: str) -> Dict[str, str]:
    """
    Input:
        "A. blue_drawer, B. green_crate, C. red_bucket"
    Output:
        {"A": "blue_drawer", "B": "green_crate", "C": "red_bucket"}
    """
    pattern = r"([A-Z])\.\s*([^,]+)"
    matches = re.findall(pattern, choice_text)
    out = {}
    for k, v in matches:
        out[k.strip()] = v.strip()
    return out


def format_choices_for_prompt(choice_text: str) -> str:
    choices = parse_choices(choice_text)
    lines = [f"{k}. {v}" for k, v in choices.items()]
    return "\n".join(lines)

# =========================
# 1. load data
# =========================
def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("The input file must be a JSON list.")

    return data


# =========================
# 2. normalize text and sample, and generate story id for deduplication
# =========================
def make_story_id(story: str) -> str:
    return hashlib.md5(story.strip().encode("utf-8")).hexdigest()


def normalize_text(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def normalize_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    story = item["story"].strip()
    question = item["question"].strip()
    # Data files use "options" (a list) instead of "choices" (a string)
    options = item.get("options", [])
    choices = ", ".join(options) if isinstance(options, list) else item.get("choices", "").strip()
    answer = item["answer"].strip()

    return {
        "sample_id": item.get("sample_id"),
        "story_id": make_story_id(story),
        "prompting_type_raw": item.get("prompting_type"),
        "deception": item.get("deception"),
        "story_length": item.get("story_length"),
        "question_order": int(item.get("question_order")),
        "story": story,
        "question": question,
        "choices_raw": choices,
        "answer": answer,
    }

# =========================
# 3. parse model output/ prediction mapping
# =========================
def extract_option_letter(output_text: str) -> Optional[str]:
    """
    get an option letter A-O from model output, if any
    """
    text = output_text.strip()

    # case 1: just a single letter (VP prompt)
    if re.fullmatch(r"[A-O]", text, flags=re.IGNORECASE):
        return text.upper()

    # case 2: Answer: K (COTP prompt)
    m = re.search(r"answer\s*[:]\s*([A-O])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # case 3: the last appearing single letter
    matches = re.findall(r"\b([A-O])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return None

def map_prediction_to_location(pred_raw: str, sample: Dict[str, Any]) -> Optional[str]:
    """
    map model output to standard choice name
    """
    choices = parse_choices(sample["choices_raw"])
    gold = normalize_text(sample["answer"])
    text = pred_raw.strip()

    # first try to extract option letter and map to choice value
    letter = extract_option_letter(text)
    if letter is not None and letter in choices:
        return normalize_text(choices[letter])

    # then try to directly output the location name
    text_norm = normalize_text(text)
    if text_norm == gold:
        return text_norm

    # if the model outputs a full sentence, check if any option value is present
    for _, value in choices.items():
        value_norm = normalize_text(value)
        if value_norm in text_norm:
            return value_norm

    return None

def judge_prediction(pred_raw: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    gold = normalize_text(sample["answer"])
    pred_final = map_prediction_to_location(pred_raw, sample)

    return {
        "pred_raw": pred_raw,
        "pred_final": pred_final,
        "gold": gold,
        "correct": int(pred_final == gold) if pred_final is not None else 0,
    }


# =========================
# BigToM Dataset Support
# =========================

def parse_bigtom_q_type(q_type: str) -> Dict[str, Any]:
    """
    Parse BigToM q_type field to extract metadata.
    Format examples: "1_forward_action_false_belief", "0_forward_belief_true_control"

    Returns dict with:
    - question_order: 0 or 1 (first digit)
    - category: forward_action, forward_belief, backward_belief, percept_to_belief
    - condition: false_belief, true_belief, false_control, true_control
    - deception: True for false_belief, False for others (inferred)
    """
    parts = q_type.split("_")

    # First digit is question_order (0 or 1 in BigToM)
    question_order = int(parts[0]) if parts[0].isdigit() else 0

    # Extract category (forward_action, forward_belief, backward_belief, percept_to_belief)
    category_parts = []
    condition = ""
    for i, part in enumerate(parts[1:], 1):
        if part in ["false", "true"]:
            # Remaining parts form the condition
            condition = "_".join(parts[i:])
            break
        category_parts.append(part)
    category = "_".join(category_parts)

    # Infer deception from condition
    deception = "false_belief" in condition

    return {
        "question_order": question_order,
        "category": category,
        "condition": condition,
        "deception": deception,
    }


def normalize_bigtom_sample(item: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    """
    Normalize a BigToM sample to the standard format used by the benchmark.

    BigToM fields:
    - q_type: encodes question type, order, and condition
    - narrative: the story text
    - question: the question text
    - true_answer: correct answer (full sentence)
    - wrong_answer: incorrect answer (full sentence)
    """
    narrative = item.get("narrative", "").strip()
    question = item.get("question", "").strip()
    true_answer = item.get("true_answer", "").strip()
    wrong_answer = item.get("wrong_answer", "").strip()
    q_type = item.get("q_type", "")

    # Parse q_type for metadata
    q_type_info = parse_bigtom_q_type(q_type)

    # Generate unique IDs
    story_id = make_story_id(narrative)
    sample_id = f"bigtom_{q_type}_{sample_idx}"

    return {
        "sample_id": sample_id,
        "story_id": story_id,
        "prompting_type_raw": q_type,
        "deception": q_type_info["deception"],
        "story_length": None,  # BigToM doesn't have this
        "question_order": q_type_info["question_order"],
        "bigtom_category": q_type_info["category"],
        "bigtom_condition": q_type_info["condition"],
        "story": narrative,
        "question": question,
        "true_answer": true_answer,
        "wrong_answer": wrong_answer,
        "answer": true_answer,  # For compatibility with existing code
        # For BigToM, we don't have A-O choices, we have binary true/wrong answers
        "choices_raw": f"TRUE: {true_answer}, WRONG: {wrong_answer}",
    }


def build_llm_judge_prompt(prediction: str, true_answer: str, wrong_answer: str) -> str:
    """
    Build a prompt for an LLM to judge whether a prediction matches the true answer.
    """
    return f"""You are judging whether a model's prediction correctly answers a question.

    Question's True Answer: {true_answer}
    Question's Wrong Answer: {wrong_answer}

    Model's Prediction: {prediction}

    Does the model's prediction match the TRUE answer (not the wrong answer)?

    Instructions:
    1. Compare the model's prediction to the TRUE answer semantically.
    2. The prediction doesn't need to be identical word-for-word, but should convey the same meaning.
    3. The prediction should NOT match the WRONG answer.

    Respond with ONLY "YES" if the prediction matches the true answer, or "NO" if it matches the wrong answer or is ambiguous/incorrect."""


def judge_prediction_bigtom(
    pred_raw: str,
    sample: Dict[str, Any],
    llm_judge_callable: Optional[Callable[[str], str]] = None
) -> Dict[str, Any]:
    """
    Judge a BigToM prediction using an LLM as judge.

    Args:
        pred_raw: The raw model prediction text
        sample: The normalized BigToM sample containing true_answer and wrong_answer
        llm_judge_callable: Optional callable to use as LLM judge.
                         If None, uses simple text matching fallback.

    Returns:
        Dict with pred_raw, pred_final, gold, correct, and judge_reasoning
    """
    true_answer = sample.get("true_answer", sample["answer"])
    wrong_answer = sample.get("wrong_answer", "")

    pred_clean = pred_raw.strip()

    # Normalize for comparison
    pred_norm = normalize_text(pred_clean)
    true_norm = normalize_text(true_answer)
    wrong_norm = normalize_text(wrong_answer)

    # Try simple matching first
    correct = None
    judge_reasoning = ""

    # Check for exact or near-exact match
    if pred_norm == true_norm or true_norm in pred_norm or pred_norm in true_norm:
        correct = 1
        judge_reasoning = "Direct match with true answer"
    elif pred_norm == wrong_norm or wrong_norm in pred_norm or pred_norm in wrong_norm:
        correct = 0
        judge_reasoning = "Direct match with wrong answer"

    # If LLM judge is available and no clear match, use it
    if correct is None and llm_judge_callable is not None:
        judge_prompt = build_llm_judge_prompt(pred_clean, true_answer, wrong_answer)
        judge_response = llm_judge_callable(judge_prompt).strip().upper()
        correct = 1 if "YES" in judge_response else 0
        judge_reasoning = f"LLM judge response: {judge_response}"

    # Fallback: if no clear match and no LLM judge, mark as incorrect
    if correct is None:
        correct = 0
        judge_reasoning = "No clear match found with either answer"

    return {
        "pred_raw": pred_raw,
        "pred_final": pred_clean,
        "gold": true_answer,
        "correct": correct,
        "judge_reasoning": judge_reasoning,
    }


def normalize_dataset_samples(raw_data: List[Dict[str, Any]], dataset_type: str = "hitom") -> List[Dict[str, Any]]:
    """
    Normalize samples from either Hi-ToM or BigToM datasets.

    Args:
        raw_data: List of raw sample dicts from JSON
        dataset_type: "hitom" or "bigtom"

    Returns:
        List of normalized samples in standard format
    """
    if dataset_type.lower() == "bigtom":
        return [normalize_bigtom_sample(item, idx) for idx, item in enumerate(raw_data)]
    else:  # default to hitom
        return [normalize_sample(item) for item in raw_data]