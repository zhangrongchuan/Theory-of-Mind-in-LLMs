import json
import re
import hashlib
from typing import Callable, Dict, List, Any, Optional

# =========================
# process choice text into dict
# =========================
def parse_choices(choice_text: Any) -> Dict[str, str]:
    """
    输入:
        "A. blue_drawer, B. green_crate, C. red_bucket"
    输出:
        {"A": "blue_drawer", "B": "green_crate", "C": "red_bucket"}
    """
    if isinstance(choice_text, list):
        choice_text = "\n".join(str(x) for x in choice_text)
    else:
        choice_text = str(choice_text)

    pattern = r"([A-Z])\.\s*(.*?)(?=(?:\s*,\s*|\s+)[A-Z]\.\s*|$)"
    matches = re.findall(pattern, choice_text.strip(), flags=re.DOTALL)
    out = {}
    for k, v in matches:
        out[k.strip()] = v.strip().rstrip(",")
    return out


def format_choices_for_prompt(choice_text: Any) -> str:
    choices = parse_choices(choice_text)
    lines = [f"{k}. {v}" for k, v in choices.items()]
    return "\n".join(lines)


def normalize_choices_raw(item: Dict[str, Any]) -> str:
    choice_data = item.get("options", item.get("choices"))
    if choice_data is None:
        raise KeyError("sample is missing 'options' or 'choices'")

    choices = parse_choices(choice_data)
    if not choices:
        raise ValueError(f"could not parse choices for sample_id={item.get('sample_id')}")

    return "\n".join(f"{letter}. {value}" for letter, value in choices.items())

# =========================
# 1. load data
# =========================
def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入文件必须是一个 JSON list。")

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
    choices = normalize_choices_raw(item)
    answer = item["answer"].strip()

    return {
        "sample_id": item.get("sample_id"),
        "story_id": make_story_id(story),
        "prompting_type_raw": item.get("prompting_type"),
        "answer_letter": item.get("gold"),
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
    m = re.search(r"answer\s*[:：]\s*([A-O])\b", text, flags=re.IGNORECASE)
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
# BigToM dataset support
# =========================
def parse_bigtom_q_type(q_type: str) -> Dict[str, Any]:
    """Parse BigToM's ``<order>_<category>_<condition>`` metadata."""
    parts = str(q_type).split("_")
    question_order = int(parts[0]) if parts and parts[0].isdigit() else 0

    category_parts = []
    condition = ""
    for index, part in enumerate(parts[1:], start=1):
        if part in {"false", "true"}:
            condition = "_".join(parts[index:])
            break
        category_parts.append(part)

    return {
        "question_order": question_order,
        "category": "_".join(category_parts),
        "condition": condition,
        "deception": "false_belief" in condition,
    }


def normalize_bigtom_sample(
    item: Dict[str, Any],
    sample_idx: int,
) -> Dict[str, Any]:
    """Normalize a raw BigToM row without changing its A/B answer semantics."""
    narrative = str(item.get("narrative", item.get("story", ""))).strip()
    question = str(item.get("question", "")).strip()
    true_answer = str(item.get("true_answer", "")).strip()
    wrong_answer = str(item.get("wrong_answer", "")).strip()
    q_type = str(item.get("q_type", ""))

    if not narrative or not question or not true_answer or not wrong_answer:
        raise ValueError(
            "BigToM samples require narrative, question, true_answer, and wrong_answer "
            f"(index={sample_idx})."
        )

    q_type_info = parse_bigtom_q_type(q_type)
    sample_id = item.get("sample_id")
    if sample_id is None:
        sample_id = f"bigtom_{q_type}_{sample_idx}"

    return {
        "sample_id": sample_id,
        "story_id": make_story_id(narrative),
        "prompting_type_raw": q_type,
        "deception": q_type_info["deception"],
        "story_length": None,
        "question_order": q_type_info["question_order"],
        "bigtom_category": q_type_info["category"],
        "bigtom_condition": q_type_info["condition"],
        "story": narrative,
        "question": question,
        "true_answer": true_answer,
        "wrong_answer": wrong_answer,
        "answer": true_answer,
        # Keep this parseable by generic A-O utilities as well as BigToM adapters.
        "choices_raw": f"A. {true_answer}\nB. {wrong_answer}",
    }


def build_llm_judge_prompt(
    prediction: str,
    true_answer: str,
    wrong_answer: str,
) -> str:
    """Retained for compatibility with older BigToM analysis scripts."""
    return f"""You are judging whether a model's prediction correctly answers a question.

Question's Correct Answer: {true_answer}
Question's Incorrect Answer: {wrong_answer}

Model's Prediction: {prediction}

Does the model's prediction match the correct answer (not the incorrect answer)?

Respond with ONLY \"YES\" if it matches the correct answer, or \"NO\" otherwise."""


def extract_bigtom_answer_label(output_text: str) -> Optional[str]:
    """Extract only a terminal BigToM A/B answer, not letters in reasoning."""
    if output_text is None:
        return None

    plain_text = re.sub(r"[*_`]", "", str(output_text)).strip()
    if re.fullmatch(r"[AB]", plain_text, flags=re.IGNORECASE):
        return plain_text.upper()

    terminal_ab_patterns = [
        r"(?:^|[\r\n])\s*([AB])[\s.!]*$",
        r"(?:final\s+answer|intermediate\s+answer|answer)\s*:\s*"
        r"(?:option\s+)?([AB])\b[\s.!]*$",
        r"(?:the\s+)?(?:final\s+|correct\s+)?answer\s+"
        r"(?:is|should\s+be|would\s+be)\s+(?:option\s+)?([AB])\b"
        r"(?:\s*,?\s*not\s+[AB])?[\s.!]*$",
        r"(?:the\s+)?(?:final\s+|correct\s+)?answer\s+"
        r"(?:is|should\s+be|would\s+be)\s+(?:option\s+)?([AB])\s*:"
        r"[^\r\n]*$",
        r"(?:i\s+(?:choose|select)|my\s+(?:choice|answer)\s+is)\s+"
        r"(?:option\s+)?([AB])\b[\s.!]*$",
    ]
    for pattern in terminal_ab_patterns:
        match = re.search(pattern, plain_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Compatibility with results produced before the A/B prompt migration.
    if re.fullmatch(r"TRUE|WRONG", plain_text, flags=re.IGNORECASE):
        return "A" if plain_text.upper() == "TRUE" else "B"

    legacy_match = re.search(
        r"(?:final\s+answer|intermediate\s+answer|answer)\s*:\s*"
        r"(TRUE|WRONG)\b[\s.!]*$",
        plain_text,
        flags=re.IGNORECASE,
    )
    if legacy_match:
        return "A" if legacy_match.group(1).upper() == "TRUE" else "B"

    return None


def judge_prediction_bigtom(
    pred_raw: str,
    sample: Dict[str, Any],
    llm_judge_callable: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Judge BigToM deterministically using its terminal A/B contract."""
    del llm_judge_callable  # API compatibility; never spend another model call judging.

    true_answer = sample.get("true_answer", sample["answer"])
    wrong_answer = sample.get("wrong_answer", "")
    pred_clean = "" if pred_raw is None else str(pred_raw).strip()

    if not pred_clean:
        return {
            "pred_raw": pred_raw,
            "pred_final": pred_clean,
            "gold": true_answer,
            "correct": 0,
            "judge_reasoning": "Empty prediction",
        }

    pred_label = extract_bigtom_answer_label(pred_clean)
    if pred_label is not None:
        return {
            "pred_raw": pred_raw,
            "pred_final": pred_label,
            "gold": true_answer,
            "correct": int(pred_label == "A"),
            "judge_reasoning": f"Parsed {pred_label} label",
        }

    pred_norm = normalize_text(pred_clean)
    if pred_norm == normalize_text(true_answer):
        return {
            "pred_raw": pred_raw,
            "pred_final": "A",
            "gold": true_answer,
            "correct": 1,
            "judge_reasoning": "Direct match with true answer",
        }
    if pred_norm == normalize_text(wrong_answer):
        return {
            "pred_raw": pred_raw,
            "pred_final": "B",
            "gold": true_answer,
            "correct": 0,
            "judge_reasoning": "Direct match with wrong answer",
        }

    return {
        "pred_raw": pred_raw,
        "pred_final": None,
        "gold": true_answer,
        "correct": 0,
        "judge_reasoning": "Missing terminal A/B label",
    }


def normalize_dataset_samples(
    raw_data: List[Dict[str, Any]],
    dataset_type: str = "hitom",
) -> List[Dict[str, Any]]:
    dataset_key = dataset_type.lower()
    if dataset_key == "bigtom":
        return [
            normalize_bigtom_sample(item, index)
            for index, item in enumerate(raw_data)
        ]
    if dataset_key == "hitom":
        return [normalize_sample(item) for item in raw_data]
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")
