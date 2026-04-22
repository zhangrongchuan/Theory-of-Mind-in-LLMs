import json
import re
import hashlib
from typing import Dict, List, Any, Optional

# =========================
# process choice text into dict
# =========================
def parse_choices(choice_text: str) -> Dict[str, str]:
    """
    输入:
        "A. blue_drawer, B. green_crate, C. red_bucket"
    输出:
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
    choices = item["choices"].strip()
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