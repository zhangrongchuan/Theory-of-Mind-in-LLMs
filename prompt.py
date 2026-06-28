import re
from typing import Dict, Any
from utils import format_choices_for_prompt

def extract_target_name(question: str) -> str:
    """
    Extracts the character name whose mental state is being queried.
    Example: 'Where does Avery really think the lettuce is?' -> 'Avery'
    """
    # Pattern for "Where does {name} [think/believe/feel/know]..."
    match = re.search(r"Where does ([A-Z][a-z]+) (?:really )?(?:think|believe|feel|know|want)", question)
    if match:
        return match.group(1)
    
    # Pattern for "How does {name} think..."
    match = re.search(r"How does ([A-Z][a-z]+) think", question)
    if match:
        return match.group(1)

    return None # Return None if no character is mentalizing (e.g., world-state questions)

def build_prompt(sample: Dict[str, Any], method: str) -> str:
    '''
    develop different thoery of mind methods by prompting here.
    '''
    method = method.upper()
    if method == "VP":
        return build_vp_prompt(sample)
    elif method == "COTP":
        return build_cotp_prompt(sample)
    raise ValueError(f"invalid method: {method}")


def build_vp_prompt(sample: Dict[str, Any]) -> str:
    choices_text = format_choices_for_prompt(sample["choices_raw"])
    return f"""
            Story:
            {sample['story']}

            Question:
            {sample['question']}

            Choices:
            {choices_text}

            Please return exactly one uppercase option letter from A to O. 
            Do not provide any explanation. 
            Do not repeat the question. 
            Do not output anything except the single letter."""


def build_cotp_prompt(sample: Dict[str, Any]) -> str:
    choices_text = format_choices_for_prompt(sample["choices_raw"])
    return f"""
            Story:
            {sample['story']}

            Question:
            {sample['question']}

            Choices:
            {choices_text}

            Think step by step, then give your final answer in the format:
            Answer: <option letter>"""


# =========================
# BigToM-specific prompt builders
# =========================

def build_vp_prompt_bigtom(sample: Dict[str, Any]) -> str:
    """
    BigToM version of VP (Verbal Prompt) - binary choice between true and wrong answer.
    """
    true_answer = sample.get("true_answer", sample.get("answer", ""))
    wrong_answer = sample.get("wrong_answer", "")

    return f"""
Story:
{sample['story']}

Question:
{sample['question']}

Possible Answers:
TRUE: {true_answer}
WRONG: {wrong_answer}

Please respond with exactly one word: either "TRUE" or "WRONG".
Do not provide any explanation.
Do not output anything except the single word."""


def build_cotp_prompt_bigtom(sample: Dict[str, Any]) -> str:
    """
    BigToM version of CoT (Chain-of-Thought) - binary choice with reasoning.
    """
    true_answer = sample.get("true_answer", sample.get("answer", ""))
    wrong_answer = sample.get("wrong_answer", "")

    return f"""
Story:
{sample['story']}

Question:
{sample['question']}

Possible Answers:
TRUE: {true_answer}
WRONG: {wrong_answer}

Think step by step about the character's mental state, then give your final answer in the format:
Answer: TRUE
or
Answer: WRONG"""