from typing import Dict,  Any
from utils import format_choices_for_prompt

def build_prompt(sample: Dict[str, Any], method: str) -> str:
    '''
    develop different thoery of mind methods by prompting here.
    '''
    method = method.upper()
    if method == "VP":
        return build_vp_prompt(sample)
    if method == "COTP":
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