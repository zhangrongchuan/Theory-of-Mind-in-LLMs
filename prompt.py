import re
from typing import Dict, Any
from utils import format_choices_for_prompt

def extract_target_name(question: str) -> str:
    """
    Extracts the character name whose mental state is being queried.
    Example: 'Where does Avery really think the lettuce is?' -> 'Avery'
    960 question ask using where does, 240 question ask using where is.
    """
    # Pattern for "Where does {name} [think/believe/feel/know]..."
    match = re.search(r"Where does ([A-Z][a-z]+) (?:really )?(?:think|believe|feel|know|want)", question)
    if match:
        return match.group(1)
    
    # # Pattern for "How does {name} think..."
    # match = re.search(r"How does ([A-Z][a-z]+) think", question)
    # if match:
    #     return match.group(1)
    
     # Pattern for "Where is the {name} [think/believe/feel/know]..."
    match = re.search(r"Where is the ([A-Z][a-z]+) (?:really )?(?:think|believe|feel|know|want)", question)
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
