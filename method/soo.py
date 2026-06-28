import re
from typing import Dict, Any, Callable
from utils import format_choices_for_prompt

class SoO:
    def __init__(self, llm_callable: Callable):
        self.llm_callable = llm_callable

    def extract_target_name(self, question: str) -> str:
        """
        Extracts the character name whose mental state is being queried.
        Example: 'Where does Avery really think the lettuce is?' -> 'Avery'
        """
        match = re.search(r"Where does ([A-Z][a-z]+) (?:really )?(?:think|believe|feel|know|want)", question)
        if match:
            return match.group(1)
        
        match = re.search(r"How does ([A-Z][a-z]+) think", question)
        if match:
            return match.group(1)

        return None

    def run(self, sample: Dict[str, Any]) -> str:
        choices_text = format_choices_for_prompt(sample["choices_raw"])
        name = self.extract_target_name(sample["question"])

        if name:
            prompt = f"""
            # Context
            {sample['story']}

            # Question
            {sample['question']}

            # Options
            {choices_text}

            Let's put ourselves in {name}'s shoes.
            """
        else:
            prompt = f"""
            # Context
            {sample['story']}

            # Question
            {sample['question']}

            # Options
            {choices_text}

            Let's think step-by-step.
            """
        
        # Pass the formatted prompt to the specialized LLM caller
        return self.llm_callable(prompt)