import re
from typing import Dict, Any, Callable


class SoOBigToM:
    """
    SoO (Simulation of Others) adapted for BigToM dataset.

    Uses binary answer format (TRUE/WRONG) instead of A-O options.
    """

    def __init__(self, llm_callable: Callable):
        self.llm_callable = llm_callable

    def extract_target_name(self, question: str) -> str:
        """
        Extracts the character name whose mental state is being queried.
        Example: 'What will Noor do?' -> 'Noor'
        """
        # Pattern for "What does/will {name}..."
        match = re.search(r"What (?:does|will) ([A-Z][a-z]+)", question)
        if match:
            return match.group(1)

        # Pattern for "Does {name} believe..."
        match = re.search(r"Does ([A-Z][a-z]+) believe", question)
        if match:
            return match.group(1)

        # Pattern for "Where does {name}..." (fallback from original)
        match = re.search(r"Where does ([A-Z][a-z]+)", question)
        if match:
            return match.group(1)

        return None

    def run(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("narrative", ""))
        question = sample["question"]
        true_answer = sample.get("true_answer", sample.get("answer", ""))
        wrong_answer = sample.get("wrong_answer", "")

        name = self.extract_target_name(question)

        if name:
            prompt = f"""
            # Context
            {story}

            # Question
            {question}

            # Possible Answers
            TRUE: {true_answer}
            WRONG: {wrong_answer}

            Let's put ourselves in {name}'s shoes.

            Think step by step about what {name} knows and believes, then give your final answer in the format:
            Answer: TRUE
            or
            Answer: WRONG
            """
        else:
            prompt = f"""
            # Context
            {story}

            # Question
            {question}

            # Possible Answers
            TRUE: {true_answer}
            WRONG: {wrong_answer}

            Let's think step-by-step.

            Give your final answer in the format:
            Answer: TRUE
            or
            Answer: WRONG
            """

        # Pass the formatted prompt to the specialized LLM caller
        return self.llm_callable(prompt)
