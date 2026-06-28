from typing import Callable, Dict, Any

from prompt import extract_target_name


class PercepToMBigToM:
    """
    PercepToM adapted for BigToM dataset.

    Uses binary answer format (TRUE/WRONG) instead of A-O options.
    """

    def __init__(self, llm_callable: Callable[[str], str]):
        """
        Initializes PercepToM with a specific model calling function.
        """
        self.llm_callable = llm_callable

    def perception_inference(self, story: str, character: str) -> str:
        prompt = f"""Story:
        {story}

        Task: Based on the story above, identify exactly what the character '{character}' has perceived (seen, heard, or witnessed). If they were absent during certain events, explicitly note what they missed.

        Perception of {character}:"""
        return self.llm_callable(prompt)

    def perception_to_belief_inference(self, perception: str, character: str) -> str:
        prompt = f"""Character: {character}
        Perception: {perception}

        Task: Based *only* on the perception provided above (and strictly ignoring any omniscient knowledge of the actual world state), what does {character} currently believe to be true about the situation?

        Belief State of {character}:"""
        return self.llm_callable(prompt)

    def answer_tom_question(
        self,
        story: str,
        belief: str,
        character: str,
        question: str,
        true_answer: str,
        wrong_answer: str
    ) -> str:
        prompt = f"""Story:
        {story}

        Belief State of {character}:
        {belief}

        Question:
        {question}

        Possible Answers:
        TRUE: {true_answer}
        WRONG: {wrong_answer}

        Task: Answer the question. You must rely primarily on the "Belief State" of {character} to answer this question, rather than the objective reality described in the "Story".
        Think step by step, then give your final answer in the format:
        Answer: TRUE
        or
        Answer: WRONG"""
        return self.llm_callable(prompt)

    def run(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("narrative", ""))
        question = sample["question"]
        true_answer = sample.get("true_answer", sample.get("answer", ""))
        wrong_answer = sample.get("wrong_answer", "")

        # 1. Identify Target Character
        character = extract_target_name(question)

        # Fallback: If no character is identified, do standard CoT
        if not character:
            fallback_prompt = f"""Story:
            {story}

            Question:
            {question}

            Possible Answers:
            TRUE: {true_answer}
            WRONG: {wrong_answer}

            Think step by step, then give your final answer in the format:
            Answer: TRUE
            or
            Answer: WRONG"""
            return self.llm_callable(fallback_prompt)

        # 2. PercepToM Pipeline
        perception = self.perception_inference(story, character)
        belief = self.perception_to_belief_inference(perception, character)
        final_answer = self.answer_tom_question(
            story, belief, character, question, true_answer, wrong_answer
        )

        # Return final reasoning trace and answer
        return final_answer
