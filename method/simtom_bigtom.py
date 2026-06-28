from typing import Any, Callable, Dict, Optional

from prompt import extract_target_name


class SimToMBigToM:
    """
    SimToM adapted for BigToM dataset.

    BigToM differences from Hi-ToM:
    - Uses 'narrative' instead of 'story'
    - Binary answers (true_answer vs wrong_answer) instead of A-O options
    - Full sentence answers instead of location codes
    """

    def __init__(
        self,
        llm_callable: Callable[[str], str],
        target_extractor: Callable[[str], Optional[str]] = extract_target_name,
    ):
        self.llm_callable = llm_callable
        self.target_extractor = target_extractor
        self.last_perspective_prompt: Optional[str] = None
        self.last_perspective: Optional[str] = None
        self.last_qa_prompt: Optional[str] = None

    def build_perspective_prompt(self, story: str, character: str) -> str:
        return f"""The following is a narrative about a character.

                Your job is to output only the events that the specified character, {character}, knows about.

                Here are the rules:
                1. A character knows about all events that they directly experience (see, hear, do).
                2. A character does NOT know about events that happen while they are absent or not paying attention.
                3. A character does NOT know about events that occurred before they entered the scene.
                4. Preserve the original event wording whenever possible. Do not add explanations or events that are not in the narrative.

                Narrative:
                {story}

                What events does {character} know about?
                Only output the events according to the above rules."""

    def take_perspective(self, story: str, character: str) -> str:
        self.last_perspective_prompt = self.build_perspective_prompt(story, character)
        self.last_perspective = self.llm_callable(self.last_perspective_prompt).strip()
        return self.last_perspective

    def build_qa_prompt(
        self,
        perspective: str,
        character: str,
        question: str,
        true_answer: str,
        wrong_answer: str,
    ) -> str:
        return f"""{perspective}

                You are {character}.
                Based only on the above information from your perspective, answer the following question:
                {question}

                Possible Answers:
                TRUE: {true_answer}
                WRONG: {wrong_answer}

                You must choose one of the above answers. Think briefly if needed, then give your final answer in the format:
                Answer: TRUE
                or
                Answer: WRONG"""

    def build_no_target_prompt(
        self,
        story: str,
        question: str,
        true_answer: str,
        wrong_answer: str,
    ) -> str:
        """Build a prompt when no target character is identified (world-state question)."""
        return f"""Story:
                {story}

                Question:
                {question}

                Possible Answers:
                TRUE: {true_answer}
                WRONG: {wrong_answer}

                The question does not specify a character perspective. Answer based on the story.
                Think briefly if needed, then give your final answer in the format:
                Answer: TRUE
                or
                Answer: WRONG"""

    def run(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("narrative", ""))
        question = sample["question"]
        true_answer = sample.get("true_answer", sample.get("answer", ""))
        wrong_answer = sample.get("wrong_answer", "")

        character = self.target_extractor(question)
        if not character:
            self.last_perspective = story
            self.last_qa_prompt = self.build_no_target_prompt(
                story=story,
                question=question,
                true_answer=true_answer,
                wrong_answer=wrong_answer,
            )
            return self.llm_callable(self.last_qa_prompt)

        perspective = self.take_perspective(story, character)
        self.last_qa_prompt = self.build_qa_prompt(
            perspective=perspective,
            character=character,
            question=question,
            true_answer=true_answer,
            wrong_answer=wrong_answer,
        )
        return self.llm_callable(self.last_qa_prompt)
