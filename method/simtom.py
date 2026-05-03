from typing import Any, Callable, Dict, Optional

from prompt import extract_target_name
from utils import format_choices_for_prompt


class SimToM:
    """
    Implements SIMTOM from "Think Twice: Perspective-Taking Improves Large
    Language Models' Theory-of-Mind Capabilities".

    SIMTOM performs two inference passes:
    1. Perspective-taking: filter the story to events the queried character
       knows about.
    2. Question-answering: answer the original question using that filtered
       perspective instead of the full omniscient story.
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
        return f"""The following is a sequence of events about some characters,
                that takes place in multiple locations.

                Your job is to output only the events that the specified character,
                {character}, knows about.

                Here are the rules:
                1. A character knows about all events that they do.
                2. If a character is in a room or location, that character knows about all
                other events that happen in that location. This includes other characters
                entering, leaving, moving objects, object locations, public claims, and
                ordinary actions.
                3. If a character leaves a location and is not in that location, they no longer
                know about events that happen there. However, they can re-enter the location.
                4. A private communication is known only to its speaker and listener.
                5. A public claim is known to characters who are present in the public location
                where it is made.
                6. Preserve the original event numbers and wording whenever possible. Do not
                add explanations, beliefs, or events that are not in the story.

                Story:
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
        choices_text: str) -> str:
        return f"""{perspective}

                You are {character}.
                Based only on the above information from your perspective, answer the following
                question:
                {question}

                Choices:
                {choices_text}

                You must choose one of the above choices. Think briefly if needed, then give
                your final answer in the format:
                Answer: <option letter>"""

    def build_no_target_prompt(
        self,
        story: str,
        question: str,
        choices_text: str,
    ) -> str:
        return f"""Story:
                {story}

                Question:
                {question}

                Choices:
                {choices_text}

                The question does not specify a character perspective. Answer based on the
                story. Think briefly if needed, then give your final answer in the format:
                Answer: <option letter>"""

    def run(self, sample: Dict[str, Any]) -> str:
        # story = sample.get("story", sample.get("context", ""))
        story = sample["story"]
        question = sample["question"]
        choices_text = format_choices_for_prompt(sample["choices_raw"])
        character = self.target_extractor(question)
        if not character:
            self.last_perspective = story
            self.last_qa_prompt = self.build_no_target_prompt(
                story=story,
                question=question,
                choices_text=choices_text,
            )
            return self.llm_callable(self.last_qa_prompt)

        perspective = self.take_perspective(story, character)
        self.last_qa_prompt = self.build_qa_prompt(
            perspective=perspective,
            character=character,
            question=question,
            choices_text=choices_text,
        )
        return self.llm_callable(self.last_qa_prompt)
