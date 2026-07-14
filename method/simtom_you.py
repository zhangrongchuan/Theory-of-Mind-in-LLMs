"""
SimToM-You: A variant of SimToM that replaces the target character with 'you'
to create a first-person perspective-taking experience for the LLM.

For questions with order > 0, the target character (the one whose mental state
is being queried) is replaced with 'you' in both the story and question.
This makes the LLM feel like it IS that character.

For questions with order 0, the behavior is identical to standard SimToM.
"""

import re
from typing import Any, Callable, Dict, Optional

from prompt import extract_target_name
from utils import format_choices_for_prompt


class SimToMYou:
    """
    SimToM-You: A first-person variant of SIMTOM.

    When question_order > 0:
    1. Extracts the target character from the question
    2. Replaces all occurrences of that character's name with "you" in the story
    3. Replaces the character name with "you" in the question (e.g.,
       "Where does Ava think..." becomes "Where do you think...")
    4. Then proceeds with SIMTOM's two-stage inference

    When question_order == 0:
    - Behavior is identical to standard SimToM
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
        self.last_modified_story: Optional[str] = None
        self.last_modified_question: Optional[str] = None

    def replace_character_with_you(self, text: str, character: str) -> str:
        """
        Replace all occurrences of a character's name with 'you' in the text.
        Handles word boundaries to avoid partial matches (e.g., 'Ava' in 'Ava' vs 'Avalanche').
        """
        # Use word boundary to match the character name as a whole word
        # Handle common variations: standalone, after period/space, before comma/period/space
        pattern = r'\b' + re.escape(character) + r'\b'
        return re.sub(pattern, 'you', text)

    def replace_question_subject_with_you(self, question: str, character: str) -> str:
        """
        Replace the character reference at the start of a question with 'you'.
        For example: "Where does Ava think..." -> "Where do you think..."
                     "How does Ava think..." -> "How do you think..."
        """
        # Pattern for "Where does {character} [think/believe/feel/know/want]..."
        where_pattern = rf"Where does {re.escape(character)} "
        if re.search(where_pattern, question):
            return re.sub(where_pattern, "Where do you ", question)

        # Pattern for "How does {character} think..."
        how_pattern = rf"How does {re.escape(character)} "
        if re.search(how_pattern, question):
            return re.sub(how_pattern, "How do you ", question)

        # If no specific pattern matched, just replace the character name
        return self.replace_character_with_you(question, character)

    def modify_story_and_question(
        self,
        story: str,
        question: str,
        character: str,
        question_order: int,
    ) -> tuple[str, str]:
        """
        Modify story and question by replacing the target character with 'you'
        if question_order > 0. Otherwise, return original story and question.
        """
        if question_order == 0:
            return story, question

        # Replace character with "you" in the story
        modified_story = self.replace_character_with_you(story, character)

        # Replace character with "you" in the question, handling verb agreement
        modified_question = self.replace_question_subject_with_you(question, character)

        return modified_story, modified_question

    def build_perspective_prompt(self, story: str, character: str, is_you: bool = False) -> str:
        """
        Build the perspective-taking prompt.
        When is_you=True, the prompt is framed from a first-person perspective.
        """
        if is_you:
            return f"""The following is a sequence of events about some characters,
                that takes place in multiple locations.

                Your job is to output only the events that you know about.

                Here are the rules:
                1. You know about all events that you do.
                2. If you are in a room or location, you know about all
                other events that happen in that location. This includes other characters
                entering, leaving, moving objects, object locations, public claims, and
                ordinary actions.
                3. If you leave a location and are not in that location, you no longer
                know about events that happen there. However, you can re-enter the location.
                4. A private communication is known only to its speaker and listener.
                5. A public claim is known to characters who are present in the public location
                where it is made.
                6. Preserve the original event numbers and wording whenever possible. Do not
                add explanations, beliefs, or events that are not in the story.

                Story:
                {story}

                What events do you know about?
                Only output the events according to the above rules."""
        else:
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

    def take_perspective(self, story: str, character: str, is_you: bool = False) -> str:
        self.last_perspective_prompt = self.build_perspective_prompt(story, character, is_you)
        self.last_perspective = self.llm_callable(self.last_perspective_prompt).strip()
        return self.last_perspective

    def build_qa_prompt(
        self,
        perspective: str,
        character: str,
        question: str,
        choices_text: str,
        is_you: bool = False,
    ) -> str:
        """Build the question-answering prompt."""
        if is_you:
            return f"""{perspective}

                Based only on the above information from your perspective, answer the following
                question:
                {question}

                Choices:
                {choices_text}

                You must choose one of the above choices. Think briefly if needed, then give
                your final answer in the format:
                Answer: <option letter>"""
        else:
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
        """Build a prompt when no target character is identified."""
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
        """
        Run the SimToM-You pipeline.

        For question_order > 0:
        1. Extract target character
        2. Replace character with "you" in story and question
        3. Take perspective (using "you" framing)
        4. Answer question (without "You are X" since it's already first-person)

        For question_order == 0:
        - Same as standard SimToM
        """
        story = sample["story"]
        question = sample["question"]
        question_order = sample.get("question_order", 0)
        choices_text = format_choices_for_prompt(sample["choices_raw"])

        character = self.target_extractor(question)
        if not character:
            # No target character found, use original story/question
            self.last_modified_story = story
            self.last_modified_question = question
            self.last_perspective = story
            self.last_qa_prompt = self.build_no_target_prompt(
                story=story,
                question=question,
                choices_text=choices_text,
            )
            return self.llm_callable(self.last_qa_prompt)

        # Determine if we need "you" transformation
        is_you = question_order > 0

        # Modify story and question if order > 0
        modified_story, modified_question = self.modify_story_and_question(
            story, question, character, question_order
        )
        self.last_modified_story = modified_story
        self.last_modified_question = modified_question

        # Take perspective from the modified story
        perspective = self.take_perspective(modified_story, character, is_you)

        # Build QA prompt
        self.last_qa_prompt = self.build_qa_prompt(
            perspective=perspective,
            character=character,
            question=modified_question,
            choices_text=choices_text,
            is_you=is_you,
        )
        return self.llm_callable(self.last_qa_prompt)

