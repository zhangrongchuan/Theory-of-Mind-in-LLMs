"""
SimToM-You: A variant of SimToM adapted for BigToM dataset.

Replaces the target character with 'you' to create a first-person perspective.
Uses binary answer format (A/B).
"""

import re
from typing import Any, Callable, Dict, Optional

from prompt import extract_target_name


class SimToMYouBigToM:
    """
    SimToM-You adapted for BigToM dataset.

    When question_order > 0:
    1. Extracts the target character from the question
    2. Replaces all occurrences of that character's name with "you" in the story
    3. Replaces the character name with "you" in the question
    4. Then proceeds with SimToM's two-stage inference

    Uses binary answer format (A/B) instead of A-O options.
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
        """
        pattern = r'\b' + re.escape(character) + r'\b'
        return re.sub(pattern, 'you', text)

    def replace_question_subject_with_you(self, question: str, character: str) -> str:
        """
        Replace the character reference at the start of a question with 'you'.
        Handles BigToM question formats like:
        - "What will {character} do?" -> "What will you do?"
        - "Does {character} believe..." -> "Do you believe..."
        """
        # Pattern for "What will {character}..."
        what_pattern = rf"What will {re.escape(character)} "
        if re.search(what_pattern, question):
            return re.sub(what_pattern, "What will you ", question)

        # Pattern for "What does {character}..."
        what_does_pattern = rf"What does {re.escape(character)} "
        if re.search(what_does_pattern, question):
            return re.sub(what_does_pattern, "What do you ", question)

        # Pattern for "Does {character}..."
        does_pattern = rf"Does {re.escape(character)} "
        if re.search(does_pattern, question):
            return re.sub(does_pattern, "Do you ", question)

        # Fallback: just replace the character name
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

        # Replace character with "you" in the question
        modified_question = self.replace_question_subject_with_you(question, character)

        return modified_story, modified_question

    def build_perspective_prompt(self, story: str, character: str, is_you: bool = False) -> str:
        """Build the perspective-taking prompt."""
        if is_you:
            return f"""The following is a narrative about you.

Your job is to output only the events that you know about.

Here are the rules:
1. You know about all events that you directly experience (see, hear, do).
2. You do NOT know about events that happen while you are absent or not paying attention.
3. You do NOT know about events that occurred before you entered the scene.
4. Preserve the original event wording whenever possible. Do not add explanations or events that are not in the narrative.

Narrative:
{story}

What events do you know about?
Only output the events according to the above rules."""
        else:
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

    def take_perspective(self, story: str, character: str, is_you: bool = False) -> str:
        self.last_perspective_prompt = self.build_perspective_prompt(story, character, is_you)
        self.last_perspective = self.llm_callable(self.last_perspective_prompt).strip()
        return self.last_perspective

    def build_qa_prompt(
        self,
        perspective: str,
        character: str,
        question: str,
        true_answer: str,
        wrong_answer: str,
        is_you: bool = False,
    ) -> str:
        """Build the question-answering prompt."""
        if is_you:
            return f"""{perspective}

Based only on the above information from your perspective, answer the following question:
{question}

Possible Answers:
A: {true_answer}
B: {wrong_answer}

You must choose one of the above answers. Think briefly if needed, then give your final answer in the format:
Answer: A
or
Answer: B"""
        else:
            return f"""{perspective}

You are {character}.
Based only on the above information from your perspective, answer the following question:
{question}

Possible Answers:
A: {true_answer}
B: {wrong_answer}

You must choose one of the above answers. Think briefly if needed, then give your final answer in the format:
Answer: A
or
Answer: B"""

    def build_no_target_prompt(
        self,
        story: str,
        question: str,
        true_answer: str,
        wrong_answer: str,
    ) -> str:
        """Build a prompt when no target character is identified."""
        return f"""Story:
{story}

Question:
{question}

Possible Answers:
A: {true_answer}
B: {wrong_answer}

The question does not specify a character perspective. Answer based on the story.
Think briefly if needed, then give your final answer in the format:
Answer: A
or
Answer: B"""

    def run(self, sample: Dict[str, Any]) -> str:
        """
        Run the SimToM-You pipeline for BigToM.
        """
        story = sample.get("story", sample.get("narrative", ""))
        question = sample["question"]
        question_order = sample.get("question_order", 0)
        true_answer = sample.get("true_answer", sample.get("answer", ""))
        wrong_answer = sample.get("wrong_answer", "")

        character = self.target_extractor(question)
        if not character:
            # No target character found, use original story/question
            self.last_modified_story = story
            self.last_modified_question = question
            self.last_perspective = story
            self.last_qa_prompt = self.build_no_target_prompt(
                story=story,
                question=question,
                true_answer=true_answer,
                wrong_answer=wrong_answer,
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
            true_answer=true_answer,
            wrong_answer=wrong_answer,
            is_you=is_you,
        )
        return self.llm_callable(self.last_qa_prompt)

