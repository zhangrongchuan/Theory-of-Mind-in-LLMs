import re
from typing import Any, Callable, Dict, List, Optional

from utils import format_choices_for_prompt


class DiscreteWorldModel:
    """
    Implements Discrete World Models (DWM) prompting from
    "A Notion of Complexity for Theory of Mind via Discrete World Models".

    DWM splits a ToM story into sequential chunks. After each chunk, it asks the
    model for a compact state description of the environment and agents' beliefs.
    The final QA prompt receives the original story plus the generated discrete
    state descriptions.
    """

    def __init__(self, llm_callable: Callable[[str], str], num_splits: int = 3):
        if num_splits < 1:
            raise ValueError("num_splits must be at least 1.")

        self.llm_callable = llm_callable
        self.num_splits = num_splits
        self.last_chunks: List[str] = []
        self.last_state_prompts: List[str] = []
        self.last_state_descriptions: List[str] = []
        self.last_qa_prompt: Optional[str] = None

    def split_story(self, story: str) -> List[str]:
        events = self._split_numbered_events(story)
        if not events:
            events = [line.strip() for line in story.splitlines() if line.strip()]
        if not events:
            return [story.strip()]

        split_count = min(self.num_splits, len(events))
        chunks = []
        for index in range(split_count):
            start = round(index * len(events) / split_count)
            end = round((index + 1) * len(events) / split_count)
            chunks.append("\n".join(events[start:end]).strip())
        return [chunk for chunk in chunks if chunk]

    def build_state_prompt(self, chunks_so_far: List[str], previous_states: List[str]) -> str:
        prior_state_text = self._format_previous_states(previous_states)
        dialogue_so_far = "\n".join(chunks_so_far)

        return f"""I give you a phrase of a dialogue between agents. I will
                reveal more parts of it later. At the end, I will give you a question you must
                answer.

                For each phrase, you must:
                1. Write down a succinct description of what each agent knows about the
                environment and about the other agents.
                2. Keep the description short and do not produce redundant information.
                3. Make implicit state changes explicit, especially object locations, who is
                present, what each agent observed, and each agent's beliefs.

                Previous discrete state descriptions:
                {prior_state_text}

                Dialogue so far:
                {dialogue_so_far}

                Now provide a succinct description of the current state of the environment and
                each agent's beliefs. Prefix each consideration with #DWM#."""

    def describe_states(self, story: str) -> List[str]:
        chunks = self.split_story(story)
        self.last_chunks = chunks
        self.last_state_prompts = []
        self.last_state_descriptions = []

        for index in range(len(chunks)):
            chunks_so_far = chunks[: index + 1]
            prompt = self.build_state_prompt(
                chunks_so_far=chunks_so_far,
                previous_states=self.last_state_descriptions,
            )
            state_description = self.llm_callable(prompt).strip()
            self.last_state_prompts.append(prompt)
            self.last_state_descriptions.append(state_description)

        return self.last_state_descriptions

    def build_qa_prompt(
        self,
        story: str,
        question: str,
        choices_text: str,
        state_descriptions: List[str],
    ) -> str:
        state_text = self._format_previous_states(state_descriptions)

        return f"""Consider the following dialogue where multiple agents
                interact.

                Here's the dialogue:
                {story}

                Here are discrete world-model descriptions generated while reading the
                dialogue:
                {state_text}

                This is the end of the dialogue. Now, this is a question for you to answer.
                Question:
                {question}

                Choices:
                {choices_text}

                Use the discrete world-model descriptions to track the environment and each
                agent's beliefs. Think step by step, then give your final answer in the format:
                Answer: <option letter>"""

    def run(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("context", ""))
        question = sample["question"]
        choices_text = format_choices_for_prompt(sample["choices_raw"])
        state_descriptions = self.describe_states(story)

        self.last_qa_prompt = self.build_qa_prompt(
            story=story,
            question=question,
            choices_text=choices_text,
            state_descriptions=state_descriptions,
        )
        return self.llm_callable(self.last_qa_prompt)

    def _split_numbered_events(self, story: str) -> List[str]:
        clean_story = "\n".join(
            line.strip()
            for line in story.splitlines()
            if line.strip() and not line.lower().startswith("read the following story")
        )
        pattern = r"(?ms)^\s*(\d+\s+.*?)(?=^\s*\d+\s+|\Z)"
        return [match.strip() for match in re.findall(pattern, clean_story)]

    def _format_previous_states(self, states: List[str]) -> str:
        if not states:
            return "No previous state descriptions."

        return "\n\n".join(
            f"State description {index}:\n{state}"
            for index, state in enumerate(states, start=1)
        )
