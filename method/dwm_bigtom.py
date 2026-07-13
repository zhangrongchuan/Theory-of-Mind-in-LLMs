import re
from typing import Any, Callable, Dict, List, Optional


class DiscreteWorldModelBigToM:
    """
    Discrete World Models (DWM) adapted for BigToM dataset.

    Uses binary answer format (A/B) instead of A-O options.
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
        """Split the story into chunks for incremental processing."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', story.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [story.strip()]

        split_count = min(self.num_splits, len(sentences))
        chunks = []
        for index in range(split_count):
            start = round(index * len(sentences) / split_count)
            end = round((index + 1) * len(sentences) / split_count)
            chunks.append(" ".join(sentences[start:end]).strip())
        return [chunk for chunk in chunks if chunk]

    def build_state_prompt(self, chunks_so_far: List[str], previous_states: List[str]) -> str:
        prior_state_text = self._format_previous_states(previous_states)
        dialogue_so_far = "\n".join(chunks_so_far)

        return f"""I give you a phrase of a narrative. I will reveal more parts of it later. At the end, I will give you a question you must answer.

        For each phrase, you must:
        1. Write down a succinct description of what each character knows about the environment and about the other characters.
        2. Keep the description short and do not produce redundant information.
        3. Make implicit state changes explicit, especially character beliefs, intentions, and what each character observed.

        Previous state descriptions:
        {prior_state_text}

        Narrative so far:
        {dialogue_so_far}

        Now provide a succinct description of the current state of the environment and each character's beliefs. Prefix each consideration with #DWM#."""

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
        true_answer: str,
        wrong_answer: str,
        state_descriptions: List[str],
    ) -> str:
        state_text = self._format_previous_states(state_descriptions)

        return f"""Consider the following narrative.

        Here's the narrative:
        {story}

        Here are discrete world-model descriptions generated while reading the narrative:
        {state_text}

        This is the end of the narrative. Now, this is a question for you to answer.
        Question:
        {question}

        Possible Answers:
        A: {true_answer}
        B: {wrong_answer}

        Use the discrete world-model descriptions to track the narrative and each character's beliefs.
        Think step by step, then give your final answer in the format:
        Answer: A
        or
        Answer: B"""

    def run(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("narrative", ""))
        question = sample["question"]
        true_answer = sample.get("true_answer", sample.get("answer", ""))
        wrong_answer = sample.get("wrong_answer", "")

        state_descriptions = self.describe_states(story)

        self.last_qa_prompt = self.build_qa_prompt(
            story=story,
            question=question,
            true_answer=true_answer,
            wrong_answer=wrong_answer,
            state_descriptions=state_descriptions,
        )
        return self.llm_callable(self.last_qa_prompt)

    def _format_previous_states(self, states: List[str]) -> str:
        if not states:
            return "No previous state descriptions."

        return "\n\n".join(
            f"State description {index}:\n{state}"
            for index, state in enumerate(states, start=1)
        )

