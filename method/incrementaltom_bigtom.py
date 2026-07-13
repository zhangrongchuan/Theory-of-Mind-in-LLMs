import re
from typing import Callable, Dict, Any, List, Optional, Tuple

from prompt import extract_target_name


class IncrementalToMBigToM:
    """
    Incremental Theory of Mind adapted for BigToM dataset.

    Splits narratives into chunks and processes incrementally.
    Uses binary answer format (A/B) instead of A-O options.
    """

    def __init__(self, llm_callable: Callable[[str], str], chunk_size: int = 3):
        self.llm_callable = llm_callable
        self.chunk_size = chunk_size

    def split_story_into_sentences(self, story: str) -> List[str]:
        """Split a story into sentences while preserving sentence structure."""
        raw_sentences = re.split(r'(?<=[.!?])\s+', story.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def chunk_sentences(self, sentences: List[str], chunk_size: int) -> List[List[str]]:
        """Group sentences into chunks of size chunk_size."""
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def ask_intermediate_question(
        self,
        story_so_far: str,
        question: str,
        true_answer: str,
        wrong_answer: str,
        previous_answer: Optional[str],
        character: Optional[str]
    ) -> str:
        """Ask the ToM question based on the story processed so far."""
        context_part = ""
        if previous_answer:
            context_part = f"""
            Current Belief State (based on previous parts of the narrative):
            {previous_answer}

            Important: Consider this belief state when answering, as it represents what has been understood so far."""

        char_focus = f" focusing on {character}" if character else ""

        prompt = f"""Narrative so far:
            {story_so_far}
            {context_part}

            Question: {question}

            Possible Answers:
            A: {true_answer}
            B: {wrong_answer}

            Task: Based on the narrative so far{f' and the current belief state' if previous_answer else ''}, answer the question{char_focus}.
            Think step by step, then give your intermediate answer in the format:
            Intermediate Answer: A
            or
            Intermediate Answer: B

            Explain your reasoning briefly, then provide the answer."""

        return self.llm_callable(prompt)

    def extract_answer_from_response(self, response: str) -> str:
        """Extract the A/B answer from an intermediate response."""
        matches = re.findall(
            r'(?:Intermediate Answer|Answer)[:\s]+(?:Option\s+)?([AB])\b',
            response,
            re.IGNORECASE,
        )
        if matches:
            return matches[-1].upper()

        # Backward compatibility for older TRUE/WRONG prompts.
        fallback_matches = re.findall(r'(?:Intermediate Answer|Answer)[:\s]+(TRUE|WRONG)\b', response, re.IGNORECASE)
        if fallback_matches:
            legacy_answer = fallback_matches[-1].upper()
            return "A" if legacy_answer == "TRUE" else "B"

        # Return full response if no clear answer
        return response.strip()

    def format_intermediate_answer(self, response: str, true_answer: str, wrong_answer: str) -> str:
        """Format the intermediate answer for context passing."""
        answer = self.extract_answer_from_response(response)
        if answer == "A":
            return f"Checkpoint selected A: {true_answer}"
        if answer == "B":
            return f"Checkpoint selected B: {wrong_answer}"
        return f"Checkpoint answer could not be parsed. Raw response: {response.strip()}"

    def ask_final_question(
        self,
        full_story: str,
        intermediate_answers: List[Tuple[int, str]],
        question: str,
        true_answer: str,
        wrong_answer: str,
        character: Optional[str]
    ) -> str:
        """Ask the final question with full context including all intermediate answers."""
        # Build context from intermediate answers
        intermediate_context = ""
        if intermediate_answers:
            intermediate_context = "Previous Understanding Checkpoints:\n"
            for chunk_idx, answer in intermediate_answers:
                intermediate_context += f"  After chunk {chunk_idx}: {answer}\n"
            intermediate_context += "\n"

        char_focus = f"Focus particularly on what '{character}' knows and believes based on their experiences in the narrative." if character else ""

        prompt = f"""Full Narrative:
        {full_story}

        {intermediate_context}Question: {question}

        Possible Answers:
        A: {true_answer}
        B: {wrong_answer}

        Task: Answer the question considering the full narrative and any intermediate understandings noted above.
        {char_focus}
        Think step by step, tracking what each character knows at each point in the narrative, then give your final answer in the format:
        Answer: A
        or
        Answer: B"""

        return self.llm_callable(prompt)

    def run(
        self,
        sample: Dict[str, Any],
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute the IncrementalToM pipeline for BigToM.

        Returns:
            Dict with "response" and "belief_state_tracking" keys.
        """
        # Use provided chunk_size or default
        n = chunk_size if chunk_size is not None else self.chunk_size

        story = sample.get("story", sample.get("narrative", ""))
        question = sample["question"]
        true_answer = sample.get("true_answer", sample.get("answer", ""))
        wrong_answer = sample.get("wrong_answer", "")

        # Identify target character if applicable
        character = extract_target_name(question)

        # Handle case where chunk_size is larger than the story
        sentences = self.split_story_into_sentences(story)

        if len(sentences) <= n:
            # Story is short enough to process in one go - use standard approach
            prompt = f"""Narrative:
            {story}

            Question:
            {question}

            Possible Answers:
            A: {true_answer}
            B: {wrong_answer}

            Think step by step, then give your final answer in the format:
            Answer: A
            or
            Answer: B"""
            return {
                "response": self.llm_callable(prompt),
                "belief_state_tracking": []
            }

        # Split into chunks
        chunks = self.chunk_sentences(sentences, n)

        # Process each chunk, maintaining intermediate answers
        intermediate_answers = []
        belief_state_tracking = []
        previous_answer = None

        for i, chunk in enumerate(chunks[:-1], 1):  # Process all but the last chunk
            story_so_far = " ".join(sentences[:i * n])

            # Ask intermediate question
            response = self.ask_intermediate_question(
                story_so_far=story_so_far,
                question=question,
                true_answer=true_answer,
                wrong_answer=wrong_answer,
                previous_answer=previous_answer,
                character=character
            )

            # Format and store the answer
            formatted_answer = self.format_intermediate_answer(response, true_answer, wrong_answer)
            intermediate_answers.append((i, formatted_answer))
            belief_state_tracking.append({
                "chunk": i,
                "raw_response": response,
                "formatted_answer": formatted_answer
            })

            # Update previous_answer for context in next iteration
            previous_answer = formatted_answer

        # Final question with full story and all intermediate context
        final_response = self.ask_final_question(
            full_story=story,
            intermediate_answers=intermediate_answers,
            question=question,
            true_answer=true_answer,
            wrong_answer=wrong_answer,
            character=character
        )

        return {
            "response": final_response,
            "belief_state_tracking": belief_state_tracking
        }

