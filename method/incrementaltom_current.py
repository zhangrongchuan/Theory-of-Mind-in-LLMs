import re
from typing import Callable, Dict, Any, List, Optional, Tuple
from prompt import extract_target_name
from utils import format_choices_for_prompt


class IncrementalToM:
    """
    Incremental Theory of Mind method.

    Splits stories into chunks of n sentences. After each chunk, asks the ToM question
    and uses the intermediate answer as "context" for processing subsequent chunks.
    This simulates maintaining a running mental state as the story unfolds.
    """

    def __init__(self, llm_callable: Callable[[str], str], chunk_size: int = 3):
        """
        Initialize IncrementalToM.

        Args:
            llm_callable: Function to call the LLM (e.g., call_model_hf)
            chunk_size: Number of sentences per chunk (n). Can be overridden per run.
        """
        self.llm_callable = llm_callable
        self.chunk_size = chunk_size

    def split_story_into_sentences(self, story: str) -> List[str]:
        """
        Split a story into sentences while preserving sentence structure.
        Handles common sentence endings and special cases.
        """
        # Split on sentence boundaries (periods, exclamation marks, question marks)
        # Keep the delimiter by using a capturing group in the split pattern
        raw_sentences = re.split(r'(?<=[.!?])\s+', story.strip())

        # Clean up empty sentences and whitespace
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def chunk_sentences(self, sentences: List[str], chunk_size: int) -> List[List[str]]:
        """
        Group sentences into chunks of size chunk_size.
        """
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def ask_intermediate_question(
        self,
        story_so_far: str,
        question: str,
        choices_text: str,
        previous_answer: Optional[str],
        character: Optional[str]
    ) -> str:
        """
        Ask the ToM question based on the story processed so far.

        If there's a previous intermediate answer, it is included as "context"
        representing what the character currently believes.
        """
        context_part = ""
        if previous_answer:
            context_part = f"""
        Current Belief State (based on previous parts of the story):
        {previous_answer}

        Important: Consider this belief state when answering, as it represents what has been understood so far."""

        prompt = f"""Story so far:
        {story_so_far}
        {context_part}

        Question: {question}

        Choices:
        {choices_text}

        Task: Based on the story so far{f' and the current belief state' if previous_answer else ''}, answer the question.
        Think step by step, then give your intermediate answer in the format:
        Intermediate Answer: <option letter>

        Explain your reasoning briefly, then provide the answer."""

        return self.llm_callable(prompt)

    def extract_answer_from_response(self, response: str) -> str:
        """
        Extract the answer letter from an intermediate response.
        Falls back to returning the full response if no clear answer is found.
        """
        # Look for "Intermediate Answer: X" or "Answer: X" pattern
        match = re.search(r'(?:Intermediate Answer|Answer)[:\s]+([A-O])\b', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Try to find any single letter A-O
        matches = re.findall(r'\b([A-O])\b', response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

        # Return full response if no clear answer
        return response.strip()

    def format_intermediate_answer(
        self,
        response: str,
        choices_text: str
    ) -> str:
        """
        Format the intermediate answer with its full choice text for better context.
        """
        answer_letter = self.extract_answer_from_response(response)

        # Parse choices to get full text
        choices = {}
        for line in choices_text.split('\n'):
            match = re.match(r'^([A-O])\.\s*(.+)$', line.strip())
            if match:
                choices[match.group(1)] = match.group(2)

        if answer_letter in choices:
            return f"{answer_letter}. {choices[answer_letter]}"
        return answer_letter

    def ask_final_question(
        self,
        full_story: str,
        intermediate_answers: List[Tuple[int, str]],
        question: str,
        choices_text: str,
        character: Optional[str]
    ) -> str:
        """
        Ask the final question with full context including all intermediate answers.
        """
        # Build context from intermediate answers
        intermediate_context = ""
        if intermediate_answers:
            intermediate_context = "Previous Understanding Checkpoint:\n"
            for chunk_idx, answer in intermediate_answers:
                intermediate_context += f"  After chunk {chunk_idx}: {answer}\n"
            intermediate_context += "\n"

        prompt = f"""Full Story:
        {full_story}

        {intermediate_context}Question: {question}

        Choices:
        {choices_text}

        Task: Answer the question considering the full story and any intermediate understandings noted above.
        {f"Focus particularly on what '{character}' knows and believes based on their experiences in the story." if character else ""}
        Think step by step, tracking what each character knows at each point in the story, then give your final answer in the format:
        Answer: <option letter>"""

        return self.llm_callable(prompt)

    def run(
        self,
        sample: Dict[str, Any],
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute the IncrementalToM pipeline.

        Args:
            sample: The test sample containing story, question, choices, etc.
            chunk_size: Override the default chunk size for this run.

        Returns:
            A dictionary containing:
                - 'response': The final answer from the LLM (string)
                - 'belief_state_tracking': List of intermediate answers (Previous Understanding Checkpoints)
        """
        # Use provided chunk_size or default
        n = chunk_size if chunk_size is not None else self.chunk_size

        story = sample.get("story", sample.get("context", ""))
        question = sample["question"]
        choices_text = format_choices_for_prompt(sample["choices_raw"])

        # Identify target character if applicable
        character = extract_target_name(question)

        # Handle case where chunk_size is larger than the story
        sentences = self.split_story_into_sentences(story)

        if len(sentences) <= n:
            # Story is short enough to process in one go - use standard approach
            prompt = f"""Story:
            {story}

            Question:
            {question}

            Choices:
            {choices_text}

            Think step by step, then give your final answer in the format:
            Answer: <option letter>"""
            return {
                "response": self.llm_callable(prompt),
                "belief_state_tracking": []
            }

        # Split into chunks
        chunks = self.chunk_sentences(sentences, n)

        # Process each chunk, maintaining intermediate answers
        intermediate_answers = []
        previous_answer = None

        for i, chunk in enumerate(chunks[:-1], 1):  # Process all but the last chunk
            story_so_far = " ".join(sentences[:i * n])

            # Ask intermediate question
            response = self.ask_intermediate_question(
                story_so_far=story_so_far,
                question=question,
                choices_text=choices_text,
                previous_answer=previous_answer,
                character=character
            )

            # Format and store the answer
            formatted_answer = self.format_intermediate_answer(response, choices_text)
            intermediate_answers.append((i, formatted_answer))

            # Update previous_answer for context in next iteration
            previous_answer = formatted_answer

        # Final question with full story and all intermediate context
        full_story = story
        final_response = self.ask_final_question(
            full_story=full_story,
            intermediate_answers=intermediate_answers,
            question=question,
            choices_text=choices_text,
            character=character
        )

        # Convert intermediate_answers tuples to a more JSON-serializable format
        belief_state_tracking = [
            {"chunk_index": chunk_idx, "answer": answer}
            for chunk_idx, answer in intermediate_answers
        ]

        return {
            "response": final_response,
            "belief_state_tracking": belief_state_tracking
        }


class IncrementalToM_Variant(IncrementalToM):
    """
    Variant of IncrementalToM that uses belief state tracking instead of
    just storing intermediate answers.
    """

    def ask_intermediate_question(
        self,
        story_so_far: str,
        question: str,
        choices_text: str,
        previous_belief: Optional[str],
        character: Optional[str]
    ) -> str:
        """
        Ask the ToM question and also request a belief state summary.
        """
        belief_part = ""
        if previous_belief:
            belief_part = f"""
            Current Belief State:
            {previous_belief}
            """

        char_focus = f"about {character}" if character else ""

        prompt = f"""Story so far:
        {story_so_far}
        {belief_part}

        Question: {question}

        Choices:
        {choices_text}

        Task: Based on the story so far, answer the question{f' focusing on {character}' if character else ''}.
        Also provide a brief "Belief State" summary {char_focus} capturing what is currently known.

        Format your response as:
        Intermediate Answer: <option letter>
        Belief State: <brief summary of current understanding>"""

        return self.llm_callable(prompt)

    def extract_belief_state(self, response: str) -> str:
        """
        Extract the belief state summary from the response.
        """
        match = re.search(r'Belief State[:\s]+(.+?)(?=\n|$)', response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def run(
        self,
        sample: Dict[str, Any],
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute the variant pipeline with belief state tracking.

        Returns:
            Dictionary containing 'response' and 'belief_state_tracking' keys
        """
        n = chunk_size if chunk_size is not None else self.chunk_size

        story = sample.get("story", sample.get("context", ""))
        question = sample["question"]
        choices_text = format_choices_for_prompt(sample["choices_raw"])
        character = extract_target_name(question)

        sentences = self.split_story_into_sentences(story)

        if len(sentences) <= n:
            prompt = f"""Story:
            {story}

            Question:
            {question}

            Choices:
            {choices_text}

            Think step by step, then give your final answer in the format:
            Answer: <option letter>"""
            return {
                "response": self.llm_callable(prompt),
                "belief_state_tracking": []
            }

        chunks = self.chunk_sentences(sentences, n)
        intermediate_answers = []
        previous_belief = None

        for i, chunk in enumerate(chunks[:-1], 1):
            story_so_far = " ".join(sentences[:i * n])

            response = self.ask_intermediate_question(
                story_so_far=story_so_far,
                question=question,
                choices_text=choices_text,
                previous_belief=previous_belief,
                character=character
            )

            # Extract both answer and belief state
            answer = self.extract_answer_from_response(response)
            belief = self.extract_belief_state(response)

            formatted_answer = self.format_intermediate_answer(response, choices_text)
            intermediate_answers.append((i, formatted_answer))

            # Pass the belief state forward
            previous_belief = belief

        # Final question
        final_response = self.ask_final_question(
            full_story=story,
            intermediate_answers=intermediate_answers,
            question=question,
            choices_text=choices_text,
            character=character
        )

        # Convert intermediate_answers to belief_state_tracking format
        belief_state_tracking = [
            {"chunk_index": chunk_idx, "answer": answer}
            for chunk_idx, answer in intermediate_answers
        ]

        return {
            "response": final_response,
            "belief_state_tracking": belief_state_tracking
        }
