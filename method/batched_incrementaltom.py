import json
import re
import os
from typing import Callable, Dict, Any, List, Optional, Tuple
from pathlib import Path
from prompt import extract_target_name
from utils import format_choices_for_prompt


class SampleState:
    """Tracks the processing state of a single sample."""

    def __init__(
        self,
        sample_id: Any,
        story: str,
        question: str,
        choices_text: str,
        character: Optional[str],
        sentences: List[str],
        chunks: List[List[str]],
        total_chunks: int
    ):
        self.sample_id = sample_id
        self.story = story
        self.question = question
        self.choices_text = choices_text
        self.character = character
        self.sentences = sentences
        self.chunks = chunks
        self.total_chunks = total_chunks

        # Progress tracking
        self.current_chunk_idx = 0  # Which chunk we're about to process
        self.intermediate_answers: List[Tuple[int, str]] = []  # (chunk_idx, answer)
        self.previous_answer: Optional[str] = None
        self.final_answer: Optional[str] = None
        self.is_complete = False

        # For short stories (single chunk)
        self.needs_chunking = total_chunks > 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "choices_text": self.choices_text,
            "character": self.character,
            "sentences": self.sentences,
            "chunks": self.chunks,
            "total_chunks": self.total_chunks,
            "current_chunk_idx": self.current_chunk_idx,
            "intermediate_answers": self.intermediate_answers,
            "previous_answer": self.previous_answer,
            "final_answer": self.final_answer,
            "is_complete": self.is_complete,
            "needs_chunking": self.needs_chunking,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], story: str) -> "SampleState":
        """Deserialize state from dictionary."""
        state = cls(
            sample_id=data["sample_id"],
            story=story,
            question=data["question"],
            choices_text=data["choices_text"],
            character=data["character"],
            sentences=data["sentences"],
            chunks=data["chunks"],
            total_chunks=data["total_chunks"],
        )
        state.current_chunk_idx = data["current_chunk_idx"]
        state.intermediate_answers = [(int(i), a) for i, a in data["intermediate_answers"]]
        state.previous_answer = data["previous_answer"]
        state.final_answer = data["final_answer"]
        state.is_complete = data["is_complete"]
        state.needs_chunking = data["needs_chunking"]
        return state


class BatchedIncrementalToMRunner:
    """
    Batched Incremental ToM runner that processes multiple samples in parallel
    while maintaining sequential chunk processing within each sample.

    Supports full resume capability by saving state after each batch operation.
    """

    def __init__(
        self,
        llm_callable: Callable[[List[str]], List[str]],
        chunk_size: int = 3,
        batch_size: int = 8,
        state_dir: Optional[str] = None,
    ):
        """
        Initialize the batched runner.

        Args:
            llm_callable: Function that takes a list of prompts and returns a list of responses
                         (batched LLM call for GPU efficiency)
            chunk_size: Number of sentences per chunk
            batch_size: Number of samples to process in parallel
            state_dir: Directory to save/load state for resumability (default: ./.incrementaltom_state)
        """
        self.llm_callable = llm_callable
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.state_dir = Path(state_dir) if state_dir else Path(".incrementaltom_state")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Sample states keyed by sample_id
        self.sample_states: Dict[Any, SampleState] = {}

    def _get_state_file(self, output_path: str) -> Path:
        """Get the state file path based on output path."""
        # Use output filename to derive state filename
        output_name = Path(output_path).stem
        return self.state_dir / f"{output_name}_chunk{self.chunk_size}_batch{self.batch_size}_state.json"

    def save_state(self, output_path: str) -> None:
        """Save current processing state to disk."""
        state_file = self._get_state_file(output_path)

        # Count complete vs incomplete for debug
        complete_count = sum(1 for s in self.sample_states.values() if s.is_complete)
        incomplete_count = len(self.sample_states) - complete_count

        state_data = {
            "chunk_size": self.chunk_size,
            "batch_size": self.batch_size,
            "samples": {
                str(sid): state.to_dict()
                for sid, state in self.sample_states.items()
            },
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        print(f"  [State] Saved {len(self.sample_states)} samples ({complete_count} complete, {incomplete_count} incomplete) to {state_file}")

    def load_state(self, output_path: str) -> bool:
        """Load processing state from disk if it exists."""
        state_file = self._get_state_file(output_path)
        print(f"  [State] Looking for state file: {state_file}")
        print(f"  [State] State file exists: {state_file.exists()}")

        if not state_file.exists():
            print(f"  [State] No state file found. Starting fresh.")
            return False

        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            print(f"  [State] Found state file with {len(state_data.get('samples', {}))} samples")
            print(f"  [State] Saved chunk_size={state_data.get('chunk_size')}, batch_size={state_data.get('batch_size')}")
            print(f"  [State] Current chunk_size={self.chunk_size}, batch_size={self.batch_size}")

            # Verify compatibility
            if state_data.get("chunk_size") != self.chunk_size:
                print(f"  [State] ERROR: chunk_size mismatch! Cannot resume with different chunk size.")
                return False
            if state_data.get("batch_size") != self.batch_size:
                print(f"  [State] WARNING: batch_size mismatch. Continuing anyway.")

            # Restore sample states
            self.sample_states = {}
            for sid, sdata in state_data["samples"].items():
                # Need to get original story - will be set during initialization
                self.sample_states[sid] = SampleState.from_dict(sdata, story="")

            # Debug: Show resume status of loaded samples
            chunk_distribution = {}
            complete_count = 0
            for sid, state in self.sample_states.items():
                if state.is_complete:
                    complete_count += 1
                else:
                    chunk_idx = state.current_chunk_idx
                    chunk_distribution[chunk_idx] = chunk_distribution.get(chunk_idx, 0) + 1

            print(f"  [State] Loaded {len(self.sample_states)} samples: {complete_count} complete, {len(self.sample_states) - complete_count} incomplete")
            if chunk_distribution:
                print(f"  [State] Incomplete samples by current_chunk_idx: {chunk_distribution}")

            return True
        except Exception as e:
            print(f"  [State] Error loading state: {e}")
            import traceback
            traceback.print_exc()
            return False

    def clear_state(self, output_path: str) -> None:
        """Clear state file after successful completion."""
        state_file = self._get_state_file(output_path)
        if state_file.exists():
            state_file.unlink()
            print(f"  [State] Cleared {state_file}")

    def split_story_into_sentences(self, story: str) -> List[str]:
        """Split a story into sentences."""
        raw_sentences = re.split(r'(?<=[.!?])\s+', story.strip())
        return [s.strip() for s in raw_sentences if s.strip()]

    def chunk_sentences(self, sentences: List[str], chunk_size: int) -> List[List[str]]:
        """Group sentences into chunks."""
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def extract_answer_from_response(self, response: str) -> str:
        """Extract answer letter from response."""
        match = re.search(r'(?:Intermediate Answer|Answer)[:\s]+([A-O])\b', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        matches = re.findall(r'\b([A-O])\b', response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        return response.strip()

    def format_intermediate_answer(self, response: str, choices_text: str) -> str:
        """Format answer with full choice text."""
        answer_letter = self.extract_answer_from_response(response)

        choices = {}
        for line in choices_text.split('\n'):
            match = re.match(r'^([A-O])\.\s*(.+)$', line.strip())
            if match:
                choices[match.group(1)] = match.group(2)

        if answer_letter in choices:
            return f"{answer_letter}. {choices[answer_letter]}"
        return answer_letter

    def build_intermediate_prompt(
        self,
        state: SampleState,
        chunk_idx: int
    ) -> str:
        """Build prompt for intermediate question at given chunk."""
        # Build story so far
        story_so_far = " ".join(state.sentences[: (chunk_idx + 1) * self.chunk_size])

        context_part = ""
        if state.previous_answer:
            context_part = f"""
            Current Belief State (based on previous parts of the story):
            {state.previous_answer}

            Important: Consider this belief state when answering, as it represents what has been understood so far."""

                    return f"""Story so far:
            {story_so_far}
            {context_part}

            Question: {state.question}

            Choices:
            {state.choices_text}

            Task: Based on the story so far{f' and the current belief state' if state.previous_answer else ''}, answer the question.
            Think step by step, then give your intermediate answer in the format:
            Intermediate Answer: <option letter>

            Explain your reasoning briefly, then provide the answer."""

    def build_final_prompt(self, state: SampleState) -> str:
        """Build final prompt for a sample."""
        # Build context from intermediate answers
        intermediate_context = ""
        if state.intermediate_answers:
            intermediate_context = "Previous Understanding Checkpoints:\n"
            for chunk_idx, answer in state.intermediate_answers:
                intermediate_context += f"  After chunk {chunk_idx}: {answer}\n"
            intermediate_context += "\n"

        char_focus = f"Focus particularly on what '{state.character}' knows and believes." if state.character else ""

        return f"""Full Story:
        {state.story}

        {intermediate_context}Question: {state.question}

        Choices:
        {state.choices_text}

        Task: Answer the question considering the full story and intermediate understandings above.
        {char_focus}
        Think step by step, tracking what each character knows at each point, then give your final answer:
        Answer: <option letter>"""

    def build_simple_prompt(self, state: SampleState) -> str:
        """Build simple prompt for short stories (no chunking needed)."""
        return f"""Story:
        {state.story}

        Question: {state.question}

        Choices:
        {state.choices_text}

        Think step by step, then give your final answer in the format:
        Answer: <option letter>"""

    def initialize_samples(self, samples: List[Dict[str, Any]]) -> None:
        """Initialize sample states for a batch of samples."""
        for sample in samples:
            sample_id = str(sample["sample_id"])

            # Skip if already initialized (from loaded state)
            if sample_id in self.sample_states:
                # Update story reference
                self.sample_states[sample_id].story = sample.get("story", sample.get("context", ""))
                continue

            story = sample.get("story", sample.get("context", ""))
            question = sample["question"]
            choices_text = format_choices_for_prompt(sample["choices_raw"])
            character = extract_target_name(question)

            sentences = self.split_story_into_sentences(story)
            chunks = self.chunk_sentences(sentences, self.chunk_size)
            total_chunks = len(chunks)

            state = SampleState(
                sample_id=sample_id,
                story=story,
                question=question,
                choices_text=choices_text,
                character=character,
                sentences=sentences,
                chunks=chunks,
                total_chunks=total_chunks,
            )

            self.sample_states[sample_id] = state

    def process_intermediate_batch(
        self,
        states: List[SampleState],
        chunk_idx: int
    ) -> None:
        """Process intermediate questions for a batch of samples at given chunk index."""
        # Build prompts for all samples
        prompts = [self.build_intermediate_prompt(state, chunk_idx) for state in states]

        # Batch LLM call
        responses = self.llm_callable(prompts)

        # Update states with responses
        for state, response in zip(states, responses):
            formatted_answer = self.format_intermediate_answer(response, state.choices_text)
            state.intermediate_answers.append((chunk_idx + 1, formatted_answer))
            state.previous_answer = formatted_answer
            state.current_chunk_idx = chunk_idx + 1

    def process_final_batch(self, states: List[SampleState]) -> Dict[str, str]:
        """Process final questions for a batch of samples. Returns sample_id -> answer mapping."""
        # Build prompts
        prompts = []
        for state in states:
            if state.needs_chunking:
                prompts.append(self.build_final_prompt(state))
            else:
                prompts.append(self.build_simple_prompt(state))

        # Batch LLM call
        responses = self.llm_callable(prompts)

        # Update states and collect results
        results = {}
        for state, response in zip(states, responses):
            state.final_answer = response
            state.is_complete = True
            results[state.sample_id] = response

        return results

    def run(
        self,
        samples: List[Dict[str, Any]],
        output_path: str,
        resume: bool = False,
        completed_sample_ids: set = None,
    ) -> Dict[str, str]:
        """
        Run batched incremental ToM on a list of samples.

        Args:
            samples: List of samples to process (pass ALL samples, filtering is handled internally)
            output_path: Output file path (used for state file naming)
            resume: Whether to resume from saved state
            completed_sample_ids: Set of sample_ids that are already complete (from output file)

        Returns:
            Dictionary mapping sample_id to final answer
        """
        completed_sample_ids = completed_sample_ids or set()

        # Load state if resuming
        if resume:
            self.load_state(output_path)

        # Initialize all samples
        print(f"[BatchedIncrementalToM] Initializing {len(samples)} samples...")
        self.initialize_samples(samples)

        # Mark samples that are already complete in the output file
        if completed_sample_ids:
            print(f"[BatchedIncrementalToM] Marking {len(completed_sample_ids)} samples as complete (from output file)")
            for sid in completed_sample_ids:
                sid_str = str(sid)
                if sid_str in self.sample_states:
                    self.sample_states[sid_str].is_complete = True

        # Debug: Show sample_states after initialization
        print(f"[BatchedIncrementalToM] After initialization: {len(self.sample_states)} samples in state")

        # Track which samples need processing
        incomplete_states = [
            state for state in self.sample_states.values()
            if not state.is_complete
        ]

        complete_states = [
            state for state in self.sample_states.values()
            if state.is_complete
        ]

        print(f"[BatchedIncrementalToM] {len(complete_states)} samples already complete, {len(incomplete_states)} need processing")

        if not incomplete_states:
            print("[BatchedIncrementalToM] All samples already complete!")
            return {sid: state.final_answer for sid, state in self.sample_states.items()}

        # Show chunk distribution of incomplete samples
        chunk_distribution = {}
        for state in incomplete_states:
            chunk_idx = state.current_chunk_idx
            chunk_distribution[chunk_idx] = chunk_distribution.get(chunk_idx, 0) + 1
        print(f"[BatchedIncrementalToM] Incomplete samples by current_chunk_idx: {chunk_distribution}")

        # Group samples by remaining chunks needed
        max_chunks = max(
            state.total_chunks for state in incomplete_states
            if state.needs_chunking
        ) if any(s.needs_chunking for s in incomplete_states) else 1

        # Process chunk by chunk across all samples
        for chunk_idx in range(max_chunks - 1):  # All but last chunk
            # Find samples that need this chunk processed
            batch_states = [
                state for state in incomplete_states
                if state.needs_chunking
                and state.current_chunk_idx == chunk_idx
                and not state.is_complete
            ]

            if not batch_states:
                continue

            print(f"  Processing chunk {chunk_idx + 1}/{max_chunks} for {len(batch_states)} samples...")

            # Process in sub-batches
            num_sub_batches = (len(batch_states) + self.batch_size - 1) // self.batch_size
            for i in range(0, len(batch_states), self.batch_size):
                sub_batch = batch_states[i:i + self.batch_size]
                sub_batch_num = i // self.batch_size + 1
                print(f"    Sub-batch {sub_batch_num}/{num_sub_batches}: {len(sub_batch)} samples")
                self.process_intermediate_batch(sub_batch, chunk_idx)

                # Save state after each sub-batch for finer resume granularity
                # (in case job is killed mid-chunk)
                self.save_state(output_path)

            # Also save at end of chunk (redundant but ensures consistency)
            self.save_state(output_path)

        # Process final questions for all incomplete samples
        print(f"[BatchedIncrementalToM] Processing final questions for {len(incomplete_states)} samples...")
        final_results = {}

        for i in range(0, len(incomplete_states), self.batch_size):
            sub_batch = incomplete_states[i:i + self.batch_size]
            print(f"  Final sub-batch {i//self.batch_size + 1}/{(len(incomplete_states) + self.batch_size - 1)//self.batch_size}: {len(sub_batch)} samples")
            batch_results = self.process_final_batch(sub_batch)
            final_results.update(batch_results)

            # Save state after each final batch
            self.save_state(output_path)

        # Collect all results
        all_results = {
            sid: state.final_answer
            for sid, state in self.sample_states.items()
        }

        # Clear state on successful completion
        self.clear_state(output_path)

        return all_results


class BatchedIncrementalToM:
    """
    Wrapper class that provides a single-sample interface compatible with
    the existing method interface in main.py, but uses batched processing
    internally for GPU efficiency.
    """

    def __init__(
        self,
        llm_callable: Callable[[str], str],
        chunk_size: int = 3,
        batch_size: int = 8,
    ):
        """
        Initialize with a single-prompt LLM callable.

        Note: For GPU batching, you should wrap your LLM call to handle batches.
        See create_batched_llm_callable() helper.
        """
        self.single_llm_callable = llm_callable
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        # Will be initialized on first run
        self.batched_runner: Optional[BatchedIncrementalToMRunner] = None

    def run(
        self,
        sample: Dict[str, Any],
        chunk_size: Optional[int] = None,
    ) -> str:
        """
        Run on a single sample (for compatibility).

        Note: For efficient GPU batching, use BatchedIncrementalToMRunner directly
        with a batched LLM callable.
        """
        n = chunk_size if chunk_size is not None else self.chunk_size

        # Create runner with single-sample wrapper
        def batched_wrapper(prompts: List[str]) -> List[str]:
            return [self.single_llm_callable(p) for p in prompts]

        runner = BatchedIncrementalToMRunner(
            llm_callable=batched_wrapper,
            chunk_size=n,
            batch_size=1,  # Process one at a time for single-sample mode
        )

        results = runner.run([sample], output_path="/tmp/single_sample", resume=False)
        return results.get(str(sample["sample_id"]), "")


def create_batched_llm_callable(
    base_callable: Callable[[str], str],
    batch_size: int = 8,
) -> Callable[[List[str]], List[str]]:
    """
    Create a batched LLM callable from a single-prompt callable.

    This is a simple sequential batcher. For true GPU parallelism,
    your model callable should natively support batching.

    Args:
        base_callable: Function that takes a single prompt and returns response
        batch_size: Number of prompts to process before yielding

    Returns:
        Function that takes a list of prompts and returns list of responses
    """
    def batched_callable(prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            responses.append(base_callable(prompt))
        return responses
    return batched_callable


# For true GPU batching with HuggingFace models
def create_hf_batched_llm_callable(
    model,
    tokenizer,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> Callable[[List[str]], List[str]]:
    """
    Create a batched LLM callable for HuggingFace models with true GPU parallelism.

    Args:
        model: HuggingFace model (on GPU)
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for processing
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        Batched callable for use with BatchedIncrementalToMRunner
    """
    import torch

    def batched_callable(prompts: List[str]) -> List[str]:
        responses = []

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode
            batch_responses = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            responses.extend(batch_responses)

        return responses

    return batched_callable
