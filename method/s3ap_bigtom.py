import json
import re
from typing import Any, Callable, Dict, Optional


BIGTOM_S3AP_INSTRUCTIONS = """
You are analyzing a narrative for Theory of Mind reasoning.

Guidelines:
1. A character witnesses everything they directly experience (actions they take, things they see/hear).
2. A character does NOT know about events they did not witness (things that happen while they're away, hidden changes).
3. Track what each character knows at each point in the narrative.
4. Note when characters have false beliefs due to missing information.
5. Include character mental states (beliefs, intentions) when relevant.
"""


S3AP_FORMAT_INSTRUCTIONS = """
Return only valid JSON. The top-level value must be a JSON array. Each item is
one timestep with exactly these keys:
- "timestep": string describing the event
- "state": string describing the world state before the action
- "observations": object mapping each character to what they observe
- "mental_states": object mapping each character to their beliefs/intentions
- "actions": object mapping each character to their action (if any)

Use "none" when a character observes nothing or takes no action.
"""


def extract_json_candidate(text: str) -> Optional[str]:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        return stripped

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start != -1 and array_end > array_start:
        return stripped[array_start : array_end + 1]

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start != -1 and object_end > object_start:
        return stripped[object_start : object_end + 1]

    return None


def parse_json_from_text(text: str) -> Optional[Any]:
    candidate = extract_json_candidate(text)
    if candidate is None:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def normalize_json_text(text: str) -> str:
    parsed = parse_json_from_text(text)
    if parsed is None:
        return text.strip()
    return json.dumps(parsed, ensure_ascii=False, indent=2)


class S3APBigToM:
    """
    S3AP adapted for BigToM dataset.

    Uses binary answer format (A/B) instead of A-O options.
    """

    def __init__(self, llm_callable: Callable[[str], str]):
        self.llm_callable = llm_callable
        self.last_parser_prompt: Optional[str] = None
        self.last_qa_prompt: Optional[str] = None
        self.last_s3ap_representation: Optional[str] = None

    def build_parser_prompt(self, story: str) -> str:
        return f"""Please analyze the following narrative.

                #### Context:
                {story}

                #### Task specific instructions:
                {BIGTOM_S3AP_INSTRUCTIONS.strip()}

                #### Format instructions:
                {S3AP_FORMAT_INSTRUCTIONS.strip()}

                Create a structured S3AP representation of the narrative. Track what each character observes and believes at each timestep."""

    def parse_story(self, story: str) -> str:
        self.last_parser_prompt = self.build_parser_prompt(story)
        raw_representation = self.llm_callable(self.last_parser_prompt)
        representation = normalize_json_text(raw_representation)
        self.last_s3ap_representation = representation
        return representation

    def build_qa_prompt(
        self,
        story: str,
        s3ap_representation: str,
        question: str,
        true_answer: str,
        wrong_answer: str,
    ) -> str:
        return f"""## Context
                {story}

                ## Structured Representation
                (to help you better understand the narrative and character mental states)
                {s3ap_representation}

                ## Task
                Question:
                {question}

                Possible Answers:
                A: {true_answer}
                B: {wrong_answer}

                Use the original narrative and the structured representation to reason about characters' observations, beliefs, and mental states.
                Think step by step, then give your final answer in the format:
                Answer: A
                or
                Answer: B"""

    def run(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("narrative", ""))
        question = sample["question"]
        true_answer = sample.get("true_answer", sample.get("answer", ""))
        wrong_answer = sample.get("wrong_answer", "")

        s3ap_representation = self.parse_story(story)
        self.last_qa_prompt = self.build_qa_prompt(
            story=story,
            s3ap_representation=s3ap_representation,
            question=question,
            true_answer=true_answer,
            wrong_answer=wrong_answer,
        )
        return self.llm_callable(self.last_qa_prompt)

