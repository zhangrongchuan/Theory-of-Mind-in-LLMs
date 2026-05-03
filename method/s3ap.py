import json
import re
from typing import Any, Callable, Dict, Optional, Sequence

from utils import format_choices_for_prompt


HITOM_S3AP_INSTRUCTIONS = """
You are dissecting the HiToM scenarios. You should assume the following:
1. An agent witnesses everything and every movement before exiting a location.
2. An agent A can infer another agent B's mental state only if A and B have
   been in the same location, or have private or public interactions.
3. Every agent tends to lie. What a character tells others does not affect
   that character's actual belief.
4. Agents in private communications know that others will not hear them, but
   they know that anyone can hear public claims.
5. In each agent's observation, include object locations if the agent is in the
   same location as the object.
"""


S3AP_FORMAT_INSTRUCTIONS = """
Return only valid JSON. The top-level value must be a JSON array. Each item is
one social world timestep with exactly these keys:
- "timestep": string
- "state": string describing the world state before the action
- "observations": object mapping each agent name to that agent's observation
- "actions": object mapping each agent name to that agent's action

Use "none" when an agent observes nothing or takes no action. Use
"<same_as_state />" when an agent's observation covers the full current state.
Use "<mental_state>...</mental_state>" for beliefs, intentions, emotions, or
other internal information that is not directly observable by others.
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


class S3AP:
    """
    Implements the S3AP static social-reasoning method from Social World Models.

    The parser step is query-independent: it receives only the story and builds
    a structured social world representation. The QA step then answers using the
    original story plus that structured representation as extra information.
    """

    def __init__(self, llm_callable: Callable[[str], str]):
        self.llm_callable = llm_callable
        self.last_parser_prompt: Optional[str] = None
        self.last_qa_prompt: Optional[str] = None
        self.last_s3ap_representation: Optional[str] = None

    def build_parser_prompt(self, story: str) -> str:
        return f"""Please analyze the following narrative/context.

                #### Context:
                {story}

                #### Task specific instructions:
                {HITOM_S3AP_INSTRUCTIONS.strip()}

                #### Format instructions:
                {S3AP_FORMAT_INSTRUCTIONS.strip()}

                Create a query-independent S3AP representation of the narrative. Preserve the
                event order and track which agents can observe each event. Do not answer any
                downstream question here."""

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
        choices_text: str,
    ) -> str:
        return f"""## Context
                {story}

                ## Extra Info
                (to help you better understand the social world state)
                {s3ap_representation}

                ## Task
                Question:
                {question}

                Choices:
                {choices_text}

                Use the original story and the S3AP representation to reason about agents'
                observations, beliefs, and higher-order beliefs. Think step by step, then give
                your final answer in the format:
                Answer: <option letter>"""

    def run(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("context", ""))
        question = sample["question"]
        choices_text = format_choices_for_prompt(sample["choices_raw"])

        s3ap_representation = self.parse_story(story)
        self.last_qa_prompt = self.build_qa_prompt(
            story=story,
            s3ap_representation=s3ap_representation,
            question=question,
            choices_text=choices_text,
        )
        return self.llm_callable(self.last_qa_prompt)


class ForeseeAndAct:
    """
    Generic one-step Foresee-and-Act workflow for interactive social agents.

    This is intentionally not wired into HiToM evaluation because HiToM is a
    static multiple-choice benchmark. The class is model-agnostic and can be
    connected later to an agent model and a social-world-model callable.
    """

    def __init__(
        self,
        action_llm_callable: Callable[[str], str],
        swm_llm_callable: Optional[Callable[[str], str]] = None,
    ):
        self.action_llm_callable = action_llm_callable
        self.swm_llm_callable = swm_llm_callable or action_llm_callable
        self.last_candidate_prompt: Optional[str] = None
        self.last_simulation_prompt: Optional[str] = None
        self.last_refinement_prompt: Optional[str] = None
        self.last_candidate_action: Optional[str] = None
        self.last_predicted_state: Optional[str] = None
        self.last_refined_action: Optional[str] = None

    def build_candidate_action_prompt(
        self,
        agent: str,
        social_world_state: str,
        goal: str,
        action_space: Sequence[str],
        history: str = "",
        format_instructions: str = "Return the action as a JSON string.",
    ) -> str:
        actions = "\n".join(f"- {action}" for action in action_space) if action_space else "Any valid action."
        history_block = history if history else "No prior interaction history."

        return f"""You are {agent}.

                Interaction history:
                {history_block}

                Current social world state:
                {social_world_state}

                Your goal:
                {goal}

                Available actions:
                {actions}

                Sample one intended action that can help you pursue the goal.
                {format_instructions}"""

    def sample_action(
        self,
        agent: str,
        social_world_state: str,
        goal: str,
        action_space: Sequence[str],
        history: str = "",
        format_instructions: str = "Return the action as a JSON string.",
    ) -> str:
        self.last_candidate_prompt = self.build_candidate_action_prompt(
            agent=agent,
            social_world_state=social_world_state,
            goal=goal,
            action_space=action_space,
            history=history,
            format_instructions=format_instructions,
        )
        self.last_candidate_action = self.action_llm_callable(self.last_candidate_prompt)
        return self.last_candidate_action

    def build_social_world_model_prompt(
        self,
        social_world_state: str,
        agent: str,
        intended_action: str,
        history: str = "",
    ) -> str:
        history_block = history if history else "No prior interaction history."

        return f"""You are a social world model.

                Interaction history:
                {history_block}

                Current S3AP social world state:
                {social_world_state}

                The focal agent is {agent}. The focal agent's intended action is:
                {intended_action}

                Predict the next social world state after this intended action. Include how the
                environment may respond, what other agents observe, and the agents' likely
                mental states. Return only valid S3AP JSON with fields "timestep", "state",
                "observations", and "actions"."""

    def predict_next_state(
        self,
        social_world_state: str,
        agent: str,
        intended_action: str,
        history: str = "",
    ) -> str:
        self.last_simulation_prompt = self.build_social_world_model_prompt(
            social_world_state=social_world_state,
            agent=agent,
            intended_action=intended_action,
            history=history,
        )
        raw_state = self.swm_llm_callable(self.last_simulation_prompt)
        self.last_predicted_state = normalize_json_text(raw_state)
        return self.last_predicted_state

    def build_refinement_prompt(
        self,
        agent: str,
        history: str,
        intended_action: str,
        predicted_state: str,
        goal: str,
        format_instructions: str = "Return only a JSON string including the action type and argument.",
    ) -> str:
        history_block = history if history else "No prior interaction history."

        return f"""You are {agent}.

                Here is the interaction history between you and the other agent so far:
                {history_block}

                Here is your intended action:
                {intended_action}

                Here is the predicted mental state after you take the intended action. Use it
                to generate a better action for achieving your goal:
                {predicted_state}

                Your goal:
                {goal}

                Please generate a refined action so that you can achieve your goal better.
                {format_instructions}"""

    def refine_action(
        self,
        agent: str,
        history: str,
        intended_action: str,
        predicted_state: str,
        goal: str,
        format_instructions: str = "Return only a JSON string including the action type and argument.",
    ) -> str:
        self.last_refinement_prompt = self.build_refinement_prompt(
            agent=agent,
            history=history,
            intended_action=intended_action,
            predicted_state=predicted_state,
            goal=goal,
            format_instructions=format_instructions,
        )
        self.last_refined_action = self.action_llm_callable(self.last_refinement_prompt)
        return self.last_refined_action

    def run(
        self,
        agent: str,
        social_world_state: str,
        goal: str,
        action_space: Sequence[str],
        history: str = "",
        action_format_instructions: str = "Return the action as a JSON string.",
        refinement_format_instructions: str = "Return only a JSON string including the action type and argument.",
    ) -> str:
        intended_action = self.sample_action(
            agent=agent,
            social_world_state=social_world_state,
            goal=goal,
            action_space=action_space,
            history=history,
            format_instructions=action_format_instructions,
        )
        predicted_state = self.predict_next_state(
            social_world_state=social_world_state,
            agent=agent,
            intended_action=intended_action,
            history=history,
        )
        return self.refine_action(
            agent=agent,
            history=history,
            intended_action=intended_action,
            predicted_state=predicted_state,
            goal=goal,
            format_instructions=refinement_format_instructions,
        )
