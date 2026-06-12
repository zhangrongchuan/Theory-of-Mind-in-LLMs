from typing import Any, Callable, Dict, Optional

from method.shared_epistemic_core import SharedEpistemicCore
from prompt import extract_target_name
from utils import format_choices_for_prompt


class PercepToM:
    """
    Complexity-adaptive PercepToM.

    Shallow mental-state questions use a perspective-only context. Deeply
    nested questions use a shared epistemic core so that only mutually
    attributable events can cross every belief layer.
    """

    def __init__(self, llm_callable: Callable[[str], str]):
        self.llm_callable = llm_callable
        self.last_initial_answer: Optional[str] = None
        self.last_checked_answer: Optional[str] = None
        self.last_check_raw: Optional[str] = None
        self.last_validation_evidence: Optional[str] = None
        self.last_recursive_answer: Optional[str] = None
        self.last_candidate_answers: Optional[Dict[str, Any]] = None
        self.last_route: Optional[str] = None
        self.last_representation: Optional[str] = None

    def perception_inference(self, story: str, character: str) -> str:
        prompt = f"""Story:
        {story}

        Task: Based on the story above, identify exactly what the character '{character}' has perceived (seen, heard, or witnessed). If they left the room or were absent during certain events, explicitly note what they missed.

        Perception of {character}:"""
        return self.llm_callable(prompt)

    def perception_to_belief_inference(self, perception: str, character: str) -> str:
        prompt = f"""Character: {character}
        Perception: {perception}

        Task: Based *only* on the perception provided above (and strictly ignoring any omniscient knowledge of the actual world state), what does {character} currently believe to be true about the situation and the locations of objects/people?

        Belief State of {character}:"""
        return self.llm_callable(prompt)

    def answer_tom_question(
        self,
        story: str,
        belief: str,
        character: str,
        question: str,
        choices_text: str,
    ) -> str:
        prompt = f"""Story:
        {story}

        Belief State of {character}: 
        {belief}

        Question: 
        {question}

        Choices:
        {choices_text}

        Task: Answer the question. You must rely primarily on the "Belief State" of {character} to answer this question, rather than the objective reality described in the "Story".
        Think step by step, then give your final answer in the format:
        Answer: <option letter>"""
        return self.llm_callable(prompt)

    def take_perspective(self, story: str, character: str) -> str:
        prompt = f"""The following is a sequence of events about some characters,
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
        return self.llm_callable(prompt).strip()

    def answer_from_perspective(
        self,
        perspective: str,
        character: str,
        question: str,
        choices_text: str,
    ) -> str:
        prompt = f"""{perspective}

                You are {character}.
                Based only on the above information from your perspective, answer the following
                question:
                {question}

                Choices:
                {choices_text}

                You must choose one of the above choices. Think briefly if needed, then give
                your final answer in the format:
                Answer: <option letter>"""
        return self.llm_callable(prompt)

    def answer_world_state(self, story: str, question: str, choices_text: str) -> str:
        prompt = f"""Story:
                {story}

                Question:
                {question}

                Choices:
                {choices_text}

                The question does not specify a character perspective. Answer based on the
                story. Think briefly if needed, then give your final answer in the format:
                Answer: <option letter>"""
        return self.llm_callable(prompt)

    def run(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("context", ""))
        question = sample["question"]
        choices_text = format_choices_for_prompt(sample["choices_raw"])
        question_order = int(sample.get("question_order", 0))
        character = extract_target_name(question)

        self.last_initial_answer = None
        self.last_checked_answer = None
        self.last_check_raw = None
        self.last_validation_evidence = None
        self.last_recursive_answer = None
        self.last_candidate_answers = None
        self.last_route = None
        self.last_representation = None

        if not character:
            self.last_route = "objective_story"
            answer = self.answer_world_state(story, question, choices_text)
            self.last_initial_answer = answer
            return answer

        if question_order <= 2:
            self.last_route = "perspective_isolation"
            perspective = self.take_perspective(story, character)
            self.last_representation = perspective
            answer = self.answer_from_perspective(
                perspective,
                character,
                question,
                choices_text,
            )
            self.last_checked_answer = answer
            return answer

        if question_order >= 3:
            self.last_route = "shared_epistemic_core"
            solver = SharedEpistemicCore(self.llm_callable)
            answer = solver.run(sample)
            self.last_representation = solver.last_core
            self.last_validation_evidence = solver.last_core
            self.last_checked_answer = answer
            return answer

        self.last_route = "perception_belief"
        perception = self.perception_inference(story, character)
        belief = self.perception_to_belief_inference(perception, character)
        self.last_representation = belief
        answer = self.answer_tom_question(
            story,
            belief,
            character,
            question,
            choices_text,
        )
        self.last_initial_answer = answer
        return answer
