from typing import Callable, Dict, Any
from prompt import extract_target_name
from utils import format_choices_for_prompt

class PercepToM:
    def __init__(self, llm_callable: Callable[[str], str]):
        """
        Initializes PercepToM with a specific model calling function 
        (e.g., call_model_ollama, call_model_deepseek).
        """
        self.llm_callable = llm_callable
    
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

    def answer_tom_question(self, story: str, belief: str, character: str, question: str, choices_text: str) -> str:
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

    def run(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("context", ""))
        question = sample["question"]
        choices_text = format_choices_for_prompt(sample["choices_raw"])
        
        # 1. Identify Target Character
        character = extract_target_name(question)
        
        # Fallback: If no character is identified (e.g. world-state question), do standard CoT
        if not character:
            fallback_prompt = f"Story:\n{story}\n\nQuestion:\n{question}\n\nChoices:\n{choices_text}\n\nThink step by step, then give your final answer in the format:\nAnswer: <option letter>"
            return self.llm_callable(fallback_prompt)

        # 2. PercepToM Pipeline
        perception = self.perception_inference(story, character)
        belief = self.perception_to_belief_inference(perception, character)
        final_answer = self.answer_tom_question(story, belief, character, question, choices_text)
        
        # Return final reasoning trace and answer
        return final_answer