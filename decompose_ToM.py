import re
from typing import Optional, Dict, Any

class DecomposeToM:
    def __init__(self, llm_callable, max_recursion_depth: int = 3):
        """
        Initializes the Decompose-ToM inference algorithm.
        
        :param llm_callable: A function that takes a prompt string and returns a text response.
                             (e.g., a wrapper around the Gemini or OpenAI API).
        :param max_recursion_depth: Maximum depth for recursive ToM reasoning (higher-order ToM).
        """
        self.llm = llm_callable
        self.max_depth = max_recursion_depth

    def identify_subject(self, question: str) -> Optional[str]:
        """
        Step 1: Identifies whose perspective the question is asking about.
        If the question is objective and requires no ToM, returns None.
        """
        prompt = f"""
        Analyze the following question and identify the specific agent or subject whose perspective, 
        belief, or knowledge is being queried. 
        If the question is about objective reality and not about someone's mental state, return "NONE".
        Respond ONLY with the name of the agent or "NONE".
        
        Question: "{question}"
        Agent:"""
        
        response = self.llm(prompt).strip()
        return None if "NONE" in response.upper() else response

    def reframe_question(self, question: str, subject: str) -> str:
        """
        Step 2: Reframes the question from the perspective of the identified subject.
        Example: "Where does John think the apple is?" -> "Where is the apple?"
        """
        prompt = f"""
        Reframe the following Theory of Mind question to be a direct question from the perspective 
        of the subject: {subject}. Remove references to their own belief.
        
        Original Question: "{question}"
        Reframed Direct Question:"""
        
        return self.llm(prompt).strip()

    def update_world_model(self, story: str, subject: str) -> str:
        """
        Step 3: Updates the world model (story/context) based on what the subject actually observed.
        Filters out events that happened while the subject was absent or unaware.
        """
        prompt = f"""
        You are simulating the exact knowledge and memory of "{subject}".
        Read the following sequence of events. Based ONLY on what {subject} observed or was told, 
        reconstruct the story. Omit any events, movements, or dialogue that {subject} is unaware of 
        (e.g., things that happened before they entered a room or after they left).
        
        Original Story:
        {story}
        
        {subject}'s World Model (Observed Story):"""
        
        return self.llm(prompt).strip()

    def knowledge_availability(self, world_model: str, question: str) -> bool:
        """
        Step 4: Checks if the subject's updated world model contains enough information 
        to answer the reframed question.
        """
        prompt = f"""
        Based on the following known context, is it possible to answer the question?
        Return ONLY "YES" or "NO".
        
        Context:
        {world_model}
        
        Question: "{question}"
        Answer:"""
        
        response = self.llm(prompt).strip().upper()
        return "YES" in response

    def direct_qa(self, context: str, question: str) -> str:
        """
        Base QA function for when ToM decomposition is complete or unneeded.
        """
        prompt = f"""
        Context: {context}
        Question: {question}
        Answer the question accurately based on the context provided.
        Answer:"""
        return self.llm(prompt).strip()

    def run(self, story: str, question: str, current_depth: int = 0) -> str:
        """
        Main Recursive Algorithm: Executes the Decompose-ToM pipeline.
        Handles higher-order ToM by recursively simulating nested perspectives.
        """
        # Base case: Prevent infinite recursion
        if current_depth >= self.max_depth:
            return self.direct_qa(story, question)

        # 1. Subject Identification
        subject = self.identify_subject(question)
        
        # If no subjective perspective is queried, answer directly based on current world model
        if not subject:
            return self.direct_qa(story, question)

        # 2. World Model Updation (Simulate pretend-play for the subject)
        agent_world_model = self.update_world_model(story, subject)

        # 3. Question-Reframing (Shift the query to the agent's perspective)
        reframed_question = self.reframe_question(question, subject)

        # 4. Knowledge Availability (Check if the agent actually knows the answer)
        has_knowledge = self.knowledge_availability(agent_world_model, reframed_question)

        if not has_knowledge:
            return f"Based on the events, {subject} does not have enough information to know the answer."

        # Recursive Call: 
        # If the reframed question STILL contains a ToM query (e.g., "What does Alice think Bob thinks?"),
        # the algorithm recursively decomposes it again using the inner agent's world model.
        return self.run(agent_world_model, reframed_question, current_depth + 1)