from typing import Any, Callable, Dict, Optional

from utils import format_choices_for_prompt


METHOD_NAME = "SHAREDEVIDENCETOM"


class SharedEvidenceToM:
    """
    Backward-compatible implementation of the old SharedEpistemicCore prompt.

    The public method name is SHAREDEVIDENCETOM, but the model-facing prompts
    intentionally remain byte-for-byte compatible with the old shared core
    method so prior results stay comparable across all question orders.
    """

    def __init__(self, llm_callable: Callable[[str], str]):
        self.llm_callable = llm_callable
        self.last_evidence_prompt: Optional[str] = None
        self.last_evidence: Optional[str] = None
        self.last_qa_prompt: Optional[str] = None
        self.last_answer: Optional[str] = None

        # Backward-compatible names for older runners and analysis scripts.
        self.last_core_prompt: Optional[str] = None
        self.last_core: Optional[str] = None

    def build_evidence_prompt(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("context", ""))
        return f"""/no_think
         Extract the SHARED EPISTEMIC CORE needed for this deeply nested belief question.

         Story:
         {story}

         Question:
         {sample["question"]}

         Do not answer the question. Follow these rules:
         1. Read only the Question to identify the ordered belief chain. Include only
            names that occur in the Question, in the same order as the repeated
            "X thinks Y thinks ..." nesting. Do not add any other story character.
         2. Ignore unrelated objects and unrelated story episodes.
         3. The shared core is an INTERSECTION, not a union. Output a target-object
            event only when every single agent in the extracted chain directly knows
            the event and can attribute that knowledge to the other agents in the chain.
            If even one chain agent does not know it, exclude it.
         4. Track presence exactly: all named entrants remain present until their own
            exit. If any chain agent exited before an event, that event is not in the
            shared core. Reuniting in a waiting room does not reveal earlier events.
         5. A private claim belongs in the shared core only if every chain agent knows
            it. A public claim belongs only if every chain agent was present for it.
         6. Preserve the original wording and chronological order. Do not add beliefs,
            explanations, inferred events, or an answer.
         7. Output only the requested structure. Do not explain your decisions.
         8. Include the initial target-location statement when every chain agent was
            present for it. Never omit it merely because later movements exist.

         Output:
         CHAIN: ...
         TARGET: ...
         SHARED TARGET-OBJECT EVENTS:
         ..."""

    def build_core_prompt(self, sample: Dict[str, Any]) -> str:
        return self.build_evidence_prompt(sample)

    def build_qa_prompt(self, evidence: str, sample: Dict[str, Any]) -> str:
        choices_text = format_choices_for_prompt(sample["choices_raw"])
        return f"""Shared epistemic core:
         {evidence}

         Question:
         {sample["question"]}

         Choices:
         {choices_text}

         Answer the deeply nested belief question using only the shared epistemic core.
         The answer is the target object's location after the last shared physical
         location event. Claims communicate information but do not physically move the
         object. Do not import any event absent from the shared core.

         End with exactly:
         Answer: <option letter>"""

    def run(self, sample: Dict[str, Any]) -> str:
        self.last_evidence_prompt = self.build_evidence_prompt(sample)
        self.last_core_prompt = self.last_evidence_prompt
        self.last_evidence = self.llm_callable(self.last_evidence_prompt).strip()
        self.last_core = self.last_evidence
        self.last_qa_prompt = self.build_qa_prompt(self.last_evidence, sample)
        self.last_answer = self.llm_callable(self.last_qa_prompt)
        return self.last_answer
