from typing import Any, Dict

from method.shared_evidence_tom import SharedEvidenceToM


class SharedEvidenceToMBigToM(SharedEvidenceToM):
    """BigToM prompt adapter for the existing SharedEvidenceToM pipeline."""

    def build_evidence_prompt(self, sample: Dict[str, Any]) -> str:
        story = sample.get("story", sample.get("narrative", ""))
        return f"""/no_think
         Extract the SHARED EPISTEMIC CORE needed for this belief question.

         Narrative:
         {story}

         Question:
         {sample["question"]}

         Do not answer the question. Follow these rules:
         1. Read only the Question to identify the relevant character or ordered
            belief chain. Include only names that occur in the Question.
         2. Ignore unrelated events.
         3. The shared core is an INTERSECTION, not a union. Output an event only
            when every relevant agent directly knows the event and can attribute
            that knowledge to the other relevant agents.
         4. Track perception exactly. Events unseen or unheard by a relevant agent
            are not in that agent's epistemic core.
         5. Preserve the original wording and chronological order. Do not add beliefs,
            explanations, inferred events, or an answer.
         6. Output only the requested structure. Do not explain your decisions.

         Output:
         CHAIN: ...
         TARGET: ...
         SHARED TARGET EVENTS:
         ..."""

    def build_qa_prompt(self, evidence: str, sample: Dict[str, Any]) -> str:
        true_answer = sample.get("true_answer", sample.get("answer", ""))
        wrong_answer = sample.get("wrong_answer", "")
        return f"""Shared epistemic core:
         {evidence}

         Question:
         {sample["question"]}

         Possible Answers:
         A: {true_answer}
         B: {wrong_answer}

         Answer the belief question using only the shared epistemic core.
         Do not import any event absent from the shared core.

         End with exactly:
         Answer: A
         or
         Answer: B"""
