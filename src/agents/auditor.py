import re
import json

from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

_AUDITOR_PROMPT = """You are a factual auditor. Verify whether every specific claim in
the AI-generated answer is directly supported by the source excerpts below.

SOURCE EXCERPTS:
{context}

AI-GENERATED ANSWER:
{answer}

Ask: "Can every claim be traced to a sentence in the sources?"

Respond ONLY with valid JSON, no markdown:
{{
  "faithfulness_score": <float 0.0-1.0>,
  "verdict": "PASS" | "FAIL",
  "unsupported_claims": ["claim1", "claim2"],
  "reasoning": "one sentence summary"
}}

Scoring:
- 1.0  = every claim directly supported
- 0.7+ = mostly supported, minor inferences acceptable -> PASS
- <0.7 = unsupported claims present -> FAIL

Special case: if the answer states it cannot find information, always score 1.0 PASS.
"""

_SAFE_ANSWER = (
    "I cannot verify this answer against the provided privacy policies. "
    "Please consult the original documents directly."
)

def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)

def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {doc.page_content[:500]}" for i, doc in enumerate(docs, 1))

class AuditorAgent:
    """Scores answer faithfulness and optionally triggers regeneration."""

    def __init__(self, llm_model: str = "qwen2.5:7b"):
        self.llm = OllamaLLM(model=llm_model, temperature=0)

    def audit(self, answer: str, docs: list[Document]) -> dict:
        """
        Score the answer against the retrieved documents.

        Args:
            answer: LLM-generated answer string.
            docs:   Documents used to generate the answer.

        Returns:
            Dict with faithfulness_score, verdict, unsupported_claims, reasoning.
        """
        if not docs:
            return {
                "faithfulness_score": 0.0,
                "verdict": "FAIL",
                "unsupported_claims": ["No source documents were retrieved."],
                "reasoning": "Answer has no grounding — no documents were retrieved.",
            }

        raw = self.llm.invoke(
            _AUDITOR_PROMPT.format(context=_format_context(docs), answer=answer)
        )

        try:
            result = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            print(f"[Auditor] JSON parse failed — defaulting to cautious PASS.")
            result = {
                "faithfulness_score": 0.5,
                "verdict": "PASS",
                "unsupported_claims": [],
                "reasoning": "Audit parse failed — defaulting to cautious PASS.",
            }

        result.setdefault("faithfulness_score", 0.5)
        result.setdefault("verdict", "PASS")
        result.setdefault("unsupported_claims", [])
        result.setdefault("reasoning", "")
        return result

    def audit_and_regenerate(
        self,
        question: str,
        answer: str,
        docs: list[Document],
        generate_fn,
        max_retries: int = 1,
    ) -> tuple[str, dict]:
        """
        Audit the answer. If FAIL, regenerate once with strict mode.

        Args:
            question:    Original user question.
            answer:      Initial LLM answer.
            docs:        Retrieved source documents.
            generate_fn: Callable matching generate_answer(question, docs, strict).
            max_retries: How many regeneration attempts before safe fallback.

        Returns:
            Tuple of (final_answer, audit_result).
        """
        audit = self.audit(answer, docs)

        if audit["verdict"] == "PASS":
            return answer, audit

        print(f"[Auditor] FAIL (score={audit['faithfulness_score']:.2f}). Regenerating...")

        for attempt in range(max_retries):
            new_answer = generate_fn(question, docs, strict=True)
            new_audit = self.audit(new_answer, docs)
            if new_audit["verdict"] == "PASS":
                print(f"[Auditor] Passed on retry {attempt + 1}.")
                return new_answer, new_audit

        print("[Auditor] Still failing after retries — returning safe fallback.")
        return _SAFE_ANSWER, audit
