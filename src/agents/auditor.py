import re
import json
from langchain_core.documents import Document

def _get_llm(model: str = "qwen2.5:7b"):
    import os
    if os.environ.get("HF_TOKEN"):
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="conversational",
            huggingfacehub_api_token=os.environ["HF_TOKEN"],
            temperature=0.01,
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=endpoint)
    else:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=model, temperature=0)

def _llm_invoke(llm, prompt: str) -> str:
    raw = llm.invoke(prompt)
    return raw.content if hasattr(raw, "content") else raw

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
  "verdict": "PASS" | "WARN" | "FAIL",
  "unsupported_claims": ["claim1", "claim2"],
  "reasoning": "one sentence summary"
}}

Scoring:
- 1.0        = every claim directly supported
- 0.85-1.0   = well-supported, minor inferences acceptable          -> PASS
- 0.70-0.84  = mostly supported, some unverified claims present     -> WARN
- < 0.70     = significant unsupported claims                       -> FAIL

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

def _apply_thresholds(result: dict) -> dict:
    score = result.get("faithfulness_score", 0.5)
    if score >= 0.85:
        result["verdict"] = "PASS"
    elif score >= 0.70:
        result["verdict"] = "WARN"
    else:
        result["verdict"] = "FAIL"
    return result

class AuditorAgent:
    def __init__(self, llm_model: str = "qwen2.5:7b"):
        self.llm = _get_llm(llm_model)

    def audit(self, answer: str, docs: list[Document]) -> dict:
        if not docs:
            return {
                "faithfulness_score": 0.0,
                "verdict": "FAIL",
                "unsupported_claims": ["No source documents were retrieved."],
                "reasoning": "Answer has no grounding.",
            }
        raw = _llm_invoke(self.llm, _AUDITOR_PROMPT.format(context=_format_context(docs), answer=answer))
        try:
            result = _extract_json(raw)
        except (json.JSONDecodeError, AttributeError):
            print("[Auditor] JSON parse failed — defaulting to cautious WARN.")
            result = {
                "faithfulness_score": 0.5,
                "verdict": "WARN",
                "unsupported_claims": [],
                "reasoning": "Audit parse failed — could not fully verify answer.",
            }
        result.setdefault("faithfulness_score", 0.5)
        result.setdefault("unsupported_claims", [])
        result.setdefault("reasoning", "")
        return _apply_thresholds(result)

    def audit_and_regenerate(self, question, answer, docs, generate_fn, max_retries=1):
        audit = self.audit(answer, docs)
        if audit["verdict"] in ("PASS", "WARN"):
            return answer, audit
        print(f"[Auditor] FAIL (score={audit['faithfulness_score']:.2f}). Regenerating...")
        for attempt in range(max_retries):
            new_answer = generate_fn(question, docs, strict=True)
            new_audit = self.audit(new_answer, docs)
            if new_audit["verdict"] in ("PASS", "WARN"):
                print(f"[Auditor] Cleared FAIL on retry {attempt + 1} ({new_audit['verdict']}).")
                return new_answer, new_audit
        print("[Auditor] Still FAIL after retries — returning safe fallback.")
        return _SAFE_ANSWER, audit