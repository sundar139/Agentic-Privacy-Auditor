import os
import re
from langchain_core.documents import Document

# N3: each keyword maps to the exact formatting instruction to append to the prompt.
# More specific = more reliably followed by the 7B model.
_FORMAT_KEYWORDS: dict[str, str] = {
    "bulleted list":       "\n\nFORMAT (mandatory): Output your answer as a markdown bulleted list. Each point on its own line starting with '- '. Do not use prose paragraphs.",
    "bullet points":       "\n\nFORMAT (mandatory): Output your answer as a markdown bulleted list. Each point on its own line starting with '- '. Do not use prose paragraphs.",
    "numbered list":       "\n\nFORMAT (mandatory): Output your answer as a numbered markdown list. Each point on its own line starting with '1.', '2.', etc. Do not use prose paragraphs.",
    "as a list":           "\n\nFORMAT (mandatory): Output your answer as a markdown bulleted list. Each point on its own line starting with '- '.",
    "in a table":          "\n\nFORMAT (mandatory): Output your answer as a markdown table with a header row. Use | column | column | syntax.",
    "as a table":          "\n\nFORMAT (mandatory): Output your answer as a markdown table with a header row. Use | column | column | syntax.",
    "markdown table":      "\n\nFORMAT (mandatory): Output your answer as a markdown table with a header row. Use | column | column | syntax.",
    "format as markdown":  "\n\nFORMAT (mandatory): Output your answer using markdown formatting — use headers (##), bullets (-), or bold (**text**) as appropriate.",
    "format your response":"\n\nFORMAT (mandatory): Strictly follow the output format the user specified in their question.",
    "as markdown":         "\n\nFORMAT (mandatory): Output your answer using markdown formatting.",
}


def _extract_format_instruction(question: str) -> str:
    """
    N3 fix: return the specific formatting instruction for the first format
    keyword found in the question. Returns empty string when none is present.
    """
    q = question.lower()
    for kw, instruction in _FORMAT_KEYWORDS.items():
        if kw in q:
            return instruction
    return ""

_BASE_PROMPT = """You are a strict Privacy Policy Compliance Auditor.
Answer the question using ONLY the privacy policy excerpts provided below.
If the answer is not present in the excerpts, state it clearly using one of these forms:
  - "The provided policy does not contain any information about <topic>." — write the actual topic name; never use a bracket placeholder like [topic].
  - "I cannot find this information in the provided privacy policies." — use when retrieval may have failed.
Never add information beyond what is written in the excerpts.

IMPORTANT INSTRUCTIONS:
- If the question contains a false or unsupported premise (e.g. assumes a specific
  timeframe, action, or feature the excerpts do not mention or contradict), you MUST:
  (1) deny the false premise — "The policy does not state X";
  (2) then IMMEDIATELY state what the policy actually says about the underlying real
      topic — e.g. "The actual [topic] stated in the policy is [Y]." NEVER stop at
      step 1 alone; always complete the answer with the real policy content.
  Do NOT silently return "I cannot find this information" when relevant text is present.
  WATCH FOR these common false premises: "immediately", "within 24 hours", "never sells",
  "always encrypts", "guarantees deletion" — if the policy says something different,
  state the correction then provide the real answer.
- If the user requests a specific output format (e.g. bulleted list, table, numbered steps), \
honour that format strictly using only text from the excerpts.
- If the user asks to "quote" or "extract" specific text, reproduce the exact words from \
the excerpt verbatim inside quotation marks. Do not paraphrase quoted content.
- When comparing legally similar terms (e.g. "sell" vs. "share", "collect" vs. "use"), \
treat each term as a distinct legal concept and address them separately.
- When listing named entities (companies, services, partners), include ONLY proper company
  or service names (e.g. Google, DoubleClick, Amazon). Do NOT include mailing addresses,
  P.O. boxes, department names, or generic descriptions.
- CRITICAL — Jurisdiction / Governing Law: If the question asks about legal jurisdiction,
  governing law, dispute resolution, or applicable law, you MUST scan every excerpt for
  phrases like "law of", "governed by", "subject to the laws", "applicable law",
  "courts of", "jurisdiction of", or any U.S. state or country name appearing near
  "law", "dispute", or "jurisdiction". If ANY such phrase appears, quote it verbatim and
  explicitly state the jurisdiction. NEVER say the information is absent when such a
  phrase is present anywhere in the excerpts.

--- PRIVACY POLICY EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Question: {question}

Answer:"""

_STRICT_PROMPT = """You are an extremely strict Privacy Policy Compliance Auditor.
Use ONLY the exact information written in the excerpts below. Do not paraphrase
beyond what is written. Do not draw on any background knowledge.
If the topic is absent from the excerpts, state: "The provided policy does not contain any information about <topic>." — use the actual topic name, never a bracket placeholder like [topic].
If uncertain whether retrieval failed, respond: "I cannot find this information in the provided privacy policies."

IMPORTANT INSTRUCTIONS:
- If the question contains a false or unsupported premise (e.g. assumes a specific
  timeframe, action, or feature the excerpts do not mention or contradict), you MUST:
  (1) deny the false premise — "The policy does not state X";
  (2) then IMMEDIATELY state what the policy actually says about the underlying real
      topic — e.g. "The actual [topic] stated in the policy is [Y]." NEVER stop at
      step 1 alone; always complete the answer with the real policy content.
  Do NOT silently return "I cannot find this information" when relevant text is present.
  WATCH FOR these common false premises: "immediately", "within 24 hours", "never sells",
  "always encrypts", "guarantees deletion" — if the policy says something different,
  state the correction then provide the real answer.
- If the user requests a specific output format (e.g. bulleted list, table, numbered steps), \
honour that format strictly using only text from the excerpts.
- If the user asks to "quote" or "extract" specific text, reproduce the exact words from \
the excerpt verbatim inside quotation marks. Do not paraphrase quoted content.
- When comparing legally similar terms (e.g. "sell" vs. "share", "collect" vs. "use"), \
treat each term as a distinct legal concept and address them separately.
- When listing named entities (companies, services, partners), include ONLY proper company
  or service names (e.g. Google, DoubleClick, Amazon). Do NOT include mailing addresses,
  P.O. boxes, department names, or generic descriptions.
- CRITICAL — Jurisdiction / Governing Law: If the question asks about legal jurisdiction,
  governing law, dispute resolution, or applicable law, you MUST scan every excerpt for
  phrases like "law of", "governed by", "subject to the laws", "applicable law",
  "courts of", "jurisdiction of", or any U.S. state or country name appearing near
  "law", "dispute", or "jurisdiction". If ANY such phrase appears, quote it verbatim and
  explicitly state the jurisdiction. NEVER say the information is absent when such a
  phrase is present anywhere in the excerpts.

--- PRIVACY POLICY EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Question: {question}

Strictly grounded answer:"""


# Regex catches any residual bracket placeholders the LLM emits despite prompt instructions.
_PLACEHOLDER_RE = re.compile(
    r"[Tt]he (?:provided )?policy does not mention \[([^\]]+)\]",
    re.IGNORECASE,
)


def _clean_placeholder_answer(text: str) -> str:
    """Replace LLM bracket-template placeholders (e.g. [topic]) with natural language."""
    def _subst(m: re.Match) -> str:
        topic = m.group(1).strip().rstrip(".")
        return f"The provided policy does not contain any information about {topic}"
    return _PLACEHOLDER_RE.sub(_subst, text)


# Jurisdiction / governing-law phrase scanner — detects governing law clauses in docs.
# Used to pre-extract and inject as a hard hint so the LLM cannot overlook them.
_JURISDICTION_PATTERNS = re.compile(
    r"(?:"
    r"law\s+of(?:\s+the)?\s+(?:state\s+of\s+)?\w+"
    r"|governed\s+by(?:\s+the)?\s+laws?\s+of\s+\w+"
    r"|subject\s+to(?:\s+the)?\s+laws?\s+of\s+\w+"
    r"|applicable\s+law\s+of\s+\w+"
    r"|courts?\s+of\s+\w+"
    r"|jurisdiction\s+of\s+\w+"
    r"|disputes?\s+(?:will\s+be\s+)?(?:resolved|governed|handled)\s+(?:in|under|by|pursuant\s+to)"
    r")",
    re.IGNORECASE,
)

_JURISDICTION_Q_KEYWORDS = frozenset([
    "jurisdiction", "govern", "law", "dispute", "legal", "court", "applicable",
    "washington", "california", "delaware", "new york", "arbitration", "venue",
])


def _extract_jurisdiction_hint(docs: list[Document], question: str) -> str | None:
    """
    Pre-scan retrieved docs for governing-law / jurisdiction phrases and return
    the matching sentence(s) as a hint string, or None if no matches.
    Only activates when the question appears to be jurisdiction-related.
    """
    q_lower = question.lower()
    if not any(kw in q_lower for kw in _JURISDICTION_Q_KEYWORDS):
        return None
    hits: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        text = doc.page_content
        for m in _JURISDICTION_PATTERNS.finditer(text):
            start = text.rfind(".", 0, m.start())
            end = text.find(".", m.end())
            sentence = text[
                start + 1 if start >= 0 else 0 : end + 1 if end >= 0 else len(text)
            ].strip()
            if sentence and sentence not in seen:
                hits.append(sentence)
                seen.add(sentence)
    return "\n".join(hits) if hits else None


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        header = (
            f"[{i}] Policy: {m.get('policy_id', 'unknown')} | "
            f"URL: {m.get('url', 'unknown')} | "
            f"Date: {m.get('collection_date', 'unknown')}"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def _generate_with_ollama(prompt: str, model: str = "qwen2.5:7b") -> str:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=model, temperature=0)
    return llm.invoke(prompt)


def _is_quota_error(exc: Exception) -> bool:
    """Returns True for HTTP 402 Payment Required / 429 Too Many Requests errors."""
    msg = str(exc).lower()
    return any(code in msg for code in ("402", "429", "payment required", "too many requests", "rate limit"))


def _is_server_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(code in msg for code in ("500", "502", "503", "504", "service unavailable", "internal server error"))


def _generate_with_hf(prompt: str, _retries: int = 3, _base_delay: float = 8.0) -> str:
    """
    BUG #1 fix: exponential backoff for HF quota/rate-limit errors.
    Retries up to _retries times with doubling delays before falling back
    to Ollama. No external retry library needed.
    """
    import time
    from huggingface_hub import InferenceClient
    client = InferenceClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        token=os.environ["HF_TOKEN"],
    )
    last_exc = None
    for attempt in range(_retries):
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.01,
                stop=["Question:", "---"],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc):
                delay = _base_delay * (2 ** attempt)   # 8s, 16s, 32s
                print(f"[Generation] HF 429/402 on attempt {attempt+1}/{_retries} — "
                      f"waiting {delay:.0f}s before retry.")
                time.sleep(delay)
                continue  # retry
            if _is_server_error(exc):
                delay = _base_delay * (2 ** attempt)
                print(f"[Generation] HF 5xx on attempt {attempt+1}/{_retries} — "
                      f"waiting {delay:.0f}s before retry.")
                time.sleep(delay)
                continue
            raise  # non-retryable error — propagate immediately

    # All retries exhausted — try local Ollama as final fallback
    try:
        print("[Generation] HF retries exhausted — attempting Ollama fallback.")
        return _generate_with_ollama(prompt)
    except Exception:
        raise RuntimeError(
            "⚠️ The HuggingFace inference service is temporarily unavailable "
            "(quota or rate limit reached). Please top up your HF credits, "
            "wait a few minutes, or run the app locally with Ollama."
        ) from last_exc


def generate_answer(
    question: str,
    docs: list[Document],
    llm_model: str = "qwen2.5:7b",
    strict: bool = False,
) -> str:
    context = _format_context(docs)
    prompt = (_STRICT_PROMPT if strict else _BASE_PROMPT).format(
        context=context, question=question
    )
    # N3: append format instruction if the user specified one
    prompt += _extract_format_instruction(question)
    # Jurisdiction hint: programmatically pre-extract governing-law phrases so
    # the LLM cannot overlook them even when its grounding behaviour is overly cautious.
    jurisdiction_hint = _extract_jurisdiction_hint(docs, question)
    if jurisdiction_hint:
        prompt += (
            "\n\nJURISDICTION PRE-EXTRACTION: The following sentence(s) were found in "
            "the excerpts above by keyword scan. You MUST reference this in your answer:\n"
            f"{jurisdiction_hint}"
        )
    if os.environ.get("HF_TOKEN"):
        return _clean_placeholder_answer(_generate_with_hf(prompt))
    else:
        return _clean_placeholder_answer(_generate_with_ollama(prompt, llm_model))