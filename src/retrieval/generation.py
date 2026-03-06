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
If the answer is not present, respond exactly with:
"I cannot find this information in the provided privacy policies."
Never add information beyond what is written in the excerpts.

IMPORTANT INSTRUCTIONS:
- If the question contains a false or unsupported premise (e.g. assumes a specific
  timeframe, action, or feature the excerpts do not mention or contradict), you MUST
  correct the premise explicitly — e.g. "The policy does not state X; instead it
  states Y" — then provide what the policy actually says. Do NOT silently return
  "I cannot find this information" when excerpts contain relevant contradicting text.
  WATCH FOR these common false premises: "immediately", "within 24 hours", "never sells",
  "always encrypts", "guarantees deletion" — if the policy says something different,
  state the correction before answering.
- If the user requests a specific output format (e.g. bulleted list, table, numbered steps), \
honour that format strictly using only text from the excerpts.
- If the user asks to "quote" or "extract" specific text, reproduce the exact words from \
the excerpt verbatim inside quotation marks. Do not paraphrase quoted content.
- When comparing legally similar terms (e.g. "sell" vs. "share", "collect" vs. "use"), \
treat each term as a distinct legal concept and address them separately.

--- PRIVACY POLICY EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Question: {question}

Answer:"""

_STRICT_PROMPT = """You are an extremely strict Privacy Policy Compliance Auditor.
Use ONLY the exact information written in the excerpts below. Do not paraphrase
beyond what is written. Do not draw on any background knowledge.
If uncertain, respond: "I cannot find this information in the provided privacy policies."

IMPORTANT INSTRUCTIONS:
- If the question contains a false or unsupported premise (e.g. assumes a specific
  timeframe, action, or feature the excerpts do not mention or contradict), you MUST
  correct the premise explicitly — e.g. "The policy does not state X; instead it
  states Y" — then provide what the policy actually says. Do NOT silently return
  "I cannot find this information" when excerpts contain relevant contradicting text.
  WATCH FOR these common false premises: "immediately", "within 24 hours", "never sells",
  "always encrypts", "guarantees deletion" — if the policy says something different,
  state the correction before answering.
- If the user requests a specific output format (e.g. bulleted list, table, numbered steps), \
honour that format strictly using only text from the excerpts.
- If the user asks to "quote" or "extract" specific text, reproduce the exact words from \
the excerpt verbatim inside quotation marks. Do not paraphrase quoted content.
- When comparing legally similar terms (e.g. "sell" vs. "share", "collect" vs. "use"), \
treat each term as a distinct legal concept and address them separately.

--- PRIVACY POLICY EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Question: {question}

Strictly grounded answer:"""


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


def _generate_with_hf(prompt: str) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        token=os.environ["HF_TOKEN"],
    )
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.01,
            stop=["Question:", "---"],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        if _is_quota_error(exc):
            # Attempt Ollama fallback before giving up
            try:
                print("[Generation] HF 402/429 — attempting Ollama fallback.")
                return _generate_with_ollama(prompt)
            except Exception:
                raise RuntimeError(
                    "⚠️ The HuggingFace inference service is temporarily unavailable "
                    "(quota or rate limit reached). Please top up your HF credits, "
                    "wait a few minutes, or run the app locally with Ollama."
                ) from exc
        if _is_server_error(exc):
            raise RuntimeError(
                "⚠️ The inference service returned a server error. "
                "Please try again in a moment."
            ) from exc
        raise


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
    if os.environ.get("HF_TOKEN"):
        return _generate_with_hf(prompt)
    else:
        return _generate_with_ollama(prompt, llm_model)