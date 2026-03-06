import os
import re
from langchain_core.documents import Document

# N3: keywords that signal the user wants a specific output format
_FORMAT_KEYWORDS = [
    "bulleted list", "bullet points", "numbered list", "as a list",
    "in a table", "markdown", "format as", "format your response",
    "as markdown",
]

def _extract_format_instruction(question: str) -> str:
    """
    N3 fix: detect formatting requests and return an explicit instruction
    to append to the prompt so the LLM always honours the user's format.
    Returns empty string when no format keyword is present.
    """
    q = question.lower()
    for kw in _FORMAT_KEYWORDS:
        if kw in q:
            # Find the original-case fragment from the keyword onward
            idx = q.find(kw)
            fragment = question[max(0, idx - 5):].strip()
            return (
                f"\n\nFORMAT REQUIREMENT (mandatory): The user explicitly requested a "
                f"specific output format. You MUST honour this format using only text "
                f"from the excerpts: \"{fragment}\""
            )
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


def _generate_with_hf(prompt: str) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        token=os.environ["HF_TOKEN"],
    )
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.01,
        stop=["Question:", "---"],
    )
    return response.choices[0].message.content.strip()


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