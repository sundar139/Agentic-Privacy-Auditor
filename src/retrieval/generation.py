import os
from langchain_core.documents import Document

_BASE_PROMPT = """You are a strict Privacy Policy Compliance Auditor.
Answer the question using ONLY the privacy policy excerpts provided below.
If the answer is not present, respond exactly with:
"I cannot find this information in the provided privacy policies."
Never add information beyond what is written in the excerpts.

--- PRIVACY POLICY EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Question: {question}

Answer:"""

_STRICT_PROMPT = """You are an extremely strict Privacy Policy Compliance Auditor.
Use ONLY the exact information written in the excerpts below. Do not paraphrase
beyond what is written. Do not draw on any background knowledge.
If uncertain, respond: "I cannot find this information in the provided privacy policies."

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
    response = client.text_generation(
        prompt,
        max_new_tokens=1024,
        temperature=0.01,  # Near-zero but HF API requires > 0
        do_sample=False,
        stop_sequences=["Question:", "---"],
    )
    return response.strip()

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

    if os.environ.get("HF_TOKEN"):
        return _generate_with_hf(prompt)
    else:
        return _generate_with_ollama(prompt, llm_model)