import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings.embedding_manager import get_embedding_model
from embeddings.vector_store import load_vector_store
from agents.planner import PlannerAgent
from agents.auditor import AuditorAgent
from retrieval.generation import generate_answer

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Privacy Policy Auditor",
    page_icon="🔍",
    layout="wide",
)

# ── Pipeline (cached — loads once per session) ────────────────────────────────

@st.cache_resource(show_spinner="Loading AI pipeline...")
def load_pipeline():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")
    embeddings = get_embedding_model()
    vectorstore = load_vector_store(embeddings, VECTOR_STORE_DIR)
    return PlannerAgent(vectorstore), AuditorAgent()


planner, auditor = load_pipeline()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Privacy Auditor")
    st.markdown("**Agentic RAG** over 115 real website privacy policies.")
    st.divider()

    st.markdown("### How it works")
    st.markdown(
        "1. **Planner** classifies your query\n"
        "2. **Retriever** fetches relevant segments\n"
        "3. **LLM** generates a grounded answer\n"
        "4. **Auditor** validates for hallucinations"
    )
    st.divider()

    st.markdown("### Query Types")
    st.markdown(
        "- 🟢 **SIMPLE** — broad question across all policies\n"
        "- 🔵 **FILTERED** — mention a specific site\n"
        "- 🟣 **COMPARE** — compare two named sites"
    )
    st.divider()

    st.markdown("### Try these")
    samples = [
        "What types of personal data do websites collect?",
        "What does amazon.com say about third-party data sharing?",
        "How does google.com handle cookies?",
        "Compare how nytimes.com and theatlantic.com treat user data.",
        "Which policies mention data retention periods?",
    ]
    for sample in samples:
        if st.button(sample, use_container_width=True, key=sample):
            st.session_state["question_input"] = sample

    st.divider()
    st.markdown(
        "Built by [Rohith Sundar](https://www.linkedin.com/in/rohithsundarj/) · "
        "Dataset: [OPP-115 Corpus](https://usableprivacy.org/data)"
    )

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🔍 Agentic Privacy & Compliance Auditor")
st.caption("Ask questions about 115 real website privacy policies — answers validated for hallucinations.")

question = st.text_input(
    label="Your question",
    placeholder="e.g. What does amazon.com say about sharing data with third parties?",
    key="question_input",
)

if st.button("🔎 Analyze", type="primary") and question.strip():

    with st.spinner("Planner routing your query..."):
        docs, plan = planner.retrieve(question)

    # Query type badge
    qtype = plan.get("type", "SIMPLE")
    badge = {"SIMPLE": "🟢", "FILTERED": "🔵", "COMPARE": "🟣"}.get(qtype, "⚪")
    st.markdown(f"**Query type:** {badge} `{qtype}` — {plan.get('reasoning', '')}")

    if not docs:
        st.warning("No relevant segments found. Try rephrasing your question.")
        st.stop()

    with st.spinner("Generating answer..."):
        answer = generate_answer(question, docs)

    with st.spinner("Auditor validating..."):
        final_answer, audit = auditor.audit_and_regenerate(
            question=question,
            answer=answer,
            docs=docs,
            generate_fn=generate_answer,
        )

    # ── Answer ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Answer")
    st.markdown(final_answer)

    # ── Audit result ──────────────────────────────────────────────────────────
    score = audit.get("faithfulness_score", 0)
    verdict = audit.get("verdict", "PASS")
    unsupported = audit.get("unsupported_claims", [])

    col1, col2 = st.columns([1, 3])
    with col1:
        (st.success if verdict == "PASS" else st.error)(
            f"{'✅' if verdict == 'PASS' else '❌'} Audit: {verdict}"
        )
        st.metric("Faithfulness Score", f"{score:.2f} / 1.00")
    with col2:
        st.markdown(f"**Auditor reasoning:** {audit.get('reasoning', '')}")
        if unsupported:
            st.markdown("**Unsupported claims flagged:**")
            for claim in unsupported:
                st.markdown(f"- ⚠ {claim}")

    # ── Sources ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"📂 Source Documents ({len(docs)} retrieved)")

    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        with st.expander(
            f"[{i}] {m.get('policy_id', 'unknown')} — {m.get('url', '')}",
            expanded=(i == 1),
        ):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Policy ID:** `{m.get('policy_id', 'N/A')}`")
            c2.markdown(f"**Collection Date:** `{m.get('collection_date', 'N/A')}`")
            c3.markdown(f"**Annotations:** `{m.get('annotation_count', '0')}`")
            st.markdown("**Segment text:**")
            st.markdown(
                f"<div style='background:#f8f9fa;padding:12px;border-radius:6px;"
                f"font-size:0.9em;border-left:3px solid #dee2e6;color:#212529'>"
                f"{doc.page_content[:800]}</div>",
                unsafe_allow_html=True,
            )

elif st.session_state.get("question_input") == "" :
    pass  # Don't show warning on initial page load
