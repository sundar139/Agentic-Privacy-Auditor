import os
import sys
import json
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Privacy Policy Auditor",
    page_icon="🔍",
    layout="wide",
)

# ── HF_TOKEN startup check (only enforced on HF Spaces) ──────────────────────
# SPACE_ID is set automatically by HuggingFace Spaces; not present locally.
if os.environ.get("SPACE_ID") and not os.environ.get("HF_TOKEN"):
    st.error(
        "❌ **HF_TOKEN is not configured.**\n\n"
        "Go to **Space Settings → Secrets** and add `HF_TOKEN` with your "
        "HuggingFace access token. The app cannot start without it."
    )
    st.stop()


# ── Vector store path resolution ──────────────────────────────────────────────

def _resolve_vector_store_path() -> str:
    candidates = [
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "vector_store",
        ),
        "/app/data/vector_store",
        os.path.join(os.getcwd(), "data", "vector_store"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


# ── Lazy pipeline loader ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from embeddings.embedding_manager import get_embedding_model
    from embeddings.vector_store import load_vector_store
    from agents.planner import PlannerAgent
    from agents.auditor import AuditorAgent

    vector_store_dir = _resolve_vector_store_path()
    embeddings = get_embedding_model()
    vectorstore = load_vector_store(embeddings, vector_store_dir)
    planner = PlannerAgent(vectorstore)
    auditor = AuditorAgent()
    return planner, auditor, vectorstore   # ← include vectorstore for corpus browser


def get_pipeline():
    try:
        return _load_pipeline(), None
    except Exception as e:
        return None, str(e)


# ── Corpus site list (cached once after pipeline loads) ───────────────────────

@st.cache_data(show_spinner=False)
def _get_corpus_sites(_vectorstore) -> list[str]:
    """Return sorted list of unique URLs from the vector store metadata."""
    try:
        all_meta = _vectorstore.get()["metadatas"]
        sites = sorted({
            m.get("url", "").strip()
            for m in all_meta
            if m.get("url", "").strip() not in ("", "unknown")
        })
        return sites
    except Exception:
        return []


# ── Q9: Pre-LLM restricted intent classifier ─────────────────────────────────
# Blocks tool-execution requests (code, scraping, etc.) BEFORE the LLM is called.
# The Auditor only checks factual faithfulness — it cannot catch these.

_RESTRICTED_INTENT_PATTERNS = [
    # Code generation
    r"\bwrite\s+(a\s+)?(python|javascript|js|sql|bash|script|code|program|function|class)\b",
    r"\b(generate|create|give me)\s+(a\s+)?(script|code|scraper|crawler|bot|program)\b",
    r"\bscrape\b", r"\bcrawler?\b", r"\bbeautifulsoup\b", r"\brequests\.get\b",
    r"\bimport\s+(requests|bs4|scrapy|selenium|playwright)\b",
    # Prompt injection / persona override
    # Pattern covers "ignore [any words] instructions" to handle multi-word variants
    # like "ignore all previous instructions", "ignore all prior instructions", etc.
    r"\bignore\s+(?:\w+\s+){0,3}instructions?\b",
    r"\byou\s+are\s+now\s+a\b",
    r"\byou\s+are\s+no\s+longer\b",
    r"\bact\s+as\s+(a\s+)?(different|new|another)\b",
    r"\bforget\s+(everything|all|your)\b",
    r"\bpretend\s+(?:you\s+are|to\s+be)\b",
    r"\bstop\s+being\s+a\b",
    r"\bdan\s+mode\b", r"\bjailbreak\b",
    # Clearly out-of-scope tool requests
    r"\bsend\s+(an?\s+)?email\b", r"\bpost\s+(to|on)\s+(twitter|reddit|slack)\b",
]

# Privacy-related keywords that indicate a valid policy question exists in the query
_PRIVACY_KEYWORDS = [
    "privacy", "policy", "data", "collect", "share", "sell", "retain", "deletion",
    "tracking", "cookie", "consent", "opt", "third party", "personal information",
    "security", "encrypt", "gdpr", "ccpa", "do not track", "advertising",
]

import re as _re

_HARDCODED_REFUSAL = (
    "🚫 **I am a legal compliance assistant and can only answer questions about privacy policies.**\n\n"
    "This tool does **not**:\n"
    "- Write code, scripts, or web scrapers\n"
    "- Execute instructions unrelated to privacy policy analysis\n"
    "- Respond to persona-override or prompt injection attempts\n\n"
    "Please ask a question about a specific website's privacy policy."
)


def _check_restricted_intent(question: str) -> tuple[bool, bool]:
    """
    Returns (has_restricted, has_valid_privacy) tuple.
    - has_restricted: True if question contains a restricted intent pattern
    - has_valid_privacy: True if question also contains a valid privacy question
    This enables Q9 multi-intent split: answer the valid part, refuse only the restricted part.
    """
    q = question.lower()
    has_restricted = any(_re.search(p, q) for p in _RESTRICTED_INTENT_PATTERNS)
    has_valid_privacy = any(kw in q for kw in _PRIVACY_KEYWORDS)
    return has_restricted, has_valid_privacy


def _strip_restricted_part(question: str) -> str:
    """
    Q9 fix: for multi-intent queries, extract just the privacy-policy question
    by splitting on common conjunctions and keeping only the valid sentence(s).
    """
    # Split on ". Also", "and also", ". Additionally", "and then", ". Then"
    import re
    parts = re.split(
        r"\.\s+(?:also|additionally|then|finally|next|after that)\b"
        r"|\band\s+(?:also\s+)?(?:write|create|generate|give me|scrape|make)\b"
        r"|\.\s+(?:write|create|generate|give me|scrape|make)\b",
        question,
        flags=re.IGNORECASE,
    )
    # Keep only parts that contain a privacy keyword
    valid_parts = [p.strip() for p in parts if any(kw in p.lower() for kw in _PRIVACY_KEYWORDS)]
    return " ".join(valid_parts) if valid_parts else question


# ── Plan diagnostic warnings ──────────────────────────────────────────────────

def _show_corpus_warnings(plan: dict) -> None:
    if plan.get("corpus_miss"):
        site = plan.get("corpus_miss_site", "that website")
        st.warning(
            f"⚠️ **'{site}' is not in the OPP-115 corpus.**\n\n"
            f"The dataset covers 115 specific websites collected in 2015–2016. "
            f"The answer below is drawn from semantically similar policies and may "
            f"**not** reflect **{site}**'s actual privacy policy.",
        )

    compare_misses = plan.get("compare_misses", [])
    if compare_misses:
        missing_str = ", ".join(f"**{s}**" for s in compare_misses)
        st.warning(
            f"⚠️ The following site(s) are **not in the OPP-115 corpus**: {missing_str}.\n\n"
            f"The comparison below only covers the site(s) that were found.",
        )

    if plan.get("year_filter"):
        st.info(f"ℹ️ Results pre-filtered to policies collected in **{plan['year_filter']}**.")

    if plan.get("quote_query") and plan.get("section_filter"):
        st.info(f"ℹ️ Section-targeted retrieval applied: `{plan['section_filter']}`")


# ── Corpus information (OPP-115 dataset scope) ────────────────────────────────

_OPP115_INFO = """
The **OPP-115 Corpus** contains privacy policies from **115 real websites**,
collected in **2015–2016** and fully annotated by legal experts across 8 categories:

| Category | What it covers |
|---|---|
| First Party Collection/Use | What data the site collects and how it uses it |
| Third Party Sharing | Data shared with or collected by third parties |
| User Choice/Control | Opt-out, consent, and preference options |
| User Access, Edit & Deletion | Rights to access, modify, or delete data |
| Data Retention | How long data is stored |
| Data Security | How data is protected |
| Policy Change | How users are notified of changes |
| Do Not Track | Response to browser DNT signals |

⚠️ **Scope note:** Policies reflect 2015–2016 language. Sites not in the list below
cannot be accurately answered — the Auditor will warn you if a site is missing.
"""

# The 115 sites in the OPP-115 corpus (used in the sidebar "Companies in Corpus" expander)
_OPP115_SITES = sorted([
    "aol.com", "apple.com", "att.com", "cbsnews.com", "chase.com",
    "cnet.com", "comcast.com", "craigslist.org", "ebay.com", "espn.com",
    "facebook.com", "foxnews.com", "go.com", "huffingtonpost.com", "imdb.com",
    "instagram.com", "linkedin.com", "live.com", "mapquest.com", "match.com",
    "mediafire.com", "microsoft.com", "mlb.com", "msn.com", "myspace.com",
    "nba.com", "netflix.com", "nfl.com", "nytimes.com", "paypal.com",
    "pinterest.com", "reddit.com", "salesforce.com", "scribd.com", "shutterstock.com",
    "snapchat.com", "spotify.com", "target.com", "theatlantic.com", "ticketmaster.com",
    "time.com", "tmz.com", "tripadvisor.com", "tumblr.com", "twitter.com",
    "usnews.com", "verizon.com", "vevo.com", "vimeo.com", "vine.co",
    "washingtonpost.com", "weather.com", "webmd.com", "whitepages.com", "wikia.com",
    "wikipedia.org", "wordpress.com", "yahoo.com", "yelp.com", "youtube.com",
    "accuweather.com", "bankofamerica.com", "bbc.com", "bestbuy.com", "bing.com",
    "blogger.com", "booking.com", "businessinsider.com", "buzzfeed.com", "capitalone.com",
    "cars.com", "citibank.com", "classmates.com", "cnn.com", "coldwellbanker.com",
    "costco.com", "creditkarma.com", "dailymotion.com", "dealnews.com", "deviantart.com",
    "dictionary.com", "digg.com", "directv.com", "discovery.com", "dropbox.com",
    "drugstore.com", "ehow.com", "etsy.com", "expedia.com", "fanfiction.net",
    "fandango.com", "flickr.com", "foodnetwork.com", "foxsports.com", "gamefaqs.com",
    "gamespot.com", "genius.com", "gofundme.com", "goodreads.com", "groupon.com",
    "homedepot.com", "hotels.com", "houzz.com", "hulu.com", "icloud.com",
    "iheartradio.com", "investopedia.com", "irs.gov", "kmart.com", "kohls.com",
    "last.fm", "livestrong.com", "lowes.com", "macys.com", "mayoclinic.org",
    "merriam-webster.com", "metacritic.com", "mlslistings.com", "monster.com", "msnbc.com",
    "nhl.com", "npr.org", "opentable.com", "pandora.com", "pbs.org",
    "photobucket.com", "pricerunner.com", "quora.com", "realtor.com", "reference.com",
    "sci-news.com", "theweek.com", "vikings.com", "walgreens.com", "wellsfargo.com",
])


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
        "- 🟣 **COMPARE** — compare two named sites\n"
        "- 🟡 **AMBIGUOUS** — vague pronoun (will ask to clarify)"
    )
    st.divider()

    st.markdown("### Try these")
    samples = [
        "What types of personal data do websites collect?",
        "What does nytimes.com say about third-party data sharing?",
        "How does washingtonpost.com handle cookies?",
        "Compare how nytimes.com and theatlantic.com treat user data.",
        "Which policies mention data retention periods?",
        "Do they sell my data?",
    ]
    for sample in samples:
        if st.button(sample, use_container_width=True, key=sample):
            st.session_state["question_input"] = sample

    st.divider()

    # Corpus scope info
    with st.expander("📚 Dataset Scope & Categories"):
        st.markdown(_OPP115_INFO)

    # Full company list
    with st.expander("🏢 Companies in Corpus (115 sites)"):
        st.caption(
            "These are the 115 websites whose 2015–2016 privacy policies are in this dataset. "
            "Questions about any other site will return a corpus-miss warning."
        )
        # Show as a compact multi-column list
        cols = st.columns(3)
        for i, site in enumerate(_OPP115_SITES):
            cols[i % 3].markdown(f"• {site}")

    # Live corpus browser (loads after pipeline is warmed up)
    with st.expander("🔎 Browse & Pre-fill a Site"):
        if "corpus_sites" in st.session_state:
            sites = st.session_state["corpus_sites"]
            if sites:
                st.caption(f"{len(sites)} sites detected in vector store")
                selected = st.selectbox(
                    "Pick a site to pre-fill question:",
                    ["— select —"] + sites,
                    key="site_browser",
                )
                if selected and selected != "— select —":
                    if st.button("Use this site", key="use_site"):
                        st.session_state["question_input"] = (
                            f"What does {selected} say about data collection?"
                        )
            else:
                st.caption("Run a query first to load the site list.")
        else:
            st.caption("Run a query first to load the site list.")

    st.divider()
    st.markdown(
        "Built by [Rohith Sundar](https://www.linkedin.com/in/rohithsundarj/) · "
        "Dataset: [OPP-115 Corpus](https://usableprivacy.org/data)"
    )


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🔍 Agentic Privacy & Compliance Auditor")
st.caption(
    "Ask questions about 115 real website privacy policies — "
    "answers are validated for hallucinations."
)

question = st.text_input(
    label="Your question",
    placeholder="e.g. What does nytimes.com say about sharing data with third parties?",
    key="question_input",
)

analyze_clicked = st.button("🔎 Analyze", type="primary")

# ── Analysis flow ─────────────────────────────────────────────────────────────

if analyze_clicked and not question.strip():
    st.warning("Please enter a question before clicking Analyze.")

elif analyze_clicked and question.strip():

    # ── Q9: Intent check — block before any LLM call ─────────────────────────
    has_restricted, has_valid_privacy = _check_restricted_intent(question)
    if has_restricted:
        if has_valid_privacy:
            # Q9 multi-intent: warn about the restricted part, strip it, continue
            st.warning(
                "⚠️ **Part of your request is outside scope.**\n\n"
                "I am a legal compliance assistant and **cannot** write code or scripts. "
                "I will answer the privacy policy question only and ignore the rest."
            )
            question = _strip_restricted_part(question)
            if not question.strip():
                st.error(_HARDCODED_REFUSAL)
                st.stop()
        else:
            # Entire query is restricted — hard block
            st.error(_HARDCODED_REFUSAL)
            st.stop()

    # ── Step 1/4: Pipeline ────────────────────────────────────────────────────
    with st.spinner("⚙️ Step 1/4 — Loading AI pipeline..."):
        pipeline, load_error = get_pipeline()

    if load_error:
        st.error(f"❌ Failed to load pipeline: {load_error}")
        st.markdown(
            "**Possible causes:**\n"
            "- Vector store not found — run `python src/ingestion/ingest.py` first\n"
            "- Ollama not running — start with `ollama serve`\n"
            f"\n**Vector store path checked:** `{_resolve_vector_store_path()}`"
        )
        st.stop()

    planner, auditor, vectorstore = pipeline

    # Cache available sites in session_state for corpus browser
    if "corpus_sites" not in st.session_state:
        st.session_state["corpus_sites"] = _get_corpus_sites(vectorstore)

    # ── Step 2/4: Planner + Retrieval ─────────────────────────────────────────
    with st.spinner("🧭 Step 2/4 — Planner routing and retrieving segments..."):
        try:
            result = planner.retrieve(question)
            # BUG 1 safety: retrieve() must always return (list, dict)
            if not isinstance(result, tuple) or len(result) != 2:
                st.error(f"❌ Planner returned unexpected result type: {type(result)}")
                st.stop()
            docs, plan = result
        except Exception as e:
            st.error(f"❌ Planner failed: {e}")
            st.stop()

    query_type = plan.get("type", "SIMPLE")
    badge = {"SIMPLE": "🟢", "FILTERED": "🔵", "COMPARE": "🟣", "AMBIGUOUS": "🟡"}.get(query_type, "⚪")
    st.markdown(f"**Query type:** {badge} `{query_type}` — {plan.get('reasoning', '')}")

    # OFF-TOPIC / prompt-injection guard
    if plan.get("off_topic"):
        st.error(
            "🔒 **Off-topic query detected.**\n\n"
            "I'm a Privacy Policy Auditor. I can only answer questions about privacy policies.\n\n"
            "Try asking something like:\n"
            "- *What data does nytimes.com collect?*\n"
            "- *Does google.com share data with third parties?*"
        )
        st.stop()

    # AMBIGUOUS: stop and ask for clarification
    if plan.get("ambiguous"):
        st.warning(
            "🟡 **Please clarify your question.**\n\n"
            "Your question uses a vague reference (e.g. 'they', 'it', 'the company') "
            "without naming a specific website.\n\n"
            "Try rephrasing with a site name, for example:\n"
            "- *What does **nytimes.com** say about data collection?*\n"
            "- *Does **washingtonpost.com** sell user data?*\n\n"
            "You can browse the 115 available sites in the sidebar under **Browse Available Sites**."
        )
        st.stop()

    _show_corpus_warnings(plan)

    # COMPARE: stop if ALL sites are missing
    if query_type == "COMPARE":
        all_sites = [sq.get("filters", {}).get("url", "") for sq in plan.get("sub_queries", [])]
        all_sites = [s for s in all_sites if s]
        misses = plan.get("compare_misses", [])
        if all_sites and set(misses) >= set(all_sites):
            st.error(
                "❌ None of the requested sites are in the OPP-115 corpus. "
                "No comparison can be generated."
            )
            st.stop()

    if not docs:
        st.warning(
            "No relevant policy segments found. "
            "Try rephrasing or mentioning a specific website."
        )
        st.stop()

    # ── Step 3/4: Generate ────────────────────────────────────────────────────
    with st.spinner("🧠 Step 3/4 — LLM generating grounded answer..."):
        try:
            from retrieval.generation import generate_answer
            answer = generate_answer(question, docs)
        except Exception as e:
            st.error(f"❌ Generation failed: {e}")
            st.stop()

    # ── Step 4/4: Audit ───────────────────────────────────────────────────────
    with st.spinner("✅ Step 4/4 — Auditor validating faithfulness..."):
        try:
            from retrieval.generation import generate_answer
            final_answer, audit = auditor.audit_and_regenerate(
                question=question,
                answer=answer,
                docs=docs,
                generate_fn=generate_answer,
            )
        except Exception as e:
            st.warning(f"⚠ Auditor error: {e}. Showing unvalidated answer.")
            final_answer = answer
            audit = {
                "faithfulness_score": 0.0,
                "verdict": "UNKNOWN",
                "unsupported_claims": [],
                "reasoning": "Audit could not be completed.",
            }

    # ── Save to history ───────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "question":  question,
        "answer":    final_answer,
        "score":     audit.get("faithfulness_score"),   # may be None on UNVERIFIED
        "verdict":   audit.get("verdict", "UNKNOWN"),
        "query_type": query_type,
        "docs":      docs,
        "audit":     audit,
    })

    # ── Answer ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Answer")
    # Render via unsafe_allow_html so Streamlit doesn't strip markdown list/table syntax
    st.markdown(final_answer, unsafe_allow_html=False)

    # ── Audit result ──────────────────────────────────────────────────────────
    score       = audit.get("faithfulness_score")   # may be None on UNVERIFIED
    verdict     = audit.get("verdict", "PASS")
    unsupported = audit.get("unsupported_claims", [])

    col1, col2 = st.columns([1, 3])
    with col1:
        if verdict == "PASS":
            st.success("✅ Audit: PASS")
        elif verdict == "WARN":
            st.warning("⚠️ Audit: WARN")
        elif verdict == "FAIL":
            st.error("❌ Audit: FAIL")
        elif verdict == "UNVERIFIED":
            st.warning("⚠️ Audit: UNVERIFIED")
        else:
            st.warning("⚠ Audit: UNKNOWN")
        if score is not None:
            st.metric("Faithfulness Score", f"{score:.2f} / 1.00")
        else:
            st.caption("Score: N/A (auditor unavailable)")
    with col2:
        if verdict == "UNVERIFIED":
            reason = audit.get("reasoning", "")
            # Show the clean RuntimeError message, not raw HF URL
            display_reason = reason if reason.startswith("⚠️") else "Auditor could not run — answer shown unvalidated."
            st.warning(f"**{display_reason}**\n\nThe answer above was generated but could not be verified for faithfulness. Treat it with caution.")
        elif verdict == "WARN":
            st.markdown(
                "**⚠️ Warning:** Some claims could not be fully verified against "
                "the source documents. Use this answer with caution and consult "
                "the source segments below."
            )
        reasoning = audit.get("reasoning", "")
        if reasoning:
            st.markdown(f"**Auditor reasoning:** {reasoning}")
        if unsupported:
            st.markdown("**Unsupported claims flagged:**")
            for claim in unsupported:
                st.markdown(f"- ⚠ {claim}")

    # ── Export ────────────────────────────────────────────────────────────────
    export_data = {
        "question":   question,
        "query_type": query_type,
        "answer":     final_answer,
        "audit":      audit,
        "sources": [
            {
                "policy_id": d.metadata.get("policy_id"),
                "url":       d.metadata.get("url"),
                "section":   d.metadata.get("section"),
                "date":      d.metadata.get("collection_date"),
                "text":      d.page_content,
            }
            for d in docs
        ],
    }
    st.download_button(
        label="📥 Export Report (JSON)",
        data=json.dumps(export_data, indent=2, ensure_ascii=False),
        file_name="privacy_audit_report.json",
        mime="application/json",
    )

    # ── Source documents ──────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"📂 Source Documents ({len(docs)} retrieved)")

    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        with st.expander(
            f"[{i}] {m.get('policy_id', 'unknown')} — {m.get('url', '')}",
            expanded=(i == 1),
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Policy ID:** `{m.get('policy_id', 'N/A')}`")
            c2.markdown(f"**Collection Date:** `{m.get('collection_date', 'N/A')}`")
            c3.markdown(f"**Section:** `{m.get('section', 'N/A')}`")
            c4.markdown(f"**Annotations:** `{m.get('annotation_count', '0')}`")

            st.markdown("**Segment text:**")
            preview = doc.page_content[:1500]
            full    = doc.page_content
            st.markdown(
                "<div style='"
                "background:#f8f9fa;"
                "padding:12px;"
                "border-radius:6px;"
                "font-size:0.9em;"
                "border-left:3px solid #dee2e6;"
                "color:#212529"
                f"'>{preview}{'…' if len(full) > 1500 else ''}</div>",
                unsafe_allow_html=True,
            )
            # Full segment toggle
            if len(full) > 1500:
                with st.expander("Show full segment"):
                    st.text(full)


# ── Query History ─────────────────────────────────────────────────────────────

if st.session_state.get("history"):
    st.divider()
    st.subheader("📜 Query History")
    history = st.session_state["history"]
    st.caption(f"{len(history)} queries this session")

    for idx, entry in enumerate(reversed(history), 1):
        v = entry["verdict"]
        v_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "UNVERIFIED": "⚠️"}.get(v, "⚪")
        score_str = f"{entry['score']:.2f}" if entry.get("score") is not None else "N/A"
        q_icon = {"SIMPLE": "🟢", "FILTERED": "🔵", "COMPARE": "🟣", "AMBIGUOUS": "🟡"}.get(
            entry.get("query_type", ""), "⚪"
        )
        with st.expander(
            f"{idx}. {q_icon} {entry['question'][:80]}{'…' if len(entry['question']) > 80 else ''} "
            f"— {v_icon} {v} ({score_str})"
        ):
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(f"**A:** {entry['answer']}")
            st.caption(f"Faithfulness: {score_str} | Verdict: {v} | Type: {entry.get('query_type', '—')}")