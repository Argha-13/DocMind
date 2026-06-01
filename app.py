
import streamlit as st
import tempfile
import os
import html as html_lib
from copy import deepcopy

st.set_page_config(
    page_title="DocMind",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE  — FIX 8: mutable defaults via factory, not shared references
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "messages":      [],        # [{role, content, confidence, sources}]
        "chat_history":  [],        # LangChain HumanMessage / AIMessage objects
        "vectorstore":   None,
        "doc_name":      None,
        "top_k":         4,
        "min_score":     0.10,
        "is_processing": False,
        "pending_query": None,      # staged query text
        "submit_count":  0,         # FIX 2: dedup key (int), not text comparison
        "processed_count": 0,       # matches submit_count after processing
        "input_key":     0,         # FIX 10: rotated to clear input widget
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            # deepcopy ensures each session gets its own list/dict instances
            st.session_state[k] = deepcopy(v)

_init_state()

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  (unchanged premium design)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,300;9..144,600&display=swap');

:root {
    --bg:      #07070f;
    --bg2:     #0e0e1c;
    --bg3:     #141424;
    --bg4:     #1c1c30;
    --border:  #22223a;
    --border2: #2e2e50;
    --accent:  #6366f1;
    --accentL: #818cf8;
    --accentD: #3730a3;
    --text:    #ede9fe;
    --text2:   #8b87b8;
    --text3:   #45456a;
    --green:   #34d399;
    --amber:   #fbbf24;
    --red:     #f87171;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}
.block-container { padding: 0 !important; max-width: 860px !important; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

.stTextInput > div > div > input {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 13px 18px !important;
    height: 48px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.15) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text3) !important; }
.stTextInput label { display: none !important; }
.stTextInput { margin: 0 !important; }

.stButton > button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    height: 48px !important;
    padding: 0 20px !important;
    transition: all .2s !important;
    box-shadow: 0 2px 16px rgba(99,102,241,.25) !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--accentL) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,.4) !important;
}
.stButton > button:active { transform: none !important; }
.stButton > button:disabled {
    background: var(--bg4) !important;
    color: var(--text3) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* Form submit button (Send →) — same accent style */
[data-testid="stFormSubmitButton"] > button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    height: 48px !important;
    padding: 0 20px !important;
    transition: all .2s !important;
    box-shadow: 0 2px 16px rgba(99,102,241,.3) !important;
    width: 100% !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--accentL) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,.45) !important;
}
[data-testid="stFormSubmitButton"] > button:active { transform: none !important; }
[data-testid="stFormSubmitButton"] > button:disabled {
    background: var(--bg4) !important;
    color: var(--text3) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    transform: none !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1.5px dashed var(--border2) !important;
    border-radius: 12px !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div { color: var(--text3) !important; font-size: 12px !important; }
[data-testid="stFileUploader"] label { color: var(--text2) !important; font-size: 12px !important; }
[data-testid="stSlider"] label { color: var(--text2) !important; font-size: 12px !important; }
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    margin: 12px 0 0 !important;
}
[data-testid="stExpander"]:hover { border-color: var(--border2) !important; }
[data-testid="stExpander"] summary {
    padding: 13px 18px !important;
    color: var(--text2) !important;
    font-size: 13px !important;
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg2) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }
[data-testid="stExpander"] > div > div { padding: 0 18px 16px !important; }

/* Nav */
.topnav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 24px 16px;
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
    background: var(--bg); backdrop-filter: blur(20px);
}
.nav-brand { display: flex; align-items: center; gap: 10px; }
.nav-gem {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, var(--accent), var(--accentL));
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center; font-size: 13px;
    animation: gemGlow 3s ease infinite;
}
@keyframes gemGlow {
    0%,100% { box-shadow: 0 0 12px rgba(99,102,241,.4); }
    50%      { box-shadow: 0 0 24px rgba(99,102,241,.8); }
}
.nav-name { font-family: 'Fraunces', serif; font-size: 18px; font-weight: 600; color: var(--text); }
.nav-doc-pill {
    display: flex; align-items: center; gap: 7px;
    background: var(--bg3); border: 1px solid var(--border2);
    border-radius: 20px; padding: 5px 13px;
    font-size: 12px; font-family: 'JetBrains Mono', monospace; color: var(--accentL);
    max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    animation: fadeUp .3s ease;
}
.nav-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 8px var(--green);
    animation: glowDot 2s ease infinite; flex-shrink: 0;
}
@keyframes glowDot { 0%,100%{box-shadow:0 0 5px var(--green)} 50%{box-shadow:0 0 12px var(--green)} }
@keyframes fadeUp  { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

/* Welcome */
.welcome {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 340px; gap: 16px; text-align: center; padding: 48px 32px;
    animation: fadeUp .5s ease;
}
.orb {
    width: 80px; height: 80px; border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, var(--accentL), var(--accentD));
    display: flex; align-items: center; justify-content: center; font-size: 30px;
    animation: orbFloat 5s ease infinite;
}
@keyframes orbFloat {
    0%,100% { transform: translateY(0) scale(1); box-shadow: 0 0 60px rgba(99,102,241,.3); }
    50%      { transform: translateY(-10px) scale(1.03); box-shadow: 0 0 90px rgba(99,102,241,.55); }
}
.welcome h2 { font-family:'Fraunces',serif; font-size:30px; font-weight:600; color:var(--text); }
.welcome p  { font-size:14px; color:var(--text2); max-width:300px; line-height:1.8; }
.steps { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; margin-top:4px; }
.step {
    background: var(--bg2); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 16px; width: 140px; text-align:left;
    transition: border-color .25s, transform .25s; animation: fadeUp .5s ease backwards;
}
.step:hover { border-color: var(--accent); transform: translateY(-3px); }
.step:nth-child(1){animation-delay:.1s} .step:nth-child(2){animation-delay:.2s} .step:nth-child(3){animation-delay:.3s}
.step-n { font-family:'JetBrains Mono',monospace; font-size:9px; color:var(--accent); letter-spacing:2px; text-transform:uppercase; margin-bottom:8px; }
.step-t { font-size:12px; color:var(--text2); line-height:1.5; }

/* Messages */
.msg { display:flex; gap:11px; animation:fadeUp .25s ease backwards; margin-bottom:4px; }
.msg.user { flex-direction:row-reverse; }
.av {
    width:30px; height:30px; border-radius:8px; flex-shrink:0; margin-top:2px;
    display:flex; align-items:center; justify-content:center;
    font-size:11px; font-family:'JetBrains Mono',monospace; font-weight:500;
}
.msg.user .av { background:var(--accent); color:#fff; }
.msg.ai   .av { background:var(--bg4); color:var(--accentL); border:1px solid var(--border2); }
.mb { max-width:78%; display:flex; flex-direction:column; gap:6px; }
.msg.user .mb { align-items:flex-end; }
.bbl {
    padding:11px 15px; border-radius:13px; font-size:14px; line-height:1.75;
    word-break:break-word;
}
.msg.user .bbl { background:var(--accent); color:#fff; border-bottom-right-radius:3px; box-shadow:0 3px 16px rgba(99,102,241,.3); }
.msg.ai   .bbl { background:var(--bg3); border:1px solid var(--border); color:var(--text); border-bottom-left-radius:3px; }
.bbl.error { background:rgba(248,113,113,.08) !important; border-color:var(--red) !important; color:var(--red) !important; }
.cbar {
    display:flex; align-items:center; gap:7px; padding:5px 10px;
    background:var(--bg2); border:1px solid var(--border);
    border-radius:7px; font-size:10px; font-family:'JetBrains Mono',monospace; color:var(--text3);
}
.ctrack { flex:1; height:2px; background:var(--border2); border-radius:2px; overflow:hidden; }
.cfill  { height:100%; border-radius:2px; transition:width .6s ease; }
.chips  { display:flex; flex-wrap:wrap; gap:4px; }
.chip {
    background:var(--bg2); border:1px solid var(--border); border-radius:5px;
    padding:2px 7px; font-size:10px; font-family:'JetBrains Mono',monospace; color:var(--text3);
    transition:border-color .2s;
}
.chip:hover { border-color:var(--accent); }
.chip b { color:var(--accentL); }
.ibar-hint {
    display:flex; align-items:center; justify-content:center; gap:6px;
    font-size:11px; font-family:'JetBrains Mono',monospace; color:var(--text3); margin-bottom:6px;
}
.pdot { width:5px; height:5px; border-radius:50%; background:var(--green); animation:glowDot 2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    emb = MistralAIEmbeddings(model="mistral-embed")
    llm = ChatMistralAI(model="mistral-small-2603", temperature=0.2)
    return emb, llm


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORSTORE  — FIX 4: temp file always cleaned up via finally
# ══════════════════════════════════════════════════════════════════════════════
def build_vectorstore(pdf_bytes: bytes) -> "Chroma":
    emb, _ = load_models()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        docs = PyPDFLoader(tmp_path).load()

        # For very large PDFs cap at 300 pages to avoid RAM exhaustion
        MAX_PAGES = 300
        if len(docs) > MAX_PAGES:
            docs = docs[:MAX_PAGES]
            st.warning(
                f"⚠ Document has more than {MAX_PAGES} pages. "
                f"Only the first {MAX_PAGES} pages were indexed to keep performance stable.",
                icon="⚠️",
            )

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1500,        # larger chunks = fewer embeddings = faster for big docs
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        ).split_documents(docs)

        return Chroma.from_documents(
            documents=chunks, embedding=emb, persist_directory=None
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL  — FIX 5: clamp scores to [0,1] before threshold comparison
# ══════════════════════════════════════════════════════════════════════════════
def retrieve_context(query: str, vs, top_k: int, min_score: float):
    raw = vs.similarity_search_with_relevance_scores(query=query, k=top_k)
    if not raw:
        return []
    # Clamp: cosine-based distances can return values outside [0,1]
    clamped = [(doc, max(0.0, min(1.0, score))) for doc, score in raw]
    filtered = [(doc, score) for doc, score in clamped if score >= min_score]
    # Safety: always return at least the single best result
    if not filtered:
        filtered = [max(clamped, key=lambda x: x[1])]
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT
# ══════════════════════════════════════════════════════════════════════════════
_SYSTEM = (
    "You are DocMind, a helpful AI assistant that answers questions strictly "
    "from the provided document context.\n"
    "Rules:\n"
    "- Answer ONLY from the context below. Never use outside knowledge.\n"
    "- If the answer is absent, say exactly: "
    "\"I could not find the answer in the document.\"\n"
    "- Use the conversation history to resolve follow-up questions.\n"
    "- Be concise and direct."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",
     "Conversation history:\n{history}\n\n"
     "Document context:\n{context}\n\n"
     "Question: {question}"),
])

MAX_HISTORY_EXCHANGES = 5  # keep last N human+AI pairs = 2N messages


def _trim_and_format_history(history: list) -> str:
    """Trim to last MAX_HISTORY_EXCHANGES exchanges and format as plain text."""
    trimmed = history[-(MAX_HISTORY_EXCHANGES * 2):]
    lines = []
    for msg in trimmed:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines) if lines else "None"


# ══════════════════════════════════════════════════════════════════════════════
#  RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def rag_query(query: str, vs, history: list, top_k: int, min_score: float) -> dict:
    """Full pipeline. Never raises — all errors returned as answer strings."""
    try:
        _, llm = load_models()
        pairs = retrieve_context(query, vs, top_k, min_score)

        if not pairs:
            return {"answer": "I could not find the answer in the document.",
                    "sources": [], "confidence": 0.0}

        context    = "\n\n".join(d.page_content for d, _ in pairs)
        sources    = [{"page": d.metadata.get("page", "?"), "score": round(s, 3)}
                      for d, s in pairs]
        confidence = round(max(s for _, s in pairs), 3)
        history_str = _trim_and_format_history(history)

        response = llm.invoke(
            _PROMPT.invoke({"history": history_str, "context": context, "question": query})
        )
        return {"answer": response.content, "sources": sources, "confidence": confidence}

    except Exception as exc:
        return {"answer": f"⚠ Error: {exc}", "sources": [], "confidence": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS PENDING QUERY
#  FIX 2: dedup via submit_count integer, not text comparison
#  FIX 3: try/finally guarantees is_processing resets even on unexpected raises
#  FIX 6: trim history BEFORE passing to rag_query
# ══════════════════════════════════════════════════════════════════════════════
def process_pending():
    # Only process if a new submission exists that hasn't been processed yet
    if st.session_state.submit_count <= st.session_state.processed_count:
        return
    q = st.session_state.pending_query
    if not q or not st.session_state.vectorstore:
        st.session_state.processed_count = st.session_state.submit_count
        return

    st.session_state.pending_query   = None
    st.session_state.is_processing   = True

    try:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.chat_history.append(HumanMessage(content=q))

        # FIX 6: trim BEFORE the LLM call so it never sees a bloated history
        if len(st.session_state.chat_history) > MAX_HISTORY_EXCHANGES * 2:
            st.session_state.chat_history = (
                st.session_state.chat_history[-(MAX_HISTORY_EXCHANGES * 2):]
            )

        result = rag_query(
            q,
            st.session_state.vectorstore,
            st.session_state.chat_history,   # already trimmed
            st.session_state.top_k,
            st.session_state.min_score,
        )

        st.session_state.messages.append({
            "role":       "ai",
            "content":    result["answer"],
            "confidence": result["confidence"],
            "sources":    result["sources"],
        })
        st.session_state.chat_history.append(AIMessage(content=result["answer"]))

    finally:
        # FIX 3: always release the processing lock
        st.session_state.is_processing   = False
        st.session_state.processed_count = st.session_state.submit_count


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: TOP NAV
# ══════════════════════════════════════════════════════════════════════════════
def render_top_nav():
    pill = ""
    if st.session_state.doc_name:
        safe = html_lib.escape(st.session_state.doc_name[:32])
        pill = f'<div class="nav-doc-pill"><div class="nav-dot"></div>{safe}</div>'
    st.markdown(f"""
    <div class="topnav">
        <div class="nav-brand">
            <div class="nav-gem">◈</div>
            <div class="nav-name">DocMind</div>
        </div>
        {pill}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: UPLOAD PANEL
#  FIX 7: can_index logic — only block if BOTH name matches AND vs exists
# ══════════════════════════════════════════════════════════════════════════════
def render_upload_panel():
    label = "📄  Upload Document"
    if st.session_state.doc_name:
        label += f" · {st.session_state.doc_name[:28]}"

    with st.expander(label, expanded=not bool(st.session_state.vectorstore)):
        uploaded_file = st.file_uploader(
            "Choose a PDF", type=["pdf"], label_visibility="collapsed"
        )

        col_btn, col_status = st.columns([2, 3])
        with col_btn:
            # FIX 7: allow re-index if vectorstore is None, even if name matches
            already_indexed = (
                uploaded_file is not None
                and st.session_state.doc_name == uploaded_file.name
                and st.session_state.vectorstore is not None
            )
            index_clicked = st.button(
                "⟳  Index Document",
                disabled=(uploaded_file is None or already_indexed),
                use_container_width=True,
                key="index_btn",
            )

        with col_status:
            if st.session_state.vectorstore:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;height:48px;padding-left:4px">
                    <div style="width:8px;height:8px;border-radius:50%;background:#34d399;
                                box-shadow:0 0 10px #34d399;animation:glowDot 2s infinite"></div>
                    <span style="font-size:12px;font-family:'JetBrains Mono',monospace;
                                 color:#34d399">Indexed &amp; ready</span>
                </div>""", unsafe_allow_html=True)
            elif st.session_state.is_processing:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;height:48px;padding-left:4px">
                    <div style="width:8px;height:8px;border-radius:50%;background:#fbbf24;
                                animation:blink .8s infinite"></div>
                    <span style="font-size:12px;font-family:'JetBrains Mono',monospace;
                                 color:#fbbf24">Processing…</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;height:48px;padding-left:4px">
                    <div style="width:8px;height:8px;border-radius:50%;background:#45456a"></div>
                    <span style="font-size:12px;font-family:'JetBrains Mono',monospace;
                                 color:#45456a">No document loaded</span>
                </div>""", unsafe_allow_html=True)

        if index_clicked and uploaded_file:
            # Reset everything for fresh document
            st.session_state.messages       = []
            st.session_state.chat_history   = []
            st.session_state.vectorstore    = None
            st.session_state.pending_query  = None
            st.session_state.submit_count   = 0
            st.session_state.processed_count = 0
            st.session_state.input_key      += 1
            st.session_state.is_processing  = True

            with st.spinner("Reading & embedding document…"):
                try:
                    vs = build_vectorstore(uploaded_file.getvalue())
                    st.session_state.vectorstore  = vs
                    st.session_state.doc_name     = uploaded_file.name
                except Exception as exc:
                    st.error(f"Failed to index: {exc}")
                finally:
                    st.session_state.is_processing = False
            st.rerun()

        with st.expander("⚙  Retrieval settings", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.top_k = st.slider(
                    "Top-K chunks", 1, 8, st.session_state.top_k, key="slider_topk"
                )
            with c2:
                st.session_state.min_score = st.slider(
                    "Min score", 0.0, 0.5, st.session_state.min_score, 0.05,
                    key="slider_score"
                )


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: CHAT
# ══════════════════════════════════════════════════════════════════════════════
def render_chat():
    msgs_box = st.container(height=430)
    with msgs_box:
        if not st.session_state.messages:
            if not st.session_state.vectorstore:
                st.markdown("""
                <div class="welcome">
                    <div class="orb">◈</div>
                    <h2>Upload your document</h2>
                    <p>Use the panel above to upload and index a PDF, then start chatting.</p>
                    <div class="steps">
                        <div class="step"><div class="step-n">01 · Upload</div>
                            <div class="step-t">Open the panel and pick a PDF</div></div>
                        <div class="step"><div class="step-n">02 · Index</div>
                            <div class="step-t">Click Index Document</div></div>
                        <div class="step"><div class="step-n">03 · Ask</div>
                            <div class="step-t">Type your question below</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="welcome">
                    <div class="orb" style="background:radial-gradient(circle at 30% 30%,
                         #34d399,#065f46)">✓</div>
                    <h2>Ready to chat</h2>
                    <p>Your document is indexed. Ask anything about it below.</p>
                </div>""", unsafe_allow_html=True)
        else:
            for i, msg in enumerate(st.session_state.messages):
                delay      = f"animation-delay:{i * 0.03:.2f}s"
                raw_content = msg["content"]
                # FIX 9: detect error flag on RAW content before escaping
                is_err     = raw_content.startswith("⚠")
                safe_content = html_lib.escape(raw_content)

                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="msg user" style="{delay}">
                        <div class="av">U</div>
                        <div class="mb"><div class="bbl">{safe_content}</div></div>
                    </div>""", unsafe_allow_html=True)
                else:
                    conf     = msg.get("confidence", 0)
                    pct      = int(conf * 100)
                    col      = "#34d399" if conf >= .7 else "#fbbf24" if conf >= .4 else "#f87171"
                    bbl_cls  = "bbl error" if is_err else "bbl"
                    chips    = "".join(
                        f'<div class="chip">p.<b>{s["page"]}</b>&nbsp;·&nbsp;{s["score"]}</div>'
                        for s in msg.get("sources", [])[:5]
                    )
                    # conf_html is OUTSIDE .bbl — keeps answer text and metadata separate
                    if is_err:
                        conf_html = ""
                    else:
                        conf_html = (
                            f'<div class="cbar">'
                            f'<span>CONF</span>'
                            f'<div class="ctrack"><div class="cfill" style="width:{pct}%;background:{col}"></div></div>'
                            f'<span style="color:{col}">{pct}%</span>'
                            f'</div>'
                            f'<div class="chips">{chips}</div>'
                        )
                    st.markdown(
                        f'<div class="msg ai" style="{delay}">'
                        f'<div class="av">AI</div>'
                        f'<div class="mb">'
                        f'<div class="{bbl_cls}">{safe_content}</div>'
                        f'{conf_html}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            ai_msgs = [m for m in st.session_state.messages if m["role"] == "ai"]
            if ai_msgs:
                avg = sum(m.get("confidence", 0) for m in ai_msgs) / len(ai_msgs)
                st.markdown(f"""
                <div style="display:flex;align-items:center;padding:10px 4px 0;
                            border-top:1px solid #22223a;margin-top:8px">
                    <span style="font-size:11px;font-family:'JetBrains Mono',monospace;color:#45456a">
                        {len(ai_msgs)} turns · avg conf {avg:.0%}
                    </span>
                </div>""", unsafe_allow_html=True)

    if st.session_state.messages:
        if st.button("✕  Clear conversation", use_container_width=True, key="clear_btn"):
            st.session_state.messages        = []
            st.session_state.chat_history    = []
            st.session_state.submit_count    = 0
            st.session_state.processed_count = 0
            st.session_state.pending_query   = None
            st.session_state.input_key      += 1
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: INPUT BAR
#  FIX 1: trigger ONLY on button click or Enter (form submit), not bare text
#  FIX 2: use submit_count integer for dedup, not text comparison
#  FIX 10: rotate input_key after submit so widget clears on next render
# ══════════════════════════════════════════════════════════════════════════════
def render_input_bar():
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    if st.session_state.vectorstore:
        st.markdown(
            '<div class="ibar-hint"><div class="pdot"></div>'
            'Document indexed · ready</div>',
            unsafe_allow_html=True,
        )

    # Use st.form so Enter key submits without re-running on every keystroke
    with st.form(key=f"chat_form_{st.session_state.input_key}", clear_on_submit=True):
        qcol, bcol = st.columns([5, 1])
        with qcol:
            query = st.text_input(
                "q",
                placeholder="Ask anything about the document…",
                label_visibility="collapsed",
                disabled=not st.session_state.vectorstore,
            )
        with bcol:
            send = st.form_submit_button(
                "Send →",
                use_container_width=True,
                disabled=not st.session_state.vectorstore,
            )

    # FIX 1 & 2: only trigger on an actual form submission with non-empty text
    if send and query.strip() and not st.session_state.is_processing:
        st.session_state.pending_query = query.strip()
        st.session_state.submit_count += 1
        # FIX 10: rotating the key causes st.form to render fresh (empty) next run
        st.session_state.input_key    += 1
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN  — process before render so answer is visible on same rerun
# ══════════════════════════════════════════════════════════════════════════════
process_pending()
render_top_nav()
render_upload_panel()
render_chat()
render_input_bar()
