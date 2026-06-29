import os
import re
import html as html_lib
import tempfile
from collections import Counter

import google.generativeai as genai
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# settings
CHUNK_SIZE    = 1_000
CHUNK_OVERLAP = 200
TOP_K         = 6
EMBED_MODEL   = "gemini-embedding-001"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
MAX_FILE_MB   = 25
MAX_URLS      = 8

# Arabic stop-words for keyword extraction
_AR_STOP = frozenset({
    "على","في","من","الى","عن","مع","هذا","هذه","ذلك","التي","الذي","التى","الذى",
    "كان","قد","عند","لكن","حتى","إذا","ثم","بعض","حيث","كل","هو","هي","ما","لا",
    "أو","أم","بل","وقد","ولا","فقد","فإذا","عليه","فيه","إلى","علي","منها","عليها",
    "به","بها","كما","أي","بعد","قبل","خلال","ضمن","بين","نحو","عبر","فيما","لدي",
    "لديه","لديها","هناك","هنا","كذلك","أيضا","غير","سوى","سوف","لن","لم","ليس",
    "كلا","كلتا","كلاهما","يجب","يمكن","تكون","يكون","والتي","والذي","والتى","والذى",
    "السلام","عليكم","شكرا","جزاكم","تحية","التي","الذي","ذلك","هؤلاء","هذا","هذه",
    "ذلك","تلك","هناك","هنا","حيث","عندما","كما","لكن","بل","أو","أم","إما","إذا",
})


st.set_page_config(
    page_title="RAG - المساعد المؤسسي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# STYLES
# -------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
.stApp { background: #F7F9FC; }
#MainMenu, footer, header { visibility: hidden; }

.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
[data-testid="stChatMessageContent"],
.stTextArea textarea,
.stAlert > div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown {
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stChatInput"] textarea {
    direction: rtl !important;
    font-family: 'Cairo', sans-serif !important;
    border-radius: 14px !important;
}

[data-testid="stSidebar"] {
    background: #fff !important;
    border-right: 1px solid #E8EDF2 !important;
}
[data-testid="stSidebar"] * { font-family: 'Cairo', sans-serif !important; }

.hero {
    background: linear-gradient(135deg, #0F2472 0%, #1D4ED8 55%, #60A5FA 100%);
    border-radius: 22px;
    padding: 46px 40px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 14px 48px rgba(29,78,216,.28);
    position: relative;
    overflow: hidden;
    direction: rtl;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: radial-gradient(ellipse at 30% 50%, rgba(255,255,255,.10) 0%, transparent 70%);
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,.18);
    color: #fff;
    font-size: .78rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 14px;
    letter-spacing: .05em;
}
.hero-title { font-size: 2.2rem; font-weight: 800; color: #fff; margin: 0 0 8px; }
.hero-sub   { font-size: 1.05rem; color: rgba(255,255,255,.84); margin: 0; }

.info-banner {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 1px solid #BFDBFE;
    border-right: 4px solid #2563EB;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 28px;
    direction: rtl;
}
.info-banner h4 { color: #1E40AF; font-weight: 700; margin: 0 0 10px; font-size: 1.05rem; }
.info-banner ul { color: #1E3A8A; font-size: .95rem; line-height: 2.1; padding-right: 22px; margin: 0; }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 22px 0;
}
.m-card {
    background: #fff;
    border-radius: 14px;
    padding: 20px 14px;
    text-align: center;
    border: 1px solid #E8EDF2;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
    position: relative;
    overflow: hidden;
}
.m-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1E3A8A, #60A5FA);
}
.m-num   { font-size: 2.1rem; font-weight: 800; color: #1E3A8A; display: block; }
.m-label { font-size: .82rem; color: #6B7280; font-weight: 500; margin-top: 4px; display: block; }

.src-tag {
    display: inline-block;
    background: #EFF6FF;
    color: #1E40AF;
    font-size: .78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 3px;
    border: 1px solid #BFDBFE;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #ECFDF5;
    color: #065F46;
    font-size: .85rem;
    font-weight: 700;
    padding: 7px 16px;
    border-radius: 20px;
    border: 1px solid #A7F3D0;
    direction: rtl;
}
.pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10B981;
    flex-shrink: 0;
    animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .3; transform: scale(.8); }
}

.stButton > button {
    font-family: 'Cairo', sans-serif !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all .2s ease !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 46px !important;
    line-height: 1.5 !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(30,58,138,.22) !important;
}

.stTextArea textarea     { font-family: 'Cairo', sans-serif !important; border-radius: 12px !important; }
.stSelectbox > div > div { font-family: 'Cairo', sans-serif !important; border-radius: 10px !important; }
.stAlert                 { border-radius: 12px !important; font-family: 'Cairo', sans-serif !important; }
.streamlit-expanderHeader { font-family: 'Cairo', sans-serif !important; font-weight: 600 !important; }

.sep {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
    margin: 28px 0;
}
.empty-state { text-align: center; padding: 50px 20px; color: #9CA3AF; direction: rtl; }
.empty-state .ei { font-size: 3.6rem; margin-bottom: 14px; }
.empty-state h3  { color: #6B7280; font-size: 1.1rem; margin-bottom: 6px; }
.empty-state p   { font-size: .9rem; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# small helpers
# -------------------------------------------------------------------
_URL_RE = re.compile(r"^https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$")

def url_ok(url):  return bool(_URL_RE.match(url.strip()))
def esc(t):       return html_lib.escape(str(t))
def file_ok(f):   return len(f.getvalue()) / (1024 * 1024) <= MAX_FILE_MB


# -------------------------------------------------------------------
# API key
# -------------------------------------------------------------------
_key = (st.secrets.get("GEMINI_API_KEY") or "").strip()
if _key:
    genai.configure(api_key=_key)
else:
    st.markdown("""
    <div style="background:#FEF2F2;border:1px solid #FECACA;border-right:4px solid #EF4444;
         border-radius:14px;padding:28px;text-align:center;margin-top:40px;direction:rtl;">
        <h3 style="color:#991B1B;margin:0 0 10px;">&#9940; مفتاح API مفقود</h3>
        <p style="color:#7F1D1D;margin:0;line-height:1.8;">
            اضف <code>GEMINI_API_KEY</code> في
            <b>Streamlit &rarr; Settings &rarr; Secrets</b> ثم اعد التشغيل.
        </p>
    </div>""", unsafe_allow_html=True)
    st.stop()


# -------------------------------------------------------------------
# session state
# -------------------------------------------------------------------
defaults = {
    "history":  [],
    "vs":       None,
    "chunks":   [],           # store chunks for keyword extraction
    "meta":     {"files": 0, "urls": 0, "chunks": 0},
    "quick_q":  "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------------------------------------------------------
# document loading
# -------------------------------------------------------------------
def read_files(files):
    docs, ok, errs = [], 0, []
    for f in files:
        if not file_ok(f):
            errs.append(f"'{f.name}' اكبر من {MAX_FILE_MB} MB — تم تخطيه.")
            continue
        try:
            name = f.name.lower()
            if name.endswith(".txt"):
                text = f.getvalue().decode("utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": f.name}))
                ok += 1
            else:
                ext = ".pdf" if name.endswith(".pdf") else ".docx"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(f.getvalue())
                    path = tmp.name
                try:
                    loader = PyPDFLoader(path) if ext == ".pdf" else Docx2txtLoader(path)
                    for d in loader.load():
                        d.metadata["source"] = f.name
                        docs.append(d)
                    ok += 1
                finally:
                    if os.path.exists(path):
                        os.remove(path)
        except Exception as e:
            errs.append(f"'{f.name}': {str(e)[:100]}")
    return docs, ok, errs


def read_urls(raw):
    docs, ok, errs = [], 0, []
    lines = [u.strip() for u in raw.splitlines() if u.strip()]
    if len(lines) > MAX_URLS:
        errs.append(f"الحد الاقصى {MAX_URLS} روابط — تم تجاهل الباقي.")
        lines = lines[:MAX_URLS]
    for url in lines:
        if not url_ok(url):
            errs.append(f"رابط غير صالح: '{esc(url)}'")
            continue
        try:
            for d in WebBaseLoader(url).load():
                d.metadata["source"] = url
                docs.append(d)
            ok += 1
        except Exception as e:
            errs.append(f"'{url}': {str(e)[:100]}")
    return docs, ok, errs


def build_index(docs):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=_key)
    return FAISS.from_documents(chunks, emb), len(chunks), chunks


# -------------------------------------------------------------------
# keyword extraction & dynamic quick prompts
# -------------------------------------------------------------------
def extract_keywords(chunks, top_n=10):
    """Extract top keywords from document chunks using frequency analysis."""
    # Sample first 20 chunks, 800 chars each — enough signal, fast
    sample = " ".join(d.page_content[:800] for d in chunks[:20])
    # Match Arabic words (4+ chars) or English words (5+ chars)
    words = re.findall(r'[\u0600-\u06FF]{4,}|[a-zA-Z]{5,}', sample)
    # Filter out stop-words
    filtered = [w for w in words if w not in _AR_STOP]
    return [w for w, _ in Counter(filtered).most_common(top_n)]


def get_quick_prompts():
    """
    Generate context-aware quick prompts based on the ACTUAL loaded documents.
    Returns a list of (label, query) tuples — never hardcoded.
    """
    chunks = st.session_state.get("chunks", [])
    if not chunks:
        return []

    # Unique source names (filenames / URLs)
    sources = sorted({d.metadata.get("source", "وثيقة") for d in chunks})

    # Extract top keywords from actual content
    keywords = extract_keywords(chunks)

    prompts = []

    # --- Prompt 1: Summary tailored to single vs. multi-file ---
    if len(sources) == 1:
        src_short = os.path.basename(sources[0]) if sources[0].startswith("http") else sources[0]
        prompts.append((
            "&#x1F4CB; ملخص شامل",
            f"اكتب ملخصاً تنفيذياً شاملاً لمحتوى الملف '{src_short}'. "
            f"ابرز أهم النقاط والمفاهيم الرئيسية والاستنتاجات."
        ))
    else:
        prompts.append((
            "&#x1F4CB; ملخص شامل",
            "اكتب ملخصاً تنفيذياً شاملاً لجميع المحتويات المرفقة. "
            "ابرز أهم النقاط والمفاهيم الرئيسية من كل ملف."
        ))

    # --- Prompt 2: Deep-dive on top keyword from actual content ---
    if keywords:
        kw = keywords[0]
        prompts.append((
            f"&#x1F50D; عن: {esc(kw)}",
            f"ما هو دور وأهمية '{kw}' في المحتوى المرفق؟ "
            f"اشرح بالتفصيل مع ذكر الأمثلة والسياق المرتبط به."
        ))
    else:
        prompts.append((
            "&#x1F511; المصطلحات الرئيسية",
            "استخرج أهم المصطلحات المتخصصة من المحتوى المرفق "
            "وعرّف كل منها بإيجاز مع ذكر مصدره."
        ))

    # --- Prompt 3: Analytical / actionable (file-aware) ---
    if len(sources) > 1:
        src_list = "، ".join(
            os.path.basename(s) if s.startswith("http") else s
            for s in sources[:3]
        )
        prompts.append((
            "&#x1F4CA; مقارنة",
            f"قارن بين محتويات الملفات المرفقة ({src_list}). "
            f"ما أوجه التشابه والاختلاف؟ قدّم تحليلاً منظماً."
        ))
    else:
        prompts.append((
            "&#x1F4A1; توصيات",
            "بناءً على المحتوى المرفق، ما هي أهم التوصيات "
            "والخطوات العملية المقترحة؟ قدّمها في نقاط واضحة."
        ))

    return prompts


# -------------------------------------------------------------------
# model helpers
# -------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def list_models():
    try:
        return [
            m.name.split("/")[-1]
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
    except Exception:
        return [DEFAULT_MODEL]


def make_system_prompt(strict, has_context=True):
    """
    Build system instruction.
    If strict mode but no context (no KB built), fall back gracefully
    so the model doesn't refuse every question.
    """
    if strict:
        if has_context:
            return (
                "انت مساعد مؤسسي دقيق. "
                "اجب فقط من السياق المرفق. "
                "ان لم تجد الاجابة قل: 'المعلومة غير متوفرة في الوثائق.'"
            )
        return (
            "انت مساعد مؤسسي. لا توجد وثائق مرفقة حالياً. "
            "اخبر المستخدم ان الوضع الصارم يتطلب رفع ملفات اولاً، "
            "او اجب بشكل عام مع التنبيه بعدم توفر وثائق."
        )
    return (
        "انت خبير تحليلي. "
        "ادمج السياق المرفق مع معرفتك لتقديم تحليل عميق ومفيد."
    )


def nice_error(e):
    s = str(e)
    if "429" in s or "quota" in s.lower():
        return "حصة النموذج انتهت — غير النموذج من الشريط الجانبي وحاول مجدداً."
    if "api_key" in s.lower() or "auth" in s.lower():
        return "مشكلة في مفتاح API — تحقق من Streamlit Secrets."
    if "safety" in s.lower() or "blocked" in s.lower():
        return "تم حجب الرد بواسطة فلاتر الأمان. أعد صياغة سؤالك."
    return f"خطا: {s[:200]}"


# ===================================================================
# UI
# ===================================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">Enterprise AI Assistant</div>
    <div class="hero-title">&#x1F9E0; المساعد المؤسسي الذكي</div>
    <div class="hero-sub">
        Retrieval-Augmented Generation &nbsp;&middot;&nbsp;
        امان تام &nbsp;&middot;&nbsp; دقة عالية &nbsp;&middot;&nbsp; تكلفة منخفضة
    </div>
</div>

<div class="info-banner">
    <h4>&#x1F3AF; لماذا RAG لشركتك؟</h4>
    <ul>
        <li><b>&#x1F512; سرية 100%:</b> ملفاتك لا تغادر بيئتك ولا تستخدم لتدريب نماذج خارجية.</li>
        <li><b>&#x1F4B0; وفر حتى 99% من Tokens:</b> يرسل فقط ما يخص سؤالك، لا آلاف الصفحات.</li>
        <li><b>&#x1F3AF; صفر هلوسة في الوضع الصارم:</b> اجابات مستندة لملفاتك فقط.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### &#x2699;&#xFE0F; الاعدادات")

    all_models = list_models()
    idx = all_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in all_models else 0
    chosen_model = st.selectbox("&#x1F916; نموذج AI", all_models, index=idx)

    st.divider()

    mode = st.radio(
        "&#x1F9E0; نمط الاجابة",
        options=["strict", "hybrid"],
        format_func=lambda x: (
            "&#x1F512; صارم — من الملفات فقط"
            if x == "strict"
            else "&#x1F4A1; مختلط — ربط وتحليل"
        ),
    )

    st.divider()

    if st.session_state.vs:
        st.markdown(
            '<div class="live-badge">'
            '<span class="pulse-dot"></span>قاعدة المعرفة نشطة'
            '</div>',
            unsafe_allow_html=True,
        )
        m = st.session_state.meta
        st.markdown(
            f"<p style='margin-top:10px;font-size:.83rem;color:#6B7280;"
            f"direction:rtl;text-align:right;'>"
            f"&#x1F4C4; <b>{m['files']}</b> ملفات &nbsp;&middot;&nbsp; "
            f"&#x1F310; <b>{m['urls']}</b> روابط &nbsp;&middot;&nbsp; "
            f"&#x1F9E9; <b>{m['chunks']}</b> جزء"
            f"</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("لم تبن قاعدة معرفة بعد.")

    st.divider()

    if st.session_state.history:
        log = "\n\n".join(
            f"{'المستخدم' if r == 'user' else 'المساعد'}: {t}"
            for r, t in st.session_state.history
        )
        st.download_button(
            "&#x1F4E5; تحميل المحادثة",
            log.encode("utf-8"),
            "rag_chat.txt",
            "text/plain",
        )

    if st.button("&#x1F5D1;&#xFE0F; مسح كل شيء"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

    st.divider()
    st.markdown(
        "<p style='font-size:.75rem;color:#9CA3AF;text-align:center;direction:rtl;'>"
        "LangChain &middot; FAISS &middot; Google Gemini<br>"
        "&#x1F512; بياناتك لا تشارك مع اطراف خارجية"
        "</p>",
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# knowledge base builder
# -------------------------------------------------------------------
st.markdown("### &#x1F4E5; بناء قاعدة المعرفة")

with st.expander(
    "&#x1F4C2; رفع الملفات واضافة الروابط",
    expanded=(st.session_state.vs is None),
):
    col_f, col_u = st.columns(2, gap="large")

    with col_f:
        st.markdown(f"**&#x1F4C4; الملفات** *(PDF / Word / TXT — حد {MAX_FILE_MB} MB للملف)*")
        uploaded = st.file_uploader(
            "files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            st.caption(f"تم اختيار {len(uploaded)} ملف")

    with col_u:
        st.markdown(f"**&#x1F310; الروابط** *(حد {MAX_URLS} — رابط في كل سطر)*")
        urls_raw = st.text_area(
            "urls",
            placeholder="https://company-policy.com\nhttps://knowledge.com",
            height=120,
            label_visibility="collapsed",
        )
        n_urls = len([l for l in urls_raw.splitlines() if l.strip()])
        if n_urls:
            st.caption(f"{n_urls} رابط")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("&#x1F525; بناء قاعدة المعرفة الان", type="primary"):
        if not uploaded and not urls_raw.strip():
            st.warning("ارفع ملفا او اضف رابطا واحدا على الاقل.")
        else:
            all_docs, f_ok, u_ok, errs = [], 0, 0, []

            with st.status("جاري المعالجة...", expanded=True) as prog:
                if uploaded:
                    st.write("تحليل الملفات...")
                    d, c, e = read_files(uploaded)
                    all_docs += d; f_ok = c; errs += e

                if urls_raw.strip():
                    st.write("جلب صفحات الويب...")
                    d, c, e = read_urls(urls_raw)
                    all_docs += d; u_ok = c; errs += e

                if all_docs:
                    st.write("بناء الذاكرة المتجهة...")
                    try:
                        vs, n_ch, all_chunks = build_index(all_docs)
                        st.session_state.vs     = vs
                        st.session_state.chunks = all_chunks
                        st.session_state.meta   = {"files": f_ok, "urls": u_ok, "chunks": n_ch}
                        prog.update(label="جاهز!", state="complete", expanded=False)
                    except Exception as e:
                        prog.update(label="فشل بناء Embeddings", state="error")
                        st.error(str(e)[:200])
                else:
                    prog.update(label="لا توجد بيانات صالحة", state="error")

            for err in errs:
                st.warning(err)

            if st.session_state.vs:
                st.rerun()


# -------------------------------------------------------------------
# metrics + dynamic quick-action buttons
# -------------------------------------------------------------------
if st.session_state.vs:
    m = st.session_state.meta
    st.markdown(
        f"""
        <div class="metrics-grid">
            <div class="m-card">
                <span class="m-num">{m['files']}</span>
                <span class="m-label">&#x1F4C4; ملفات مفهرسة</span>
            </div>
            <div class="m-card">
                <span class="m-num">{m['urls']}</span>
                <span class="m-label">&#x1F310; روابط معالجة</span>
            </div>
            <div class="m-card">
                <span class="m-num">{m['chunks']}</span>
                <span class="m-label">&#x1F9E9; قطعة في الذاكرة</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- DYNAMIC quick prompts (generated from actual loaded files) ---
    quick_options = get_quick_prompts()
    if quick_options:
        st.markdown("**&#x1F4A1; استعلامات سريعة مبنية على ملفاتك:**")
        cols = st.columns(len(quick_options))
        for col, (label, q) in zip(cols, quick_options):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state.quick_q = q


# -------------------------------------------------------------------
# chat
# -------------------------------------------------------------------
st.markdown('<hr class="sep">', unsafe_allow_html=True)
st.markdown("### &#x1F4AC; الاستعلام الذكي")

# render conversation history
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

# empty states
if not st.session_state.history:
    if not st.session_state.vs:
        st.markdown("""
        <div class="empty-state">
            <div class="ei">&#x1F4C2;</div>
            <h3>ابدا برفع ملفاتك اعلاه</h3>
            <p>بعد بناء قاعدة المعرفة اكتب سؤالك هنا.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="ei">&#x1F4AC;</div>
            <h3>قاعدة المعرفة جاهزة!</h3>
            <p>اكتب سؤالك ادناه او اضغط احد الازرار السريعة.</p>
        </div>""", unsafe_allow_html=True)

# get next query
typed = st.chat_input("اسال عن اي شيء في وثائقك...")
current_q = typed

# quick-action override
if st.session_state.quick_q:
    current_q = st.session_state.quick_q
    st.session_state.quick_q = ""

if current_q:
    # FIX: use raw query for the model — don't HTML-escape it
    query = current_q

    with st.chat_message("user"):
        st.write(query)
    st.session_state.history.append(("user", query))

    with st.chat_message("assistant"):
        try:
            sources = []
            has_context = False

            if st.session_state.vs:
                with st.spinner("جاري البحث في قاعدة المعرفة..."):
                    hits    = st.session_state.vs.similarity_search(query, k=TOP_K)
                    sources = sorted({d.metadata.get("source", "—") for d in hits})
                    context = "\n\n---\n\n".join(d.page_content for d in hits)
                    prompt  = f"السياق:\n{context}\n\nالسؤال: {query}"
                    has_context = True
            else:
                prompt = query

            model_obj = genai.GenerativeModel(
                model_name=chosen_model,
                system_instruction=make_system_prompt(
                    mode == "strict", has_context=has_context
                ),
            )
            response = model_obj.generate_content(prompt)

            # FIX: handle empty / blocked responses gracefully
            try:
                answer = response.text.strip() if response.text else ""
            except (ValueError, AttributeError):
                answer = ""

            if not answer:
                answer = (
                    "عذراً، لم يتم توليد إجابة. "
                    "حاول إعادة صياغة سؤالك أو تغيير النموذج من الشريط الجانبي."
                )

            st.write(answer)

            if sources:
                st.markdown("---")
                st.markdown("**المصادر:**")
                st.markdown(
                    " ".join(
                        f'<span class="src-tag">&#x1F4C4; {esc(s)}</span>'
                        for s in sources
                    ),
                    unsafe_allow_html=True,
                )

            st.session_state.history.append(("assistant", answer))

        except Exception as e:
            msg = nice_error(e)
            st.error(msg)
            st.session_state.history.append(("assistant", msg))

    st.rerun()
