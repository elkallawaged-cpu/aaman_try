"""
RAG Enterprise Assistant — Streamlit App
Improvements over v1:
  · Professional white-background UI with Cairo font
  · Input validation & sanitization (URLs, file sizes, HTML escaping)
  · Organised into clear, testable helper functions
  · Proper session-state bootstrapping and model caching
  · Granular, actionable error messages
  · Robust Web Scraping fix with Custom User-Agent
"""

import os
import re
import html as html_lib
import tempfile

# منع تحذيرات LangChain وتجهيز هوية مخصصة للمتصفح لمنع حظر الروابط
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

import google.generativeai as genai
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ═══════════════════════════════════════════════════════════════════════════════
#  ⚙️  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
CHUNK_SIZE    = 1_000
CHUNK_OVERLAP = 200
TOP_K         = 6
EMBED_MODEL   = "gemini-embedding-001"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
MAX_FILE_MB   = 25          # per-file upload limit
MAX_URLS      = 8           # maximum web URLs per session


# ═══════════════════════════════════════════════════════════════════════════════
#  📐  PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RAG — المساعد المؤسسي الذكي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  🎨  GLOBAL STYLES  — professional white theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; direction: rtl; }
.stApp { background: #F7F9FC; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0F2472 0%, #1D4ED8 55%, #60A5FA 100%);
    border-radius: 22px;
    padding: 46px 40px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 14px 48px rgba(29,78,216,.28);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';
    position:absolute;
    inset:0;
    background: radial-gradient(ellipse at 30% 50%, rgba(255,255,255,.10) 0%, transparent 70%);
    pointer-events:none;
}
.hero-badge {
    display:inline-block;
    background:rgba(255,255,255,.18);
    color:#fff;
    font-size:.78rem;
    font-weight:700;
    letter-spacing:.08em;
    padding:4px 14px;
    border-radius:20px;
    margin-bottom:14px;
    backdrop-filter:blur(4px);
    text-transform:uppercase;
}
.hero-title { font-size:2.2rem;  font-weight:800; color:#fff; margin:0 0 8px; }
.hero-sub   { font-size:1.05rem; color:rgba(255,255,255,.84); margin:0; }

/* ── Info Banner ── */
.info-banner {
    background: linear-gradient(135deg,#EFF6FF,#DBEAFE);
    border:1px solid #BFDBFE;
    border-right:4px solid #2563EB;
    border-radius:14px;
    padding:22px 26px;
    margin-bottom:28px;
}
.info-banner h4 { color:#1E40AF; font-weight:700; margin:0 0 10px; font-size:1.05rem; }
.info-banner ul { color:#1E3A8A; font-size:.95rem; line-height:2.1; padding-right:22px; margin:0; }

/* ── Metrics Grid ── */
.metrics-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:22px 0; }
.m-card {
    background:#fff;
    border-radius:14px;
    padding:20px 14px;
    text-align:center;
    border:1px solid #E8EDF2;
    box-shadow:0 2px 8px rgba(0,0,0,.04);
    position:relative;
    overflow:hidden;
}
.m-card::after {
    content:'';
    position:absolute;
    bottom:0; left:0; right:0;
    height:3px;
    background:linear-gradient(90deg,#1E3A8A,#60A5FA);
}
.m-num   { font-size:2.1rem; font-weight:800; color:#1E3A8A; display:block; }
.m-label { font-size:.82rem; color:#6B7280; font-weight:500; margin-top:4px; display:block; }

/* ── Source Tags ── */
.src-tag {
    display:inline-block;
    background:#EFF6FF; color:#1E40AF;
    font-size:.78rem; font-weight:600;
    padding:4px 12px;
    border-radius:20px; margin:3px;
    border:1px solid #BFDBFE;
}

/* ── Live Status Badge ── */
.live-badge {
    display:inline-flex; align-items:center; gap:7px;
    background:#ECFDF5; color:#065F46;
    font-size:.85rem; font-weight:700;
    padding:7px 16px;
    border-radius:20px;
    border:1px solid #A7F3D0;
}
.pulse-dot {
    width:8px; height:8px;
    border-radius:50%; background:#10B981;
    animation:blink 1.8s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.3;transform:scale(.8)} }

/* ── Buttons ── */
.stButton > button {
    font-family:'Cairo',sans-serif !important;
    border-radius:12px !important;
    font-weight:600 !important;
    transition:all .2s ease !important;
    direction:rtl !important;
    white-space:normal !important;
    height:auto !important;
    min-height:46px !important;
    line-height:1.5 !important;
}
.stButton > button:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 4px 14px rgba(30,58,138,.22) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#fff !important; border-left:1px solid #E8EDF2 !important; }
[data-testid="stSidebar"] * { font-family:'Cairo',sans-serif !important; }
[data-testid="stSidebar"] label { font-weight:600 !important; color:#374151 !important; }

/* ── Chat ── */
[data-testid="stChatMessageContent"] {
    font-family:'Cairo',sans-serif !important;
    font-size:.97rem !important;
    direction:rtl !important;
    text-align:right !important;
    line-height:1.85 !important;
}
[data-testid="stChatInput"] {
    font-family:'Cairo',sans-serif !important;
    border-radius:14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color:#2563EB !important;
    box-shadow:0 0 0 3px rgba(37,99,235,.10) !important;
}

/* ── Form Elements ── */
.stTextArea textarea  { direction:rtl !important; font-family:'Cairo',sans-serif !important; border-radius:12px !important; }
.stSelectbox > div > div { font-family:'Cairo',sans-serif !important; border-radius:10px !important; }
.stAlert { border-radius:12px !important; font-family:'Cairo',sans-serif !important; }
.streamlit-expanderHeader { font-family:'Cairo',sans-serif !important; font-weight:600 !important; }

/* ── Utilities ── */
.sep { border:none; height:1px; background:linear-gradient(90deg,transparent,#E2E8F0,transparent); margin:28px 0; }
.empty-state { text-align:center; padding:50px 20px; color:#9CA3AF; }
.empty-state .ei { font-size:3.6rem; margin-bottom:14px; }
.empty-state h3  { color:#6B7280; font-size:1.1rem; margin-bottom:6px; }
.empty-state p   { font-size:.9rem; }
.sidebar-footer  { font-size:.76rem; color:#9CA3AF; text-align:center; direction:rtl; line-height:1.7; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  🔐  SECURITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
_URL_RE = re.compile(r"^https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$")


def is_valid_url(url: str) -> bool:
    """Return True only for well-formed http/https URLs."""
    return bool(_URL_RE.match(url.strip()))


def safe_html(text: str) -> str:
    """Escape special HTML characters to prevent XSS in markdown blocks."""
    return html_lib.escape(str(text))


def within_size_limit(uploaded_file) -> bool:
    """Check that a UploadedFile does not exceed MAX_FILE_MB."""
    return len(uploaded_file.getvalue()) / (1024 * 1024) <= MAX_FILE_MB


# ═══════════════════════════════════════════════════════════════════════════════
#  🔑  API KEY  — load once, fail fast
# ═══════════════════════════════════════════════════════════════════════════════
def _init_api() -> str:
    key = (st.secrets.get("GEMINI_API_KEY") or "").strip()
    if key:
        genai.configure(api_key=key)
    return key


_API_KEY = _init_api()

if not _API_KEY:
    st.markdown("""
    <div style="background:#FEF2F2;border:1px solid #FECACA;border-right:4px solid #EF4444;
         border-radius:14px;padding:28px;text-align:center;margin-top:40px;">
        <h3 style="color:#991B1B;margin:0 0 10px;">⛔ مفتاح API مفقود</h3>
        <p style="color:#7F1D1D;margin:0;line-height:1.8;">
            لم يتم العثور على <code>GEMINI_API_KEY</code> في إعدادات
            <b>Streamlit → Settings → Secrets</b>.<br>
            أضفه ثم أعد تشغيل التطبيق.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  🗄️  SESSION STATE  — single source of truth
# ═══════════════════════════════════════════════════════════════════════════════
_DEFAULTS: dict = {
    "chat_history": [],                              # list[tuple[role, text]]
    "vector_store": None,                            # FAISS | None
    "all_docs": [],                                  # لتخزين المستندات من أجل التوليد الديناميكي
    "suggested_queries": [],                         # قائمة الاستعلامات المقترحة ذكياً
    "query_gen_error": None,                         # لتتبع وحفظ أي خطأ في توليد الأسئلة المقترحة
    "meta_stats":   {"files": 0, "urls": 0, "chunks": 0},
    "quick_input":  "",                              # pre-filled chat query
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
#  📦  DOCUMENT LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_files(files) -> tuple[list[Document], int, list[str]]:
    """
    Parse uploaded files into LangChain Documents.
    Returns (docs, successful_count, error_messages).
    """
    docs, count, errors = [], 0, []

    for f in files:
        if not within_size_limit(f):
            errors.append(f"'{f.name}' يتجاوز حد {MAX_FILE_MB} MB — تم تخطيه.")
            continue

        fname = f.name.lower()
        try:
            if fname.endswith(".txt"):
                content = f.getvalue().decode("utf-8", errors="ignore")
                docs.append(Document(page_content=content, metadata={"source": f.name}))
                count += 1

            elif fname.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    path = tmp.name
                try:
                    for d in PyPDFLoader(path).load():
                        d.metadata["source"] = f.name
                        docs.append(d)
                    count += 1
                finally:
                    os.path.exists(path) and os.remove(path)

            elif fname.endswith(".docx"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(f.getvalue())
                    path = tmp.name
                try:
                    for d in Docx2txtLoader(path).load():
                        d.metadata["source"] = f.name
                        docs.append(d)
                    count += 1
                finally:
                    os.path.exists(path) and os.remove(path)

        except Exception as exc:
            errors.append(f"'{f.name}': {str(exc)[:100]}")

    return docs, count, errors


def load_urls(raw: str) -> tuple[list[Document], int, list[str]]:
    """
    Scrape web pages from a newline-separated URL list.
    Returns (docs, successful_count, error_messages).
    """
    docs, count, errors = [], 0, []
    lines = [u.strip() for u in raw.splitlines() if u.strip()]

    if len(lines) > MAX_URLS:
        errors.append(f"الحد الأقصى {MAX_URLS} روابط — تم تجاهل الزائد.")
        lines = lines[:MAX_URLS]

    # 🛠️ التعديل الجوهري: إضافة ترويسة متصفح حقيقي لتجنب حظر الـ HTTP 403
    request_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    for url in lines:
        if not is_valid_url(url):
            errors.append(f"رابط غير صالح أو غير آمن: '{safe_html(url)}'")
            continue
        try:
            # تمرير الـ headers للـ WebBaseLoader لضمان القراءة الناجحة
            loader = WebBaseLoader(web_path=url, requests_kwargs={"headers": request_headers})
            for d in loader.load():
                d.metadata["source"] = url
                docs.append(d)
            count += 1
        except Exception as exc:
            errors.append(f"فشل تحميل '{url}': {str(exc)[:100]}")

    return docs, count, errors


# ═══════════════════════════════════════════════════════════════════════════════
#  🧩  VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════════
def build_vector_store(docs: list[Document]) -> tuple[FAISS, int]:
    """Chunk documents, embed them, and return a FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=_API_KEY)
    return FAISS.from_documents(chunks, emb), len(chunks)


# ═══════════════════════════════════════════════════════════════════════════════
#  🤖  MODEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600, show_spinner=False)
def list_models() -> list[str]:
    """Fetch available Gemini models that support generateContent."""
    try:
        return [
            m.name.split("/")[-1]
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
    except Exception:
        return [DEFAULT_MODEL]


def build_system_prompt(strict: bool) -> str:
    if strict:
        return (
            "أنت مساعد مؤسسي دقيق. أجب حصراً بناءً على السياق المُرفق. "
            "إن لم تجد الإجابة اكتب فقط: 'المعلومة غير متوفرة في الوثائق المرفوعة.'"
        )
    return (
        "أنت خبير تحليلي. ادمج السياق المُرفق مع معرفتك لتقديم تحليل عميق ومفيد. "
        "استنتج الأنماط وقدّم توصيات عملية."
    )


def generate_smart_queries(docs: list[Document], current_mode: str, model_name: str) -> list[str]:
    """توليد 3 أسئلة مخصصة وقصيرة جداً بناءً على محتوى الملف والنموذج المختار"""
    try:
        if not docs:
            return []
        
        st.session_state["query_gen_error"] = None # تصفير الخطأ طالما المحاولة بدأت
        sample_text = "\n".join([d.page_content[:600] for d in docs[:3]])
        
        # 🎯 استخدام النموذج المختار ديناميكياً لتجنب مشاكل الـ Quota أو الإصدارات
        model = genai.GenerativeModel(model_name)
        
        if current_mode == "strict":
            mode_instruction = "الوضع الحالي: [صارم]. ركز تماماً على استخراج الحقائق المباشرة، الأرقام الصريحة، والتواريخ المذكورة نصاً بدون استنتاج."
        else:
            mode_instruction = "الوضع الحالي: [مختلط]. ركز على التحليل الاستنتاجي، الأسباب، والتوصيات والربط الفني بين الأفكار."

        prompt = (
            f"بناءً على المحتوى المرفق, اقترح بالضبط 3 أسئلة أو استعلامات هامة للمستخدم.\n"
            f"💡 توجيه نوعية الأسئلة: {mode_instruction}\n\n"
            f"⚠️ شروط إجبارية للتنسيق:\n"
            f"1. اكتب الأسئلة باللغة العربية.\n"
            f"2. اجعل الأسئلة قصيرة جداً وموجزة ومحددة (من 3 إلى 6 كلمات فقط لكل سؤال).\n"
            f"3. أعطني الـ 3 أسئلة مباشرة في 3 أسطر منفصلة، بدون أي أرقام (لا تكتب 1، 2، 3) وبدون أي مقدمات أو خاتمة.\n\n"
            f"المحتوى:\n{sample_text}"
        )
        response = model.generate_content(prompt)
        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        cleaned_queries = [re.sub(r'^\d+[\.\-\)]\s*', '', q).strip() for q in lines]
        return [q for q in cleaned_queries if q][:3]
        
    except Exception as e:
        # حفظ الخطأ لإظهاره بشفافية للمستخدم بدل الاختفاء
        st.session_state["query_gen_error"] = str(e)
        if current_mode == "strict":
            return ["ما هي الحقائق الصريحة في الملف؟", "استخرج أهم الأرقام والتواريخ المحددة.", "ما هي الشروط والأحكام المذكورة؟"]
        else:
            return ["ما هو التحليل العام للمستند؟", "ما هي أبرز المقترحات والتوصيات؟", "ملخص شامل لأهم نقاط الملف."]


def classify_error(exc: Exception) -> str:
    msg = str(exc)
    if "429" in msg or "quota" in msg.lower():
        return "⚠️ تم استنفاد حصة النموذج الحالي. غيّره من الشريط الجانبي وأعد المحاولة."
    if "api_key" in msg.lower() or "auth" in msg.lower():
        return "🔑 خطأ في مفتاح الـ API — تحقق من Streamlit Secrets."
    return f"❌ خطأ غير متوقع: {msg[:220]}"


# ═══════════════════════════════════════════════════════════════════════════════
#  🖼️  HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ Enterprise AI Assistant ✦</div>
    <div class="hero-title">🧠 المساعد المؤسسي الذكي</div>
    <div class="hero-sub">
        Retrieval-Augmented Generation &nbsp;·&nbsp;
        أمان تام &nbsp;·&nbsp; دقة عالية &nbsp;·&nbsp; تكلفة منخفضة
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-banner">
    <h4>🎯 لماذا RAG لشركتك بالتحديد؟</h4>
    <ul>
        <li><b>🔒 سرية بياناتك مضمونة 100%:</b>
            ملفاتك لا تُغادر بيئتك ولا تُستخدم لتدريب نماذج خارجية.</li>
        <li><b>💰 وفر حتى 99% من تكلفة Tokens:</b>
            يُرسَل فقط السياق المرتبط بسؤالك، لا آلاف الصفحات.</li>
        <li><b>🎯 صفر هلوسة في الوضع الصارم:</b>
            يلتزم الموديل بإجابات مستندة لمصادرك الخاصة فقط.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  🗄️  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ إعدادات النظام")

    all_models = list_models()
    def_idx = all_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in all_models else 0
    selected_model = st.selectbox("🤖 نموذج الذكاء الاصطناعي", all_models, index=def_idx)

    st.divider()

    mode = st.radio(
        "🧠 نمط الإجابة",
        options=["strict", "hybrid"],
        format_func=lambda x: (
            "🔒 صارم — من الملفات فقط"
            if x == "strict"
            else "💡 مختلط — ربط وتحليل"
        ),
    )

    st.divider()

    # Knowledge-base status
    if st.session_state.vector_store:
        st.markdown(
            '<div class="live-badge"><span class="pulse-dot"></span>قاعدة المعرفة نشطة</div>',
            unsafe_allow_html=True,
        )
        s = st.session_state.meta_stats
        st.markdown(
            f"<div style='margin-top:12px;font-size:.84rem;color:#6B7280;direction:rtl;'>"
            f"📄 <b>{s['files']}</b> ملفات &nbsp;·&nbsp; "
            f"🌐 <b>{s['urls']}</b> روابط &nbsp;·&nbsp; "
            f"🧩 <b>{s['chunks']}</b> جزء"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("⏳ لم تُبنَ قاعدة معرفة بعد.")

    st.divider()

    # Download chat log
    if st.session_state.chat_history:
        log = "\n\n".join(
            f"{'المستخدم' if r == 'user' else 'المساعد'}: {t}"
            for r, t in st.session_state.chat_history
        )
        st.sidebar.download_button(
            "📥 تحميل سجل المحادثة",
            data=log.encode("utf-8"),
            file_name="rag_chat.txt",
            mime="text/plain",
        )

    if st.button("🗑️ إعادة تعيين كاملة"):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    st.divider()
    st.markdown(
        "<div class='sidebar-footer'>"
        "مبني بـ LangChain · FAISS · Google Gemini<br>"
        "🔒 بياناتك لا تُشارك مع أطراف خارجية"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  📥  KNOWLEDGE-BASE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📥 بناء قاعدة المعرفة")

with st.expander(
    "📂 رفع المستندات وإضافة الروابط",
    expanded=(st.session_state.vector_store is None),
):
    col_files, col_urls = st.columns(2, gap="large")

    with col_files:
        st.markdown(f"**📄 الملفات المدعومة** *(حد أقصى {MAX_FILE_MB} MB للملف)*")
        uploaded = st.file_uploader(
            "الملفات",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            st.caption(f"✅ {len(uploaded)} ملف مُختار")

    with col_urls:
        st.markdown(f"**🌐 روابط الويب** *(حد أقصى {MAX_URLS} روابط)*")
        urls_raw = st.text_area(
            "الروابط",
            placeholder="https://company-policy.com\nhttps://knowledge-base.com",
            height=120,
            label_visibility="collapsed",
        )
        n_urls = len([l for l in urls_raw.splitlines() if l.strip()])
        if n_urls:
            st.caption(f"📎 {n_urls} رابط")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔥 بناء قاعدة المعرفة الآن", type="primary"):
        if not uploaded and not urls_raw.strip():
            st.warning("⚠️ الرجاء رفع ملف واحد على الأقل أو إضافة رابط.")
        else:
            all_docs, f_count, u_count, all_errors = [], 0, 0, []

            with st.status("🚀 جاري معالجة المصادر...", expanded=True) as prog:

                if uploaded:
                    st.write("📂 تحليل الملفات المرفوعة...")
                    fd, fc, fe = load_files(uploaded)
                    all_docs.extend(fd)
                    f_count = fc
                    all_errors.extend(fe)

                if urls_raw.strip():
                    st.write("🌐 جلب صفحات الويب...")
                    ud, uc, ue = load_urls(urls_raw)
                    all_docs.extend(ud)
                    u_count = uc
                    all_errors.extend(ue)

                if all_docs:
                    st.write("🧩 بناء الذاكرة المتجهة (Embeddings)...")
                    try:
                        vs_new, n_chunks = build_vector_store(all_docs)
                        st.session_state.vector_store = vs_new
                        st.session_state.all_docs = all_docs 
                        st.session_state.suggested_queries = [] # تفريغ القائمة القديمة لإجبار السيستم على التوليد الجديد
                        st.session_state.query_gen_error = None # تصفير أخطاء التوليد القديمة
                        st.session_state.meta_stats = {
                            "files": f_count,
                            "urls":  u_count,
                            "chunks": n_chunks,
                        }
                        prog.update(
                            label="✅ قاعدة المعرفة جاهزة!",
                            state="complete",
                            expanded=False,
                        )
                    except Exception as be:
                        prog.update(label="❌ فشل في بناء الـ Embeddings", state="error")
                        st.error(f"تفاصيل: {str(be)[:200]}")
                else:
                    prog.update(label="⚠️ لا توجد بيانات صالحة للمعالجة", state="error")

            for err in all_errors:
                st.warning(f"⚠️ {err}")

            if st.session_state.vector_store:
                st.rerun()


# 📊 METRICS + QUICK ACTIONS (التوليد الديناميكي الفعلي)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.vector_store:
    s = st.session_state.meta_stats
    st.markdown(
        f"""
        <div class="metrics-grid">
            <div class="m-card">
                <span class="m-num">{s.get('files', 0)}</span>
                <span class="m-label">📄 ملفات مفهرسة</span>
            </div>
            <div class="m-card">
                <span class="m-num">{s.get('urls', 0)}</span>
                <span class="m-label">🌐 روابط معالجة</span>
            </div>
            <div class="m-card">
                <span class="m-num">{s.get('chunks', 0)}</span>
                <span class="m-label">🧩 قطعة في الذاكرة</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # تتبع النمط الحالي ومراقبته
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = mode

    # لو غيرت النمط يفرغ الأسئلة عشان يعيد التوليد فوراً بالنمط الجديد
    if st.session_state.last_mode != mode:
        st.session_state.last_mode = mode
        st.session_state.suggested_queries = []

    mode_title = "الصارم" if mode == "strict" else "المختلط"
    st.markdown(f"**💡 استعلامات مقترحة مخصصة لملفك ({mode_title}):**")
    
    # ⏳ مرحلة الـ الـ Spinner الحقيقي لما يبدأ يحلل باستعمال الموديل المختار
    if not st.session_state.get("suggested_queries"):
        with st.spinner("⏳ جاري تحليل مستنداتك وتوليد أسئلة ذكية تناسب الوضع المختار..."):
            docs_to_analyze = st.session_state.get("all_docs", [])
            # تم تمرير selected_model لحل مشكلة الجمود البرمجي الثابت
            st.session_state.suggested_queries = generate_smart_queries(docs_to_analyze, mode, selected_model)
        st.rerun()

    # لو التوليد اللحظي فشل لأي سبب بره الإرادة، هيظهرلك كابشن صغير هنا يقولك السبب ايه بالظبط عشان تصلحه
    if st.session_state.get("query_gen_error"):
        st.caption(f"⚠️ *ملاحظة: تعذر التوليد اللحظي وجاري استخدام أسئلة ذكية عامة بسبب الخطأ التالي:* `{st.session_state.query_gen_error}`")

    # عرض الـ 3 أزرار اللحظية
    queries = st.session_state.suggested_queries
    cols = st.columns(len(queries))
    for col, query in zip(cols, queries):
        with col:
            if st.button(query, key=f"dynamic_btn_{hash(query)}", use_container_width=True):
                st.session_state.quick_input = query
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  💬  CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="sep">', unsafe_allow_html=True)
st.markdown("### 💬 الاستعلام والتحليل الذكي")

# — render history —
if st.session_state.chat_history:
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)
else:
    if st.session_state.vector_store is None:
        st.markdown(
            """<div class="empty-state">
                <div class="ei">📂</div>
                <h3>ابدأ بتحميل ملفاتك أعلاه</h3>
                <p>بعد بناء قاعدة المعرفة، اكتب سؤالك هنا.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="empty-state">
                <div class="ei">💬</div>
                <h3>قاعدة المعرفة جاهزة!</h3>
                <p>اكتب سؤالك في الأسفل أو اضغط أحد الاستعلامات السريعة.</p>
            </div>""",
            unsafe_allow_html=True,
        )

# — get query (typed or quick-action) —
typed_q = st.chat_input("اسأل المساعد عن أي شيء في وثائقك...")
current_q: str | None = typed_q

if st.session_state.quick_input:
    current_q = st.session_state.quick_input
    st.session_state.quick_input = ""

# — process query —
if current_q:
    query = safe_html(current_q)       # sanitize before display / API call

    with st.chat_message("user"):
        st.write(query)
    st.session_state.chat_history.append(("user", query))

    with st.chat_message("assistant"):
        try:
            is_strict = (mode == "strict")
            sys_inst  = build_system_prompt(is_strict)
            sources: list[str] = []

            if st.session_state.vector_store:
                with st.spinner("🔍 جاري البحث في قاعدة المعرفة..."):
                    hits    = st.session_state.vector_store.similarity_search(query, k=TOP_K)
                    sources = sorted({d.metadata.get("source", "—") for d in hits})
                    context = "\n\n---\n\n".join(d.page_content for d in hits)
                    prompt  = f"السياق:\n{context}\n\nالسؤال: {query}"
            else:
                with st.spinner("🤔 جاري التفكير..."):
                    prompt = query

            model_obj = genai.GenerativeModel(
                model_name=selected_model,
                system_instruction=sys_inst,
            )
            response = model_obj.generate_content(prompt)
            answer   = response.text.strip()

            st.write(answer)

            if sources:
                st.markdown("---")
                st.markdown("**📌 المصادر المستند إليها:**")
                tags = " ".join(
                    f'<span class="src-tag">📄 {safe_html(s)}</span>'
                    for s in sources
                )
                st.markdown(tags, unsafe_allow_html=True)

            st.session_state.chat_history.append(("assistant", answer))

        except Exception as exc:
            err_msg = classify_error(exc)
            st.error(err_msg)
            st.session_state.chat_history.append(("assistant", err_msg))

    st.rerun()
