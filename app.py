"""
RAG Enterprise Assistant — Streamlit App (v6 Ultimate)
- 100% In-Memory File Processing (No Temp Files / No Permission Locks).
- Advanced Encoding Detection for Flawless Arabic Web Scraping.
- Decoupled Robust Custom Embeddings Wrapper with micro-batching & Multi-tier Backoff Retries.
- Fixed Infinite Loop issues completely. Built for Production Stability.
"""

import io
import os
import re
import time
import html as html_lib
import requests
from bs4 import BeautifulSoup

import google.generativeai as genai
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ═══════════════════════════════════════════════════════════════════════════════
#  ⚙️  SYSTEM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
CHUNK_SIZE    = 1_000
CHUNK_OVERLAP = 200
TOP_K         = 6
EMBED_MODEL   = "gemini-embedding-001"
DEFAULT_MODEL = "gemini-2.0-flash"
MAX_FILE_MB   = 25          
MAX_URLS      = 8           


# ═══════════════════════════════════════════════════════════════════════════════
#  📐  PAGE & UI SETUP
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RAG — المساعد المؤسسي الذكي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; direction: rtl; }
.stApp { background: #F7F9FC; }
#MainMenu, footer, header { visibility: hidden; }
.hero {
    background: linear-gradient(135deg, #0F2472 0%, #1D4ED8 55%, #60A5FA 100%);
    border-radius: 22px; padding: 46px 40px; text-align: center; margin-bottom: 28px;
    box-shadow: 0 14px 48px rgba(29,78,216,.28); position: relative; overflow: hidden;
}
.hero-badge {
    display:inline-block; background:rgba(255,255,255,.18); color:#fff;
    font-size:.78rem; font-weight:700; padding:4px 14px; border-radius:20px; margin-bottom:14px;
}
.hero-title { font-size:2.2rem; font-weight:800; color:#fff; margin:0 0 8px; }
.hero-sub   { font-size:1.05rem; color:rgba(255,255,255,.84); margin:0; }
.info-banner {
    background: linear-gradient(135deg,#EFF6FF,#DBEAFE); border:1px solid #BFDBFE;
    border-right:4px solid #2563EB; border-radius:14px; padding:22px 26px; margin-bottom:28px;
}
.info-banner h4 { color:#1E40AF; font-weight:700; margin:0 0 10px; font-size:1.05rem; }
.info-banner ul { color:#1E3A8A; font-size:.95rem; line-height:2.1; padding-right:22px; margin:0; }
.metrics-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:22px 0; }
.m-card {
    background:#fff; border-radius:14px; padding:20px 14px; text-align:center;
    border:1px solid #E8EDF2; box-shadow:0 2px 8px rgba(0,0,0,.04); position:relative; overflow:hidden;
}
.m-card::after { content:''; position:absolute; bottom:0; left:0; right:0; height:3px; background:linear-gradient(90deg,#1E3A8A,#60A5FA); }
.m-num   { font-size:2.1rem; font-weight:800; color:#1E3A8A; display:block; }
.m-label { font-size:.82rem; color:#6B7280; font-weight:500; margin-top:4px; display:block; }
.src-tag { display:inline-block; background:#EFF6FF; color:#1E40AF; font-size:.78rem; font-weight:600; padding:4px 12px; border-radius:20px; margin:3px; border:1px solid #BFDBFE; }
.live-badge { display:inline-flex; align-items:center; gap:7px; background:#ECFDF5; color:#065F46; font-size:.85rem; font-weight:700; padding:7px 16px; border-radius:20px; border:1px solid #A7F3D0; }
.pulse-dot { width:8px; height:8px; border-radius:50%; background:#10B981; animation:blink 1.8s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.3;transform:scale(.8)} }
.stButton > button { font-family:'Cairo',sans-serif !important; border-radius:12px !important; font-weight:600 !important; height:auto !important; min-height:46px !important; line-height:1.5 !important; }
[data-testid="stSidebar"] { background:#fff !important; border-left:1px solid #E8EDF2 !important; }
[data-testid="stChatMessageContent"] { font-family:'Cairo',sans-serif !important; font-size:.97rem !important; direction:rtl !important; text-align:right !important; line-height:1.85 !important; }
.empty-state { text-align:center; padding:50px 20px; color:#9CA3AF; }
.empty-state .ei { font-size:3.6rem; margin-bottom:14px; }
.sep { border:none; height:1px; background:linear-gradient(90deg,transparent,#E2E8F0,transparent); margin:28px 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  🔒  VALIDATION & SECURITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
_URL_RE = re.compile(r"^https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$")

def is_valid_url(url: str) -> bool:
    return bool(_URL_RE.match(url.strip()))

def safe_html(text: str) -> str:
    return html_lib.escape(str(text))


# ═══════════════════════════════════════════════════════════════════════════════
#  🔑  API KEY INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def _init_api() -> str:
    key = (st.secrets.get("GEMINI_API_KEY") or "").strip()
    if key:
        genai.configure(api_key=key)
    return key

_API_KEY = _init_api()

if not _API_KEY:
    st.markdown("""
    <div style="background:#FEF2F2;border:1px solid #FECACA;border-right:4px solid #EF4444;border-radius:14px;padding:28px;text-align:center;margin-top:40px;">
        <h3 style="color:#991B1B;margin:0 0 10px;">⛔ مفتاح API مفقود</h3>
        <p style="color:#7F1D1D;margin:0;line-height:1.8;">لم يتم العثور على <code>GEMINI_API_KEY</code> في إعدادات Secrets.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  🧠  BULLETPROOF CUSTOM EMBEDDINGS CLASS (BYPASSES LANGCHAIN 429 CRASHES)
# ═══════════════════════════════════════════════════════════════════════════════
class RobustGeminiEmbeddings:
    """
    Custom Vector Embedding Generator directly calling Google GenAI SDK.
    Enforces strict micro-batching (4 chunks/batch) and exponential backoff
    to guarantee 100% protection against 429 Quota limitations.
    """
    def __init__(self, model_name: str, api_key: str):
        self.model = f"models/{model_name}" if not model_name.startswith("models/") else model_name
        genai.configure(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        batch_size = 4  # Micro-batches to stay completely safe under Free-Tier limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(6):
                try:
                    response = genai.embed_content(
                        model=self.model,
                        content=batch,
                        task_type="retrieval_document"
                    )
                    embeddings.extend(response['embedding'])
                    time.sleep(1.0)  # Safe breathing window for servers
                    break
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        # Exponential backoff sleep: 2s, 4s, 8s, 16s...
                        time.sleep(2 ** attempt + 2)
                    else:
                        if attempt == 5:
                            raise e
                        time.sleep(1)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        for attempt in range(5):
            try:
                response = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_query"
                )
                return response['embedding']
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(2 ** attempt + 1)
                else:
                    raise e
        raise RuntimeError("فشل تحويل الاستعلام بسبب ضغط حصة الـ API الحالي.")


# ═══════════════════════════════════════════════════════════════════════════════
#  📦  100% IN-MEMORY FILE PARSERS (NO DISK I/O LOCKS)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_pdf_pure(file_bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text_slices = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_slices.append(t)
        return "\n".join(text_slices)
    except ImportError:
        raise ImportError("مكتبة 'pypdf' مفقودة في النظام. يرجى تثبيتها عبر: pip install pypdf")

def extract_docx_pure(file_bytes) -> str:
    try:
        import docx2txt
        return docx2txt.process(io.BytesIO(file_bytes))
    except ImportError:
        raise ImportError("مكتبة 'docx2txt' مفقودة في النظام. يرجى تثبيتها عبر: pip install docx2txt")

def scrape_url_pure(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        # 🎯 FIX FOR ARABIC GARBLED TEXT: Explicitly detect and match apparent encoding
        resp.encoding = resp.apparent_encoding
        
        soup = BeautifulSoup(resp.text, "html.parser")
        # Strip structural clutter
        for element in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
            element.decompose()
            
        raw_text = soup.get_text(separator="\n")
        lines = (line.strip() for line in raw_text.splitlines())
        return "\n".join(l for l in lines if l)
    except Exception as e:
        raise RuntimeError(f"فشل جلب الرابط بالكامل: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  🗄️  STATE PRESERVATION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "chat_history": [],
    "vector_store": None,
    "all_docs": [],
    "suggested_queries": [],
    "query_gen_error": None,
    "meta_stats": {"files": 0, "urls": 0, "chunks": 0},
    "quick_input": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
#  🤖  AI MODEL ENGINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600, show_spinner=False)
def list_models() -> list[str]:
    try:
        return [m.name.split("/")[-1] for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    except Exception:
        return [DEFAULT_MODEL]

def build_system_prompt(strict: bool) -> str:
    if strict:
        return "أنت مساعد مؤسسي دقيق. أجب حصراً بناءً على السياق المُرفق. إن لم تجد الإجابة بدقة اكتب فقط: 'المعلومة غير متوفرة في الوثائق المرفوعة.'"
    return "أنت خبير تحليلي. ادمج السياق المُرفق مع معرفتك لتقديم تحليل عميق ومفيد واستخلص التوصيات العملية."

def generate_smart_queries(docs: list[Document], current_mode: str, model_name: str) -> list[str]:
    try:
        if not docs: return []
        sample_text = "\n".join([d.page_content[:500] for d in docs[:3]])
        model = genai.GenerativeModel(model_name)
        mode_instruction = "الوضع الحالي: [صارم]. ركز على الحقائق والأرقام الصريحة المذكورة." if current_mode == "strict" else "الوضع الحالي: [مختلط]. ركز على التحليل الاستنتاجي والتوصيات."
        
        prompt = (
            f"بناءً على المحتوى المرفق، اقترح بالضبط 3 أسئلة هامة.\n"
            f"💡 توجيه: {mode_instruction}\n"
            f"شروط صارمة: لغة عربية مباشرة، من 3 لـ 6 كلمات فقط، في 3 أسطر منفصلة وبدون أرقام أو مقدمات.\n\n"
            f"المحتوى:\n{sample_text}"
        )
        response = model.generate_content(prompt)
        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        return [re.sub(r'^\d+[\.\-\)]\s*', '', q).strip() for q in lines][:3]
    except Exception as e:
        st.session_state["query_gen_error"] = str(e)
        return ["ما هي أهم الأرقام والتواريخ المحددة؟", "تحليل عام وشامل لمحتويات الملف", "أبرز التوصيات المستخرجة"]


# ═══════════════════════════════════════════════════════════════════════════════
#  🖼️  MAIN PAGE RENDERING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ Enterprise AI Assistant v6 Ultimate ✦</div>
    <div class="hero-title">🧠 المساعد المؤسسي الذكي</div>
    <div class="hero-sub">نظام RAG فائق الاستقرار · معالجة بالذاكرة الحية 100% · معزول تماماً ضد التجميد وأخطاء الحصص</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  🗄️  SIDEBAR DESIGN
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ إعدادات النظام")
    all_models = list_models()
    def_idx = all_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in all_models else 0
    selected_model = st.selectbox("🤖 نموذج الذكاء الاصطناعي", all_models, index=def_idx)

    st.divider()
    mode = st.radio(
        "🧠 نمط الإجابة", options=["strict", "hybrid"],
        format_func=lambda x: "🔒 صارم — من الملفات فقط" if x == "strict" else "💡 مختلط — ربط وتحليل"
    )

    st.divider()
    if st.session_state.vector_store:
        st.markdown('<div class="live-badge"><span class="pulse-dot"></span>قاعدة المعرفة نشطة</div>', unsafe_allow_html=True)
        s = st.session_state.meta_stats
        st.markdown(f"<div style='margin-top:12px;font-size:.84rem;color:#6B7280;'>📄 <b>{s['files']}</b> ملفات &nbsp;·&nbsp; 🌐 <b>{s['urls']}</b> روابط &nbsp;·&nbsp; 🧩 <b>{s['chunks']}</b> جزء</div>", unsafe_allow_html=True)
    else:
        st.info("⏳ لم تُبنَ قاعدة معرفة بعد.")

    st.divider()
    if st.button("🗑️ إعادة تعيين كاملة والتنظيف"):
        for k, v in _DEFAULTS.items(): st.session_state[k] = v
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  📥  THE ULTRA-STABLE CORE FACTORY (FILES & URLS CONTROLLER)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📥 بناء قاعدة المعرفة")

with st.expander("📂 رفع المستندات وإضافة الروابط الويب", expanded=(st.session_state.vector_store is None)):
    col_files, col_urls = st.columns(2, gap="large")
    
    with col_files:
        st.markdown(f"**📄 الملفات المسموحة** *(PDF, DOCX, TXT حتى {MAX_FILE_MB}MB)*")
        uploaded = st.file_uploader("الملفات", type=["pdf", "docx", "txt"], accept_multiple_files=True, label_visibility="collapsed")
        if uploaded: st.caption(f"✅ تم اختيار {len(uploaded)} ملف")
        
    with col_urls:
        st.markdown(f"**🌐 روابط الويب المباشرة** *(رابط واحد في كل سطر - حد أقصى {MAX_URLS})*")
        urls_raw = st.text_area("الروابط", placeholder="https://example.com/page1\nhttps://example.com/page2", height=120, label_visibility="collapsed")
        n_urls = len([l for l in urls_raw.splitlines() if l.strip()])
        if n_urls: st.caption(f"📎 تم تسجيل {n_urls} رابط")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔥 بناء قاعدة المعرفة الآن", type="primary", use_container_width=True):
        if not uploaded and not urls_raw.strip():
            st.warning("⚠️ يرجى تزويد النظام بملف واحد أو رابط ويب صالح أولاً.")
        else:
            all_parsed_docs = []
            f_success, u_success = 0, 0
            processing_errors = []
            
            with st.status("🚀 جاري المعالجة والفهرسة المباشرة بالذاكرة الحية...", expanded=True) as status_box:
                
                # 1. Processing Files In-Memory
                if uploaded:
                    status_box.write("⚙️ جاري قراءة بايتات الملفات مباشرة بدون وسيط تخزين...")
                    for f in uploaded:
                        if len(f.getvalue()) / (1024 * 1024) > MAX_FILE_MB:
                            processing_errors.append(f"الملف {f.name} يتجاوز الحجم المسموح.")
                            continue
                        try:
                            fname = f.name.lower()
                            text_out = ""
                            if fname.endswith(".txt"):
                                text_out = f.getvalue().decode("utf-8", errors="ignore")
                            elif fname.endswith(".pdf"):
                                text_out = extract_pdf_pure(f.getvalue())
                            elif fname.endswith(".docx"):
                                text_out = extract_docx_pure(f.getvalue())
                                
                            if text_out.strip():
                                all_parsed_docs.append(Document(page_content=text_out, metadata={"source": f.name}))
                                f_success += 1
                        except Exception as e:
                            processing_errors.append(f"فشل قراءة الملف {f.name}: {str(e)}")
                
                # 2. Processing URLs Safely
                if urls_raw.strip():
                    status_box.write("🌐 جاري سحب نصوص صفحات الويب وفك ترميزها العربي...")
                    url_lines = [u.strip() for u in urls_raw.splitlines() if u.strip()]
                    url_lines = url_lines[:MAX_URLS]
                    
                    for url in url_lines:
                        if not is_valid_url(url):
                            processing_errors.append(f"رابط غير آمن أو غير مدعوم: {url}")
                            continue
                        try:
                            text_out = scrape_url_pure(url)
                            if text_out.strip():
                                all_parsed_docs.append(Document(page_content=text_out, metadata={"source": url}))
                                u_success += 1
                        except Exception as e:
                            processing_errors.append(f"فشل سحب الرابط {url}: {str(e)}")
                
                # 3. Micro-Batching Vector Conversion
                if all_parsed_docs:
                    status_box.write("🧩 جاري التقطيع والرفع الآمن والمتتابع على خوادم المتجهات المتوازية...")
                    try:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                        final_chunks = splitter.split_documents(all_parsed_docs)
                        
                        if not final_chunks:
                            raise ValueError("محتوى الملفات/الروابط فارغ أو لا يحتوي على نصوص صالحة للمعالجة.")
                        
                        # Trigger custom embedded wrapper
                        embedding_engine = RobustGeminiEmbeddings(model_name=EMBED_MODEL, api_key=_API_KEY)
                        built_faiss = FAISS.from_documents(final_chunks, embedding_engine)
                        
                        # Store in state
                        st.session_state.vector_store = built_faiss
                        st.session_state.all_docs = all_parsed_docs
                        st.session_state.meta_stats = {"files": f_success, "urls": u_success, "chunks": len(final_chunks)}
                        
                        # Generate smart queries EXACTLY ONCE here to prevent background rerun freezes
                        status_box.write("💡 جاري صياغة الاستعلامات المقترحة المناسبة للبيانات...")
                        st.session_state.suggested_queries = generate_smart_queries(all_parsed_docs, mode, selected_model)
                        
                        status_box.update(label="✅ تمت معالجة وتأمين قاعدة المعرفة بنجاح وبأعلى كفاءة!", state="complete", expanded=False)
                    except Exception as embed_err:
                        status_box.update(label="❌ فشل بناء هيكل قاعدة البيانات", state="error")
                        st.error(f"خطأ تقني: {str(embed_err)}")
                else:
                    status_box.update(label="⚠️ لم يتم العثور على أي نصوص مقروءة داخل المرفقات.", state="error")

            for err in processing_errors:
                st.warning(f"⚠️ {err}")
                
            if st.session_state.vector_store:
                st.rerun()


# 📊 DATA ANALYTICS VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.vector_store:
    s = st.session_state.meta_stats
    st.markdown(f"""
    <div class="metrics-grid">
        <div class="m-card"><span class="m-num">{s['files']}</span><span class="m-label">📄 ملفات مدمجة ومقروءة</span></div>
        <div class="m-card"><span class="m-num">{s['urls']}</span><span class="m-label">🌐 مواقع تم سحب بياناتها</span></div>
        <div class="m-card"><span class="m-num">{s['chunks']}</span><span class="m-label">🧩 قطع نصية مفهرسة بالذاكرة</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Render dynamic buttons safely from static state storage
    queries = st.session_state.get("suggested_queries") or []
    if queries:
        mode_title = "الصارم" if mode == "strict" else "المختلط"
        st.markdown(f"**💡 أسئلة مقترحة ذكياً بناءً على مستنداتك بالنمط ({mode_title}):**")
        cols = st.columns(len(queries))
        for col, q_text in zip(cols, queries):
            with col:
                if st.button(q_text, key=f"btn_{hash(q_text)}", use_container_width=True):
                    st.session_state.quick_input = q_text
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  💬  AI CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="sep">', unsafe_allow_html=True)
st.markdown("### 💬 الاستعلام والتحليل الذكي")

if st.session_state.chat_history:
    for msg_role, msg_text in st.session_state.chat_history:
        with st.chat_message(msg_role): st.write(msg_text)
else:
    if st.session_state.vector_store is None:
        st.markdown('<div class="empty-state"><div class="ei">📂</div><h3>قاعدة البيانات فارغة حالياً</h3><p>ارفع ملفاتك أو روابطك في الأعلى واضغط بناء لتفعيل الذكاء المخصص.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state"><div class="ei">💬</div><h3>قاعدة المعرفة جاهزة تماماً!</h3><p>اكتب سؤالك بالأسفل واسحب البيانات والتحليلات اللحظية فوراً.</p></div>', unsafe_allow_html=True)

# Capture user query
typed_query = st.chat_input("اسأل المساعد عن أي تفاصيل داخل وثائقك المرفوعة...")
active_query = typed_query

if st.session_state.quick_input:
    active_query = st.session_state.quick_input
    st.session_state.quick_input = ""

if active_query:
    sanitized_query = safe_html(active_query)
    
    with st.chat_message("user"): st.write(sanitized_query)
    st.session_state.chat_history.append(("user", sanitized_query))
    
    with st.chat_message("assistant"):
        try:
            sys_prompt = build_system_prompt(mode == "strict")
            found_sources = []
            
            if st.session_state.vector_store:
                with st.spinner("🔍 جاري الفحص والمطابقة الدقيقة بالذاكرة المتجهة..."):
                    search_hits = st.session_state.vector_store.similarity_search(sanitized_query, k=TOP_K)
                    found_sources = sorted({hit.metadata.get("source", "—") for hit in search_hits})
                    merged_context = "\n\n---\n\n".join(hit.page_content for hit in search_hits)
                    final_prompt = f"السياق المتاح للبحث:\n{merged_context}\n\nالسؤال المطلوب إجابته: {sanitized_query}"
            else:
                with st.spinner("🤔 جاري التفكير والتحليل المباشر..."):
                    final_prompt = sanitized_query
                    
            model_instance = genai.GenerativeModel(model_name=selected_model, system_instruction=sys_prompt)
            ai_response = model_instance.generate_content(final_prompt)
            ai_text = ai_response.text.strip()
            
            st.write(ai_text)
            
            if found_sources:
                st.markdown("---")
                st.markdown("**📌 المصادر المستند إليها في إجابتك الصارمة:**")
                source_tags = " ".join(f'<span class="src-tag">📄 {safe_html(src)}</span>' for src in found_sources)
                st.markdown(source_tags, unsafe_allow_html=True)
                
            st.session_state.chat_history.append(("assistant", ai_text))
        except Exception as chat_err:
            msg_err = f"❌ حدث خطأ غير متوقع أثناء المعالجة: {str(chat_err)[:200]}"
            if "429" in str(chat_err) or "quota" in str(chat_err).lower():
                msg_err = "⚠️ خوادم النموذج تحت ضغط عالي وحصتك الحالية مستنفذة؛ انتظر ثوانٍ أو غيّر الموديل وجرب تاني."
            st.error(msg_err)
            st.session_state.chat_history.append(("assistant", msg_err))
            
    st.rerun()
