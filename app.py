import os
import re
import html as html_lib
import tempfile

import google.generativeai as genai
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ═══════════════════════════════════════════════════════════════════════════════
#  ⚙️ الإعدادات والثوابت (CONSTANTS & CONFIG)
# ═══════════════════════════════════════════════════════════════════════════════
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
TOP_K         = 5
EMBED_MODEL   = "models/embedding-001"
DEFAULT_MODEL = "gemini-3.1-flash-lite"

st.set_page_config(page_title="RAG — المساعد المؤسسي الذكي", page_icon="🧠", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
#  🎨 التنسيقات وجماليات الواجهة (CSS STYLES)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;700&display=swap');
html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; direction: rtl; text-align: right; }
.stApp { background: #F7F9FC; }
.hero { background: linear-gradient(135deg, #0F2472, #1D4ED8); border-radius: 20px; padding: 30px; text-align: center; color: white; margin-bottom: 25px; }
.m-card { background: white; padding: 20px; border-radius: 12px; border-right: 6px solid #1D4ED8; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; margin-bottom: 15px; }
.src-tag { display:inline-block; background:#EFF6FF; color:#1E40AF; padding:4px 12px; border-radius:15px; font-size:0.8rem; margin:3px; border:1px solid #BFDBFE; font-weight: bold; }
.stButton > button { border-radius: 10px !important; font-family: 'Cairo' !important; font-weight: bold; }
.suggested-title { font-weight: bold; margin-top: 15px; margin-bottom: 5px; color: #0F2472; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  🔐 التحقق من مفتاح الـ API وتأهيله
# ═══════════════════════════════════════════════════════════════════════════════
def _init_api():
    key = st.secrets.get("GEMINI_API_KEY", "").strip()
    if not key:
        st.error("🚨 لم يتم العثور على مفتاح GEMINI_API_KEY في ملف Secrets!")
        st.stop()
    genai.configure(api_key=key)
    return key

_API_KEY = _init_api()

# ═══════════════════════════════════════════════════════════════════════════════
#  🧠 دالة توليد الأسئلة الذكية ديناميكياً (DYNAMIC QUERY GENERATOR)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_smart_queries(docs: list[Document]) -> list[str]:
    """يقوم بقراءة عينات من المستندات المرفوعة وتوليد 3 أسئلة ذكية تناسب المحتوى تماماً."""
    try:
        # أخذ مقتطفات من أول 3 وثائق لعدم تخطي حدود الـ Tokens
        sample_text = "\n".join([d.page_content[:500] for d in docs[:3]])
        model = genai.GenerativeModel(DEFAULT_MODEL)
        prompt = (
            f"أنت خبير تحليل بيانات ومستندات. بناءً على المحتوى المرفق، اقترح بالضبط 3 أسئلة تحليلية عميقة أو استعلامات هامة "
            f"يمكن للمستخدم أن يطرحها حول هذا المحتوى ليفهمه بشكل أفضل.\n"
            f"شروط مهمة جداً:\n"
            f"1. اكتب الأسئلة باللغة العربية.\n"
            f"2. اجعل الأسئلة قصيرة ومباشرة (لا تتجاوز 7 كلمات للسؤال).\n"
            f"3. أعطني الأسئلة في شكل سطر لكل سؤال، بدون أي ترقيم (مثل 1، 2، 3) وبدون أي مقدمات أو خاتمة.\n\n"
            f"المحتوى المقتطف:\n{sample_text}"
        )
        response = model.generate_content(prompt)
        # تقسيم الاستجابة إلى أسطر وتنظيفها من الترقيم العشوائي إن وجد
        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        cleaned_queries = [re.sub(r'^\d+[\.\-\)]\s*', '', q).strip() for q in lines]
        return cleaned_queries[:3]
    except Exception as e:
        # حل احتياطي في حال حدوث أي خطأ في الاتصال أو التحليل
        return [
            "ما هي الخلاصة التنفيذية لهذه المستندات؟",
            "استخرج أهم النقاط والتوصيات المذكورة.",
            "هل هناك أي شروط أو تواريخ هامة يجب مراعاتها؟"
        ]

# ═══════════════════════════════════════════════════════════════════════════════
#  📂 دالة معالجة وتحميل الملفات (FILE LOADING LOGIC)
# ═══════════════════════════════════════════════════════════════════════════════
def load_uploaded_files(uploaded_files) -> list[Document]:
    documents = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
                loaded_docs = loader.load()
            elif suffix == ".docx":
                loader = Docx2txtLoader(tmp_path)
                loaded_docs = loader.load()
            elif suffix == ".txt":
                with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                loaded_docs = [Document(page_content=text)]
            else:
                continue
            
            # حقن اسم الملف الأصلي داخل الميتا داتا لكل جزء
            for d in loaded_docs:
                d.metadata["source"] = uploaded_file.name
            documents.extend(loaded_docs)
        except Exception as e:
            st.error(f"خطأ أثناء قراءة الملف {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return documents

# ═══════════════════════════════════════════════════════════════════════════════
#  🗄️ تهيئة وإدارة الذاكرة (SESSION STATE)
# ═══════════════════════════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "meta_stats" not in st.session_state:
    st.session_state.meta_stats = {"files": 0, "chunks": 0}
if "quick_input" not in st.session_state:
    st.session_state.quick_input = ""
if "suggested_queries" not in st.session_state:
    st.session_state.suggested_queries = []

# ═══════════════════════════════════════════════════════════════════════════════
#  🎯 الواجهة الرسومية (UI INTERFACE)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero"><h1>🧠 المساعد المؤسسي الذكي (RAG)</h1><p>ارفع ملفاتك، ودع الذكاء الاصطناعي يحللها ويقترح عليك أسئلتها فوراً!</p></div>', unsafe_allow_html=True)

# شريط جانبي لمعلومات النظام والمؤشرات
with st.sidebar:
    st.markdown("### 📊 حالة قاعدة المعرفة")
    if st.session_state.vector_store:
        st.markdown(f'<div class="m-card">📝 <b>الملفات المعالجة:</b> {st.session_state.meta_stats["files"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="m-card">🧩 <b>عدد الأجزاء (Chunks):</b> {st.session_state.meta_stats["chunks"]}</div>', unsafe_allow_html=True)
        if st.button("🗑️ تفريغ الذاكرة والملفات", type="secondary"):
            st.session_state.vector_store = None
            st.session_state.suggested_queries = []
            st.session_state.chat_history = []
            st.session_state.meta_stats = {"files": 0, "chunks": 0}
            st.rerun()
    else:
        st.info("لم يتم بناء قاعدة المعرفة بعد. يرجى رفع الملفات من القائمة الرئيسية.")

# صندوق رفع الملفات وبناء الـ Vector Store
with st.expander("📂 إدارة ورفع المستندات والملفات", expanded=(st.session_state.vector_store is None)):
    uploaded_files = st.file_uploader("اختر ملفاتك (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    
    if st.button("🔥 بناء وتحديث قاعدة المعرفة الآن", type="primary"):
        if uploaded_files:
            with st.status("🚀 جاري استخراج النصوص وتقسيم البيانات...") as status:
                all_docs = load_uploaded_files(uploaded_files)
                if all_docs:
                    status.update(label="🧬 جاري توليد الـ Embeddings وبناء كشاف FAISS...")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                    split_chunks = text_splitter.split_documents(all_docs)
                    
                    # إنشاء التضمينات وقاعدة البيانات الشجرية
                    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=_API_KEY)
                    vector_store = FAISS.from_documents(split_chunks, embeddings)
                    
                    # حفظ البيانات في الـ Session State
                    st.session_state.vector_store = vector_store
                    st.session_state.meta_stats = {
                        "files": len(uploaded_files),
                        "chunks": len(split_chunks)
                    }
                    
                    status.update(label="🧠 جاري تحليل المحتوى لتوليد أسئلة ذكية...")
                    st.session_state.suggested_queries = generate_smart_queries(all_docs)
                    
                    status.update(label="✅ تم بناء قاعدة المعرفة واقتراح الأسئلة بنجاح!", state="complete")
                    st.rerun()
                else:
                    st.error("فشل استخراج أي نصوص صالحة من الملفات المرفوعة.")
        else:
            st.warning("الرجاء رفع ملف واحد على الأقل أولاً.")

# 💡 عرض الأسئلة المقترحة ديناميكياً كمفاتيح سريعة (Quick Actions)
if st.session_state.vector_store and st.session_state.suggested_queries:
    st.markdown('<p class="suggested-title">💡 أسئلة مقترحة ذكياً بناءً على ملفاتك المرفوعة:</p>', unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.suggested_queries))
    for col, query in zip(cols, st.session_state.suggested_queries):
        with col:
            # عرض جزء من النص لو طويل على حجم الزر
            display_text = query[:40] + "..." if len(query) > 40 else query
            if st.button(display_text, key=f"btn_{query}", use_container_width=True):
                st.session_state.quick_input = query
                st.rerun()

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  💬 منطقة المحادثة ومعالجة الـ RAG Loop
# ═══════════════════════════════════════════════════════════════════════════════
# عرض التاريخ السابق للمحادثة
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.markdown(" ".join([f'<span class="src-tag">📄 {html_lib.escape(s)}</span>' for s in msg["sources"]]), unsafe_allow_html=True)

# استقبال السؤال (سواء كتابة أو من خلال الأزرار المقترحة)
user_query = st.chat_input("اسألني أي شيء حول المستندات المرفوعة...")
if st.session_state.quick_input:
    user_query = st.session_state.quick_input
    st.session_state.quick_input = ""  # تصفير الإدخال السريع فوراً

if user_query:
    # عرض سؤال المستخدم فوراً في الواجهة
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("جاري قراءة وتحليل السياق والإجابة..."):
            context = ""
            sources = set()
            
            # إذا كانت قاعدة البيانات جاهزة، نقوم بالبحث السيمانتيكي
            if st.session_state.vector_store:
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": TOP_K})
                relevant_docs = retriever.get_relevant_documents(user_query)
                
                context_chunks = []
                for doc in relevant_docs:
                    context_chunks.append(doc.page_content)
                    if "source" in doc.metadata:
                        sources.add(doc.metadata["source"])
                context = "\n\n".join(context_chunks)
            
            # صياغة الـ Prompt النهائي لـ Gemini
            if context:
                full_prompt = (
                    f"أنت مساعد ذكي ومستشار مؤسسي محترف. أجب على سؤال المستخدم بكل دقة وموثوقية بالاعتماد فقط على السياق المرفق.\n"
                    f"إذا لم تكن الإجابة موجودة في السياق، وضح بلطف أنها غير متوفرة في الملفات بدلاً من اختراع معلومة.\n\n"
                    f"السياق المتاح:\n{context}\n\n"
                    f"سؤال المستخدم:\n{user_query}"
                )
            else:
                full_prompt = (
                    f"أنت مساعد ذكي ومستشار مؤسسي محترف. تنبيه: لا توجد مستندات مرفوعة حالياً في قاعدة المعرفة.\n"
                    f"أجب على سؤال المستخدم بشكل عام وذكّره برفع مستنداته إذا كان السؤال خاصاً بملف معين.\n\n"
                    f"سؤال المستخدم:\n{user_query}"
                )
            
            try:
                model = genai.GenerativeModel(DEFAULT_MODEL)
                response = model.generate_content(full_prompt)
                ai_response = response.text
                
                # عرض الإجابة في الواجهة
                st.markdown(ai_response)
                if sources:
                    st.markdown(" ".join([f'<span class="src-tag">📄 {html_lib.escape(s)}</span>' for s in sources]), unsafe_allow_html=True)
                
                # حفظ الإجابة في تاريخ الذاكرة
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ai_response,
                    "sources": list(sources)
                })
            except Exception as e:
                st.error(f"حدث خطأ أثناء معالجة الرد من الـ AI: {e}")
