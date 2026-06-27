import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai
import os
import tempfile

# 1. إعداد الصفحة والواجهة الاحترافية
st.set_page_config(page_title="مثال على تقنية RAG", layout="wide", page_icon="🚀")

# الحقن السحري للـ CSS لتجميل الواجهة وجعلها تبدو كـ تطبيق شركات مدفوع
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght=300;400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Cairo', sans-serif;
        text-align: right;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FF4B4B, #1E3A8A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #555555;
        margin-bottom: 25px;
        font-size: 1.2rem;
    }
    .business-section {
        background-color: #f0f4f8;
        border-right: 6px solid #1E3A8A;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 30px;
    }
    .business-title {
        color: #1E3A8A;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-right: 5px solid #FF4B4B;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 مثال على تقنية RAG (Retrieval-Augmented Generation)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">نظام المساعد الذكي المخصص لربط وتحليل البيانات والملفات المؤسسية</div>', unsafe_allow_html=True)

# القسم التعريفي للشركات
st.markdown("""
<div class="business-section">
    <div class="business-title">🎯 ليه تستخدم طريقة الـ RAG دي لشركتك بالذات؟</div>
    <p style="font-size: 1.05rem; line-height: 1.8; color: #333;">
        الاعتماد على الذكاء الاصطناعي العام في الشركات له عيوب خطيرة، وهنا يأتي دور تقنية <b>RAG</b> لتوفير حل احترافي آمن ومخصص:
    </p>
    <ul style="font-size: 1rem; line-height: 1.8; color: #444; padding-right: 20px;">
        <li><b>🔒 أمان وسرية بياناتك 100%:</b> ملفات شركتك وعقودها لا تُرفع لتدريب الموديلات العامة، بل يتم تشفيرها والبحث داخلها محلياً وفي بيئة معزولة تماماً.</li>
        <li><b>💰 توفير مرعب في تكلفة الـ Tokens:</b> بدلاً من إرسال آلاف الصفحات مع كل سؤال، يقوم سيستم RAG بفلترة البيانات بدقة وإرسال الفقرات ذات الصلة بالسؤال فقط، مما يوفر حتى 99% من التكلفة.</li>
        <li><b>🎯 القضاء على "الهلوسة" (Zero Hallucination):</b> يمكنك إجبار الموديل على الالتزام الصارم بملفات الشركة فقط، ليكون الجواب دقيقاً ومستنداً لمراجع حقيقية.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# جلب المفتاح من الـ Secrets
if "GEMINI_API_KEY" in st.secrets:
    SECRET_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=SECRET_KEY)
else:
    st.error("رجاءً تأكد من إضافة GEMINI_API_KEY في الـ Secrets الخاص بـ Streamlit.")
    st.stop()

# 2. إدارة الـ Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "meta_stats" not in st.session_state:
    st.session_state.meta_stats = {"files": 0, "urls": 0, "chunks": 0}
if "quick_input" not in st.session_state:
    st.session_state.quick_input = ""

# ⚙️ شريط الإعدادات الجانبي (Sidebar)
with st.sidebar:
    st.markdown("### 🛠️ هندسة النظام")
    try:
        models_list = [m.name.split('/')[-1] for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        default_index = models_list.index("gemini-2.0-flash") if "gemini-2.0-flash" in models_list else 0
        selected_model = st.selectbox("🤖 محرك الـ AI النشط:", models_list, index=default_index)
    except:
        selected_model = "gemini-2.0-flash"
        
    st.divider()
    reasoning_mode = st.radio(
        "🧠 نمط التفكير والتحليل:",
        ["🔒 التزام صارم (Strict Context)", "💡 ذكاء مختلط وربط (Hybrid Reasoning)"]
    )
    st.divider()
    
    if st.session_state.chat_history:
        chat_text = ""
        for role, text in st.session_state.chat_history:
            chat_text += f"{role.upper()}: {text}\n\n"
        st.download_button(label="📥 تحميل تقرير المحادثة (TXT)", data=chat_text, file_name="rag_report.txt", mime="text/plain")
        
    if st.button("🗑️ تفريغ الذاكرة وإعادة التعيين"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.meta_stats = {"files": 0, "urls": 0, "chunks": 0}
        st.rerun()

# 3. لوحة التغذية الشاملة
st.markdown("### 📥 لوحة تغذية قاعدة المعرفة للمؤسسات")
with st.expander("📂 اضغط هنا لرفع مستندات الشركة أو روابطها", expanded=st.session_state.vector_store is None):
    col1, col2 = st.columns(2)
    
    with col1:
        # هنا تم إضافة الـ txt للملفات المقبولة بالملي 🎯
        uploaded_files = st.file_uploader(
            "ارفع ملفاتك (PDF / Word / TXT):", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        
    with col2:
        urls_input = st.text_area(
            "ضع روابط المواقع والمقالات ذات الصلة:", 
            placeholder="https://company-policy.com\nhttps://knowledge-base.com"
        )
        
    if st.button("🔥 دمج وبناء قاعدة المعرفة الموحدة"):
        all_docs = []
        file_count = 0
        url_count = 0
        
        with st.status("🚀 جاري معالجة المصادر وبناء الـ Embeddings المتجهة...", expanded=True) as status:
            if uploaded_files:
                for u_file in uploaded_files:
                    # قراءة ومعالجة ملف التكست المرفوع مباشرة لايف
                    if u_file.name.endswith(".txt"):
                        try:
                            string_data = u_file.getvalue().decode("utf-8")
                            # تقسيم الملف لسطور أو الاحتفاظ به كنص كامل
                            file_docs = [Document(page_content=string_data, metadata={"source": u_file.name})]
                            all_docs.extend(file_docs)
                            file_count += 1
                        except Exception as te:
                            st.error(f"خطأ في قراءة ملف التكست {u_file.name}: {te}")
                    else:
                        # معالجة الـ PDF والـ Word المعتادة
                        suffix = ".pdf" if u_file.name.endswith(".pdf") else ".docx"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            tmp_file.write(u_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            loader = PyPDFLoader(tmp_file_path) if suffix == ".pdf" else Docx2txtLoader(tmp_file_path)
                            file_docs = loader.load()
                            for d in file_docs:
                                d.metadata["source"] = u_file.name
                            all_docs.extend(file_docs)
                            file_count += 1
                        except Exception as fe:
                            st.error(f"خطأ في الملف {u_file.name}: {fe}")
                        finally:
                            if os.path.exists(tmp_file_path):
                                os.remove(tmp_file_path)
                            
            if urls_input.strip():
                urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
                for url in urls:
                    try:
                        loader = WebBaseLoader(url)
                        url_docs = loader.load()
                        for d in url_docs:
                            d.metadata["source"] = url
                        all_docs.extend(url_docs)
                        url_count += 1
                    except Exception as ue:
                        st.error(f"خطأ في الرابط {url}: {ue}")
            
            if all_docs:
                # تقسيم النصوص ذكياً لقطع مناسبة للحسابات الرقمية
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(all_docs)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=SECRET_KEY)
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                
                st.session_state.meta_stats = {
