import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai
import os
import tempfile

st.set_page_config(page_title="مثال على تقنية RAG", layout="wide", page_icon="🚀")

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

if "GEMINI_API_KEY" in st.secrets:
    SECRET_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=SECRET_KEY)
else:
    st.error("رجاءً تأكد من إضافة GEMINI_API_KEY في الـ Secrets الخاص بـ Streamlit.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "meta_stats" not in st.session_state:
    st.session_state.meta_stats = {"files": 0, "urls": 0, "chunks": 0}
if "quick_input" not in st.session_state:
    st.session_state.quick_input = ""

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
                            file_docs = [Document(page_content=string_data, metadata={"source": u_file.name})]
                            all_docs.extend(file_docs)
                            file_count += 1
                        except Exception as te:
                            st.error(f"خطأ في قراءة ملف التكست {u_file.name}: {te}")
                    else:
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
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(all_docs)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=SECRET_KEY)
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                
                st.session_state.meta_stats = {
                    "files": file_count,
                    "urls": url_count,
                    "chunks": len(splits)
                }
                status.update(label="🎯 تم بناء الذاكرة المتجهة بنجاح وكل شيء جاهز!", state="complete", expanded=False)
                st.rerun()
            else:
                st.warning("برجاء إدخال بيانات صحيحة أولاً.")

if st.session_state.vector_store is not None:
    st.write("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card">📄 <b>المستندات المرفوعة:</b> {st.session_state.meta_stats["files"]} ملفات</div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card">🌐 <b>الروابط المفهرسة:</b> {st.session_state.meta_stats["urls"]} روابط</div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card">🧩 <b>أجزاء الذاكرة المتجهة:</b> {st.session_state.meta_stats["chunks"]} جُزء</div>', unsafe_allow_html=True)

    # أزرار ذكية عامة وممتازة للتصوير لايف في الفيديو
    st.write("")
    st.write("💡 **عمليات تحليل سريعة بنقرة واحدة:**")
    qp1, qp2, qp3 = st.columns(3)
    with qp1:
        if st.button("🔒 سؤال صارم: كم استغرق شحن عميل الإسكندرية وما السبب؟"):
            st.session_state.quick_input = "كم استغرق الشحن بالضبط لعميل الإسكندرية وائل غنيم وما هو السبب المكتوب في السجلات؟"
    with qp2:
        if st.button("💡 سؤال مختلط: حلل المشكلة الهندسية لشاحن 65 واط واقترح حلاً"):
            st.session_state.quick_input = "بناءً على شكوى العميل طارق البشري بخصوص حرارة الشاحن الـ 65 واط، استخدم علم الهندسة والالكترونيات لتحليل المشكلة وتقديم اقتراحات لتفاديها في التصنيع القادم."
    with qp3:
        if st.button("📈 تقرير عام: ما هي الشكوى التقنية المتكررة في تطبيق الساعة الذكية؟"):
            st.session_state.quick_input = "استخرج الشكوى الخاصة بتطبيق الساعة الذكية Pro، ولماذا تستهلك البطارية بشكل مرعب؟"

st.write("---")
st.write("### 💬 نافذة الاستعلام والتحليل الذكي")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

current_query = st.chat_input("اسأل المساعد الذكي عن أي شيء يخص مستندات الشركة...")
if st.session_state.quick_input:
    current_query = st.session_state.quick_input
    st.session_state.quick_input = "" 

if current_query:
    with st.chat_message("user"):
        st.write(current_query)
    st.session_state.chat_history.append(("user", current_query))

    with st.chat_message("assistant"):
        try:
            if "التزام صارم" in reasoning_mode:
                system_instruction = (
                    "أنت مساعد ذكي ومحترف. يجب عليك الإجابة على أسئلة المستخدم بناءً *فقط* على السياق المرفق. "
                    "إذا لم تكن الإجابة موجودة حرفياً في السياق، قل بكل وضوح 'المعلومة غير متوفرة في الملفات المرفوعة' ولا تخترع أي شيء."
                )
            else:
                system_instruction = (
                    "أنت خبير ومحلل ذكاء اصطناعي عبقري. مهمتك هي الدمج والربط بين السياق المرفق (Context) وبين "
                    "معلوماتك العامة العميقة وقدرتك على الاستنتاج والتحليل البناء. قم بربط الخيوط ببعضها لحل مسألة المستخدم بذكاء."
                )
            
            model = genai.GenerativeModel(model_name=selected_model, system_instruction=system_instruction)

            if st.session_state.vector_store is not None:
                with st.spinner("🧠 جاري فحص الذاكرة المتجهة والربط الذكي للأفكار..."):
                    retrieved_docs = st.session_state.vector_store.similarity_search(current_query, k=6)
                    sources = set([d.metadata.get("source", "مصدر غير معروف") for d in retrieved_docs])
                    
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    full_prompt = f"السياق المستخرج من مصادرك الشاملة:\n{context}\n\nسؤال المستخدم:\n{current_query}"
                    
                    response = model.generate_content(full_prompt)
                    answer = response.text
                    
                    st.write(answer)
                    
                    st.write("")
                    with st.expander("📌 المستندات والمصادر المستند إليها في الإجابة:", expanded=False):
                        for src in sources:
                            st.markdown(f"• `{src}`")
            else:
                with st.spinner("جاري التفكير المباشر..."):
                    response = model.generate_content(current_query)
                    answer = response.text
                    st.write(answer)

            st.session_state.chat_history.append(("assistant", answer))
            if current_query:
                st.rerun() 

        except Exception as api_error:
            if "429" in str(api_error) or "quota" in str(api_error).lower():
                st.warning("⚠️ كوتا الموديل مضغوطة حالياً، غير الموديل من القائمة الجانبية وجرب تاني.")
            else:
                st.error(f"حدث خطأ أثناء معالجة الرد: {api_error}")
