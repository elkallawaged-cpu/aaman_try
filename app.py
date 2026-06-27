import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
import tempfile

# 1. إعداد الصفحة والواجهة الاحترافية
st.set_page_config(page_title="AI Knowledge Hub Pro", layout="wide", page_icon="🚀")

# الحقن السحري للـ CSS لتجميل الواجهة وجعلها تبدو كـ تطبيق مدفوع
st.markdown("""
    <style>
    /* تغيير الفونت وتنسيق العنوان الرئيسي */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;700&display=swap');
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
        margin-bottom: 30px;
        font-size: 1.2rem;
    }
    /* تحسين شكل الـ Tabs والبطاقات */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: bold;
        padding: 10px 20px;
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

# عرض العناوين بالـ Styling الجديد
st.markdown('<div class="main-title">🚀 AI Knowledge Hub & Reasoning Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">الجيل القادم من أنظمة المساعدين الذكيين - ادمج مصادرك وحللها بذكاء بشري</div>', unsafe_allow_html=True)

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
    st.session_state.meta_stats = {"files": 0, "urls": 0, "chunks": 0, "sources_names": []}
if "quick_input" not in st.session_state:
    st.session_state.quick_input = ""

# ⚙️ شريط الإعدادات الجانبي (Sidebar المطور)
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
        [
            "🔒 التزام صارم (Strict Context)",
            "💡 ذكاء مختلط وربط (Hybrid Reasoning)"
        ],
        help="النمط المختلط يسمح للموديل باستخدام ذكائه الخارجي لربط المعلومات وحل المعضلات المعقدة."
    )
    
    st.divider()
    
    # ميزة صايعة: تحميل الشات
    if st.session_state.chat_history:
        chat_text = ""
        for role, text in st.session_state.chat_history:
            chat_text += f"{role.upper()}: {text}\n\n"
        st.download_button(
            label="📥 تحميل سجل المحادثة (TXT)",
            data=chat_text,
            file_name="chat_history_report.txt",
            mime="text/plain"
        )
        
    if st.button("🗑️ تفريغ الذاكرة وإعادة تعيين السيستم"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.meta_stats = {"files": 0, "urls": 0, "chunks": 0, "sources_names": []}
        st.rerun()

# 3. لوحة التغذية الشاملة بتصميم شيك جداً
st.markdown("### 📥 لوحة تغذية قاعدة المعرفة")
with st.expander("📂 اضغط هنا لإضافة أو تعديل مصادر البيانات (ملفات وروابط)", expanded=st.session_state.vector_store is None):
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_files = st.file_uploader(
            "ارفع ملفاتك (PDF / Word):", 
            type=["pdf", "docx"], 
            accept_multiple_files=True
        )
        
    with col2:
        urls_input = st.text_area(
            "ضع روابط المواقع (رابط في كل سطر):", 
            placeholder="https://example1.com\nhttps://example2.com"
        )
        
    if st.button("🔥 دمج وبناء قاعدة المعرفة الموحدة"):
        all_docs = []
        file_count = 0
        url_count = 0
        sources_list = []
        
        with st.status("🚀 جاري معالجة المصادر وبناء الـ Embeddings...", expanded=True) as status:
            if uploaded_files:
                for u_file in uploaded_files:
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
                        sources_list.append(u_file.name)
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
                        sources_list.append(url)
                    except Exception as ue:
                        st.error(f"خطأ في الرابط {url}: {ue}")
            
            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(all_docs)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=SECRET_KEY)
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                
                # تحديث الإحصائيات في الـ Session
                st.session_state.meta_stats = {
                    "files": file_count,
                    "urls": url_count,
                    "chunks": len(splits),
                    "sources_names": list(set(sources_list))
                }
                status.update(label="🎯 تم بناء الذاكرة المتجهة بنجاح وكل شيء جاهز!", state="complete", expanded=False)
                st.rerun()
            else:
                st.warning("برجاء إدخال بيانات صحيحة أولاً.")

# 📊 لوحة العدادات الذكية (تظهر فقط لو في داتا مرفوعة)
if st.session_state.vector_store is not None:
    st.write("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card">📄 <b>الملفات المرفوعة:</b> {st.session_state.meta_stats["files"]} ملفات</div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card">🌐 <b>الروابط المفهرسة:</b> {st.session_state.meta_stats["urls"]} روابط</div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card">🧩 <b>أجزاء المعرفة الذكية:</b> {st.session_state.meta_stats["chunks"]} جُزء وثائقي</div>', unsafe_allow_html=True)

    # 📝 ميزة صايعة: أزرار الأسئلة السريعة (Quick Prompts)
    st.write("")
    st.write("💡 **أسئلة مقترحة سريعة:**")
    qp1, qp2, qp3 = st.columns(3)
    with qp1:
        if st.button("📝 لخص لي كل هذه المصادر في نقاط مكثفة"):
            st.session_state.quick_input = "قم بتلخيص جميع المصادر المرفوعة بشكل منظم ومكثف في نقاط رئيسية."
    with qp2:
        if st.button("🔍 استخرج أهم 5 حقائق أو أرقام من الداتا"):
            st.session_state.quick_input = "استخرج أهم 5 حقائق، استنتاجات أو أرقام رئيسية موجودة داخل البيانات."
    with qp3:
        if st.button("❓ اقترح عليّ 3 أسئلة ذكية يمكنني طرحها هنا"):
            st.session_state.quick_input = "بناءً على محتوى المصادر، اقترح 3 أسئلة عميقة يمكنني طرحها عليك لتحليل هذه البيانات."

# 4. واجهة المحادثة المصقولة
st.write("---")
st.write("### 💬 نافذة المحادثة التحليلية")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# التقاط السؤال سواء كتبه المستخدم أو ضغط على زر سريع
current_query = st.chat_input("اسألني أي سؤال عن مصادرك الشاملة...")
if st.session_state.quick_input:
    current_query = st.session_state.quick_input
    st.session_state.quick_input = "" # تصفير المتغير بعد القراءة

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
                    "معلوماتك العامة العميقة وقدرتك على الاستنتاج والتحليل. قم بربط الخيوط ببعضها لحل مسألة المستخدم بذكاء. "
                    "وضح للمستخدم بذكاء طريقة ربطك للأمور (مثال: بالنظر إلى الملفات المرفوعة وبإضافة التحليل الهندسي المنطقي...)."
                )
            
            model = genai.GenerativeModel(model_name=selected_model, system_instruction=system_instruction)

            if st.session_state.vector_store is not None:
                with st.spinner("🧠 جاري فحص الذاكرة المتجهة والربط الذكي للأفكار..."):
                    retrieved_docs = st.session_state.vector_store.similarity_search(current_query, k=5)
                    sources = set([d.metadata.get("source", "مصدر غير معروف") for d in retrieved_docs])
                    
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    full_prompt = f"السياق المستخرج من مصادرك الشاملة:\n{context}\n\nسؤال المستخدم:\n{current_query}"
                    
                    response = model.generate_content(full_prompt)
                    answer = response.text
                    
                    st.write(answer)
                    
                    # عرض المصادر بشكل منسق واحترافي جداً
                    st.write("")
                    with st.expander("📌 المراجع والوثائق المستند إليها في هذه الإجابة:", expanded=False):
                        for src in sources:
                            st.markdown(f"• `{src}`")
            else:
                with st.spinner("جاري التفكير المباشر..."):
                    response = model.generate_content(current_query)
                    answer = response.text
                    st.write(answer)

            st.session_state.chat_history.append(("assistant", answer))
            if current_query:
                st.rerun() # لإعادة التحديث السلس للواجهة

        except Exception as api_error:
            if "429" in str(api_error) or "quota" in str(api_error).lower():
                st.warning("⚠️ كوتا الموديل ده مضغوطة حالياً، غير الموديل من القائمة الجانبية (Sidebar) وجرب تاني.")
            else:
                st.error(f"حدث خطأ أثناء معالجة الرد: {api_error}")
