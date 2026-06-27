import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
import tempfile

# 1. إعداد واجهة المستخدم
st.set_page_config(page_title="Mega AI Knowledge Hub", layout="wide")
st.title("🚀 Mega AI Knowledge Hub & Reasoning Engine")
st.subheader("اجمع كل ملفاتك وروابطك في مكان واحد، وخلي الـ AI يربط الأحداث بذكاء!")

# جلب المفتاح من الـ Secrets
if "GEMINI_API_KEY" in st.secrets:
    SECRET_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=SECRET_KEY)
else:
    st.error("رجاءً تأكد من إضافة GEMINI_API_KEY في الـ Secrets الخاص بـ Streamlit.")
    st.stop()

# ⚙️ شريط الإعدادات الجانبي (Sidebar)
with st.sidebar:
    st.header("🛠️ لوحة التحكم الذكية")
    
    # جلب الموديلات المتاحة تلقائياً
    try:
        models_list = [m.name.split('/')[-1] for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        default_index = models_list.index("gemini-2.0-flash") if "gemini-2.0-flash" in models_list else 0
        selected_model = st.selectbox("🤖 الموديل النشط:", models_list, index=default_index)
    except:
        selected_model = "gemini-2.0-flash"
        
    st.divider()
    
    # الخيار العبقري اللي طلبته: تحديد نمط تفكير الـ AI
    reasoning_mode = st.radio(
        "🧠 نمط تفكير وتحليل الـ AI:",
        [
            "🔒 التزام صارم (Strict Context): إجابات من داخل الملفات فقط بدون أي اجتهاد.",
            "💡 ذكاء مختلط (Hybrid Reasoning): الربط بين معلومات الملفات وذكاء الموديل العام للتحليل والاستنتاج."
        ]
    )
    
    st.divider()
    if st.button("🗑️ تفريغ الذاكرة والشات"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.rerun()

# 2. إدارة الـ Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 3. صندوق التغذية الشامل (Multi-Source Ingestion Container)
st.write("### 📥 لوحة تغذية قاعدة المعرفة (رفع متعدد للمصادر)")
with st.expander("📂 اضغط هنا لإضافة ملفاتك وروابطك معاً", expanded=st.session_state.vector_store is None):
    col1, col2 = st.columns(2)
    
    with col1:
        # دعم رفع أكتر من ملف في نفس الوقت (PDF و DOCX)
        uploaded_files = st.file_uploader(
            "رفع ملفات متعددة (PDF / Word):", 
            type=["pdf", "docx"], 
            accept_multiple_files=True
        )
        
    with col2:
        # إضافة أكتر من رابط موقع (رابط في كل سطر)
        urls_input = st.text_area(
            "أدخل روابط المواقع (رابط واحد في كل سطر):", 
            placeholder="https://example1.com\nhttps://example2.com"
        )
        
    st.write("---")
    if st.button("🔥 ادمج وابنِ قاعدة المعرفة الشاملة حالاً"):
        all_docs = []
        
        with st.spinner("جاري سحب البيانات من جميع المصادر وتصنيفها..."):
            # أولاً: معالجة كل الملفات المرفوعة
            if uploaded_files:
                for u_file in uploaded_files:
                    suffix = ".pdf" if u_file.name.endswith(".pdf") else ".docx"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(u_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        loader = PyPDFLoader(tmp_file_path) if suffix == ".pdf" else Docx2txtLoader(tmp_file_path)
                        file_docs = loader.load()
                        # حركة صايعة: بنجبر السورس يشيل اسم الملف الحقيقي مش المسار المؤقت
                        for d in file_docs:
                            d.metadata["source"] = u_file.name
                        all_docs.extend(file_docs)
                    except Exception as fe:
                        st.error(f"خطأ في قراءة الملف {u_file.name}: {fe}")
                    finally:
                        if os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path)
                            
            # ثانياً: معالجة كل الروابط المكتوبة
            if urls_input.strip():
                urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
                for url in urls:
                    try:
                        loader = WebBaseLoader(url)
                        url_docs = loader.load()
                        for d in url_docs:
                            d.metadata["source"] = url
                        all_docs.extend(url_docs)
                    except Exception as ue:
                        st.error(f"خطأ في سحب بيانات الرابط {url}: {ue}")
            
            # ثالثاً: تقسيم النصوص وبناء الـ Vector Store المشترك
            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(all_docs)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=SECRET_KEY)
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                st.success(f"🎯 العظمة تمت بنجاح! تم دمج وثقب {len(all_docs)} مصدر بنجاح في ذاكرة الـ AI.")
            else:
                st.warning("برجاء إضافة مصادر (ملفات أو روابط) أولاً قبل الضغط على الزر.")

# 4. واجهة المحادثة الذكية وعرض المصادر
st.divider()
st.write("### 💬 نافذة المحادثة التحليلية")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

user_query = st.chat_input("اسألني أي سؤال، وهجيبلك الإجابة مع ربط المصادر...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append(("user", user_query))

    with st.chat_message("assistant"):
        try:
            # صياغة الـ System Instructions بناءً على اختيار المستخدم من الـ Sidebar
            if "التزام صارم" in reasoning_mode:
                system_instruction = (
                    "أنت مساعد ذكي ومحترف. يجب عليك الإجابة على أسئلة المستخدم بناءً *فقط* على السياق المرفق. "
                    "إذا لم تكن الإجابة موجودة حرفياً في السياق، قل بكل وضوح 'المعلومة غير متوفرة في الملفات المرفوعة' ولا تخترع أي شيء."
                )
            else:
                system_instruction = (
                    "أنت خبير ومحلل ذكاء اصطناعي عبقري. مهمتك هي الدمج والربط بين السياق المرفق (Context) وبين "
                    "معلوماتك العامة العميقة وقدرتك على الاستنتاج والتحليل. قم بربط الخيوط ببعضها لحل مسألة المستخدم بذكاء. "
                    "إذا استخدمت معلومات خارجية للربط والتحليل، وضح ذلك بذكاء للمستخدم (مثال: بناءً على الأوراق المرفقة مضافاً إليها التحليل الهندسي...)."
                )
            
            model = genai.GenerativeModel(model_name=selected_model, system_instruction=system_instruction)

            if st.session_state.vector_store is not None:
                with st.spinner("جاري البحث في قاعدة المعرفة الموحدة والربط الذكي..."):
                    # البحث عن الفقرات الشبيهة
                    retrieved_docs = st.session_state.vector_store.similarity_search(user_query, k=5)
                    
                    # استخراج المصادر الفريدة اللي الـ AI جاب منها الكلام
                    sources = set([d.metadata.get("source", "مصدر غير معروف") for d in retrieved_docs])
                    
                    context = "\n\n".join([d.page_content for d in retrieved_docs])
                    full_prompt = f"السياق المستخرج من مصادرك الشاملة:\n{context}\n\nسؤال المستخدم التحليلي:\n{user_query}"
                    
                    response = model.generate_content(full_prompt)
                    answer = response.text
                    
                    # عرض الإجابة
                    st.write(answer)
                    
                    # 🔥 حركة إظهار السورسات بشكل شيك جداً تحت الإجابة
                    st.markdown("##### 📌 المصادر التي تم الاعتماد عليها في البحث:")
                    for src in sources:
                        st.caption(f"• `{src}`")
            else:
                # لو مفيش ملفات، يشتغل بذكائه العام مباشرة
                with st.spinner("جاري التفكير والتحليل العام..."):
                    response = model.generate_content(user_query)
                    answer = response.text
                    st.write(answer)

            st.session_state.chat_history.append(("assistant", answer))

        except Exception as api_error:
            if "429" in str(api_error) or "quota" in str(api_error).lower():
                st.warning("⚠️ كوتا الموديل ده مش قادرة، غير الموديل من القائمة اللي في الجنب (Sidebar) وجرب تاني حالاً.")
            else:
                st.error(f"حدث خطأ أثناء معالجة الرد: {api_error}")
