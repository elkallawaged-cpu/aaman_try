import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
import tempfile

# 1. إعداد واجهة المستخدم
st.set_page_config(page_title="Universal AI Knowledge Assistant", layout="wide")
st.title("📂 Multi-Source AI Knowledge Assistant (Ultimate RAG)")
st.subheader("ارفع ملفات (PDF/Word) أو ضع رابط موقع وابدأ الشات فوراً")

# جلب المفتاح من الـ Secrets
if "GEMINI_API_KEY" in st.secrets:
    SECRET_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=SECRET_KEY)
    st.info(f"⚙️ الـ API Key النشط يبدأ بـ: `{SECRET_KEY[:8]}...`")
else:
    st.error("رجاءً تأكد من إضافة GEMINI_API_KEY في الـ Secrets الخاص بـ Streamlit.")
    st.stop()

# جلب الموديلات المتاحة من جوجل مباشرة
try:
    models_list = [m.name.split('/')[-1] for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    default_index = models_list.index("gemini-2.0-flash") if "gemini-2.0-flash" in models_list else 0
    selected_model = st.selectbox("🤖 اختر الموديل النشط في حسابك:", models_list, index=default_index)
except Exception as e:
    selected_model = "gemini-2.0-flash"

# 2. إدارة الـ Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 3. واجهة إدخال البيانات (ملف أو رابط)
st.write("### 📥 تزويد قاعدة المعرفة بالبيانات")
tab1, tab2 = st.tabs(["📄 رفع ملف (PDF / Word)", "🌐 رابط موقع (URL)"])

docs = []
process_trigger = False

with tab1:
    uploaded_file = st.file_uploader("اختر ملف PDF أو Word (.docx)", type=["pdf", "docx"])
    if uploaded_file:
        if st.button("🚀 معالجة وبناء المعرفة من الملف"):
            process_trigger = True

with tab2:
    url_input = st.text_input("أدخل رابط الموقع بالكامل:", placeholder="https://example.com")
    if url_input:
        if st.button("🚀 معالجة وبناء المعرفة من الرابط"):
            process_trigger = True

# معالجة البيانات وتحويلها لـ Vectors
if process_trigger:
    with st.spinner("جاري استخراج البيانات وبناء قاعدة المعرفة المتجهة..."):
        try:
            # تصفير قاعدة المعرفة القديمة لاستقبال الجديدة
            st.session_state.vector_store = None
            
            if uploaded_file and not url_input:
                # معالجة الملفات المرفوعة
                suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = Docx2txtLoader(tmp_file_path)
                
                docs = loader.load()
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                    
            elif url_input:
                # معالجة روابط المواقع
                loader = WebBaseLoader(url_input)
                docs = loader.load()

            if docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                embeddings = GoogleGenerativeAIEmbeddings(
                    model="gemini-embedding-001", 
                    google_api_key=SECRET_KEY
                )
                
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                st.success("🎯 تم تحديث قاعدة المعرفة بنجاح! جاهز لتلقي الأسئلة.")
            else:
                st.error("فشل استخراج أي نصوص من المصدر المحدّد.")
        except Exception as e:
            st.error(f"حدث خطأ أثناء المعالجة: {e}")

# 4. واجهة المحادثة
st.divider()
st.write("### 💬 نافذة المحادثة الذكية")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

user_query = st.chat_input("اسألني أي شيء عن الملف أو الرابط المرفوع...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append(("user", user_query))

    with st.chat_message("assistant"):
        try:
            system_instruction = (
                "أنت مساعد ذكي ومحترف وظيفتك الإجابة على أسئلة المستخدم بناءً على السياق (Context) المرفق فقط. "
                "إذا كانت الإجابة غير موجودة في السياق، قل بكل وضوح 'المعلومة غير متوفرة في المصادر المرفوعة' ولا تقم باختراع إجابات."
            )
            
            model = genai.GenerativeModel(
                model_name=selected_model, 
                system_instruction=system_instruction
            )

            if st.session_state.vector_store is not None:
                with st.spinner("جاري البحث في قاعدة المعرفة..."):
                    docs = st.session_state.vector_store.similarity_search(user_query, k=4)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    full_prompt = f"السياق المستخرج:\n{context}\n\nسؤال المستخدم:\n{user_query}"
                    response = model.generate_content(full_prompt)
                    answer = response.text
            else:
                with st.spinner("جاري التفكير المباشر..."):
                    response = model.generate_content(user_query)
                    answer = response.text

            st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))

        except Exception as api_error:
            if "429" in str(api_error) or "quota" in str(api_error).lower():
                st.warning("⚠️ كوتا الموديل الحالي مضغوطة. جرب تغيير الموديل من القائمة المنسدلة فوق.")
            else:
                st.error(f"عذراً، واجه الموديل مشكلة. التفاصيل: {api_error}")
