import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
import tempfile

# 1. إعداد واجهة المستخدم
st.set_page_config(page_title="Universal AI Knowledge Assistant", layout="wide")
st.title("📂 Universal AI Knowledge Assistant (Dynamic RAG)")
st.subheader("ارفع أي ملف وابدأ الشات معاه فوراً")

# جلب المفتاح من الـ Secrets
if "GEMINI_API_KEY" in st.secrets:
    SECRET_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=SECRET_KEY)
    
    # سطر الفحص الحاسم - هيطبع أول 8 حروف من المفتاح اللي قاري منه السيرفر حالياً
    st.info(f"⚙️ الـ API Key النشط في السيرفر حالياً يبدأ بـ: `{SECRET_KEY[:8]}...` (قارنه بمفتاحك الجديد)")
else:
    st.error("رجاءً تأكد من إضافة GEMINI_API_KEY في الـ Secrets الخاص بـ Streamlit.")
    st.stop()

# 2. إدارة الـ Session State للشات والـ Vector Store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 3. رفع ملفات الـ PDF ومعالجتها
uploaded_file = st.file_uploader("اختر ملف PDF لتقديمه للمساعد الذكي", type=["pdf"])

if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("جاري معالجة الملف واستخراج البيانات وبناء قاعدة المعرفة..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001", 
                google_api_key=SECRET_KEY
            )
            
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            st.success("تم تحليل الملف وبناء قاعدة المعرفة بنجاح! يمكنك البدء بالأسئلة الآن.")
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الملف. تفاصيل: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# 4. واجهة المحادثة وعرض الرسائل
st.divider()
st.write("### 💬 نافذة المحادثة")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

user_query = st.chat_input("اسألني أي شيء عن الملف المرفوع...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append(("user", user_query))

    with st.chat_message("assistant"):
        try:
            system_instruction = (
                "أنت مساعد ذكي ومحترف وظيفتك الإجابة على أسئلة المستخدم بناءً على السياق (Context) المرفق فقط. "
                "إذا كانت الإجابة غير موجودة في السياق، قل بكل وضوح 'المعلومة غير متوفرة في الملف المرفوع' ولا تقم باختراع إجابات."
            )
            
            # العودة للموديل الصحيح والمدعوم بالكامل
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash", 
                system_instruction=system_instruction
            )

            if st.session_state.vector_store is not None:
                with st.spinner("loading..."):
                    docs = st.session_state.vector_store.similarity_search(user_query, k=4)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    full_prompt = f"السياق المستخرج من الملف:\n{context}\n\nسؤال المستخدم:\n{user_query}"
                    
                    response = model.generate_content(full_prompt)
                    answer = response.text
            else:
                with st.spinner("جاري التفكير..."):
                    response = model.generate_content(user_query)
                    answer = response.text

            st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))

        except Exception as api_error:
            if "429" in str(api_error) or "quota" in str(api_error).lower():
                st.warning("⚠️ تم تجاوز حد الطلبات. المفتاح النشط حالياً مستهلك، يرجى تحديث الـ Secrets من لوحة تحكم Streamlit أونلاين ومطابقة الحروف الظاهرة فوق.")
            else:
                st.error(f"عذراً، واجه الموديل مشكلة أثناء توليد الإجابة. التفاصيل: {api_error}")
