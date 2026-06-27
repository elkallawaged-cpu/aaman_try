import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
import tempfile

# 1. إعداد واجهة المستخدم
st.set_page_config(page_title="Universal AI Knowledge Assistant", layout="wide")
st.title("📂 Universal AI Knowledge Assistant (Dynamic RAG)")
st.subheader("ارفع أي ملف وابدأ الشات معاه فوراً")

# تعيين المفتاح كمتغير بيئة رسمي لمنع أي تضارب في الصلاحيات داخلياً
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
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

            # الـ Embeddings المستقرة المعتمِدة على متغيرات البيئة
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
            
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            st.success("تم تحليل الملف وبناء قاعدة المعرفة بنجاح! يمكنك البدء بالأسئلة الآن.")
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الملف. تفاصيل: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove
