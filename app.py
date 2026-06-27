import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import tempfile

# 1. إعداد واجهة المستخدم
st.set_page_config(page_title="Universal AI Knowledge Assistant", layout="wide")
st.title("📂 Universal AI Knowledge Assistant (Dynamic RAG)")
st.subheader("ارفع أي ملف وابدأ الشات معاه فوراً")

# جلب الـ API Key من الـ Secrets بأمان تام
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

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

            # استخدام الموديل المستقر والمعتمد
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001", 
                google_api_key=GEMINI_API_KEY
            )
            
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            st.success("تم تحليل الملف وبناء قاعدة المعرفة بنجاح! يمكنك البدء بالأسئلة الآن.")
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الملف، تأكد من صلاحية الـ API Key. تفاصيل: {e}")
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
        if st.session_state.vector_store is not None:
            with st.spinner("جاري البحث في الملف وتوليد الإجابة النموذجية..."):
                docs = st.session_state.vector_store.similarity_search(user_query, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])

                prompt_template = ChatPromptTemplate.from_template("""
                أنت مساعد ذكي ومحترف وظيفتك الإجابة على أسئلة المستخدم بناءً على السياق (Context) المرفق فقط.
                إذا كانت الإجابة غير موجودة في السياق، قل بكل وضوح "المعلومة غير متوفرة في الملف المرفوع" ولا تقم باختراع إجابات.
                
                السياق المستخرج من الملف:
                {context}
                
                سؤال المستخدم:
                {question}
                """)
                
                formatted_prompt = prompt_template.format(context=context, question=user_query)
                
                # تم تصحيح مسار الموديل هنا بإضافة الـ prefix الرسمي
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                response = model.generate_content(formatted_prompt)
                
                answer = response.text
                st.write(answer)
                st.session_state.chat_history.append(("assistant", answer))
        else:
            with st.spinner("جاري التفكير..."):
                # تم تصحيح مسار الموديل هنا أيضاً لضمان عمل الشات العام
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                response = model.generate_content(user_query)
                answer = response.text
                st.write(answer)
                st.session_state.chat_history.append(("assistant", answer))
