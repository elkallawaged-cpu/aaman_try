import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import os
import tempfile

# 1. إعداد واجهة المستخدم
st.set_page_config(page_title="Universal AI Knowledge Assistant", layout="wide")
st.title("📂 Universal AI Knowledge Assistant (Dynamic RAG)")
st.subheader("ارفع أي ملف وابدأ الشات معاه فوراً")

# جلب الـ API Key من الـ Secrets بأمان تام لمنع أي تسريب
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# 2. إدارة الـ Session State للشات والـ Vector Store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# كلاس مخصص لتوليد الـ Embeddings باستخدام موديل جوجل المحدث
class GeminiEmbeddings:
    def embed_documents(self, texts):
        return [
            genai.embed_content(
                model="models/text-embedding-004",  # تم تصحيح الاسم هنا
                content=t,
                task_type="retrieval_document"
            )["embedding"] for t in texts
        ]
    
    def embed_query(self, text):
        return genai.embed_content(
            model="models/text-embedding-004",      # تم تصحيح الاسم هنا
            content=text,
            task_type="retrieval_query"
        )["embedding"]

# 3. رفع ملفات الـ PDF ومعالجتها
uploaded_file = st.file_uploader("اختر ملف PDF لتقديمه للمساعد الذكي", type=["pdf"])

if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("جاري معالجة الملف واستخراج البيانات وبناء قاعدة المعرفة..."):
        # حفظ الملف المرفوع في ملف مؤقت لقراءته بواسطة البوت
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # تحميل نصوص ملف الـ PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            # تقسيم النصوص إلى أجزاء صغيرة ومناسبة للـ RAG
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # توليد الـ Embeddings وحفظها في قاعدة بيانات FAISS
            embeddings = GeminiEmbeddings()
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            st.success("تم تحليل الملف وبناء قاعدة المعرفة بنجاح! يمكنك البدء بالأسئلة الآن.")
        finally:
            # تنظيف ومسح الملف المؤقت من السيرفر فوراً
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# 4. واجهة المحادثة وعرض الرسائل
st.divider()
st.write("### 💬 نافذة المحادثة")

# عرض تاريخ الشات القديم للمستخدم بانتظام
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# استقبال السؤال الجديد من الـ Chat Input
user_query = st.chat_input("اسألني أي شيء عن الملف المرفوع...")

if user_query:
    # عرض سؤال اليوزر فوراً في الشات وحفظه
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append(("user", user_query))

    # توليد الرد الذكي بناءً على قاعدة المعرفة
    with st.chat_message("assistant"):
        if st.session_state.vector_store is not None:
            with st.spinner("جاري البحث في الملف وتوليد الإجابة النموذجية..."):
                # البحث عن أكثر الفقرات شبهاً بسؤال المستخدم
                docs = st.session_state.vector_store.similarity_search(user_query, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])

                # بناء الـ Prompt الاحترافي لضمان دقة الإجابة من الملف فقط
                prompt_template = ChatPromptTemplate.from_template("""
                أنت مساعد ذكي ومحترف وظيفتك الإجابة على أسئلة المستخدم بناءً على السياق (Context) المرفق فقط.
                إذا كانت الإجابة غير موجودة في السياق، قل بكل وضوح "المعلومة غير متوفرة في الملف المرفوع" ولا تقم باختراع إجابات.
                
                السياق المستخرج من الملف:
                {context}
                
                سؤال المستخدم:
                {question}
                """)
                
                formatted_prompt = prompt_template.format(context=context, question=user_query)
                
                # استدعاء الموديل السريع والقوي الفلاش لتوليد النص
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(formatted_prompt)
                
                answer = response.text
                st.write(answer)
                st.session_state.chat_history.append(("assistant", answer))
        else:
            # في حال عدم رفع أي ملف، يعمل البوت كمساعد ذكي عام بذكاء جينيرال
            with st.spinner("جاري التفكير..."):
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(user_query)
                answer = response.text
                st.write(answer)
                st.session_state.chat_history.append(("assistant", answer))
