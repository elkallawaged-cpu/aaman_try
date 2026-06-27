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

# أدخل الـ API Key الخاص بـ Gemini هنا (أو خليه يقراه من الـ Environment Variables)
# يمكنك الحصول عليه مجاناً من Google AI Studio
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# 2. إدارة الـ Session State للشات والـ Vector Store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 3. السايدبار لرفع الملفات
with st.sidebar:
    st.header("🗂️ مركز رفع المستندات")
    uploaded_files = st.file_uploader("ارفع ملفاتك هنا (PDF أو Text)", type=["pdf", "txt"], accept_multiple_files=True)
    
    process_button = st.button("⚡ معالجة وبناء المستندات")

# 4. معالجة الملفات وتحويلها لـ Vectors عند الضغط على الزر
if process_button and uploaded_files:
    with st.spinner("جاري قراءة الملفات وتحليلها ذكياً..."):
        all_docs = []
        
        for uploaded_file in uploaded_files:
            # حفظ الملف مؤقتاً لقراءته
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
            
            # قراءة محتوى الملف بناءً على نوعه
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                all_docs.extend(docs)
            elif uploaded_file.name.endswith('.txt'):
                with open(temp_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                from langchain_core.documents import Document
                all_docs.append(Document(page_content=text, metadata={"source": uploaded_file.name}))
            
            os.unlink(temp_path) # حذف الملف المؤقت
        
        # تقسيم النصوص إلى Chunks صغيرة ومناسبة
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        
        # استخدام هجين مخصص للـ Embeddings عبر LangChain أو استخدام نموذج Gemini مباشرة
        # هنا سنقوم بعمل الـ Indexing باستخدام محرك FAISS محلي مجاني
        # لتسهيل الـ Embedding بشكل مجاني بالكامل وسريع سنستخدم الـ Custom Embedding الخاص بـ Gemini
        class GeminiEmbeddings:
            def embed_documents(self, texts):
                return [genai.embed_content(model="models/embedding-004", content=t, task_type="retrieval_document")["embedding"] for t in texts]
            def embed_query(self, text):
                return genai.embed_content(model="models/embedding-004", content=text, task_type="retrieval_query")["embedding"]

        # بناء الـ Vector Store في الـ Memory
        embeddings = GeminiEmbeddings()
        st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
        st.success("🎯 تم بناء قاعدة البيانات بنجاح! المستند جاهز للأسئلة.")

# 5. منطقة الشات الرئيسي
if st.session_state.vector_store is not None:
    # عرض تاريخ الشات
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # استقبال سؤال المستخدم
    if user_query := st.chat_input("اسألني أي حاجة جوه الملفات اللي رفعتها..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # عمل Retrieval لأهم الـ Chunks المناسبة للسؤال
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.get_relevant_documents(user_query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # كتابة الـ Prompt الاحترافي للـ LLM
        prompt = f"""
        You are an expert AI assistant. Answer the user's question based strictly on the provided context. 
        If the answer cannot be found in the context, say politely that the information is not available in the uploaded documents.
        Always answer in the same language as the user's question (Arabic or English).
        
        Context:
        {context}
        
        Question:
        {user_query}
        
        Answer:
        """
        
        # توليد الإجابة باستخدام Gemini 1.5 Flash
        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير والاستخراج..."):
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                st.markdown(response.text)
                
                # إظهار المصادر تحت الإجابة لزيادة الجودة والـ Credibility
                with st.expander("🔍 المصادر المستند عليها من ملفاتك:"):
                    for doc in relevant_docs:
                        st.write(f"- {doc.metadata.get('source', 'ملف مرفوع')}: ... {doc.page_content[:150]} ...")
                        
        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
else:
    st.info("👈 من فضلك ارفع ملف أو أكتر من السايدبار واضغط على 'معالجة وبناء المستندات' عشان نبدأ الشات!")