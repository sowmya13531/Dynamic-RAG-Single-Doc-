import streamlit as st
import tempfile
from operator import itemgetter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline

st.set_page_config(page_title="RAG PDF QA", layout="centered")
st.title("📄 RAG-based PDF Question Answering")

# ------------------------------
# Cache PDF loading & splitting
# ------------------------------
@st.cache_data(show_spinner=True)
def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

# ------------------------------
# Helper: format retrieved docs
# ------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ------------------------------
# Create RAG chain
# ------------------------------
def create_rag_chain(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = PromptTemplate.from_template(
        """Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# ------------------------------
# Streamlit UI
# ------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success(f"Uploaded: {uploaded_file.name}")

    docs = load_documents(pdf_path)
    rag_chain = create_rag_chain(docs)

    question = st.text_input("Ask a question about the PDF")

    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke({"question": question})

        st.markdown("### ✅ Answer")
        st.write(answer)
