import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from operator import itemgetter

st.title("📄 RAG-based PDF Question Answering")

# ------------------------------
# Cache only the documents
# ------------------------------
@st.cache_data(show_spinner=True)
def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    return docs

# ------------------------------
# Create RAG chain (do NOT cache vectorstore)
# ------------------------------
def create_rag_chain(docs):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    hf_pipeline = pipeline('text2text-generation', model='google/flan-t5-base', max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = PromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not present, say 'I don't know'.

context:
{context}

Question:
{question}
""")
    parser = StrOutputParser()

    rag_chain = (
        {
            "context": itemgetter('question') | retriever,
            "question": itemgetter('question')
        }
        | prompt
        | llm
        | parser
    )
    return rag_chain

# ------------------------------
# Streamlit UI
# ------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save uploaded PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success(f"Uploaded file: {uploaded_file.name}")

    # Load documents (cached)
    docs = load_documents(pdf_path)

    # Build RAG chain
    rag_chain = create_rag_chain(docs)

    # Ask question
    question = st.text_input("Ask a question about the PDF:")
    if st.button("Get Answer") and question:
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke({"question": question})
        st.markdown(f"**Answer:** {response}")
