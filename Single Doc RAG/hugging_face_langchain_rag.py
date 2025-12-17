# Install dependencies
!pip install -U langchain-community langchain-text-splitters
!pip install transformers sentence-transformers faiss-cpu pypdf

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline
from operator import itemgetter

# Load PDF
loader = PyPDFLoader("Sample.pdf")
documents = loader.load()
print("Pages loaded:", len(documents))

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = splitter.split_documents(documents)
print("Chunks created:", len(docs))

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Prompt
prompt = PromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
""")

# Helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Ask question
response = rag_chain.invoke({
    "question": "What is the goal of the AgriPredict system?"
})

print(response)
