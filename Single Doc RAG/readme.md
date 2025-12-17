# 📄 RAG-Based PDF Question Answering (Google Colab)

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline implemented in **Google Colab**.
It allows users to load a PDF file, retrieve relevant document content using vector search, and generate answers using a language model.

The goal of this project is to clearly explain and experiment with **RAG fundamentals** using **LangChain, FAISS, and HuggingFace models**.

## 🎯 Project Objective

To build a simple and understandable **RAG pipeline** that:

* Reads a PDF document
* Retrieves relevant text based on a user question
* Generates an answer grounded strictly in the document content

## ✨ Features

* 📂 Load and process PDF documents
* ✂️ Split text into manageable chunks
* 🔍 Semantic search using vector embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* 🤖 Answer generation using FLAN-T5
* ⚡ Runs entirely in Google Colab (CPU)

## 🏗️ RAG Workflow Overview

```
PDF File
  ↓
Text Extraction
  ↓
Text Chunking
  ↓
Embedding Generation
  ↓
FAISS Vector Store
  ↓
Context Retrieval
  ↓
LLM Answer Generation
```

## 🧠 Step-by-Step Explanation

### 1️⃣ Load PDF

The PDF file is loaded and converted into text using **PyPDFLoader**.

```python
loader = PyPDFLoader("Sample.pdf")
documents = loader.load()
```

### 2️⃣ Split Text into Chunks

The document text is divided into overlapping chunks to preserve context.

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

### 3️⃣ Generate Embeddings

Each text chunk is converted into a numerical vector using a **Sentence Transformer** model.

```python
sentence-transformers/all-MiniLM-L6-v2
```

### 4️⃣ Store Embeddings in FAISS

FAISS enables fast similarity search over document embeddings.


### 5️⃣ Retrieve Relevant Context

Given a question, FAISS retrieves the most relevant chunks from the document.

### 6️⃣ Generate Answer (RAG)

The retrieved context and the user’s question are passed to **FLAN-T5**, which generates an answer based **only on the document context**.

## 🛠️ Tech Stack

* **Python**
* **LangChain**
* **FAISS**
* **HuggingFace Transformers**
* **Sentence-Transformers**
* **PyPDF**
* **Google Colab**

## 🚀 How to Run (Google Colab)

### 1️⃣ Install Dependencies

```python
!pip install -U langchain-community langchain-text-splitters
!pip install transformers sentence-transformers faiss-cpu pypdf
```

### 2️⃣ Upload PDF

Upload your PDF file (e.g., `Sample.pdf`) to the Colab environment.

### 3️⃣ Run the Notebook Cells

Execute each cell sequentially to:

* Load the PDF
* Build the vector store
* Create the RAG chain
* Ask questions

### 4️⃣ Ask a Question

```python
response = rag_chain.invoke({
    "question": "What is the goal of the AgriPredict system?"
})
print(response)
```

## 📁 Repository Structure

```
.
├── rag_pdf_qa_colab.ipynb   # Google Colab notebook
├── Sample.pdf              # Example PDF
├── README.md               # Project documentation
```

## 👤 Author

**Sowmya**
GitHub: [Sowmya](https://github.com/sowmya13531)

Just tell me 👍
