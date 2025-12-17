# 📄 RAG-Based PDF Question Answering App

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions about its content.
The system retrieves relevant document chunks using **FAISS + embeddings** and generates accurate, context-aware answers using a **HuggingFace LLM**, all wrapped in a **Streamlit** web interface.

🔗 **Live Demo:**
👉 [Live app on Streamlit Cloud](https://etkvpt5me74hud4yevawhw.streamlit.app/)

## ✨ Features
* 📂 Upload any PDF document
* 🔍 Semantic search using vector embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* 🤖 Context-aware answers using FLAN-T5
* 📚 Answers grounded strictly in document content
* ⚡ Fast and lightweight (CPU-friendly)
* 🌐 Deployed on Streamlit Cloud

## 🏗️ Architecture Overview

```
PDF Upload
    ↓
PyPDFLoader
    ↓
Text Chunking (RecursiveCharacterTextSplitter)
    ↓
Embeddings (sentence-transformers)
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
Prompt + LLM (FLAN-T5)
    ↓
Answer
```

## 🛠️ Tech Stack

* **Python**
* **Streamlit** – UI
* **LangChain** – RAG orchestration
* **FAISS** – Vector similarity search
* **HuggingFace Transformers** – LLM inference
* **Sentence-Transformers** – Text embeddings
* **PyPDF** – PDF parsing

## 🚀 Getting Started (Local Setup)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sowmya13531/Dynamic-RAG-Single-Doc-.git
cd RAG-Single-Doc-
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

The app will be available at:
👉 `http://localhost:8501`

## 📁 Project Structure

```
.
├── RAG_app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```


## 🧪 Example Usage

1. Upload a PDF file
2. Enter a question related to the document
3. Click **Get Answer**
4. Receive an answer generated strictly from the PDF content

If the answer is not found in the document, the app responds with:

> **"I don't know"**

## 🧠 Design Decisions

- Used **RAG** to avoid hallucinations and ensure grounded answers
- Selected **FAISS** for fast in-memory vector search
- Used **FLAN-T5** for lightweight, CPU-friendly inference
- Chunked text to balance context size and retrieval accuracy

## 📜 License

This project is licensed under the **MIT License**.
Feel free to use, modify, and distribute.

## 👤 Author

**Sowmya Kanithi**
🔗 GitHub: [Sowmya13531](https://github.com/sowmya13531)
