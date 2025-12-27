# 07.RAG-Powered-Machine-Learning-Interview-Mentor
RAG-Powered Machine Learning Interview Mentor

# 🤖 RAG-Powered Machine Learning Interview Mentor

An end-to-end AI web application that helps users prepare for Machine Learning interviews by providing accurate, context-aware answers using **Retrieval-Augmented Generation (RAG)** and Large Language Models (LLMs).

---

## 🚀 Project Overview

Preparing for ML interviews often requires searching through multiple resources. This project builds an **AI Interview Mentor** that allows users to upload PDFs (notes, books, resumes, guides) and ask questions. The system retrieves the most relevant content and generates precise answers using LLMs.

It combines:
- 🔍 Semantic search over documents
- 🧠 LLM reasoning
- 💬 Conversational, history-aware Q&A

---

## ✨ Key Features

- ✅ RAG-based question answering for ML topics  
- ✅ Upload and process PDF documents  
- ✅ Context-aware and history-aware responses  
- ✅ Semantic search using vector embeddings  
- ✅ Low-latency generation with **Groq LLM**  
- ✅ Interactive **Streamlit** web interface  
- ✅ End-to-end pipeline: PDF → Chunks → Vectors → LLM Answer  

---

## 🛠️ Tech Stack

**Programming:** Python  
**LLM & RAG:** LangChain, Groq (OpenAI API)  
**Embeddings:** Hugging Face  
**Vector Store:** FAISS  
**Web Framework:** Streamlit  
**Libraries:** PyPDF, NumPy, Pandas  
**Tools:** Git, GitHub  

---

🔄 How It Works

User uploads PDF files
Text is extracted and chunked
Embeddings are generated using Hugging Face models
Vectors stored in FAISS index
Relevant chunks retrieved for each query
Groq LLM generates accurate answers using retrieved context


