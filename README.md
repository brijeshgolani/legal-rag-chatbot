# ⚖️ Legal Document RAG Chatbot

An AI-powered chatbot that answers questions about legal documents
using Retrieval-Augmented Generation (RAG).

Built with LangChain, ChromaDB, HuggingFace Embeddings, and Ollama.

---

## Features

- Ask questions about any legal PDF document
- Powered by local LLM (llama3.2) — completely free, no API costs
- Vector similarity search using ChromaDB
- Clean chat UI built with Streamlit
- Shows source sections used to generate each answer
- Full conversation history

## Tech Stack

| Tool                           | Purpose         |
| ------------------------------ | --------------- |
| LangChain                      | RAG pipeline    |
| ChromaDB                       | Vector database |
| HuggingFace (all-MiniLM-L6-v2) | Text embeddings |
| Ollama + llama3.2:1b           | Local LLM       |
| Streamlit                      | Web UI          |

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Install Ollama

Download from https://ollama.com and run:
ollama pull llama3.2:1b

### 3. Ingest your document

python ingest.py

### 4. Launch the app

streamlit run app.py

### 5. Open browser

Go to http://localhost:8501
