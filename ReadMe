# Mini RAG System – GroqAI + Pinecone

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-%5E0.95-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **Retrieval-Augmented Generation (RAG)** system integrating **GroqAI** (Free Tier) with **Pinecone Cloud** vector database for PDF and text processing. Provides AI-generated answers with citations using uploaded documents.

---

## Features

- ✅ PDF & Text Upload
- ✅ Embeddings using `sentence-transformers` (fallback to hash-based)
- ✅ Pinecone vector database with namespaces per chat
- ✅ Simple reranking for relevance
- ✅ Citation-aware answers via GroqAI
- ✅ API with FastAPI and CORS support
- ✅ Namespace management (list, delete)
- ✅ Fast, error-handled responses
- ✅ Free-tier compatible

---

## System Requirements

- Python 3.10+
- FastAPI
- Pinecone client
- LangChain (`langchain`, `langchain_community`, `langchain_pinecone`)
- `sentence-transformers` (optional)
- `requests`, `numpy`
- `.env` file with:

```env
GROQ_API_KEY=<Your Groq API Key>
PINECONE_API_KEY=<Your Pinecone API Key>
PINECONE_ENVIRONMENT=us-east-1
```

git clone <repository-url>
cd mini-rag-groq-pinecone
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
pip install -r requirements.txt

Running the Application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
API Endpoints
Endpoint Method Description
/upload_pdf/ POST Upload a PDF and process into Pinecone vectors
/upload_text/ POST Upload text content into Pinecone
/chat/ POST Ask a question; retrieves and reranks chunks, generates answer with citations
/delete_chat/{chat_id} DELETE Delete a namespace (chat) and its vectors
/list_chats/ GET List all available chat namespaces
/debug_pinecone GET Check Pinecone connection and index status
/ GET Root endpoint with system info

Usage Examples
Upload a PDF
curl -X POST "http://localhost:8000/upload_pdf/" \
 -F "file=@example.pdf" \
 -F "chat_id=mychat"
Upload Text
curl -X POST "http://localhost:8000/upload_text/" \
 -F "text=This is a sample document." \
 -F "chat_id=mychat" \
 -F "title=Sample Text"
Ask a Question
curl -X POST "http://localhost:8000/chat/" \
 -F "chat_id=mychat" \
 -F "message=What is the main topic of the document?"
Delete a Chat
curl -X DELETE "http://localhost:8000/delete_chat/mychat"

List Available Chats
curl -X GET "http://localhost:8000/list_chats/"

How It Works

Document Upload: PDFs or text are converted into vector embeddings.

Vector Storage: Data is stored in Pinecone Cloud under a namespace per chat ID.

Retrieval: Queries retrieve relevant chunks from the namespace using cosine similarity.

Reranking: Top chunks are reranked for relevance.

Answer Generation: GroqAI produces an answer with inline citations referencing the sources.

Logging & Debugging

Detailed logs for every step (info, warning, error).

Use /debug_pinecone to inspect indexes, namespaces, and vectors.

Notes

GroqAI Free Tier limits context; prompts are truncated automatically.

If sentence-transformers is unavailable, fallback hash-based embeddings are used.

Pinecone namespaces isolate multiple chats for concurrent storage.
