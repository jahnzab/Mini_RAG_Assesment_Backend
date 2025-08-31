
import os
import re
import traceback
import time
import hashlib
import requests
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone, ServerlessSpec
import logging
import numpy as np

# --------------------- Configuration -----------------------

# Load environment variables from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Changed from GROK to GROQ
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in environment variables")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"✅ Groq API Key Loaded: {bool(GROQ_API_KEY)}")
print(f"✅ Pinecone API Key Loaded: {bool(PINECONE_API_KEY)}")
print(f"✅ Pinecone Environment: {PINECONE_ENVIRONMENT}")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# --------------------- Groq API Integration -------------------

def call_groq_api(prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
    """
    Call Groq API directly using REST requests.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Truncate prompt if too long (Groq has context limits)
    if len(prompt) > 6000:
        prompt = prompt[:6000] + "\n\n[Content truncated for API limits]"
    
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that answers questions based on provided document context. Always include citations [1], [2], etc. when referencing the provided sources."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "model": "Llama3-8b-8192",  # Free Groq model
        "max_tokens": min(max_tokens, 1000),  # Ensure within limits
        "temperature": temperature,
        "stream": False
    }
    
    try:
        logger.info(f"📤 Sending request to Groq API with model: {data['model']}")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30  # Reduced timeout
        )
        
        # Log response details for debugging
        logger.info(f"📥 Groq API response status: {response.status_code}")
        
        if not response.ok:
            error_text = response.text
            logger.error(f"❌ Groq API error response: {error_text}")
            raise HTTPException(status_code=response.status_code, detail=f"Groq API error: {error_text}")
        
        result = response.json()
        
        if "choices" not in result or not result["choices"]:
            raise HTTPException(status_code=500, detail="Invalid response from Groq API")
            
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.Timeout:
        logger.error("❌ Groq API request timed out")
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Groq API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Unexpected error calling Groq: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --------------------- Embeddings Implementation -------------------

try:
    from sentence_transformers import SentenceTransformer
    local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L3-v2")
    use_local_encoder = True
    logging.info("✅ Loaded local sentence-transformers model.")
except Exception as e:
    # Fallback to simple hash-based embeddings
    local_model = None
    use_local_encoder = False
    logging.warning(f"⚠️ Using fallback embeddings: {e}")

class GroqEmbeddings:
    """Custom embedding class compatible with LangChain using sentence-transformers"""

    def __init__(self):
        self.model = local_model
        self.dimension = 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if use_local_encoder:
            return self.model.encode(texts).tolist()
        else:
            # Fallback hash-based embeddings
            embeddings = []
            for text in texts:
                hash_obj = hashlib.sha256(text.encode())
                hash_hex = hash_obj.hexdigest()
                
                embedding = []
                for i in range(0, len(hash_hex), 2):
                    hex_pair = hash_hex[i:i+2]
                    embedding.append(int(hex_pair, 16) / 255.0)
                
                if len(embedding) < 384:
                    embedding.extend([0.0] * (384 - len(embedding)))
                else:
                    embedding = embedding[:384]
                    
                embeddings.append(embedding)
            return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

# --------------------- App Initialization ------------------

app = FastAPI(
    title="Mini RAG System - GroqAI + Pinecone",
    description="RAG system with GroqAI (Free Tier) + Pinecone + retriever + reranker",
    version="1.0.0"
)

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://mini-rag-assesment-front.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------- Utility Functions -------------------

def sanitize_index_name(chat_id: str) -> str:
    """
    Ensure index name follows Pinecone requirements:
    - Lowercase alphanumeric and hyphens only
    - Must start and end with alphanumeric
    - Max 45 characters
    """
    chat_id = re.sub(r'[^a-z0-9\-]', '-', chat_id.lower())
    chat_id = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', chat_id)
    
    if len(chat_id) < 3:
        chat_id = f"chat-{chat_id}-{int(time.time()) % 10000}"
    
    if len(chat_id) > 45:
        chat_id = chat_id[:45]
    
    chat_id = re.sub(r'-+$', '', chat_id)
    return chat_id

def get_or_create_pinecone_index(index_name: str = "rag-shared-index"):
    """Get existing or create a shared Pinecone index for all chats using namespaces."""
    try:
        existing_indexes = pc.list_indexes()
        index_names = [idx["name"] for idx in existing_indexes]

        if index_name not in index_names:
            logger.info(f"🆕 Creating new Pinecone shared index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENVIRONMENT
                )
            )
            time.sleep(10)
            logger.info(f"✅ Pinecone index '{index_name}' created successfully")
        else:
            logger.info(f"📚 Using existing Pinecone index: {index_name}")

        return pc.Index(index_name)

    except Exception as e:
        logger.error(f"❌ Error with Pinecone index '{index_name}': {str(e)}")
        raise e

def enhanced_text_chunking(documents: List[Document]) -> List[Document]:
    """
    Enhanced chunking strategy with metadata preservation for citations.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    enhanced_docs = []
    for doc_idx, doc in enumerate(documents):
        chunks = text_splitter.split_text(doc.page_content)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{doc_idx}_{chunk_idx}_{chunk[:50]}".encode()).hexdigest()[:8]
            
            enhanced_metadata = {
                **doc.metadata,
                "chunk_id": chunk_id,
                "doc_index": doc_idx,
                "chunk_index": chunk_idx,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "section": f"chunk_{chunk_idx}",
                "position": chunk_idx
            }
            enhanced_docs.append(Document(page_content=chunk, metadata=enhanced_metadata))
    
    logger.info(f"✅ Created {len(enhanced_docs)} chunks with enhanced metadata")
    return enhanced_docs

def simple_reranker(retrieved_docs: List[Document], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Simple reranker implementation."""
    query_lower = query.lower()
    scored_docs = []
    
    for doc in retrieved_docs:
        content_lower = doc.page_content.lower()
        score = 0
        query_words = query_lower.split()
        
        for word in query_words:
            if word in content_lower:
                score += content_lower.count(word) * 2
        
        if any(word in content_lower for word in query_words):
            score += 1
            
        scored_docs.append({
            "document": doc,
            "score": score,
            "metadata": doc.metadata
        })
    
    reranked = sorted(scored_docs, key=lambda x: x["score"], reverse=True)[:top_k]
    logger.info(f"✅ Reranked {len(retrieved_docs)} docs, returning top {top_k}")
    
    return reranked

def process_pdf_with_pinecone(content: bytes, chat_id: str):
    """Enhanced PDF processing with Pinecone cloud vector storage using namespaces."""
    sanitized_chat_id = sanitize_index_name(chat_id)
    logger.info(f"Starting enhanced PDF processing for chat ID: {sanitized_chat_id}")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    logger.info(f"Temporary PDF file created at: {tmp_path}")

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF.")
        
        chunked_documents = enhanced_text_chunking(documents)
        
    finally:
        os.remove(tmp_path)
        logger.info(f"Temporary file removed.")

    if not chunked_documents:
        logger.warning("No documents found in PDF. Skipping embedding process.")
        return

    embeddings = GroqEmbeddings()
    logger.info("Initialized GroqAI-compatible Embeddings.")

    index = get_or_create_pinecone_index("rag-shared-index")
    
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=sanitized_chat_id
    )
    
    vector_store.add_documents(chunked_documents)
    logger.info(f"Added {len(chunked_documents)} chunks to namespace '{sanitized_chat_id}'.")

def process_text_with_pinecone(text: str, chat_id: str, title: str = "Uploaded Text"):
    """Process text content with Pinecone cloud vector storage using namespaces."""
    sanitized_chat_id = sanitize_index_name(chat_id)
    logger.info(f"Starting text processing for chat ID: {sanitized_chat_id}")
    
    doc = Document(
        page_content=text,
        metadata={
            "source": title,
            "type": "text_upload",
            "title": title
        }
    )
    
    chunked_documents = enhanced_text_chunking([doc])
    
    if not chunked_documents:
        logger.warning("No content to process.")
        return
    
    embeddings = GroqEmbeddings()
    index = get_or_create_pinecone_index("rag-shared-index")
    
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=sanitized_chat_id
    )
    
    vector_store.add_documents(chunked_documents)
    logger.info(f"Added {len(chunked_documents)} text chunks to namespace '{sanitized_chat_id}'.")

def get_pinecone_retriever(chat_id: str):
    """Get Pinecone retriever for chat ID using namespaces."""
    sanitized_chat_id = sanitize_index_name(chat_id)
    logger.info(f"Getting Pinecone retriever for chat ID: {sanitized_chat_id}")

    existing_indexes = pc.list_indexes()
    index_names = [idx["name"] for idx in existing_indexes]

    if not index_names:
        raise HTTPException(
            status_code=404,
            detail="No Pinecone indexes found. Please upload documents first."
        )

    shared_index_name = "rag-shared-index" if "rag-shared-index" in index_names else index_names[0]

    embeddings = GroqEmbeddings()
    index = pc.Index(shared_index_name)

    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})

        if sanitized_chat_id not in namespaces:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found for chat ID '{chat_id}'. Please upload documents first."
            )
    except Exception as e:
        logger.warning(f"Could not check namespace stats: {e}")

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=sanitized_chat_id
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 8,
            "score_threshold": 0.0
        }
    )

    logger.info(f"Pinecone retriever initialized for namespace '{sanitized_chat_id}' in index '{shared_index_name}'.")
    return retriever

def generate_answer_with_citations(question: str, reranked_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate answer with inline citations using GroqAI.
    """
    start_time = time.time()
    
    context_parts = []
    source_snippets = []
    
    for i, doc_info in enumerate(reranked_docs, 1):
        doc = doc_info["document"]
        citation_id = f"[{i}]"
        # Limit context length per source
        content = doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content
        context_parts.append(f"{citation_id} {content}")
        
        source_snippets.append({
            "id": i,
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata,
            "page": doc.metadata.get("page", "Unknown"),
            "source": doc.metadata.get("source", "Unknown"),
            "chunk_id": doc.metadata.get("chunk_id", f"chunk_{i}")
        })
    
    context = "\n\n".join(context_parts)
    
    # Keep prompt concise for Groq limits
    prompt = f"""Based on the following context, answer the question with inline citations.

Context:
{context[:3000]}

Question: {question}

Provide a clear answer using only the context above. Include citations like [1], [2] for your sources.

Answer:"""
    
    try:
        answer = call_groq_api(prompt, max_tokens=800, temperature=0.1)
        
        processing_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": source_snippets,
            "processing_time": round(processing_time, 2),
            "retrieved_chunks": len(reranked_docs)
        }
    
    except Exception as e:
        logger.error(f"❌ Error generating answer with Groq: {str(e)}")
        # Return a fallback response with context
        fallback_answer = f"I found relevant information but couldn't generate a complete response. Here's what I found in the documents:\n\n"
        for snippet in source_snippets[:2]:
            fallback_answer += f"[{snippet['id']}] {snippet['content']}\n\n"
        
        return {
            "answer": fallback_answer,
            "sources": source_snippets,
            "processing_time": round(time.time() - start_time, 2),
            "retrieved_chunks": len(reranked_docs),
            "error": str(e)
        }

# --------------------- API Routes --------------------------

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), chat_id: str = Form(...)):
    """Upload PDF and process it into Pinecone cloud vector database."""
    start_time = time.time()
    
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        process_pdf_with_pinecone(content, chat_id)
        
        processing_time = time.time() - start_time
        
        return {
            "message": f"✅ PDF processed successfully and stored in Pinecone cloud",
            "chat_id": sanitize_index_name(chat_id),
            "filename": file.filename,
            "processing_time": round(processing_time, 2),
            "file_size": len(content),
            "vector_db": "Pinecone Cloud",
            "llm_model": "GroqAI (Llama3-8b-8192)"
        }
    except Exception as e:
        logger.error(f"❌ Error uploading PDF: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            content={
                "error": f"Failed to process PDF: {str(e)}",
                "processing_time": round(time.time() - start_time, 2)
            }, 
            status_code=500
        )

@app.post("/upload_text/")
async def upload_text(
    text: str = Form(...), 
    chat_id: str = Form(...), 
    title: str = Form("Uploaded Text")
):
    """Upload text content and store it in Pinecone under a namespace."""
    start_time = time.time()
    
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        process_text_with_pinecone(text, chat_id, title)
        
        processing_time = time.time() - start_time
        
        return {
            "message": "✅ Text processed successfully and stored in Pinecone cloud",
            "chat_id": sanitize_index_name(chat_id),
            "title": title,
            "processing_time": round(processing_time, 2),
            "text_length": len(text),
            "vector_db": "Pinecone Cloud (Namespace)",
            "llm_model": "GroqAI (Llama3-8b-8192)"
        }
        
    except Exception as e:
        logger.error(f"❌ Error uploading text: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            content={
                "error": f"Failed to process text: {str(e)}",
                "processing_time": round(time.time() - start_time, 2)
            }, 
            status_code=500
        )

@app.post("/chat/")
async def enhanced_chat(chat_id: str = Form(...), message: str = Form(...)):
    """
    Enhanced chat with GroqAI + Pinecone retriever + reranker pipeline and citations.
    """
    start_time = time.time()
    
    try:
        if not message.strip():
            raise HTTPException(status_code=400, detail="Empty message provided")
        
        existing_indexes = pc.list_indexes()
        if not existing_indexes:
            return JSONResponse(
                content={
                    "error": f"No indexes found. Please upload documents first.",
                    "suggestions": "Upload a PDF or text document before starting the chat."
                },
                status_code=404
            )
        
        # Step 1: Retrieve from Pinecone
        retriever = get_pinecone_retriever(chat_id)
        retrieved_docs = retriever.invoke(message)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        if not retrieved_docs:
            return {
                "response": "I couldn't find relevant information in the uploaded documents to answer your question.",
                "answer": "I couldn't find relevant information in the uploaded documents to answer your question.",
                "sources": [],
                "retrieved_chunks": 0,
                "processing_time": round(time.time() - start_time, 2),
                "reranked_chunks": 0,
                "vector_db": "Pinecone Cloud (Namespace)",
                "llm_model": "GroqAI (Llama3-8b-8192)",
                "error": None
            }
        
        # Step 2: Apply reranker
        reranked_docs = simple_reranker(retrieved_docs, message, top_k=5)
        logger.info(f"Reranked to top {len(reranked_docs)} most relevant chunks")
        
        # Step 3: Generate answer with citations using GroqAI
        result = generate_answer_with_citations(message, reranked_docs)
        
        result.update({
            "response": result["answer"],
            "retrieved_chunks": len(retrieved_docs),
            "reranked_chunks": len(reranked_docs),
            "total_processing_time": round(time.time() - start_time, 2),
            "vector_db": "Pinecone Cloud (Namespace)",
            "llm_model": "GroqAI (llama-3.1-70b-versatile)",
            "namespace": sanitize_index_name(chat_id)
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced chat: {str(e)}")
        return JSONResponse(
            content={
                "error": f"Chat processing failed: {str(e)}",
                "processing_time": round(time.time() - start_time, 2)
            }, 
            status_code=500
        )

@app.delete("/delete_chat/{chat_id}")
def delete_chat(chat_id: str):
    """Delete vectors from namespace."""
    sanitized_chat_id = sanitize_index_name(chat_id)
    
    try:
        existing_indexes = pc.list_indexes()
        index_names = [idx['name'] for idx in existing_indexes]

        if not index_names:
            raise HTTPException(status_code=404, detail="No indexes found")
        
        # Always use the shared index
        shared_index_name = "rag-shared-index" if "rag-shared-index" in index_names else index_names[0]
        index = pc.Index(shared_index_name)
        
        # Check if namespace exists first
        try:
            stats = index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            
            if sanitized_chat_id not in namespaces:
                return JSONResponse(
                    content={
                        "message": f"Chat ID '{chat_id}' not found or already deleted",
                        "namespace": sanitized_chat_id,
                        "available_namespaces": list(namespaces.keys())
                    },
                    status_code=404
                )
        except Exception as e:
            logger.warning(f"Could not check namespace stats: {e}")
        
        # Delete all vectors in the namespace
        try:
            # Use the correct delete method for namespaces
            delete_response = index.delete(delete_all=True, namespace=sanitized_chat_id)
            logger.info(f"Delete response: {delete_response}")
            logger.info(f"Successfully deleted all vectors in namespace '{sanitized_chat_id}'")
            
            return {
                "message": f"Chat ID '{chat_id}' deleted successfully from Pinecone",
                "deleted_namespace": sanitized_chat_id,
                "index_used": shared_index_name,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error deleting namespace '{sanitized_chat_id}': {str(e)}")
            return JSONResponse(
                content={
                    "message": f"Failed to delete chat ID '{chat_id}'",
                    "namespace": sanitized_chat_id,
                    "error": str(e)
                },
                status_code=500
            )

    except Exception as e:
        logger.error(f"Error in delete_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_chats/")
def list_chat_ids():
    """List all available namespaces (chat IDs) in shared indexes."""
    try:
        indexes = pc.list_indexes()
        
        if not indexes:
            return {
                "chat_ids": [],  # Match original format
                "chats": [],
                "vector_db": "Pinecone Cloud (Namespace)",
                "llm_model": "GroqAI (llama-3.1-70b-versatile)",
                "total_namespaces": 0
            }
        
        chat_info = []
        chat_ids = []  # Simple list for backward compatibility
        
        for idx in indexes:
            try:
                index = pc.Index(idx['name'])
                stats = index.describe_index_stats()
                
                namespaces = stats.namespaces if hasattr(stats, 'namespaces') else {}
                
                for namespace_name, namespace_stats in namespaces.items():
                    chat_ids.append(namespace_name)  # Simple format
                    chat_info.append({
                        "chat_id": namespace_name,
                        "namespace": namespace_name,
                        "index_name": idx['name'],
                        "vector_count": namespace_stats.vector_count if hasattr(namespace_stats, 'vector_count') else 0,
                        "dimension": stats.dimension,
                        "cloud": "Pinecone",
                        "llm_model": "GroqAI (llama-3.1-70b-versatile)",
                        "status": "ready"
                    })
                
            except Exception as e:
                logger.warning(f"Could not get stats for index {idx['name']}: {e}")
        
        return {
            "chat_ids": chat_ids,  # Original format for compatibility
            "chats": chat_info,    # Detailed format
            "vector_db": "Pinecone Cloud (Namespace)",
            "llm_model": "GroqAI (llama-3.1-70b-versatile)",
            "total_namespaces": len(chat_info)
        }
        
    except Exception as e:
        logger.error(f"Error listing chats: {str(e)}")
        # Return safe fallback to prevent frontend crashes
        return {
            "chat_ids": [],
            "chats": [],
            "error": str(e),
            "vector_db": "Pinecone Cloud (Namespace)",
            "llm_model": "GroqAI (llama-3.1-70b-versatile)",
            "total_namespaces": 0
        }

@app.get("/debug_pinecone")
def debug_pinecone():
    """Debug endpoint to check Pinecone connection and indexes."""
    try:
        # Test Pinecone connection
        indexes = pc.list_indexes()
        
        debug_info = {
            "pinecone_connected": True,
            "api_key_configured": bool(PINECONE_API_KEY),
            "environment": PINECONE_ENVIRONMENT,
            "total_indexes": len(indexes),
            "indexes": []
        }
        
        for idx in indexes:
            index_info = {
                "name": idx["name"],
                "dimension": idx.get("dimension", "unknown"),
                "metric": idx.get("metric", "unknown"),
                "status": idx.get("status", "unknown")
            }
            
            try:
                # Get detailed stats
                index = pc.Index(idx["name"])
                stats = index.describe_index_stats()
                index_info.update({
                    "total_vectors": stats.total_vector_count,
                    "namespaces": list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') else []
                })
            except Exception as e:
                index_info["error"] = str(e)
            
            debug_info["indexes"].append(index_info)
        
        return debug_info
        
    except Exception as e:
        return {
            "pinecone_connected": False,
            "error": str(e),
            "api_key_configured": bool(PINECONE_API_KEY),
            "environment": PINECONE_ENVIRONMENT,
            "suggestion": "Check your Pinecone API key and environment in .env file"
        }

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Mini RAG System - GroqAI + Pinecone (Free Tier)",
        "version": "1.0.0",
        "vector_db": "Pinecone Cloud",
        "llm_model": "GroqAI (llama-3.1-70b-versatile)",
        "features": [
            "Enhanced PDF/text processing",
            "Cloud vector storage (Pinecone)", 
            "Simple reranking",
            "Citation generation with GroqAI",
            "Fast response times",
            "Graceful error handling",
            "Free tier compatible"
        ],
        "endpoints": [
            "/upload_pdf/", 
            "/upload_text/", 
            "/chat/", 
            "/list_chats/", 
            "/delete_chat/{chat_id}",
            "/debug_pinecone",
            "/health"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
