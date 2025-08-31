import os
import re
import traceback
import time
import hashlib
import requests
import json
from typing import List, Dict, Any, Optional
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
from datetime import datetime
import asyncio

# --------------------- Configuration -----------------------

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Detect deployment environment
IS_RENDER = os.getenv("RENDER") is not None
IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production" or IS_RENDER

# Environment-specific settings
CLOUD_TIMEOUT = 60 if IS_PRODUCTION else 30
EMBEDDING_BATCH_SIZE = 10 if IS_PRODUCTION else 20
SIMILARITY_THRESHOLD = 0.1 if IS_PRODUCTION else 0.3
VECTOR_SEARCH_TOP_K = 15 if IS_PRODUCTION else 10

# Validate environment variables
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")
if not NOMIC_API_KEY:
    raise ValueError("NOMIC_API_KEY not found in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print(f"Environment: {'Production' if IS_PRODUCTION else 'Development'}")
print(f"Groq API Key Loaded: {bool(GROQ_API_KEY)}")
print(f"Nomic API Key Loaded: {bool(NOMIC_API_KEY)}")
print(f"Pinecone API Key Loaded: {bool(PINECONE_API_KEY)}")
print(f"Pinecone Environment: {PINECONE_ENVIRONMENT}")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# --------------------- Enhanced Groq API Integration -------------------

def call_groq_api(prompt: str, max_tokens: int = 1500, temperature: float = 0.1) -> str:
    """Enhanced Groq API call with better error handling and retry logic."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Smart prompt truncation
    if len(prompt) > 7000:
        prompt = prompt[:7000] + "\n\n[Content truncated for API limits]"
    
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert AI assistant that provides comprehensive answers based on document context. Always include specific citations [1], [2], etc. when referencing sources. Be detailed and thorough in your responses."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "model": "llama-3.1-8b-instant",  # Better model for production
        "max_tokens": min(max_tokens, 2000),
        "temperature": temperature,
        "stream": False
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending request to Groq API (attempt {attempt + 1})")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=CLOUD_TIMEOUT
            )
            
            logger.info(f"Groq API response status: {response.status_code}")
            
            if response.ok:
                result = response.json()
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]
                else:
                    raise HTTPException(status_code=500, detail="Invalid response format from Groq API")
            else:
                error_text = response.text
                logger.error(f"Groq API error: {error_text}")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=response.status_code, detail=f"Groq API error: {error_text}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
        except requests.exceptions.Timeout:
            logger.error(f"Groq API timeout (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
            time.sleep(2 ** attempt)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")
            time.sleep(2 ** attempt)

# --------------------- Production-Ready Nomic Embeddings -------------------

class EnhancedGroqEmbeddings:
    """Production-ready embedding class with consistency and cloud optimization."""

    def __init__(self):
        self.dimension = 384
        self.model_name = "nomic-embed-text-v1.5"
        self.base_url = "https://api-atlas.nomic.ai"
        self._cache = {}  # Embedding cache for consistency
        self.max_cache_size = 1000
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent embeddings."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        return text[:8192]  # Nomic's max context length
        
    def _get_cache_key(self, text: str, task_type: str) -> str:
        """Generate consistent cache key."""
        normalized_text = self._normalize_text(text)
        return f"{task_type}:{hashlib.sha256(normalized_text.encode()).hexdigest()[:16]}"
    
    def _manage_cache(self):
        """Manage cache size to prevent memory issues."""
        if len(self._cache) > self.max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(self._cache) // 5
            keys_to_remove = list(self._cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cache cleaned: removed {items_to_remove} entries")

    def _get_nomic_embedding(self, text: str, task_type: str = "search_document") -> List[float]:
        """Get embedding from Nomic AI with enhanced reliability."""
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return [0.0] * self.dimension
            
        cache_key = self._get_cache_key(normalized_text, task_type)
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        headers = {
            "Authorization": f"Bearer {NOMIC_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "texts": [normalized_text],
            "task_type": task_type,
            "dimensionality": self.dimension
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/embedding/text",
                    headers=headers,
                    json=data,
                    timeout=CLOUD_TIMEOUT
                )
                
                if response.ok:
                    result = response.json()
                    embedding = result["embeddings"][0]
                    
                    # Ensure correct dimension and normalize
                    if len(embedding) != self.dimension:
                        logger.warning(f"Unexpected embedding dimension: {len(embedding)}")
                        embedding = embedding[:self.dimension] + [0.0] * max(0, self.dimension - len(embedding))
                    
                    # L2 normalize for better similarity matching
                    embedding = np.array(embedding, dtype=np.float32)
                    norm = np.linalg.norm(embedding)
                    if norm > 1e-8:
                        embedding = embedding / norm
                    embedding = embedding.tolist()
                    
                    # Cache result
                    self._cache[cache_key] = embedding
                    self._manage_cache()
                    
                    return embedding
                else:
                    logger.error(f"Nomic API error (attempt {attempt+1}): {response.status_code} - {response.text}")
                    if attempt == max_retries - 1:
                        return self._fallback_embedding(normalized_text)
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"Nomic request error (attempt {attempt+1}): {str(e)}")
                if attempt == max_retries - 1:
                    return self._fallback_embedding(normalized_text)
                time.sleep(2 ** attempt)
        
        return self._fallback_embedding(normalized_text)

    def _fallback_embedding(self, text: str) -> List[float]:
        """Enhanced fallback embedding with better distribution."""
        try:
            # Use multiple hash sources for better distribution
            hash1 = hashlib.sha256(text.encode()).hexdigest()
            hash2 = hashlib.md5(text.encode()).hexdigest()
            combined_hash = hash1 + hash2
            
            embedding = []
            for i in range(0, min(len(combined_hash), self.dimension * 2), 2):
                if i + 1 < len(combined_hash):
                    hex_pair = combined_hash[i:i+2]
                    value = (int(hex_pair, 16) / 255.0 - 0.5) * 2  # Center around 0, range [-1, 1]
                    embedding.append(value)
            
            # Pad or truncate to exact dimension
            while len(embedding) < self.dimension:
                embedding.append(0.0)
            embedding = embedding[:self.dimension]
            
            # Normalize
            embedding = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            
            logger.warning(f"Using fallback embedding for text: {text[:50]}...")
            return embedding.tolist()
            
        except Exception:
            # Ultimate fallback
            return [0.1] * self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with enhanced batch processing."""
        if not texts:
            return []
            
        embeddings = []
        batch_size = EMBEDDING_BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self._get_nomic_batch_embeddings(batch, "search_document")
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: {len(embeddings)}/{len(texts)} embeddings")
                
                # Cloud-friendly delay
                if IS_PRODUCTION:
                    time.sleep(0.2)
                    
            except Exception as e:
                logger.error(f"Batch {i//batch_size + 1} failed: {str(e)}")
                for text in batch:
                    embeddings.append(self._fallback_embedding(text))
        
        logger.info(f"Generated {len(embeddings)} total embeddings")
        return embeddings

    def _get_nomic_batch_embeddings(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Get embeddings for batch with enhanced error handling."""
        normalized_texts = [self._normalize_text(text) for text in texts]
        clean_texts = [text for text in normalized_texts if text]
        
        if not clean_texts:
            return [self._fallback_embedding("") for _ in texts]
        
        headers = {
            "Authorization": f"Bearer {NOMIC_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "texts": clean_texts,
            "task_type": task_type,
            "dimensionality": self.dimension
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/embedding/text",
                headers=headers,
                json=data,
                timeout=CLOUD_TIMEOUT
            )
            
            if response.ok:
                result = response.json()
                embeddings = result["embeddings"]
                
                # Normalize all embeddings
                normalized_embeddings = []
                for emb in embeddings:
                    if len(emb) != self.dimension:
                        emb = emb[:self.dimension] + [0.0] * max(0, self.dimension - len(emb))
                    
                    emb_array = np.array(emb, dtype=np.float32)
                    norm = np.linalg.norm(emb_array)
                    if norm > 1e-8:
                        emb_array = emb_array / norm
                    normalized_embeddings.append(emb_array.tolist())
                
                return normalized_embeddings
            else:
                logger.error(f"Nomic batch API error: {response.status_code} - {response.text}")
                return [self._fallback_embedding(text) for text in clean_texts]
                
        except Exception as e:
            logger.error(f"Nomic batch embedding error: {str(e)}")
            return [self._fallback_embedding(text) for text in clean_texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query with consistency measures."""
        return self._get_nomic_embedding(text, task_type="search_query")

# --------------------- Enhanced App Configuration ------------------

app = FastAPI(
    title="Production RAG System - GroqAI + Pinecone + Nomic",
    description="Production-ready RAG system optimized for cloud deployment",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://mini-rag-assesment-front.vercel.app",
        "https://*.vercel.app",
        "https://*.render.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------- Enhanced Utility Functions -------------------

def sanitize_index_name(chat_id: str) -> str:
    """Enhanced index name sanitization."""
    chat_id = re.sub(r'[^a-z0-9\-]', '-', chat_id.lower())
    chat_id = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', chat_id)
    chat_id = re.sub(r'-+', '-', chat_id)
    
    if len(chat_id) < 3:
        timestamp = str(int(time.time()) % 100000)
        chat_id = f"chat-{chat_id}-{timestamp}"
    
    if len(chat_id) > 45:
        chat_id = chat_id[:40] + str(hash(chat_id) % 10000)
    
    chat_id = re.sub(r'-+$', '', chat_id)
    return chat_id

def get_or_create_pinecone_index(index_name: str = "rag-shared-index"):
    """Enhanced Pinecone index management with better error handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            existing_indexes = pc.list_indexes()
            index_names = [idx["name"] for idx in existing_indexes]

            if index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=PINECONE_ENVIRONMENT
                    )
                )
                # Wait longer for cloud propagation
                wait_time = 15 if IS_PRODUCTION else 10
                logger.info(f"Waiting {wait_time}s for index creation...")
                time.sleep(wait_time)
                
                # Verify creation
                updated_indexes = pc.list_indexes()
                if index_name not in [idx["name"] for idx in updated_indexes]:
                    raise Exception(f"Index {index_name} creation verification failed")
                    
                logger.info(f"Pinecone index '{index_name}' created successfully")
            else:
                logger.info(f"Using existing Pinecone index: {index_name}")

            return pc.Index(index_name)

        except Exception as e:
            logger.error(f"Pinecone index error (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(5)

def enhanced_text_chunking(documents: List[Document]) -> List[Document]:
    """Enhanced chunking with better metadata and overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Slightly larger for better context
        chunk_overlap=200,  # More overlap for better retrieval
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    enhanced_docs = []
    for doc_idx, doc in enumerate(documents):
        if not doc.page_content.strip():
            continue
            
        chunks = text_splitter.split_text(doc.page_content)
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very small chunks
                continue
                
            chunk_id = hashlib.md5(f"{doc_idx}_{chunk_idx}_{chunk[:100]}".encode()).hexdigest()[:12]
            
            enhanced_metadata = {
                **doc.metadata,
                "chunk_id": chunk_id,
                "doc_index": doc_idx,
                "chunk_index": chunk_idx,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", doc_idx),
                "chunk_size": len(chunk),
                "timestamp": datetime.now().isoformat(),
                "total_chunks": len(chunks)
            }
            
            enhanced_docs.append(Document(
                page_content=chunk.strip(),
                metadata=enhanced_metadata
            ))
    
    logger.info(f"Created {len(enhanced_docs)} enhanced chunks from {len(documents)} documents")
    return enhanced_docs

def advanced_reranker(retrieved_docs: List[Document], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Advanced reranking with multiple scoring factors."""
    if not retrieved_docs:
        return []
        
    query_lower = query.lower()
    query_words = set(query_lower.split())
    scored_docs = []
    
    for doc in retrieved_docs:
        content_lower = doc.page_content.lower()
        content_words = set(content_lower.split())
        
        # Multiple scoring factors
        exact_matches = sum(1 for word in query_words if word in content_lower)
        word_overlap = len(query_words.intersection(content_words))
        content_length_bonus = min(len(doc.page_content) / 1000, 1.0)
        
        # Position bonus (earlier chunks often more important)
        position_bonus = 1.0 / (doc.metadata.get("chunk_index", 0) + 1) * 0.1
        
        # Calculate final score
        score = (
            exact_matches * 3.0 +
            word_overlap * 2.0 +
            content_length_bonus +
            position_bonus
        )
        
        scored_docs.append({
            "document": doc,
            "score": score,
            "metadata": doc.metadata,
            "debug": {
                "exact_matches": exact_matches,
                "word_overlap": word_overlap,
                "content_length": len(doc.page_content)
            }
        })
    
    # Sort and return top results
    reranked = sorted(scored_docs, key=lambda x: x["score"], reverse=True)[:top_k]
    logger.info(f"Reranked {len(retrieved_docs)} docs to top {len(reranked)} (scores: {[round(d['score'], 2) for d in reranked[:3]]})")
    
    return reranked

def process_pdf_with_enhanced_verification(content: bytes, chat_id: str):
    """Enhanced PDF processing with verification for cloud deployment."""
    sanitized_chat_id = sanitize_index_name(chat_id)
    logger.info(f"Processing PDF for chat ID: {sanitized_chat_id}")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        if not documents:
            raise ValueError("No content extracted from PDF")
        
        chunked_documents = enhanced_text_chunking(documents)
        if not chunked_documents:
            raise ValueError("No valid chunks created from PDF")
            
    finally:
        os.remove(tmp_path)

    embeddings = EnhancedGroqEmbeddings()
    index = get_or_create_pinecone_index("rag-shared-index")

    # Upload with verification
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=sanitized_chat_id
    )
    
    logger.info(f"Adding {len(chunked_documents)} chunks to namespace '{sanitized_chat_id}'")
    vector_store.add_documents(chunked_documents)

    # Enhanced verification with retries
    verification_attempts = 3
    for attempt in range(verification_attempts):
        try:
            time.sleep(2 + attempt)  # Progressive wait
            stats = index.describe_index_stats()
            
            if hasattr(stats, 'namespaces') and sanitized_chat_id in stats.namespaces:
                vector_count = stats.namespaces[sanitized_chat_id].vector_count
                logger.info(f"Verification successful: namespace '{sanitized_chat_id}' has {vector_count} vectors")
                
                # Test retrieval immediately
                test_query = "test document content"
                test_embedding = embeddings.embed_query(test_query)
                test_results = index.query(
                    vector=test_embedding,
                    top_k=3,
                    namespace=sanitized_chat_id,
                    include_metadata=True
                )
                logger.info(f"Test retrieval: found {len(test_results.matches)} matches")
                break
            else:
                logger.warning(f"Verification attempt {attempt + 1}: namespace not found")
                if attempt == verification_attempts - 1:
                    logger.error(f"Failed to verify upload after {verification_attempts} attempts")
                    
        except Exception as e:
            logger.warning(f"Verification attempt {attempt + 1} error: {str(e)}")

def get_enhanced_pinecone_retriever(chat_id: str):
    """Enhanced retriever with comprehensive debugging and fallback strategies."""
    sanitized_chat_id = sanitize_index_name(chat_id)
    logger.info(f"Getting enhanced retriever for chat ID: {sanitized_chat_id}")

    existing_indexes = pc.list_indexes()
    if not existing_indexes:
        raise HTTPException(status_code=404, detail="No Pinecone indexes found")

    index_names = [str(idx["name"]) for idx in existing_indexes]
    shared_index_name = "rag-shared-index" if "rag-shared-index" in index_names else index_names[0]
    
    embeddings = EnhancedGroqEmbeddings()
    index = pc.Index(shared_index_name)

    # Comprehensive namespace checking
    target_namespace = sanitized_chat_id
    try:
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats.total_vector_count} total vectors")

        if hasattr(stats, "namespaces"):
            namespaces = stats.namespaces
            logger.info(f"Available namespaces: {list(namespaces.keys())}")

            if sanitized_chat_id in namespaces:
                vector_count = namespaces[sanitized_chat_id].vector_count
                logger.info(f"Target namespace '{sanitized_chat_id}' has {vector_count} vectors")
                
                if vector_count == 0:
                    logger.warning(f"Namespace '{sanitized_chat_id}' exists but has 0 vectors")
                    
            else:
                logger.warning(f"Namespace '{sanitized_chat_id}' not found")
                # Try to find any namespace with vectors
                available_ns = [ns for ns, info in namespaces.items() if info.vector_count > 0]
                if available_ns:
                    target_namespace = available_ns[0]
                    logger.info(f"Using fallback namespace: '{target_namespace}'")
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No documents found for chat ID '{chat_id}'. Available: {list(namespaces.keys())}"
                    )

    except Exception as e:
        logger.error(f"Error checking namespaces: {str(e)}")

    # Create enhanced vector store
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=target_namespace
    )
    
    # Create retriever with cloud-optimized settings
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": VECTOR_SEARCH_TOP_K,
            "score_threshold": SIMILARITY_THRESHOLD
        }
    )

    logger.info(f"Enhanced retriever ready for namespace '{target_namespace}'")
    return retriever, target_namespace, index, embeddings

def generate_comprehensive_answer(question: str, reranked_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive answer with enhanced citations."""
    start_time = time.time()
    
    if not reranked_docs:
        return {
            "answer": "No relevant information found in the uploaded documents.",
            "sources": [],
            "processing_time": 0,
            "retrieved_chunks": 0
        }
    
    # Build context with better formatting
    context_parts = []
    source_snippets = []
    
    for i, doc_info in enumerate(reranked_docs, 1):
        doc = doc_info["document"]
        score = doc_info.get("score", 0)
        
        # Enhanced context with metadata
        source_info = doc.metadata.get("source", "Unknown")
        page_info = doc.metadata.get("page", "Unknown")
        
        content = doc.page_content
        if len(content) > 1000:
            content = content[:1000] + "..."
            
        context_parts.append(f"[{i}] (Score: {score:.2f}, Source: {source_info}, Page: {page_info})\n{content}")
        
        source_snippets.append({
            "id": i,
            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": round(score, 3),
            "page": page_info,
            "source": source_info,
            "chunk_id": doc.metadata.get("chunk_id", f"chunk_{i}")
        })
    
    context = "\n\n".join(context_parts)
    
    # Enhanced prompt for better answers
    prompt = f"""Based on the following context from uploaded documents, provide a comprehensive and detailed answer to the question.

CONTEXT FROM DOCUMENTS:
{context[:4000]}

QUESTION: {question}

INSTRUCTIONS:
- Provide a thorough, well-structured answer using ONLY the context above
- Include citations [1], [2], etc. for every claim you make
- If the context doesn't fully answer the question, state what information is missing
- Be specific and detailed in your response
- Structure your answer clearly with proper paragraphs

ANSWER:"""
    
    try:
        answer = call_groq_api(prompt, max_tokens=1500, temperature=0.1)
        
        processing_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": source_snippets,
            "processing_time": round(processing_time, 2),
            "retrieved_chunks": len(reranked_docs),
            "context_used": len(context),
            "model_used": "llama-3.1-70b-versatile"
        }
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        
        # Enhanced fallback response
        fallback_answer = f"I found {len(source_snippets)} relevant sections in your documents but encountered an error generating the complete response. Here's what I found:\n\n"
        
        for snippet in source_snippets[:3]:
            fallback_answer += f"[{snippet['id']}] {snippet['content']}\n\n"
        
        return {
            "answer": fallback_answer,
            "sources": source_snippets,
            "processing_time": round(time.time() - start_time, 2),
            "retrieved_chunks": len(source_snippets),
            "error": str(e)
        }

# --------------------- Enhanced API Routes --------------------------

@app.post("/upload_pdf/")
async def upload_pdf_enhanced(file: UploadFile = File(...), chat_id: str = Form(...)):
    """Enhanced PDF upload with comprehensive verification."""
    start_time = time.time()
    
    try:
        # Validation
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")

        # Process with enhanced verification
        process_pdf_with_enhanced_verification(content, chat_id)

        processing_time = time.time() - start_time
        sanitized_chat_id = sanitize_index_name(chat_id)
        
        return {
            "message": "PDF processed successfully and stored in Pinecone cloud",
            "chat_id": sanitized_chat_id,
            "filename": file.filename,
            "processing_time": round(processing_time, 2),
            "file_size_mb": round(len(content) / 1024 / 1024, 2),
            "vector_db": "Pinecone Cloud",
            "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
            "llm_model": "GroqAI (llama-3.1-70b-versatile)",
            "status": "ready_for_chat",
            "environment": "Production" if IS_PRODUCTION else "Development"
        }

    except Exception as e:
        logger.error(f"PDF upload error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            content={
                "error": f"Failed to process PDF: {str(e)}",
                "processing_time": round(time.time() - start_time, 2),
                "chat_id": sanitize_index_name(chat_id),
                "suggestions": [
                    "Check if PDF is valid and not corrupted",
                    "Ensure file size is under 50MB",
                    "Try again in a few moments if this was a network issue"
                ]
            },
            status_code=500
        )

@app.post("/upload_text/")
async def upload_text_enhanced(text: str = Form(...), chat_id: str = Form(...), title: str = Form("Uploaded Text")):
    """Enhanced text upload with verification."""
    start_time = time.time()
    
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        
        if len(text) > 1000000:  # 1MB limit for text
            raise HTTPException(status_code=400, detail="Text too long. Maximum: 1MB")

        sanitized_chat_id = sanitize_index_name(chat_id)
        
        # Create document
        doc = Document(
            page_content=text,
            metadata={
                "source": title,
                "type": "text_upload",
                "title": title,
                "upload_time": datetime.now().isoformat(),
                "char_count": len(text)
            }
        )
        
        chunked_documents = enhanced_text_chunking([doc])
        if not chunked_documents:
            raise ValueError("No valid chunks created from text")

        embeddings = EnhancedGroqEmbeddings()
        index = get_or_create_pinecone_index("rag-shared-index")

        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace=sanitized_chat_id
        )
        
        vector_store.add_documents(chunked_documents)
        
        # Verification
        time.sleep(2)
        stats = index.describe_index_stats()
        namespace_vectors = 0
        if hasattr(stats, 'namespaces') and sanitized_chat_id in stats.namespaces:
            namespace_vectors = stats.namespaces[sanitized_chat_id].vector_count

        processing_time = time.time() - start_time
        
        return {
            "message": "Text processed successfully and stored in Pinecone cloud",
            "chat_id": sanitized_chat_id,
            "title": title,
            "processing_time": round(processing_time, 2),
            "text_length": len(text),
            "chunks_created": len(chunked_documents),
            "vectors_stored": namespace_vectors,
            "vector_db": "Pinecone Cloud",
            "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
            "llm_model": "GroqAI (llama-3.1-70b-versatile)",
            "status": "ready_for_chat"
        }

    except Exception as e:
        logger.error(f"Text upload error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            content={
                "error": f"Failed to process text: {str(e)}",
                "processing_time": round(time.time() - start_time, 2)
            },
            status_code=500
        )

@app.post("/chat/")
async def enhanced_chat_endpoint(chat_id: str = Form(...), message: str = Form(...)):
    """Production-ready chat endpoint with comprehensive debugging."""
    start_time = time.time()
    
    try:
        if not message.strip():
            raise HTTPException(status_code=400, detail="Empty message provided")

        logger.info(f"Processing chat request for ID: {chat_id}, Message: {message[:100]}...")
        
        # Get enhanced retriever
        retriever, namespace, index, embeddings = get_enhanced_pinecone_retriever(chat_id)
        
        # Debug: Generate and log query embedding
        query_embedding = embeddings.embed_query(message)
        logger.info(f"Query embedding generated: dimension={len(query_embedding)}")
        
        # Primary retrieval attempt
        try:
            retrieved_docs = retriever.invoke(message)
            logger.info(f"Primary retrieval: {len(retrieved_docs)} documents")
        except Exception as e:
            logger.error(f"Primary retrieval failed: {str(e)}")
            retrieved_docs = []
        
        # Fallback retrieval if primary fails
        if not retrieved_docs:
            logger.info("Attempting fallback retrieval with direct Pinecone search...")
            try:
                # Direct search with very permissive settings
                search_results = index.query(
                    vector=query_embedding,
                    top_k=20,
                    include_metadata=True,
                    namespace=namespace
                )
                
                logger.info(f"Fallback search: {len(search_results.matches)} raw matches")
                
                # Convert to Document format
                retrieved_docs = []
                for match in search_results.matches:
                    if match.score > 0.05:  # Very low threshold
                        if 'text' in match.metadata:
                            doc = Document(
                                page_content=match.metadata['text'],
                                metadata=match.metadata
                            )
                            retrieved_docs.append(doc)
                            
                logger.info(f"Fallback retrieval: {len(retrieved_docs)} documents after filtering")
                
            except Exception as e:
                logger.error(f"Fallback retrieval also failed: {str(e)}")

        # Final check - if still no documents, try any namespace
        if not retrieved_docs:
            logger.info("Trying emergency search across all namespaces...")
            try:
                stats = index.describe_index_stats()
                if hasattr(stats, 'namespaces'):
                    for ns_name, ns_info in stats.namespaces.items():
                        if ns_info.vector_count > 0:
                            emergency_results = index.query(
                                vector=query_embedding,
                                top_k=5,
                                namespace=ns_name,
                                include_metadata=True
                            )
                            if emergency_results.matches:
                                logger.info(f"Emergency: Found {len(emergency_results.matches)} matches in namespace '{ns_name}'")
                                for match in emergency_results.matches[:3]:
                                    if match.score > 0.1 and 'text' in match.metadata:
                                        doc = Document(
                                            page_content=match.metadata['text'],
                                            metadata=match.metadata
                                        )
                                        retrieved_docs.append(doc)
                                break
            except Exception as e:
                logger.error(f"Emergency search failed: {str(e)}")

        if not retrieved_docs:
            # Get namespace info for debugging
            debug_info = {}
            try:
                stats = index.describe_index_stats()
                debug_info = {
                    "total_vectors": stats.total_vector_count,
                    "available_namespaces": list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') else [],
                    "target_namespace": namespace,
                    "query_embedding_norm": round(np.linalg.norm(query_embedding), 4)
                }
            except Exception as e:
                debug_info["stats_error"] = str(e)
            
            return {
                "response": "No relevant information found in uploaded documents. This might indicate an embedding consistency issue in the cloud environment.",
                "answer": "No relevant information found in uploaded documents.",
                "sources": [],
                "retrieved_chunks": 0,
                "reranked_chunks": 0,
                "processing_time": round(time.time() - start_time, 2),
                "vector_db": "Pinecone Cloud",
                "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
                "llm_model": "GroqAI (llama-3.1-70b-versatile)",
                "namespace": namespace,
                "debug_info": debug_info,
                "suggestions": [
                    "Try rephrasing your question",
                    "Check if documents were uploaded correctly using /list_chats/",
                    "Use /debug_search/{chat_id} endpoint for detailed debugging"
                ]
            }

        # Rerank retrieved documents
        reranked_docs = advanced_reranker(retrieved_docs, message, top_k=7)
        logger.info(f"Reranked to top {len(reranked_docs)} most relevant chunks")

        # Generate comprehensive answer
        result = generate_comprehensive_answer(message, reranked_docs)
        
        # Add system information
        result.update({
            "response": result["answer"],
            "retrieved_chunks": len(retrieved_docs),
            "reranked_chunks": len(reranked_docs),
            "total_processing_time": round(time.time() - start_time, 2),
            "vector_db": "Pinecone Cloud",
            "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
            "llm_model": "GroqAI (llama-3.1-70b-versatile)",
            "namespace": namespace,
            "environment": "Production" if IS_PRODUCTION else "Development"
        })

        return result

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            content={
                "error": f"Chat processing failed: {str(e)}",
                "processing_time": round(time.time() - start_time, 2),
                "chat_id": sanitize_index_name(chat_id),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.post("/debug_search/{chat_id}")
async def debug_search_comprehensive(chat_id: str, query: str = Form(...)):
    """Comprehensive debug endpoint for troubleshooting retrieval issues."""
    sanitized_chat_id = sanitize_index_name(chat_id)
    
    try:
        # Step 1: Index and namespace verification
        index = get_or_create_pinecone_index("rag-shared-index")
        stats = index.describe_index_stats()
        
        namespace_info = {
            "target_namespace": sanitized_chat_id,
            "total_vectors_in_index": stats.total_vector_count,
            "available_namespaces": {}
        }
        
        if hasattr(stats, 'namespaces'):
            for ns, info in stats.namespaces.items():
                namespace_info["available_namespaces"][ns] = {
                    "vector_count": info.vector_count
                }
        
        # Step 2: Embedding generation test
        embeddings = EnhancedGroqEmbeddings()
        query_embedding = embeddings.embed_query(query)
        embedding_info = {
            "dimension": len(query_embedding),
            "norm": round(float(np.linalg.norm(query_embedding)), 4),
            "sample_values": [round(float(x), 4) for x in query_embedding[:5]]
        }
        
        # Step 3: Direct Pinecone search with different thresholds
        search_tests = {}
        
        # Test with target namespace
        if sanitized_chat_id in namespace_info["available_namespaces"]:
            search_results = index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True,
                namespace=sanitized_chat_id
            )
            
            search_tests["target_namespace"] = {
                "namespace": sanitized_chat_id,
                "matches_found": len(search_results.matches),
                "top_scores": [round(float(m.score), 4) for m in search_results.matches[:5]],
                "sample_matches": []
            }
            
            for match in search_results.matches[:3]:
                search_tests["target_namespace"]["sample_matches"].append({
                    "score": round(float(match.score), 4),
                    "text_preview": match.metadata.get('text', '')[:150] + "...",
                    "source": match.metadata.get('source', 'unknown'),
                    "page": match.metadata.get('page', 'unknown')
                })
        
        # Test with any available namespace
        available_ns = [ns for ns, info in namespace_info["available_namespaces"].items() if info["vector_count"] > 0]
        if available_ns and available_ns[0] != sanitized_chat_id:
            fallback_ns = available_ns[0]
            fallback_results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace=fallback_ns
            )
            
            search_tests["fallback_namespace"] = {
                "namespace": fallback_ns,
                "matches_found": len(fallback_results.matches),
                "top_scores": [round(float(m.score), 4) for m in fallback_results.matches[:3]]
            }
        
        # Step 4: Langchain retriever test
        try:
            retriever, actual_namespace, _, _ = get_enhanced_pinecone_retriever(chat_id)
            langchain_docs = retriever.invoke(query)
            
            retriever_info = {
                "langchain_retrieval_successful": True,
                "documents_retrieved": len(langchain_docs),
                "actual_namespace_used": actual_namespace,
                "sample_doc_lengths": [len(doc.page_content) for doc in langchain_docs[:3]]
            }
        except Exception as e:
            retriever_info = {
                "langchain_retrieval_successful": False,
                "error": str(e)
            }
        
        return {
            "debug_timestamp": datetime.now().isoformat(),
            "query": query,
            "chat_id": chat_id,
            "sanitized_chat_id": sanitized_chat_id,
            "namespace_info": namespace_info,
            "embedding_info": embedding_info,
            "search_tests": search_tests,
            "retriever_info": retriever_info,
            "environment": {
                "is_production": IS_PRODUCTION,
                "is_render": IS_RENDER,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "search_top_k": VECTOR_SEARCH_TOP_K
            },
            "recommendations": [
                "If no matches found, try uploading documents again",
                "If low scores, try more specific queries",
                "If wrong namespace, check chat_id parameter",
                "Use /health endpoint to verify all services"
            ]
        }
        
    except Exception as e:
        logger.error(f"Debug search error: {str(e)}")
        return JSONResponse(
            content={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "chat_id": chat_id,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@app.delete("/delete_chat/{chat_id}")
async def delete_chat_enhanced(chat_id: str):
    """Enhanced chat deletion with comprehensive cleanup."""
    sanitized_chat_id = sanitize_index_name(chat_id)
    
    try:
        existing_indexes = pc.list_indexes()
        if not existing_indexes:
            raise HTTPException(status_code=404, detail="No indexes found")
        
        index_names = [idx['name'] for idx in existing_indexes]
        shared_index_name = "rag-shared-index" if "rag-shared-index" in index_names else index_names[0]
        index = pc.Index(shared_index_name)
        
        # Check namespace exists
        stats = index.describe_index_stats()
        namespaces = stats.namespaces if hasattr(stats, 'namespaces') else {}
        
        if sanitized_chat_id not in namespaces:
            return {
                "message": f"Chat ID '{chat_id}' not found or already deleted",
                "namespace": sanitized_chat_id,
                "available_namespaces": list(namespaces.keys()),
                "status": "not_found"
            }
        
        vector_count_before = namespaces[sanitized_chat_id].vector_count
        
        # Delete with verification
        delete_response = index.delete(delete_all=True, namespace=sanitized_chat_id)
        
        # Wait and verify deletion
        time.sleep(2)
        updated_stats = index.describe_index_stats()
        updated_namespaces = updated_stats.namespaces if hasattr(updated_stats, 'namespaces') else {}
        
        deletion_verified = sanitized_chat_id not in updated_namespaces
        
        return {
            "message": f"Chat ID '{chat_id}' deleted successfully",
            "deleted_namespace": sanitized_chat_id,
            "vectors_deleted": vector_count_before,
            "deletion_verified": deletion_verified,
            "remaining_namespaces": list(updated_namespaces.keys()),
            "index_used": shared_index_name,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Delete chat error: {str(e)}")
        return JSONResponse(
            content={
                "error": f"Failed to delete chat: {str(e)}",
                "chat_id": chat_id,
                "namespace": sanitized_chat_id
            },
            status_code=500
        )

@app.get("/list_chats/")
async def list_chats_enhanced():
    """Enhanced chat listing with detailed information."""
    try:
        indexes = pc.list_indexes()
        
        if not indexes:
            return {
                "chat_ids": [],
                "chats": [],
                "total_namespaces": 0,
                "total_vectors": 0,
                "system_info": {
                    "vector_db": "Pinecone Cloud",
                    "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
                    "llm_model": "GroqAI (llama-3.1-70b-versatile)",
                    "environment": "Production" if IS_PRODUCTION else "Development"
                }
            }
        
        chat_info = []
        chat_ids = []
        total_vectors = 0
        
        for idx in indexes:
            try:
                index = pc.Index(idx['name'])
                stats = index.describe_index_stats()
                total_vectors += stats.total_vector_count
                
                namespaces = stats.namespaces if hasattr(stats, 'namespaces') else {}
                
                for namespace_name, namespace_stats in namespaces.items():
                    vector_count = namespace_stats.vector_count if hasattr(namespace_stats, 'vector_count') else 0
                    
                    if vector_count > 0:  # Only include namespaces with content
                        chat_ids.append(namespace_name)
                        chat_info.append({
                            "chat_id": namespace_name,
                            "namespace": namespace_name,
                            "index_name": idx['name'],
                            "vector_count": vector_count,
                            "dimension": stats.dimension,
                            "status": "active",
                            "last_activity": "recently",  # Could be enhanced with actual timestamps
                            "estimated_documents": max(1, vector_count // 10)  # Rough estimate
                        })
                
            except Exception as e:
                logger.warning(f"Could not get stats for index {idx['name']}: {e}")
        
        return {
            "chat_ids": chat_ids,
            "chats": sorted(chat_info, key=lambda x: x["vector_count"], reverse=True),
            "total_namespaces": len(chat_info),
            "total_vectors": total_vectors,
            "system_info": {
                "vector_db": "Pinecone Cloud",
                "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
                "llm_model": "GroqAI (llama-3.1-70b-versatile)",
                "environment": "Production" if IS_PRODUCTION else "Development",
                "cloud_optimizations": IS_PRODUCTION
            }
        }
        
    except Exception as e:
        logger.error(f"List chats error: {str(e)}")
        return {
            "chat_ids": [],
            "chats": [],
            "error": str(e),
            "total_namespaces": 0
        }

@app.get("/health")
async def comprehensive_health_check():
    """Comprehensive health check for production monitoring."""
    health_start = time.time()
    
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "deployment": "Production" if IS_PRODUCTION else "Development",
                "is_render": IS_RENDER,
                "groq_configured": bool(GROQ_API_KEY),
                "nomic_configured": bool(NOMIC_API_KEY),
                "pinecone_configured": bool(PINECONE_API_KEY),
                "pinecone_environment": PINECONE_ENVIRONMENT
            },
            "models": {
                "embedding_model": "Nomic AI text-embedding-v1.5",
                "embedding_dimension": 384,
                "llm_model": "GroqAI llama-3.1-70b-versatile",
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "search_top_k": VECTOR_SEARCH_TOP_K
            }
        }
        
        # Test Pinecone connection
        try:
            indexes = pc.list_indexes()
            total_vectors = 0
            namespace_count = 0
            
            for idx in indexes:
                try:
                    index = pc.Index(idx["name"])
                    stats = index.describe_index_stats()
                    total_vectors += stats.total_vector_count
                    if hasattr(stats, 'namespaces'):
                        namespace_count += len(stats.namespaces)
                except:
                    pass
            
            health_status["pinecone"] = {
                "connected": True,
                "indexes_count": len(indexes),
                "total_vectors": total_vectors,
                "total_namespaces": namespace_count,
                "primary_index": "rag-shared-index"
            }
        except Exception as e:
            health_status["pinecone"] = {
                "connected": False,
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Test Groq API
        try:
            test_response = call_groq_api("Test connection", max_tokens=10, temperature=0.1)
            health_status["groq"] = {
                "connected": True,
                "model": "llama-3.1-70b-versatile",
                "test_response_length": len(test_response)
            }
        except Exception as e:
            health_status["groq"] = {
                "connected": False,
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Test Nomic embeddings
        try:
            embeddings = EnhancedGroqEmbeddings()
            test_embedding = embeddings.embed_query("health check test")
            health_status["nomic_embeddings"] = {
                "working": True,
                "dimension": len(test_embedding),
                "model": "nomic-embed-text-v1.5",
                "cache_size": len(embeddings._cache)
            }
        except Exception as e:
            health_status["nomic_embeddings"] = {
                "working": False,
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Performance metrics
        health_status["performance"] = {
            "health_check_time": round(time.time() - health_start, 3),
            "cloud_timeout": CLOUD_TIMEOUT,
            "batch_size": EMBEDDING_BATCH_SIZE
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "health_check_time": round(time.time() - health_start, 3)
        }

@app.get("/system_stats")
async def get_system_stats():
    """Get comprehensive system statistics."""
    try:
        # Pinecone statistics
        indexes = pc.list_indexes()
        total_vectors = 0
        namespace_details = []
        
        for idx in indexes:
            try:
                index = pc.Index(idx["name"])
                stats = index.describe_index_stats()
                total_vectors += stats.total_vector_count
                
                if hasattr(stats, 'namespaces'):
                    for ns, info in stats.namespaces.items():
                        namespace_details.append({
                            "namespace": ns,
                            "index": idx["name"],
                            "vectors": info.vector_count,
                            "status": "active" if info.vector_count > 0 else "empty"
                        })
            except Exception as e:
                logger.warning(f"Stats error for index {idx['name']}: {e}")
        
        return {
            "system_overview": {
                "total_indexes": len(indexes),
                "total_vectors": total_vectors,
                "total_namespaces": len(namespace_details),
                "active_namespaces": len([ns for ns in namespace_details if ns["vectors"] > 0])
            },
            "namespace_details": sorted(namespace_details, key=lambda x: x["vectors"], reverse=True),
            "configuration": {
                "embedding_model": "Nomic AI text-embedding-v1.5",
                "vector_dimension": 384,
                "llm_model": "GroqAI llama-3.1-70b-versatile",
                "environment": "Production" if IS_PRODUCTION else "Development",
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "search_top_k": VECTOR_SEARCH_TOP_K
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "timestamp": datetime.now().isoformat()},
            status_code=500
        )

@app.post("/test_embedding_consistency")
async def test_embedding_consistency(text: str = Form(...)):
    """Test embedding consistency across multiple generations."""
    try:
        embeddings = EnhancedGroqEmbeddings()
        
        # Generate same embedding multiple times
        results = []
        for i in range(3):
            embedding = embeddings.embed_query(text)
            results.append({
                "attempt": i + 1,
                "dimension": len(embedding),
                "norm": round(float(np.linalg.norm(embedding)), 6),
                "first_5_values": [round(float(x), 6) for x in embedding[:5]]
            })
            time.sleep(0.1)  # Small delay between calls
        
        # Calculate consistency
        if len(results) > 1:
            emb1 = np.array([float(x) for x in results[0]["first_5_values"]])
            emb2 = np.array([float(x) for x in results[1]["first_5_values"]])
            similarity = float(np.dot(emb1, emb2))
            
            consistency_info = {
                "similarity_between_attempts": round(similarity, 6),
                "consistent": similarity > 0.99,
                "cache_used": len(embeddings._cache) > 0
            }
        else:
            consistency_info = {"error": "Need multiple attempts for comparison"}
        
        return {
            "text_tested": text[:100],
            "embedding_attempts": results,
            "consistency_analysis": consistency_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/")
async def root():
    """Enhanced root endpoint with comprehensive API information."""
    return {
        "service": "Production RAG System",
        "version": "2.0.0",
        "status": "operational",
        "environment": "Production" if IS_PRODUCTION else "Development",
        "architecture": {
            "vector_db": "Pinecone Cloud",
            "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
            "llm_model": "GroqAI (llama-3.1-70b-versatile)",
            "chunking_strategy": "RecursiveCharacterTextSplitter",
            "reranking": "Advanced multi-factor scoring"
        },
        "features": [
            "Cloud-optimized embedding consistency",
            "Enhanced PDF/text processing",
            "Multi-level fallback retrieval",
            "Advanced reranking algorithm",
            "Comprehensive error handling",
            "Production monitoring endpoints",
            "Namespace-based chat isolation",
            "Automatic retry mechanisms",
            "Embedding caching for consistency",
            "Real-time debugging capabilities"
        ],
        "endpoints": {
            "core": [
                "POST /upload_pdf/ - Upload and process PDF documents",
                "POST /upload_text/ - Upload and process text content", 
                "POST /chat/ - Chat with your documents",
                "GET /list_chats/ - List all chat sessions",
                "DELETE /delete_chat/{chat_id} - Delete chat session"
            ],
            "debugging": [
                "POST /debug_search/{chat_id} - Debug retrieval issues",
                "POST /test_embedding_consistency - Test embedding consistency",
                "GET /system_stats - Get system statistics",
                "GET /health - Comprehensive health check"
            ]
        },
        "optimizations": {
            "cloud_timeout": CLOUD_TIMEOUT,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "vector_search_top_k": VECTOR_SEARCH_TOP_K
        },
        "timestamp": datetime.now().isoformat()
    }

# --------------------- Enhanced Process Functions -------------------

def process_text_with_enhanced_verification(text: str, chat_id: str, title: str = "Uploaded Text"):
    """Enhanced text processing with comprehensive verification."""
    sanitized_chat_id = sanitize_index_name(chat_id)
    logger.info(f"Processing text for chat ID: {sanitized_chat_id}")
    
    doc = Document(
        page_content=text,
        metadata={
            "source": title,
            "type": "text_upload",
            "title": title,
            "upload_time": datetime.now().isoformat(),
            "char_count": len(text)
        }
    )
    
    chunked_documents = enhanced_text_chunking([doc])
    if not chunked_documents:
        raise ValueError("No valid chunks created from text")

    embeddings = EnhancedGroqEmbeddings()
    index = get_or_create_pinecone_index("rag-shared-index")

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        namespace=sanitized_chat_id
    )
    
    logger.info(f"Adding {len(chunked_documents)} text chunks to namespace '{sanitized_chat_id}'")
    vector_store.add_documents(chunked_documents)
    
    # Enhanced verification
    verification_attempts = 3
    for attempt in range(verification_attempts):
        try:
            time.sleep(2 + attempt)
            stats = index.describe_index_stats()
            
            if hasattr(stats, 'namespaces') and sanitized_chat_id in stats.namespaces:
                vector_count = stats.namespaces[sanitized_chat_id].vector_count
                logger.info(f"Verification: namespace '{sanitized_chat_id}' has {vector_count} vectors")
                
                # Test immediate retrieval
                test_embedding = embeddings.embed_query("test content document")
                test_results = index.query(
                    vector=test_embedding,
                    top_k=3,
                    namespace=sanitized_chat_id,
                    include_metadata=True
                )
                logger.info(f"Test retrieval: {len(test_results.matches)} matches found")
                return
                
            else:
                logger.warning(f"Verification attempt {attempt + 1}: namespace not visible yet")
                
        except Exception as e:
            logger.warning(f"Verification attempt {attempt + 1} error: {str(e)}")
    
    logger.warning("Upload completed but verification had issues")

if __name__ == "__main__":
    import uvicorn
    
    # Production-ready server configuration
    config = {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "log_level": "info" if IS_PRODUCTION else "debug",
        "access_log": True,
        "reload": not IS_PRODUCTION
    }
    
    logger.info(f"Starting server with config: {config}")
    uvicorn.run(app, **config)



# import os
# import re
# import traceback
# import time
# import hashlib
# import requests
# from typing import List, Dict, Any
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_pinecone import PineconeVectorStore
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from tempfile import NamedTemporaryFile
# from dotenv import load_dotenv
# import pinecone
# from pinecone import Pinecone, ServerlessSpec
# import logging
# import numpy as np

# # --------------------- Configuration -----------------------

# # Load environment variables from .env
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")  # Add Nomic API key
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# if not GROQ_API_KEY:
#     raise ValueError("❌ GROQ_API_KEY not found in environment variables")
# if not NOMIC_API_KEY:
#     raise ValueError("❌ NOMIC_API_KEY not found in environment variables")
# if not PINECONE_API_KEY:
#     raise ValueError("❌ PINECONE_API_KEY not found in environment variables")

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# print(f"✅ Groq API Key Loaded: {bool(GROQ_API_KEY)}")
# print(f"✅ Nomic API Key Loaded: {bool(NOMIC_API_KEY)}")
# print(f"✅ Pinecone API Key Loaded: {bool(PINECONE_API_KEY)}")
# print(f"✅ Pinecone Environment: {PINECONE_ENVIRONMENT}")

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # --------------------- Groq API Integration -------------------

# def call_groq_api(prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
#     """
#     Call Groq API directly using REST requests.
#     """
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
    
#     # Truncate prompt if too long (Groq has context limits)
#     if len(prompt) > 6000:
#         prompt = prompt[:6000] + "\n\n[Content truncated for API limits]"
    
#     data = {
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "You are a helpful AI assistant that answers questions based on provided document context. Always include citations [1], [2], etc. when referencing the provided sources."
#             },
#             {
#                 "role": "user", 
#                 "content": prompt
#             }
#         ],
#         "model": "llama-3.1-8b-instant",  # Better Groq model
#         "max_tokens": min(max_tokens, 1000),
#         "temperature": temperature,
#         "stream": False
#     }
    
#     try:
#         logger.info(f"📤 Sending request to Groq API with model: {data['model']}")
#         response = requests.post(
#             "https://api.groq.com/openai/v1/chat/completions",
#             headers=headers,
#             json=data,
#             timeout=30
#         )
        
#         logger.info(f"📥 Groq API response status: {response.status_code}")
        
#         if not response.ok:
#             error_text = response.text
#             logger.error(f"❌ Groq API error response: {error_text}")
#             raise HTTPException(status_code=response.status_code, detail=f"Groq API error: {error_text}")
        
#         result = response.json()
        
#         if "choices" not in result or not result["choices"]:
#             raise HTTPException(status_code=500, detail="Invalid response from Groq API")
            
#         return result["choices"][0]["message"]["content"]
        
#     except requests.exceptions.Timeout:
#         logger.error("❌ Groq API request timed out")
#         raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
#     except requests.exceptions.RequestException as e:
#         logger.error(f"❌ Groq API request failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")
#     except Exception as e:
#         logger.error(f"❌ Unexpected error calling Groq: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# # --------------------- Nomic AI Embeddings Implementation -------------------

# class GroqEmbeddings:
#     """Custom embedding class using Nomic AI embeddings with 384 dimensions"""

#     def __init__(self):
#         self.dimension = 384
#         self.model_name = "nomic-embed-text-v1.5"  # Latest Nomic embedding model
#         self.base_url = "https://api-atlas.nomic.ai"
        
#     def _get_nomic_embedding(self, text: str, task_type: str = "search_document") -> List[float]:
#         """Get embedding from Nomic AI API"""
#         try:
#             # Clean and truncate text if needed
#             text = text.strip()
#             if len(text) > 8192:  # Nomic supports 8192 context length
#                 text = text[:8192]
            
#             if not text:  # Handle empty text
#                 return [0.0] * self.dimension
            
#             headers = {
#                 "Authorization": f"Bearer {NOMIC_API_KEY}",
#                 "Content-Type": "application/json"
#             }
            
#             data = {
#                 "model": self.model_name,
#                 "texts": [text],
#                 "task_type": task_type,
#                 "dimensionality": self.dimension  # Set to exactly 384 dimensions
#             }
            
#             response = requests.post(
#                 f"{self.base_url}/v1/embedding/text",
#                 headers=headers,
#                 json=data,
#                 timeout=30
#             )
            
#             if response.ok:
#                 result = response.json()
#                 return result["embeddings"][0]
#             else:
#                 logger.error(f"❌ Nomic API error: {response.status_code} - {response.text}")
#                 return self._fallback_embedding(text)
                
#         except Exception as e:
#             logger.error(f"❌ Nomic embedding error: {str(e)}")
#             # Fallback to hash-based embedding
#             return self._fallback_embedding(text)

#     def _fallback_embedding(self, text: str) -> List[float]:
#         """Fallback hash-based embedding if Nomic fails"""
#         try:
#             hash_obj = hashlib.sha256(text.encode())
#             hash_hex = hash_obj.hexdigest()
            
#             embedding = []
#             for i in range(0, len(hash_hex), 2):
#                 hex_pair = hash_hex[i:i+2]
#                 embedding.append(int(hex_pair, 16) / 255.0)
            
#             # Ensure exactly 384 dimensions
#             if len(embedding) < 384:
#                 embedding.extend([0.0] * (384 - len(embedding)))
#             else:
#                 embedding = embedding[:384]
                
#             return embedding
#         except:
#             # Ultimate fallback - random normalized vector
#             return [0.1] * 384

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """Embed a list of documents using Nomic AI"""
#         embeddings = []
        
#         # Process in batches to respect rate limits
#         batch_size = 20
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]
            
#             try:
#                 # Use batch processing for efficiency
#                 batch_embeddings = self._get_nomic_batch_embeddings(batch, task_type="search_document")
#                 embeddings.extend(batch_embeddings)
                
#                 # Log progress and add small delay for rate limits
#                 logger.info(f"📝 Processed batch {i//batch_size + 1}, total: {len(embeddings)}/{len(texts)} embeddings")
#                 time.sleep(0.1)  # Small delay to respect rate limits
                    
#             except Exception as e:
#                 logger.error(f"❌ Error embedding batch starting at {i}: {str(e)}")
#                 # Use fallback for failed batch
#                 for text in batch:
#                     embeddings.append(self._fallback_embedding(text))
        
#         logger.info(f"✅ Generated {len(embeddings)} Nomic AI embeddings")
#         return embeddings

#     def _get_nomic_batch_embeddings(self, texts: List[str], task_type: str = "search_document") -> List[List[float]]:
#         """Get embeddings for a batch of texts"""
#         try:
#             # Clean texts
#             clean_texts = []
#             for text in texts:
#                 text = text.strip()
#                 if len(text) > 8192:
#                     text = text[:8192]
#                 clean_texts.append(text)
            
#             headers = {
#                 "Authorization": f"Bearer {NOMIC_API_KEY}",
#                 "Content-Type": "application/json"
#             }
            
#             data = {
#                 "model": self.model_name,
#                 "texts": clean_texts,
#                 "task_type": task_type,
#                 "dimensionality": self.dimension
#             }
            
#             response = requests.post(
#                 f"{self.base_url}/v1/embedding/text",
#                 headers=headers,
#                 json=data,
#                 timeout=60  # Longer timeout for batches
#             )
            
#             if response.ok:
#                 result = response.json()
#                 return result["embeddings"]
#             else:
#                 logger.error(f"❌ Nomic batch API error: {response.status_code} - {response.text}")
#                 # Return fallback embeddings for the batch
#                 return [self._fallback_embedding(text) for text in texts]
                
#         except Exception as e:
#             logger.error(f"❌ Nomic batch embedding error: {str(e)}")
#             return [self._fallback_embedding(text) for text in texts]

#     def embed_query(self, text: str) -> List[float]:
#         """Embed a single query using Nomic AI"""
#         # Use query task type for search queries
#         return self._get_nomic_embedding(text, task_type="search_query")

# # --------------------- App Initialization ------------------

# app = FastAPI(
#     title="Mini RAG System - GroqAI + Pinecone + Nomic Embeddings",
#     description="RAG system with GroqAI (Free Tier) + Pinecone + Nomic AI Embeddings",
#     version="1.0.0"
# )

# # Enable CORS for frontend connection
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000","https://mini-rag-assesment-front.vercel.app"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --------------------- Utility Functions -------------------

# def sanitize_index_name(chat_id: str) -> str:
#     """
#     Ensure index name follows Pinecone requirements:
#     - Lowercase alphanumeric and hyphens only
#     - Must start and end with alphanumeric
#     - Max 45 characters
#     """
#     chat_id = re.sub(r'[^a-z0-9\-]', '-', chat_id.lower())
#     chat_id = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', chat_id)
    
#     if len(chat_id) < 3:
#         chat_id = f"chat-{chat_id}-{int(time.time()) % 10000}"
    
#     if len(chat_id) > 45:
#         chat_id = chat_id[:45]
    
#     chat_id = re.sub(r'-+$', '', chat_id)
#     return chat_id

# def get_or_create_pinecone_index(index_name: str = "rag-shared-index"):
#     """Get existing or create a shared Pinecone index for all chats using namespaces."""
#     try:
#         existing_indexes = pc.list_indexes()
#         index_names = [idx["name"] for idx in existing_indexes]

#         if index_name not in index_names:
#             logger.info(f"🆕 Creating new Pinecone shared index: {index_name}")
#             pc.create_index(
#                 name=index_name,
#                 dimension=384,
#                 metric="cosine",
#                 spec=ServerlessSpec(
#                     cloud="aws",
#                     region=PINECONE_ENVIRONMENT
#                 )
#             )
#             time.sleep(10)
#             logger.info(f"✅ Pinecone index '{index_name}' created successfully")
#         else:
#             logger.info(f"📚 Using existing Pinecone index: {index_name}")

#         return pc.Index(index_name)

#     except Exception as e:
#         logger.error(f"❌ Error with Pinecone index '{index_name}': {str(e)}")
#         raise e

# def enhanced_text_chunking(documents: List[Document]) -> List[Document]:
#     """
#     Enhanced chunking strategy with metadata preservation for citations.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=120,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
    
#     enhanced_docs = []
#     for doc_idx, doc in enumerate(documents):
#         chunks = text_splitter.split_text(doc.page_content)
#         for chunk_idx, chunk in enumerate(chunks):
#             chunk_id = hashlib.md5(f"{doc_idx}_{chunk_idx}_{chunk[:50]}".encode()).hexdigest()[:8]
            
#             enhanced_metadata = {
#                 **doc.metadata,
#                 "chunk_id": chunk_id,
#                 "doc_index": doc_idx,
#                 "chunk_index": chunk_idx,
#                 "source": doc.metadata.get("source", "unknown"),
#                 "page": doc.metadata.get("page", 0),
#                 "section": f"chunk_{chunk_idx}",
#                 "position": chunk_idx
#             }
#             enhanced_docs.append(Document(page_content=chunk, metadata=enhanced_metadata))
    
#     logger.info(f"✅ Created {len(enhanced_docs)} chunks with enhanced metadata")
#     return enhanced_docs

# def simple_reranker(retrieved_docs: List[Document], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
#     """Simple reranker implementation."""
#     query_lower = query.lower()
#     scored_docs = []
    
#     for doc in retrieved_docs:
#         content_lower = doc.page_content.lower()
#         score = 0
#         query_words = query_lower.split()
        
#         for word in query_words:
#             if word in content_lower:
#                 score += content_lower.count(word) * 2
        
#         if any(word in content_lower for word in query_words):
#             score += 1
            
#         scored_docs.append({
#             "document": doc,
#             "score": score,
#             "metadata": doc.metadata
#         })
    
#     reranked = sorted(scored_docs, key=lambda x: x["score"], reverse=True)[:top_k]
#     logger.info(f"✅ Reranked {len(retrieved_docs)} docs, returning top {top_k}")
    
#     return reranked


# def process_text_with_pinecone(text: str, chat_id: str, title: str = "Uploaded Text"):
#     """Process text content with Pinecone cloud vector storage using namespaces."""
#     sanitized_chat_id = sanitize_index_name(chat_id)
#     logger.info(f"Starting text processing for chat ID: {sanitized_chat_id}")
    
#     doc = Document(
#         page_content=text,
#         metadata={
#             "source": title,
#             "type": "text_upload",
#             "title": title
#         }
#     )
    
#     chunked_documents = enhanced_text_chunking([doc])
    
#     if not chunked_documents:
#         logger.warning("No content to process.")
#         return
    
#     embeddings = GroqEmbeddings()  # Now uses Nomic AI embeddings
#     index = get_or_create_pinecone_index("rag-shared-index")
    
#     vector_store = PineconeVectorStore(
#         index=index,
#         embedding=embeddings,
#         text_key="text",
#         namespace=sanitized_chat_id
#     )
    
#     vector_store.add_documents(chunked_documents)
#     logger.info(f"Added {len(chunked_documents)} text chunks to namespace '{sanitized_chat_id}'.")
# # Replace your get_pinecone_retriever function with this fixed version:


# # Also add debugging to your upload function to see where vectors are actually going:




# def process_pdf_with_pinecone(content: bytes, chat_id: str):
#     """Process PDF and store embeddings in Pinecone using Nomic AI embeddings."""
#     sanitized_chat_id = sanitize_index_name(chat_id)
#     logger.info(f"Starting PDF processing for chat ID: {sanitized_chat_id}")

#     # Save PDF temporarily
#     with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(content)
#         tmp_path = tmp.name
#     logger.info(f"Temporary PDF file created at: {tmp_path}")

#     try:
#         loader = PyPDFLoader(tmp_path)
#         documents = loader.load()
#         logger.info(f"Loaded {len(documents)} pages from PDF.")
        
#         chunked_documents = enhanced_text_chunking(documents)
#     finally:
#         os.remove(tmp_path)
#         logger.info(f"Temporary file removed.")

#     if not chunked_documents:
#         logger.warning("No documents found in PDF. Skipping embedding process.")
#         return

#     embeddings = GroqEmbeddings()  # Nomic AI embeddings
#     logger.info("Initialized Nomic AI Embeddings.")

#     index = get_or_create_pinecone_index("rag-shared-index")

#     # Upload chunks
#     logger.info(f"Adding {len(chunked_documents)} chunks to namespace '{sanitized_chat_id}'")
#     vector_store = PineconeVectorStore(
#         index=index,
#         embedding=embeddings,
#         text_key="text",
#         namespace=sanitized_chat_id
#     )
#     vector_store.add_documents(chunked_documents)

#     # Verify upload
#     try:
#         time.sleep(1)
#         stats = index.describe_index_stats()
#         logger.info(f"After upload - Total vectors: {stats.total_vector_count}")
#         if hasattr(stats, 'namespaces') and sanitized_chat_id in stats.namespaces:
#             logger.info(f"✅ Namespace '{sanitized_chat_id}' has {stats.namespaces[sanitized_chat_id].vector_count} vectors")
#         else:
#             logger.warning(f"⚠️ Namespace '{sanitized_chat_id}' not found after upload!")
#     except Exception as e:
#         logger.warning(f"Could not verify upload: {e}")


# def get_pinecone_retriever(chat_id: str):
#     """Retrieve Pinecone retriever for a chat ID with proper namespace handling."""
#     sanitized_chat_id = sanitize_index_name(chat_id)
#     logger.info(f"Getting Pinecone retriever for chat ID: {sanitized_chat_id}")

#     existing_indexes = pc.list_indexes()
#     if not existing_indexes:
#         raise HTTPException(
#             status_code=404,
#             detail="No Pinecone indexes found. Please upload documents first."
#         )

#     # Ensure index name is a string
#     index_names = []
#     for idx in existing_indexes:
#         if hasattr(idx, "name"):
#             index_names.append(str(idx.name))
#         else:
#             index_names.append(str(idx))
    
#     shared_index_name = "rag-shared-index" if "rag-shared-index" in index_names else index_names[0]
#     embeddings = GroqEmbeddings()
#     index = pc.Index(shared_index_name)

#     target_namespace = sanitized_chat_id
#     try:
#         stats = index.describe_index_stats()
#         logger.info(f"Total vectors in index: {stats.total_vector_count}")

#         if hasattr(stats, "namespaces"):
#             namespaces = stats.namespaces
#             logger.info(f"Available namespaces: {list(namespaces.keys())}")

#             if sanitized_chat_id in namespaces:
#                 vector_count = namespaces[sanitized_chat_id].vector_count
#                 logger.info(f"✅ Found namespace '{sanitized_chat_id}' with {vector_count} vectors")
#                 target_namespace = sanitized_chat_id
#             else:
#                 # fallback to default namespace with vectors
#                 default_ns = [ns for ns in namespaces if namespaces[ns].vector_count > 0]
#                 if default_ns:
#                     target_namespace = default_ns[0]
#                     logger.info(f"⚠️ Using fallback namespace: '{target_namespace}'")
#                 else:
#                     raise HTTPException(
#                         status_code=404,
#                         detail=f"No documents found. Available namespaces: {list(namespaces.keys())}"
#                     )
#         else:
#             logger.warning("No namespace information available. Proceeding with intended namespace.")

#     except Exception as e:
#         logger.warning(f"Could not check namespace stats: {e}")

#     vector_store = PineconeVectorStore(
#         index=index,
#         embedding=embeddings,
#         text_key="text",
#         namespace=target_namespace
#     )
      
#     retriever = vector_store.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 5}  # retrieve all for debugging
#     )

#     logger.info(f"✅ Pinecone retriever initialized for namespace '{target_namespace}' in index '{shared_index_name}'")
#     return retriever

# def generate_answer_with_citations(question: str, reranked_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Generate answer with inline citations using GroqAI.
#     """
#     start_time = time.time()
    
#     context_parts = []
#     source_snippets = []
    
#     for i, doc_info in enumerate(reranked_docs, 1):
#         doc = doc_info["document"]
#         citation_id = f"[{i}]"
#         # Limit context length per source
#         content = doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content
#         context_parts.append(f"{citation_id} {content}")
        
#         source_snippets.append({
#             "id": i,
#             "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
#             "metadata": doc.metadata,
#             "page": doc.metadata.get("page", "Unknown"),
#             "source": doc.metadata.get("source", "Unknown"),
#             "chunk_id": doc.metadata.get("chunk_id", f"chunk_{i}")
#         })
    
#     context = "\n\n".join(context_parts)
    
#     # Keep prompt concise for Groq limits
#     prompt = f"""Based on the following context, answer the question with inline citations.

# Context:
# {context[:3000]}

# Question: {question}

# Provide a clear answer using only the context above. Include citations like [1], [2] for your sources.

# Answer:"""
    
#     try:
#         answer = call_groq_api(prompt, max_tokens=800, temperature=0.1)
        
#         processing_time = time.time() - start_time
        
#         return {
#             "answer": answer,
#             "sources": source_snippets,
#             "processing_time": round(processing_time, 2),
#             "retrieved_chunks": len(reranked_docs)
#         }
    
#     except Exception as e:
#         logger.error(f"❌ Error generating answer with Groq: {str(e)}")
#         # Return a fallback response with context
#         fallback_answer = f"I found relevant information but couldn't generate a complete response. Here's what I found in the documents:\n\n"
#         for snippet in source_snippets[:2]:
#             fallback_answer += f"[{snippet['id']}] {snippet['content']}\n\n"
        
#         return {
#             "answer": fallback_answer,
#             "sources": source_snippets,
#             "processing_time": round(time.time() - start_time, 2),
#             "retrieved_chunks": len(reranked_docs),
#             "error": str(e)
#         }

# # --------------------- API Routes --------------------------


# @app.post("/upload_pdf/")
# async def upload_pdf(file: UploadFile = File(...), chat_id: str = Form(...)):
#     """Upload PDF and process it into Pinecone cloud vector database."""
#     start_time = time.time()
#     try:
#         if not file.filename.lower().endswith(".pdf"):
#             raise HTTPException(status_code=400, detail="Only PDF files are supported")

#         content = await file.read()
#         if len(content) == 0:
#             raise HTTPException(status_code=400, detail="Empty file uploaded")

#         # Use robust function
#         process_pdf_with_pinecone(content, chat_id)

#         processing_time = time.time() - start_time
#         return {
#             "message": "✅ PDF processed successfully and stored in Pinecone cloud",
#             "chat_id": sanitize_index_name(chat_id),
#             "filename": file.filename,
#             "processing_time": round(processing_time, 2),
#             "file_size": len(content),
#             "vector_db": "Pinecone Cloud",
#             "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
#             "llm_model": "GroqAI (Llama3-8b-8192)"
#         }

#     except Exception as e:
#         logger.error(f"❌ Error uploading PDF: {str(e)}")
#         traceback.print_exc()
#         return JSONResponse(
#             content={"error": f"Failed to process PDF: {str(e)}",
#                      "processing_time": round(time.time() - start_time, 2)},
#             status_code=500
#         )


# @app.post("/upload_text/")
# async def upload_text(text: str = Form(...), chat_id: str = Form(...), title: str = Form("Uploaded Text")):
#     """Upload text content and store it in Pinecone under a namespace."""
#     start_time = time.time()
#     try:
#         if not text.strip():
#             raise HTTPException(status_code=400, detail="Empty text provided")

#         process_text_with_pinecone(text, chat_id, title)

#         processing_time = time.time() - start_time
#         return {
#             "message": "✅ Text processed successfully and stored in Pinecone cloud",
#             "chat_id": sanitize_index_name(chat_id),
#             "title": title,
#             "processing_time": round(processing_time, 2),
#             "text_length": len(text),
#             "vector_db": "Pinecone Cloud (Namespace)",
#             "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
#             "llm_model": "GroqAI (Llama3-8b-8192)"
#         }

#     except Exception as e:
#         logger.error(f"❌ Error uploading text: {str(e)}")
#         traceback.print_exc()
#         return JSONResponse(
#             content={"error": f"Failed to process text: {str(e)}",
#                      "processing_time": round(time.time() - start_time, 2)},
#             status_code=500
#         )


# @app.post("/chat/")
# async def enhanced_chat(chat_id: str = Form(...), message: str = Form(...)):
#     """Enhanced chat with GroqAI + Pinecone retriever + reranker + citations."""
#     start_time = time.time()
#     try:
#         if not message.strip():
#             raise HTTPException(status_code=400, detail="Empty message provided")

#         # Step 1: Get retriever
#         retriever = get_pinecone_retriever(chat_id)
#         retrieved_docs = retriever.invoke(message)
#         logger.info(f"Retrieved {len(retrieved_docs)} documents")

#         if not retrieved_docs:
#             return {
#                 "response": "No relevant information found in uploaded documents.",
#                 "answer": "No relevant information found in uploaded documents.",
#                 "sources": [],
#                 "retrieved_chunks": 0,
#                 "reranked_chunks": 0,
#                 "processing_time": round(time.time() - start_time, 2),
#                 "vector_db": "Pinecone Cloud (Namespace)",
#                 "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
#                 "llm_model": "GroqAI (Llama3-8b-8192)",
#                 "namespace": sanitize_index_name(chat_id),
#                 "error": None
#             }

#         # Step 2: Rerank top-k chunks
#         reranked_docs = simple_reranker(retrieved_docs, message, top_k=5)
#         logger.info(f"Reranked to top {len(reranked_docs)} most relevant chunks")

#         # Step 3: Generate answer with citations
#         result = generate_answer_with_citations(message, reranked_docs)

#         result.update({
#             "response": result["answer"],
#             "retrieved_chunks": len(retrieved_docs),
#             "reranked_chunks": len(reranked_docs),
#             "total_processing_time": round(time.time() - start_time, 2),
#             "vector_db": "Pinecone Cloud (Namespace)",
#             "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
#             "llm_model": "GroqAI (Llama3-8b-8192)",
#             "namespace": sanitize_index_name(chat_id)
#         })

#         return result

#     except Exception as e:
#         logger.error(f"❌ Error in enhanced chat: {str(e)}")
#         traceback.print_exc()
#         return JSONResponse(
#             content={"error": f"Chat processing failed: {str(e)}",
#                      "processing_time": round(time.time() - start_time, 2)},
#             status_code=500
#         )

# @app.delete("/delete_chat/{chat_id}")
# def delete_chat(chat_id: str):
#     """Delete vectors from namespace."""
#     sanitized_chat_id = sanitize_index_name(chat_id)
    
#     try:
#         existing_indexes = pc.list_indexes()
#         index_names = [idx['name'] for idx in existing_indexes]

#         if not index_names:
#             raise HTTPException(status_code=404, detail="No indexes found")
        
#         # Always use the shared index
#         shared_index_name = "rag-shared-index" if "rag-shared-index" in index_names else index_names[0]
#         index = pc.Index(shared_index_name)
        
#         # Check if namespace exists first
#         try:
#             stats = index.describe_index_stats()
#             namespaces = stats.get("namespaces", {})
            
#             if sanitized_chat_id not in namespaces:
#                 return JSONResponse(
#                     content={
#                         "message": f"Chat ID '{chat_id}' not found or already deleted",
#                         "namespace": sanitized_chat_id,
#                         "available_namespaces": list(namespaces.keys())
#                     },
#                     status_code=404
#                 )
#         except Exception as e:
#             logger.warning(f"Could not check namespace stats: {e}")
        
#         # Delete all vectors in the namespace
#         try:
#             delete_response = index.delete(delete_all=True, namespace=sanitized_chat_id)
#             logger.info(f"Delete response: {delete_response}")
#             logger.info(f"Successfully deleted all vectors in namespace '{sanitized_chat_id}'")
            
#             return {
#                 "message": f"Chat ID '{chat_id}' deleted successfully from Pinecone",
#                 "deleted_namespace": sanitized_chat_id,
#                 "index_used": shared_index_name,
#                 "status": "success"
#             }
            
#         except Exception as e:
#             logger.error(f"Error deleting namespace '{sanitized_chat_id}': {str(e)}")
#             return JSONResponse(
#                 content={
#                     "message": f"Failed to delete chat ID '{chat_id}'",
#                     "namespace": sanitized_chat_id,
#                     "error": str(e)
#                 },
#                 status_code=500
#             )

#     except Exception as e:
#         logger.error(f"Error in delete_chat: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/list_chats/")
# def list_chat_ids():
#     """List all available namespaces (chat IDs) in shared indexes."""
#     try:
#         indexes = pc.list_indexes()
        
#         if not indexes:
#             return {
#                 "chat_ids": [],
#                 "chats": [],
#                 "vector_db": "Pinecone Cloud (Namespace)",
#                 "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
#                 "llm_model": "GroqAI (Llama3-8b-8192)",
#                 "total_namespaces": 0
#             }
        
#         chat_info = []
#         chat_ids = []
        
#         for idx in indexes:
#             try:
#                 index = pc.Index(idx['name'])
#                 stats = index.describe_index_stats()
                
#                 namespaces = stats.namespaces if hasattr(stats, 'namespaces') else {}
                
#                 for namespace_name, namespace_stats in namespaces.items():
#                     chat_ids.append(namespace_name)
#                     chat_info.append({
#                         "chat_id": namespace_name,
#                         "namespace": namespace_name,
#                         "index_name": idx['name'],
#                         "vector_count": namespace_stats.vector_count if hasattr(namespace_stats, 'vector_count') else 0,
#                         "dimension": stats.dimension,
#                         "cloud": "Pinecone",
#                         "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
#                         "llm_model": "GroqAI (Llama3-8b-8192)",
#                         "status": "ready"
#                     })
                
#             except Exception as e:
#                 logger.warning(f"Could not get stats for index {idx['name']}: {e}")
        
#         return {
#             "chat_ids": chat_ids,
#             "chats": chat_info,
#             "vector_db": "Pinecone Cloud (Namespace)",
#             "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
#             "llm_model": "GroqAI (Llama3-8b-8192)",
#             "total_namespaces": len(chat_info)
#         }
        
#     except Exception as e:
#         logger.error(f"Error listing chats: {str(e)}")
#         return {
#             "chat_ids": [],
#             "chats": [],
#             "error": str(e),
#             "vector_db": "Pinecone Cloud (Namespace)",
#             "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
#             "llm_model": "GroqAI (llama-3.1-70b-versatile)",
#             "total_namespaces": 0
#         }

# @app.get("/debug_pinecone")
# def debug_pinecone():
#     """Debug endpoint to check Pinecone connection and indexes."""
#     try:
#         # Test Pinecone connection
#         indexes = pc.list_indexes()
        
#         debug_info = {
#             "pinecone_connected": True,
#             "api_key_configured": bool(PINECONE_API_KEY),
#             "environment": PINECONE_ENVIRONMENT,
#             "total_indexes": len(indexes),
#             "indexes": []
#         }
        
#         for idx in indexes:
#             index_info = {
#                 "name": idx["name"],
#                 "dimension": idx.get("dimension", "unknown"),
#                 "metric": idx.get("metric", "unknown"),
#                 "status": idx.get("status", "unknown")
#             }
            
#             try:
#                 # Get detailed stats
#                 index = pc.Index(idx["name"])
#                 stats = index.describe_index_stats()
#                 index_info.update({
#                     "total_vectors": stats.total_vector_count,
#                     "namespaces": list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') else []
#                 })
#             except Exception as e:
#                 index_info["error"] = str(e)
            
#             debug_info["indexes"].append(index_info)
        
#         return debug_info
        
#     except Exception as e:
#         return {
#             "pinecone_connected": False,
#             "error": str(e),
#             "api_key_configured": bool(PINECONE_API_KEY),
#             "environment": PINECONE_ENVIRONMENT,
#             "suggestion": "Check your Pinecone API key and environment in .env file"
#         }

# @app.get("/health")
# def health_check():
#     """Comprehensive health check endpoint for debugging deployment issues."""
#     try:
#         health_status = {
#             "status": "healthy",
#             "timestamp": time.time(),
#             "environment": {
#                 "groq_api_key_configured": bool(GROQ_API_KEY),
#                 "nomic_api_key_configured": bool(NOMIC_API_KEY),
#                 "pinecone_api_key_configured": bool(PINECONE_API_KEY),
#                 "pinecone_environment": PINECONE_ENVIRONMENT
#             },
#             "models": {
#                 "embedding_model": "Nomic AI text-embedding-v1.5",
#                 "embedding_dimension": 384,
#                 "llm_model": "GroqAI llama-3.1-70b-versatile"
#             }
#         }
        
#         # Test Pinecone connection
#         try:
#             indexes = pc.list_indexes()
#             health_status["pinecone"] = {
#                 "connected": True,
#                 "indexes_count": len(indexes),
#                 "indexes": [idx["name"] for idx in indexes]
#             }
#         except Exception as e:
#             health_status["pinecone"] = {
#                 "connected": False,
#                 "error": str(e)
#             }
#             health_status["status"] = "degraded"
        
#         # Test GroqAI connection with a simple request
#         try:
#             test_response = call_groq_api("Hello, this is a test.", max_tokens=10, temperature=0.1)
#             health_status["groq"] = {
#                 "connected": True,
#                 "test_response_length": len(test_response)
#             }
#         except Exception as e:
#             health_status["groq"] = {
#                 "connected": False,
#                 "error": str(e)
#             }
#             health_status["status"] = "degraded"
        
#         # Test Nomic embeddings
#         try:
#             embeddings = GroqEmbeddings()
#             test_embedding = embeddings.embed_query("test")
#             health_status["nomic_embeddings"] = {
#                 "working": True,
#                 "dimension": len(test_embedding),
#                 "model": "nomic-embed-text-v1.5"
#             }
#         except Exception as e:
#             health_status["nomic_embeddings"] = {
#                 "working": False,
#                 "error": str(e)
#             }
#             health_status["status"] = "degraded"
        
#         return health_status
        
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e),
#             "timestamp": time.time()
#         }

# @app.get("/")
# def root():
#     """Root endpoint with API information."""
#     return {
#         "message": "Mini RAG System - GroqAI + Pinecone + Nomic Embeddings",
#         "version": "1.0.0",
#         "vector_db": "Pinecone Cloud",
#         "embedding_model": "Nomic AI text-embedding-v1.5 (384d)",
#         "llm_model": "GroqAI (llama-3.1-70b-versatile)",
#         "features": [
#             "Enhanced PDF/text processing",
#             "Cloud vector storage (Pinecone)", 
#             "Nomic AI embeddings (384d)",
#             "Simple reranking",
#             "Citation generation with GroqAI",
#             "Fast response times",
#             "Graceful error handling",
#             "Free tier compatible"
#         ],
#         "endpoints": [
#             "/upload_pdf/", 
#             "/upload_text/", 
#             "/chat/", 
#             "/list_chats/", 
#             "/delete_chat/{chat_id}",
#             "/debug_pinecone",
#             "/health"
#         ]
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



