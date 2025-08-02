import os
# Disable GPU loading for FAISS at application startup
os.environ['FAISS_NO_AVX2'] = '1'
os.environ['FAISS_NO_GPU'] = '1'

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from utils.pdf_utils import extract_text_from_pdf, chunk_text
from utils.embed_utils import insert_into_faiss, search_similar_chunks
import google.generativeai as genai
import asyncio
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from cachetools import TTLCache
import httpx
from functools import lru_cache
import time
import logging
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI(title="PDF Bot API", root_path="/api/v1")

# Authentication
security = HTTPBearer()
AUTH_TOKEN = os.getenv("AUTHORIZE_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the authentication token"""
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Enhanced Caching with larger sizes and longer TTL
document_cache = TTLCache(maxsize=300, ttl=7200)  # Increased from 200 to 300
qa_cache = TTLCache(maxsize=3000, ttl=3600)  # Increased from 2000 to 3000
embedding_cache = TTLCache(maxsize=3000, ttl=7200)  # Increased from 2000 to 3000

# HTTP client with optimized connection pooling
try:
    http_client = httpx.AsyncClient(
        timeout=15.0,  # Reduced timeout
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
        http2=True  # Enable HTTP/2 for better performance
    )
except ImportError:
    # Fallback if HTTP/2 is not available
    logger.warning("HTTP/2 not available, using HTTP/1.1")
    http_client = httpx.AsyncClient(
        timeout=15.0,
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200)
    )

# Optimized thread pools for 15-second response time with sentence transformers
# Calculate optimal thread counts based on CPU cores
cpu_count = multiprocessing.cpu_count()
cpu_executor = ThreadPoolExecutor(max_workers=min(cpu_count * 8, 64))   # Optimized for sentence transformers
io_executor = ThreadPoolExecutor(max_workers=min(cpu_count * 6, 48))    # Optimized for sentence transformers
process_executor = ProcessPoolExecutor(max_workers=min(cpu_count * 4, 32))  # Optimized for sentence transformers

# Compiled regex patterns
whitespace_pattern = re.compile(r'\s+')
newline_pattern = re.compile(r'\n+')

class HackRXRequest(BaseModel):
    documents: str
    questions: list[str]

def get_content_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

def is_local_path(string: str) -> bool:
    return Path(string).exists()

async def download_file_from_url_async(url: str) -> bytes:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/pdf,*/*'
    }
    response = await http_client.get(url, headers=headers)
    response.raise_for_status()

    content_type = response.headers.get('content-type', '').lower()
    if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
        if not response.content.startswith(b'%PDF-'):
            raise HTTPException(status_code=400, detail="URL does not point to a valid PDF")
    return response.content

def read_local_file(file_path: str) -> bytes:
    path = Path(file_path)
    if not path.exists() or not path.suffix.lower() == '.pdf':
        raise HTTPException(status_code=400, detail="Invalid file path or not a PDF")
    return path.read_bytes()

@lru_cache(maxsize=256)  # Increased cache size
def clean_extracted_text(text: str) -> str:
    text = whitespace_pattern.sub(' ', text)
    text = text.replace('\f', '\n').replace('\r', '')
    text = newline_pattern.sub('\n', text)
    return text.strip()

async def process_document_content_async(content: bytes, source_name: str) -> dict:
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    content_hash = get_content_hash(content)
    
    # Check cache first
    if content_hash in document_cache:
        logger.info(f"Cache hit for document: {content_hash}")
        return document_cache[content_hash]

    # Process in thread pool with timeout
    loop = asyncio.get_event_loop()
    
    def process_sync():
        start_time = time.time()
        text = extract_text_from_pdf(content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in document")

        text = clean_extracted_text(text)
        chunks = chunk_text(text, chunk_size=10000, overlap=0)  # Page-based chunking for sentence transformers
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to chunk document")

        # Use FAISS instead of Pinecone
        insert_into_faiss(chunks, metadata={
            "source": source_name,
            "total_chunks": len(chunks),
            "original_text_length": len(text),
            "content_hash": content_hash
        })

        processing_time = time.time() - start_time
        logger.info(f"Document processing completed in {processing_time:.2f}s with sentence transformers")

        return {
            "chunks_created": len(chunks),
            "text_length": len(text),
            "status": "success",
            "content_hash": content_hash,
            "processing_time": processing_time
        }

    # Add timeout to prevent hanging
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(cpu_executor, process_sync),
            timeout=25.0  # Reduced timeout for 15-second target
        )
        document_cache[content_hash] = result
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Document processing timed out")

def process_chunk_with_embedding(chunk_data):
    """Process a single chunk with embedding - for multithreading"""
    chunk, metadata = chunk_data
    try:
        from utils.embed_utils import get_embedding
        embedding = get_embedding(chunk, "retrieval_document")
        return {
            'embedding': embedding,
            'metadata': metadata,
            'success': True
        }
    except Exception as e:
        return {
            'embedding': None,
            'metadata': metadata,
            'success': False,
            'error': str(e)
        }

async def process_questions_multithreaded(questions: list, content_hash: str) -> list:
    """Process multiple questions concurrently with sentence transformers for 15-second target"""
    if not questions:
        return []
    
    logger.info(f"Processing {len(questions)} questions with sentence transformers...")
    
    # Create tasks for concurrent processing
    tasks = []
    for question in questions:
        if question.strip():
            task = process_question_async(question, content_hash)
            tasks.append(task)
    
    # Enhanced concurrency with adaptive semaphore
    max_concurrent = min(len(tasks), 4)  # Optimized for 15-second target
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(task):
        async with semaphore:
            return await task
    
    # Process in batches for better resource management
    batch_size = 3  # Process 3 questions at a time for 15-second target
    all_results = []
    
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        
        # Run batch concurrently
        batch_results = await asyncio.gather(
            *[process_with_semaphore(task) for task in batch], 
            return_exceptions=True
        )
        
        # Handle exceptions in batch
        for result in batch_results:
            if isinstance(result, Exception):
                all_results.append(f"Error processing question: {str(result)}")
            else:
                all_results.append(result)
        
        logger.info(f"Completed batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
    
    return all_results

async def process_questions_parallel_optimized(questions: list, content_hash: str) -> list:
    """Advanced parallel processing with different strategies based on question count"""
    if not questions:
        return []
    
    question_count = len(questions)
    logger.info(f"Processing {question_count} questions with parallel optimization...")
    
    # Different strategies based on question count
    if question_count <= 3:
        # Small batch: Process all concurrently
        return await process_questions_small_batch(questions, content_hash)
    elif question_count <= 8:
        # Medium batch: Process in parallel with batching
        return await process_questions_medium_batch(questions, content_hash)
    else:
        # Large batch: Process with advanced parallelism
        return await process_questions_large_batch(questions, content_hash)

async def process_questions_small_batch(questions: list, content_hash: str) -> list:
    """Process small batches (≤3 questions) with maximum concurrency"""
    tasks = [process_question_async(q, content_hash) for q in questions if q.strip()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(f"Error processing question: {str(result)}")
        else:
            processed_results.append(result)
    
    return processed_results

async def process_questions_medium_batch(questions: list, content_hash: str) -> list:
    """Process medium batches (4-8 questions) with controlled parallelism"""
    tasks = [process_question_async(q, content_hash) for q in questions if q.strip()]
    
    # Use semaphore for controlled concurrency
    semaphore = asyncio.Semaphore(3)  # Allow 3 concurrent questions for 15-second target
    
    async def process_with_semaphore(task):
        async with semaphore:
            return await task
    
    results = await asyncio.gather(*[process_with_semaphore(task) for task in tasks], return_exceptions=True)
    
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(f"Error processing question: {str(result)}")
        else:
            processed_results.append(result)
    
    return processed_results

async def process_questions_large_batch(questions: list, content_hash: str) -> list:
    """Process large batches (>8 questions) with advanced batching and parallelism"""
    tasks = [process_question_async(q, content_hash) for q in questions if q.strip()]
    
    # Enhanced concurrency with adaptive semaphore
    max_concurrent = min(len(tasks), 6)  # Optimized for 15-second target
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(task):
        async with semaphore:
            return await task
    
    # Process in smaller batches for better resource management
    batch_size = 4  # Smaller batches for 15-second target
    all_results = []
    
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        
        # Run batch concurrently
        batch_results = await asyncio.gather(
            *[process_with_semaphore(task) for task in batch], 
            return_exceptions=True
        )
        
        # Handle exceptions in batch
        for result in batch_results:
            if isinstance(result, Exception):
                all_results.append(f"Error processing question: {str(result)}")
            else:
                all_results.append(result)
        
        logger.info(f"Completed large batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
    
    return all_results

async def process_questions_ultra_fast(questions: list, content_hash: str) -> list:
    """Ultra-fast parallel processing optimized for 12+ questions to achieve 30-second target"""
    if not questions:
        return []
    
    question_count = len(questions)
    logger.info(f"Processing {question_count} questions with ultra-fast optimization...")
    
    # Ultra-aggressive strategy for 12+ questions
    if question_count >= 12:
        return await process_questions_ultra_aggressive(questions, content_hash)
    elif question_count >= 8:
        return await process_questions_high_concurrency(questions, content_hash)
    else:
        return await process_questions_parallel_optimized(questions, content_hash)

async def process_questions_ultra_aggressive(questions: list, content_hash: str) -> list:
    tasks = [process_question_async(q, content_hash) for q in questions if q.strip()]

    # Increase concurrency and batch size
    max_concurrent = min(len(tasks), 16)  # Optimized for 15-second target
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(task):
        async with semaphore:
            return await task

    logger.info(f"Starting ultra-aggressive processing with {max_concurrent} concurrent questions")

    batch_size = 4  # Optimized for 15-second target
    all_results = []

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_with_semaphore(task) for task in batch],
            return_exceptions=True
        )

        for result in batch_results:
            if isinstance(result, Exception):
                all_results.append(f"Error processing question: {str(result)}")
            else:
                all_results.append(result)

        logger.info(f"Completed ultra-aggressive batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")

    return all_results

async def process_questions_high_concurrency(questions: list, content_hash: str) -> list:
    """High concurrency processing for 8-11 questions"""
    tasks = [process_question_async(q, content_hash) for q in questions if q.strip()]
    
    # High concurrency for medium batches
    max_concurrent = min(len(tasks), 8)  # Optimized for 15-second target
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(task):
        async with semaphore:
            return await task
    
    # Process with high concurrency
    results = await asyncio.gather(*[process_with_semaphore(task) for task in tasks], return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(f"Error processing question: {str(result)}")
        else:
            processed_results.append(result)
    
    return processed_results

async def process_question_async(question: str, content_hash: str) -> str:
    if not question.strip():
        return "Invalid question"

    # Check cache
    cache_key = f"{content_hash}:{question}"
    if cache_key in qa_cache:
        logger.info(f"Cache hit for question: {question[:50]}...")
        return qa_cache[cache_key]

    loop = asyncio.get_event_loop()
    
    try:
        # Use top_k=1 for faster processing with sentence transformers
        search_task = loop.run_in_executor(
            io_executor, lambda: search_similar_chunks(question, top_k=1)
        )
        context_chunks = await asyncio.wait_for(search_task, timeout=1.5)

        if not context_chunks:
            return "Sorry, no relevant information found."

        context = "\n\n".join(context_chunks)

        # Generate answer using Gemini with tighter timeout
        prompt = f"""
You are an expert insurance document analyst. Answer the question using only the provided context.

Instructions:
- Answer in 1–2 sentences only
- Start Yes/No questions with Yes or No
- Include numbers/percentages if present
- Keep answer under 40 words
- Be factual and concise
- Give the response soon to the user within 20 seconds

Context:
{context}

Question: {question}

Answer:"""

        answer_task = loop.run_in_executor(
            cpu_executor, lambda: chat_model.generate_content(prompt)
        )
        response = await asyncio.wait_for(answer_task, timeout=6.0)
        answer = response.text.strip()

        qa_cache[cache_key] = answer
        return answer

    except asyncio.TimeoutError:
        return "Sorry, the question processing timed out. Please try again."
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return f"Error processing question: {str(e)}"


@app.post("/hackrx/run")
async def hackrx_run(request: HackRXRequest, token: str = Depends(verify_token)):
    start_time = time.time()
    
    try:
        document_source = request.documents.strip()
        questions = request.questions

        if not document_source or not questions:
            raise HTTPException(status_code=400, detail="Missing document or questions")

        # Get document content
        if is_url(document_source):
            content = await download_file_from_url_async(document_source)
            source_name = f"URL: {document_source}"
        elif is_local_path(document_source):
            content = read_local_file(document_source)
            source_name = f"Local: {document_source}"
        else:
            raise HTTPException(status_code=400, detail="Invalid document source")

        # Process document
        processing_result = await process_document_content_async(content, source_name)
        content_hash = processing_result["content_hash"]

        # Filter valid questions
        valid_questions = [q for q in questions if q.strip()]
        if not valid_questions:
            return {"answers": ["Invalid question"] * len(questions)}

        # Process questions with enhanced multithreading
        answers = await process_questions_ultra_fast(valid_questions, content_hash)
        
        # Map back to original order (with invalids)
        result_answers = []
        valid_idx = 0
        for question in questions:
            if question.strip():
                result_answers.append(answers[valid_idx])
                valid_idx += 1
            else:
                result_answers.append("Invalid question")

        # Clean up formatting
        cleaned_answers = [
            answer.replace("\\n", "\n").replace('\\"', '"').strip()
            for answer in result_answers
        ]

        return {
            "answers": cleaned_answers
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint (no auth required)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "cache_sizes": {
        "document_cache": len(document_cache),
        "qa_cache": len(qa_cache),
        "embedding_cache": len(embedding_cache)
    }}

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    cpu_executor.shutdown(wait=True)
    io_executor.shutdown(wait=True)
    process_executor.shutdown(wait=True)
    logger.info("All thread pools and HTTP client closed")

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment (Render sets PORT env var)
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        workers=1
    )
