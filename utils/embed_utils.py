# embed_utils.py - OPTIMIZED VERSION WITH FAISS AND SENTENCE TRANSFORMERS
import os
# Disable GPU loading for FAISS
os.environ['FAISS_NO_AVX2'] = '1'
os.environ['FAISS_NO_GPU'] = '1'

import faiss
import numpy as np
import re
import pickle
import hashlib
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging
from pathlib import Path
import joblib
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# FAISS Index Configuration
FAISS_INDEX_PATH = "faiss_index"
FAISS_METADATA_PATH = "faiss_metadata.pkl"

# Initialize Sentence Transformer model
try:
    # Use a fast and efficient model for 15-second response time
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions, very fast
    DIMENSION = 384
    logger.info("Loaded all-MiniLM-L6-v2 sentence transformer model")
except Exception as e:
    logger.error(f"Error loading sentence transformer model: {e}")
    # Fallback to a smaller model
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 384 dimensions, even faster
        DIMENSION = 384
        logger.info("Loaded paraphrase-MiniLM-L3-v2 sentence transformer model")
    except Exception as e2:
        logger.error(f"Error loading fallback model: {e2}")
        raise Exception("Could not load any sentence transformer model")

# Caching for embeddings and search results
embedding_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour
search_cache = TTLCache(maxsize=500, ttl=1800)  # 30 minutes

class FAISSVectorStore:
    """High-performance local vector store using CPU-only FAISS"""
    
    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.vector_count = 0
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        try:
            if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
                # Load existing index
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                with open(FAISS_METADATA_PATH, 'rb') as f:
                    self.metadata = pickle.load(f)
                self.vector_count = len(self.metadata)
                logger.info(f"Loaded existing FAISS index with {self.vector_count} vectors")
            else:
                # Create new CPU-only index
                self.index = faiss.IndexFlatIP(self.dimension)  # CPU-only inner product for cosine similarity
                self.metadata = []
                self.vector_count = 0
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error loading/creating FAISS index: {e}")
            # Fallback to new CPU-only index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self.vector_count = 0
    
    def _save_index(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            with open(FAISS_METADATA_PATH, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved FAISS index with {self.vector_count} vectors")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def add_vectors(self, vectors: List[np.ndarray], metadata_list: List[Dict[str, Any]]):
        """Add vectors to the index - OPTIMIZED for CPU speed"""
        if not vectors or not metadata_list:
            return
        
        # Convert to numpy array and normalize for cosine similarity
        vectors_np = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_np)
        
        # Add to index in one batch for CPU speed
        self.index.add(vectors_np)
        
        # Add metadata
        self.metadata.extend(metadata_list)
        self.vector_count += len(vectors)
        
        # Save less frequently for speed
        if self.vector_count % 5000 == 0:  # Optimized for 15-second target
            self._save_index()
        
        logger.info(f"Added {len(vectors)} vectors to FAISS index. Total: {self.vector_count}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 8, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if self.vector_count == 0:
            return []
        
        # Normalize query vector
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, min(top_k * 2, self.vector_count))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= similarity_threshold:
                results.append({
                    'text': self.metadata[idx].get('text', ''),
                    'score': float(score),
                    'metadata': self.metadata[idx]
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_vectors': self.vector_count,
            'dimension': self.dimension,
            'index_type': 'FAISS FlatIP',
            'metadata_count': len(self.metadata)
        }
    
    def clear(self):
        """Clear all vectors"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.vector_count = 0
        self._save_index()
        logger.info("Cleared FAISS index")

# Initialize global CPU-only vector store
vector_store = FAISSVectorStore()

def get_embedding(text: str, task_type: str = "retrieval_document") -> np.ndarray:
    """Get embedding using sentence transformers for 15-second response time"""
    # Create cache key
    cache_key = f"{hashlib.md5(text.encode()).hexdigest()}:{task_type}"
    
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    try:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Clean text before embedding - OPTIMIZED for sentence transformers
        text = text.strip()
        if len(text) > 4000:  # Increased for page-based chunks
            text = text[:4000]
            logger.info(f"Truncated text to 4000 characters for sentence transformer")
        
        # Use sentence transformer for embedding
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        embedding = embedding.astype(np.float32)
        
        # Ensure exactly DIMENSION dimensions
        if len(embedding) < DIMENSION:
            # Pad with zeros
            padding = np.zeros(DIMENSION - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        elif len(embedding) > DIMENSION:
            # Truncate
            embedding = embedding[:DIMENSION]
        
        # Cache the result
        embedding_cache[cache_key] = embedding
        return embedding
        
    except Exception as e:
        logger.error(f"Error getting embedding for text: {str(e)}")
        raise

def preprocess_query(query: str) -> List[str]:
    """Generate minimal query variations for ultra-fast search"""
    query = query.strip().lower()
    variations = [query]
    
    # Minimal insurance term mappings for speed
    term_mappings = {
        'ncd': ['no claim discount'],
        'no claim discount': ['ncd'],
        'waiting period': ['wait period'],
        'pre-existing': ['ped'],
        'maternity': ['pregnancy'],
        'room rent': ['accommodation'],
        'icu': ['intensive care'],
        'ayush': ['ayurveda'],
        'organ donor': ['transplant'],
        'health checkup': ['checkup'],
        'hospital': ['medical facility'],
        'grace period': ['grace time']
    }
    
    # Add only essential variations
    for key, synonyms in term_mappings.items():
        if key in query:
            for synonym in synonyms[:1]:  # Only first synonym for speed
                new_query = query.replace(key, synonym)
                if new_query not in variations:
                    variations.append(new_query)
                    break  # Only add one variation per term
    
    logger.info(f"Generated {len(variations)} query variations for: '{query}'")
    return list(set(variations))  # Remove duplicates

def insert_into_faiss(chunks: List[str], metadata: Dict[str, Any] = {}):
    """Insert chunks into FAISS index with sentence transformers for 15-second response time"""
    try:
        if not chunks:
            logger.info("No chunks to insert")
            return
        
        logger.info(f"Starting to process {len(chunks)} chunks with sentence transformers...")
        
        # Enhanced metadata with processing info
        base_metadata = {
            **metadata,
            'insertion_timestamp': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'processing_version': '6.0_faiss_sentence_transformers'
        }
        
        # Prepare chunk data for multithreading
        chunk_data = []
        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                logger.info(f"Skipping empty chunk at index {i}")
                continue
            
            chunk_metadata = {
                **base_metadata,
                'text': chunk,
                'chunk_index': i,
                'chunk_length': len(chunk),
                'chunk_words': len(chunk.split()),
                'chunk_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
            }
            chunk_data.append((chunk, chunk_metadata))
        
        # Process embeddings with advanced multithreading
        vectors = []
        metadata_list = []
        successful_chunks = 0
        failed_chunks = 0
        
        # Use ProcessPoolExecutor for CPU-intensive embedding generation
        import multiprocessing
        
        # Calculate optimal worker count based on system resources
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(cpu_count * 2, 8)  # Optimized for sentence transformers
        
        # Process in batches for better memory management
        batch_size = 15  # Optimized for 15-second target
        
        for batch_start in range(0, len(chunk_data), batch_size):
            batch_end = min(batch_start + batch_size, len(chunk_data))
            batch_data = chunk_data[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(chunk_data) + batch_size - 1)//batch_size}")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit batch tasks
                future_to_chunk = {
                    executor.submit(get_embedding, chunk, "retrieval_document"): (i + batch_start, chunk_metadata) 
                    for i, (chunk, chunk_metadata) in enumerate(batch_data)
                }
                
                # Collect batch results
                batch_vectors = []
                batch_metadata = []
                
                for future in as_completed(future_to_chunk):
                    chunk_idx, chunk_metadata = future_to_chunk[future]
                    try:
                        embedding = future.result()
                        batch_vectors.append(embedding)
                        batch_metadata.append(chunk_metadata)
                        successful_chunks += 1
                        logger.info(f"Processed chunk {chunk_idx + 1}/{len(chunk_data)}")
                    except Exception as e:
                        failed_chunks += 1
                        logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                        continue
                
                # Add batch to main results
                vectors.extend(batch_vectors)
                metadata_list.extend(batch_metadata)
        
        # Add to FAISS index
        if vectors:
            vector_store.add_vectors(vectors, metadata_list)
            logger.info(f"Successfully inserted {successful_chunks} vectors into FAISS using sentence transformers")
            logger.info(f"Failed to process {failed_chunks} chunks")
            
            # Get updated stats
            stats = vector_store.get_stats()
            logger.info(f"Total vectors in index: {stats['total_vectors']}")
            
        else:
            logger.info("No valid vectors to insert")
            
    except Exception as e:
        logger.error(f"Error inserting into FAISS: {str(e)}")
        raise

def search_similar_chunks(query: str, top_k: int = 1, similarity_threshold: float = 0.3) -> List[str]:
    """Ultra-fast similarity search optimized for 15-second target with sentence transformers"""
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Create cache key
        cache_key = f"{hashlib.md5(query.encode()).hexdigest()}:{top_k}:{similarity_threshold}"
        if cache_key in search_cache:
            return search_cache[cache_key]
        
        logger.info(f"Searching for: '{query}' (top_k={top_k}, threshold={similarity_threshold})")
        
        # Generate minimal query variations for speed
        query_variations = preprocess_query(query)
        
        all_matches = []
        seen_texts = set()
        
        # Search with only 1 query variation for maximum speed
        for variation in query_variations[:1]:  # Single variation for 15-second target
            try:
                query_vector = get_embedding(variation, "retrieval_query")
                results = vector_store.search(query_vector, top_k * 2, similarity_threshold)
                
                # Filter and deduplicate
                for result in results:
                    text = result['text']
                    score = result['score']
                    
                    if text not in seen_texts:
                        all_matches.append({
                            'text': text,
                            'score': score,
                            'query_variation': variation
                        })
                        seen_texts.add(text)
                        
            except Exception as variation_error:
                logger.error(f"Error searching with variation '{variation}': {str(variation_error)}")
                continue
        
        # Sort by score and return top results
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        final_results = [match['text'] for match in all_matches[:top_k]]
        
        logger.info(f"Found {len(final_results)} relevant chunks (from {len(all_matches)} total matches)")
        
        # Cache the results
        search_cache[cache_key] = final_results
        return final_results
        
    except Exception as e:
        logger.error(f"Error searching similar chunks: {str(e)}")
        # Fallback to simple search
        try:
            logger.info("Attempting fallback search...")
            query_vector = get_embedding(query, "retrieval_query")
            results = vector_store.search(query_vector, top_k, similarity_threshold)
            return [result['text'] for result in results]
        except Exception as fallback_error:
            logger.error(f"Fallback search also failed: {str(fallback_error)}")
            raise

def get_index_statistics() -> Dict[str, Any]:
    """Get comprehensive statistics about the FAISS index"""
    try:
        stats = vector_store.get_stats()
        return {
            **stats,
            'index_path': FAISS_INDEX_PATH,
            'metadata_path': FAISS_METADATA_PATH,
            'embedding_cache_size': len(embedding_cache),
            'search_cache_size': len(search_cache)
        }
    except Exception as e:
        logger.error(f"Error getting index statistics: {str(e)}")
        return {'error': str(e)}

def clear_index(confirm: bool = False):
    """Clear all vectors from the index (use with caution!)"""
    if not confirm:
        logger.warning("This will delete ALL vectors from the index. Call with confirm=True to proceed.")
        return
    
    try:
        vector_store.clear()
        logger.info(f"Successfully cleared all vectors from FAISS index")
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        raise

def search_by_metadata(metadata_filter: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
    """Search vectors by metadata filters"""
    try:
        # Filter metadata first
        filtered_metadata = []
        for i, meta in enumerate(vector_store.metadata):
            if all(meta.get(key) == value for key, value in metadata_filter.items()):
                filtered_metadata.append((i, meta))
        
        # Get vectors for filtered indices
        if filtered_metadata:
            indices = [idx for idx, _ in filtered_metadata[:top_k]]
            # Note: This is a simplified implementation. For production, you might want to maintain a separate metadata index
            return [
                {
                    'text': vector_store.metadata[idx].get('text', ''),
                    'metadata': vector_store.metadata[idx],
                    'score': 1.0  # Placeholder score
                }
                for idx in indices
            ]
        return []
    except Exception as e:
        logger.error(f"Error searching by metadata: {str(e)}")
        raise

# Backward compatibility functions
def insert_into_pinecone(chunks: List[str], metadata: Dict[str, Any] = {}):
    """Backward compatibility - redirects to FAISS"""
    return insert_into_faiss(chunks, metadata)