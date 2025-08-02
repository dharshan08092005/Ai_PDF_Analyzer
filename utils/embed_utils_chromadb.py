# embed_utils_chromadb.py - ALTERNATIVE VERSION WITH CHROMADB
import os
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
import chromadb
from chromadb.config import Settings
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

class ChromaDBVectorStore:
    """High-performance local vector store using ChromaDB"""
    
    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension
        self.client = None
        self.collection = None
        self.vector_count = 0
        self._load_or_create_collection()
    
    def _load_or_create_collection(self):
        """Load existing collection or create new one"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection("insurance_documents")
                self.vector_count = self.collection.count()
                logger.info(f"Loaded existing ChromaDB collection with {self.vector_count} vectors")
            except:
                # Create new collection
                self.collection = self.client.create_collection(
                    name="insurance_documents",
                    metadata={"description": "Insurance document embeddings"}
                )
                self.vector_count = 0
                logger.info("Created new ChromaDB collection")
                
        except Exception as e:
            logger.error(f"Error loading/creating ChromaDB collection: {e}")
            raise
    
    def add_vectors(self, vectors: List[np.ndarray], metadata_list: List[Dict[str, Any]]):
        """Add vectors to the collection - OPTIMIZED for CPU speed"""
        if not vectors or not metadata_list:
            return
        
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for i, (vector, metadata) in enumerate(zip(vectors, metadata_list)):
                doc_id = f"doc_{self.vector_count + i}_{hashlib.md5(str(metadata).encode()).hexdigest()[:8]}"
                ids.append(doc_id)
                embeddings.append(vector.tolist())
                documents.append(metadata.get('text', ''))
                metadatas.append(metadata)
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_documents = documents[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
            
            self.vector_count += len(vectors)
            logger.info(f"Added {len(vectors)} vectors to ChromaDB collection. Total: {self.vector_count}")
            
        except Exception as e:
            logger.error(f"Error adding vectors to ChromaDB: {e}")
            raise
    
    def search(self, query_vector: np.ndarray, top_k: int = 8, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if self.vector_count == 0:
            return []
        
        try:
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k * 2
            )
            
            # Process results
            processed_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
                    similarity_score = 1.0 - distance
                    
                    if similarity_score >= similarity_threshold:
                        processed_results.append({
                            'text': doc,
                            'score': similarity_score,
                            'metadata': metadata
                        })
            
            # Sort by score and return top_k
            processed_results.sort(key=lambda x: x['score'], reverse=True)
            return processed_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'total_vectors': self.vector_count,
            'dimension': self.dimension,
            'index_type': 'ChromaDB',
            'collection_name': 'insurance_documents'
        }
    
    def clear(self):
        """Clear all vectors"""
        try:
            self.client.delete_collection("insurance_documents")
            self.collection = self.client.create_collection(
                name="insurance_documents",
                metadata={"description": "Insurance document embeddings"}
            )
            self.vector_count = 0
            logger.info("Cleared ChromaDB collection")
        except Exception as e:
            logger.error(f"Error clearing ChromaDB collection: {e}")
            raise

# Initialize global vector store
vector_store = ChromaDBVectorStore()

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

def insert_into_chromadb(chunks: List[str], metadata: Dict[str, Any] = {}):
    """Insert chunks into ChromaDB collection with sentence transformers for 15-second response time"""
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
            'processing_version': '6.0_chromadb_sentence_transformers'
        }
        
        # Prepare chunk data for processing
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
        
        # Process embeddings with multithreading
        vectors = []
        metadata_list = []
        successful_chunks = 0
        failed_chunks = 0
        
        # Calculate optimal worker count
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(cpu_count * 2, 8)  # Optimized for sentence transformers
        
        # Process in batches
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
        
        # Add to ChromaDB collection
        if vectors:
            vector_store.add_vectors(vectors, metadata_list)
            logger.info(f"Successfully inserted {successful_chunks} vectors into ChromaDB using sentence transformers")
            logger.info(f"Failed to process {failed_chunks} chunks")
            
            # Get updated stats
            stats = vector_store.get_stats()
            logger.info(f"Total vectors in collection: {stats['total_vectors']}")
            
        else:
            logger.info("No valid vectors to insert")
            
    except Exception as e:
        logger.error(f"Error inserting into ChromaDB: {str(e)}")
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
    """Get comprehensive statistics about the ChromaDB collection"""
    try:
        stats = vector_store.get_stats()
        return {
            **stats,
            'embedding_cache_size': len(embedding_cache),
            'search_cache_size': len(search_cache)
        }
    except Exception as e:
        logger.error(f"Error getting collection statistics: {str(e)}")
        return {'error': str(e)}

def clear_index(confirm: bool = False):
    """Clear all vectors from the collection (use with caution!)"""
    if not confirm:
        logger.warning("This will delete ALL vectors from the collection. Call with confirm=True to proceed.")
        return
    
    try:
        vector_store.clear()
        logger.info(f"Successfully cleared all vectors from ChromaDB collection")
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        raise

# Backward compatibility functions
def insert_into_faiss(chunks: List[str], metadata: Dict[str, Any] = {}):
    """Backward compatibility - redirects to ChromaDB"""
    return insert_into_chromadb(chunks, metadata) 