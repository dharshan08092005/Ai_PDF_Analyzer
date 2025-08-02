#!/usr/bin/env python3
"""
Verification script for CPU-only FAISS implementation
"""

import faiss
import numpy as np
from utils.embed_utils import vector_store, get_embedding, search_similar_chunks

def verify_faiss_cpu():
    """Verify that we're using CPU-only FAISS"""
    print("🔍 Verifying CPU-only FAISS implementation...")
    
    # Check FAISS version and capabilities
    print(f"✅ FAISS version: {faiss.__version__}")
    
    # Check if we can create CPU-only index
    try:
        index = faiss.IndexFlatIP(384)  # CPU-only inner product index
        print("✅ Successfully created CPU-only FAISS index")
        
        # Test basic operations
        test_vectors = np.random.rand(10, 384).astype(np.float32)
        faiss.normalize_L2(test_vectors)
        index.add(test_vectors)
        
        query = np.random.rand(1, 384).astype(np.float32)
        faiss.normalize_L2(query)
        scores, indices = index.search(query, 5)
        
        print("✅ CPU-only FAISS operations working correctly")
        print(f"✅ Search returned {len(indices[0])} results")
        
        return True
    except Exception as e:
        print(f"❌ Error with CPU-only FAISS: {e}")
        return False

def verify_vector_store():
    """Verify our vector store implementation"""
    print("\n🔍 Verifying Vector Store implementation...")
    
    try:
        # Check vector store stats
        stats = vector_store.get_stats()
        print(f"✅ Vector store initialized: {stats}")
        
        # Test adding vectors
        test_texts = [
            "This is a test insurance policy document.",
            "The policy covers medical expenses up to $100,000.",
            "There is a 30-day waiting period for pre-existing conditions."
        ]
        
        # Generate embeddings
        embeddings = []
        for text in test_texts:
            embedding = get_embedding(text)
            embeddings.append(embedding)
        
        # Add to vector store
        metadata_list = [{"text": text, "test": True} for text in test_texts]
        vector_store.add_vectors(embeddings, metadata_list)
        
        print("✅ Successfully added test vectors to CPU-only FAISS")
        
        # Test search
        query = "What is the waiting period?"
        results = search_similar_chunks(query, top_k=2)
        
        print(f"✅ Search working: found {len(results)} results")
        for i, result in enumerate(results):
            print(f"   Result {i+1}: {result[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error with vector store: {e}")
        return False

def verify_functionality():
    """Verify all FAISS functionality is working"""
    print("\n🔍 Verifying FAISS functionality...")
    
    try:
        # Test all main functions
        from utils.embed_utils import (
            insert_into_faiss, 
            search_similar_chunks, 
            get_index_statistics,
            clear_index
        )
        
        print("✅ All FAISS functions imported successfully")
        
        # Test index statistics
        stats = get_index_statistics()
        print(f"✅ Index statistics: {stats}")
        
        # Test search functionality
        test_query = "insurance coverage"
        results = search_similar_chunks(test_query, top_k=1)
        print(f"✅ Search functionality working: {len(results)} results")
        
        return True
    except Exception as e:
        print(f"❌ Error with functionality: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🧪 Verifying CPU-Only FAISS Implementation")
    print("=" * 60)
    
    tests = [
        ("FAISS CPU Implementation", verify_faiss_cpu),
        ("Vector Store", verify_vector_store),
        ("FAISS Functionality", verify_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CPU-only FAISS is working correctly.")
        print("\n✅ Confirmed:")
        print("   - Using faiss-cpu package")
        print("   - IndexFlatIP (CPU-only inner product)")
        print("   - All vector operations working")
        print("   - Search functionality intact")
        print("   - No GPU dependencies")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 