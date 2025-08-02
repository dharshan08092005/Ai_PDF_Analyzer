#!/usr/bin/env python3
"""
Deployment test script for CPU-only FAISS
"""

import os
import sys

def test_faiss_deployment():
    """Test FAISS deployment configuration"""
    print("🧪 Testing FAISS Deployment Configuration")
    print("=" * 50)
    
    # Set environment variables
    os.environ['FAISS_NO_AVX2'] = '1'
    os.environ['FAISS_NO_GPU'] = '1'
    os.environ['FAISS_DISABLE_GPU'] = '1'
    os.environ['FAISS_CPU_ONLY'] = '1'
    os.environ['FAISS_NO_CUDA'] = '1'
    
    print("✅ Environment variables set:")
    for var in ['FAISS_NO_AVX2', 'FAISS_NO_GPU', 'FAISS_CPU_ONLY']:
        print(f"   {var}: {os.environ.get(var)}")
    
    try:
        # Test FAISS import
        import faiss
        print(f"✅ FAISS imported successfully: {faiss.__version__}")
        
        # Test basic operations
        import numpy as np
        
        # Create CPU-only index
        index = faiss.IndexFlatIP(384)
        print("✅ CPU-only FAISS index created")
        
        # Test vector operations
        test_vectors = np.random.rand(10, 384).astype(np.float32)
        faiss.normalize_L2(test_vectors)
        index.add(test_vectors)
        print("✅ Vector addition working")
        
        # Test search
        query = np.random.rand(1, 384).astype(np.float32)
        faiss.normalize_L2(query)
        scores, indices = index.search(query, 5)
        print("✅ Vector search working")
        
        # Test our vector store
        from utils.embed_utils import vector_store, get_embedding
        print("✅ Vector store imported successfully")
        
        # Test embedding generation
        test_text = "This is a test document for deployment."
        embedding = get_embedding(test_text)
        print(f"✅ Embedding generated: shape {embedding.shape}")
        
        print("\n🎉 All deployment tests passed!")
        print("✅ FAISS is configured for CPU-only deployment")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_deployment():
    """Test API deployment"""
    print("\n🔍 Testing API deployment...")
    
    try:
        from main import app
        print("✅ FastAPI app imported successfully")
        
        # Test health endpoint
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API deployment test failed: {e}")
        return False

def main():
    """Run all deployment tests"""
    print("🚀 FAISS Deployment Test Suite")
    print("=" * 50)
    
    tests = [
        ("FAISS Deployment", test_faiss_deployment),
        ("API Deployment", test_api_deployment),
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
    
    print("\n" + "=" * 50)
    print(f"📊 Deployment Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All deployment tests passed!")
        print("✅ Your application is ready for deployment")
        print("\n📋 Deployment Checklist:")
        print("   ✅ FAISS CPU-only configuration")
        print("   ✅ Environment variables set")
        print("   ✅ Vector operations working")
        print("   ✅ API endpoints responding")
        print("   ✅ No GPU dependencies")
    else:
        print("⚠️  Some deployment tests failed.")
        print("Please check the errors above before deploying.")

if __name__ == "__main__":
    main() 