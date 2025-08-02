#!/usr/bin/env python3
"""
Test script for sentence transformer integration
"""

import asyncio
import time
from main import app
from fastapi.testclient import TestClient

def test_sentence_transformer_setup():
    """Test that sentence transformers are properly initialized"""
    try:
        from utils.embed_utils import model, DIMENSION
        print(f"✅ Sentence transformer model loaded successfully")
        print(f"✅ Model dimension: {DIMENSION}")
        return True
    except Exception as e:
        print(f"❌ Error loading sentence transformer: {e}")
        return False

def test_faiss_integration():
    """Test FAISS integration with sentence transformers"""
    try:
        from utils.embed_utils import vector_store, get_embedding
        print(f"✅ FAISS vector store initialized")
        
        # Test embedding generation
        test_text = "This is a test document for insurance coverage."
        embedding = get_embedding(test_text)
        print(f"✅ Embedding generated successfully, shape: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing FAISS integration: {e}")
        return False

def test_api_endpoint():
    """Test the API endpoint with sentence transformers"""
    try:
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Error testing API endpoint: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Sentence Transformer Integration")
    print("=" * 50)
    
    tests = [
        ("Sentence Transformer Setup", test_sentence_transformer_setup),
        ("FAISS Integration", test_faiss_integration),
        ("API Endpoint", test_api_endpoint),
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
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Sentence transformer integration is working.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 