#!/usr/bin/env python3
"""
Performance test script for the optimized PDF Bot API
"""
import asyncio
import time
import httpx
import json
import os
from pathlib import Path

# Test configuration
API_BASE_URL = "http://localhost:8000/api/v1"
AUTH_TOKEN = os.getenv("AUTHORIZE_TOKEN", "your-auth-token")

# Test data
TEST_DOCUMENT_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
TEST_QUESTIONS = [
    "What is the main topic of this document?",
    "What are the key features mentioned?",
    "What is the document about?",
    "What are the main points?"
]

async def test_performance():
    """Test the API performance"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        
        print("🚀 Starting Performance Test...")
        print("=" * 50)
        
        # Test 1: Health check
        print("\n1. Testing Health Check...")
        start_time = time.time()
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            health_time = time.time() - start_time
            print(f"✅ Health check: {health_time:.3f}s")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Cache sizes: {data.get('cache_sizes', {})}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return
        
        # Test 2: Document processing and Q&A
        print("\n2. Testing Document Processing and Q&A...")
        start_time = time.time()
        
        payload = {
            "documents": TEST_DOCUMENT_URL,
            "questions": TEST_QUESTIONS
        }
        
        try:
            response = await client.post(
                f"{API_BASE_URL}/hackrx/run",
                headers=headers,
                json=payload
            )
            
            total_time = time.time() - start_time
            print(f"✅ Total processing time: {total_time:.3f}s")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Processing time reported: {data.get('processing_time', 'N/A')}s")
                print(f"   Document hash: {data.get('document_hash', 'N/A')}")
                print(f"   Number of answers: {len(data.get('answers', []))}")
                
                # Show first answer as sample
                answers = data.get('answers', [])
                if answers:
                    print(f"   Sample answer: {answers[0][:100]}...")
                
                # Performance analysis
                if total_time < 30:
                    print("🎉 EXCELLENT: Response time under 30 seconds!")
                elif total_time < 60:
                    print("✅ GOOD: Response time under 60 seconds")
                else:
                    print("⚠️  SLOW: Response time over 60 seconds")
                    
            else:
                print(f"❌ Request failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        # Test 3: Cache performance (second request should be faster)
        print("\n3. Testing Cache Performance...")
        start_time = time.time()
        
        try:
            response = await client.post(
                f"{API_BASE_URL}/hackrx/run",
                headers=headers,
                json=payload
            )
            
            cache_time = time.time() - start_time
            print(f"✅ Cached request time: {cache_time:.3f}s")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                reported_time = data.get('processing_time', 0)
                print(f"   Reported processing time: {reported_time:.3f}s")
                
                # Calculate cache improvement
                if 'total_time' in locals():
                    improvement = ((total_time - cache_time) / total_time) * 100
                    print(f"   Cache improvement: {improvement:.1f}%")
                    
        except Exception as e:
            print(f"❌ Cache test failed: {e}")
        
        print("\n" + "=" * 50)
        print("🏁 Performance test completed!")

async def test_concurrent_requests():
    """Test concurrent request handling"""
    print("\n🔄 Testing Concurrent Requests...")
    
    async def make_request(client, request_id):
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        payload = {
            "documents": TEST_DOCUMENT_URL,
            "questions": [f"Test question {request_id}"]
        }
        
        start_time = time.time()
        try:
            response = await client.post(
                f"{API_BASE_URL}/hackrx/run",
                headers=headers,
                json=payload
            )
            duration = time.time() - start_time
            return f"Request {request_id}: {duration:.3f}s (Status: {response.status_code})"
        except Exception as e:
            duration = time.time() - start_time
            return f"Request {request_id}: {duration:.3f}s (Error: {e})"
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Test 3 concurrent requests
        tasks = [make_request(client, i) for i in range(1, 4)]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"Concurrent requests completed in: {total_time:.3f}s")
        for result in results:
            print(f"  {result}")

if __name__ == "__main__":
    print("PDF Bot API Performance Test")
    print("=" * 50)
    
    # Check if server is running
    try:
        import requests
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server is not responding properly")
            exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please start the server first with: python main.py")
        exit(1)
    
    # Run tests
    asyncio.run(test_performance())
    asyncio.run(test_concurrent_requests()) 