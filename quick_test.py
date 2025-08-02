#!/usr/bin/env python3
"""
Quick test script to verify optimizations work
"""
import asyncio
import time
import httpx
import os

# Test configuration
API_BASE_URL = "http://localhost:8000/api/v1"
AUTH_TOKEN = os.getenv("AUTHORIZE_TOKEN", "your-auth-token")

# Simple test document
TEST_DOCUMENT_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
TEST_QUESTIONS = [
    "What is this document about?",
    "What are the main features?"
]

async def quick_test():
    """Quick performance test"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        
        print("🚀 Quick Performance Test")
        print("=" * 40)
        
        # Test document processing
        print("\n📄 Testing document processing...")
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
            print(f"✅ Total time: {total_time:.2f}s")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                reported_time = data.get('processing_time', 0)
                print(f"   Reported time: {reported_time:.2f}s")
                
                if total_time < 30:
                    print("🎉 SUCCESS: Under 30 seconds!")
                elif total_time < 60:
                    print("✅ ACCEPTABLE: Under 60 seconds")
                else:
                    print("⚠️  SLOW: Over 60 seconds")
                    
                # Show answers
                answers = data.get('answers', [])
                if answers:
                    print(f"\n📝 Sample answer: {answers[0][:100]}...")
                    
            else:
                print(f"❌ Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("Quick Optimization Test")
    print("=" * 40)
    
    # Check if server is running
    try:
        import requests
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server not responding")
            exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please start the server first with: python start_server.py")
        exit(1)
    
    # Run test
    asyncio.run(quick_test()) 