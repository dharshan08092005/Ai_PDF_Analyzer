#!/usr/bin/env python3
"""
Test script for PDF Bot API deployment
"""

import requests
import json
import time

def test_api_endpoints(base_url="http://localhost:7860"):
    """Test all API endpoints"""
    
    print("🧪 Testing PDF Bot API endpoints...")
    print(f"Base URL: {base_url}")
    print("-" * 50)
    
    # Test 1: Root endpoint (no auth required)
    print("1. Testing root endpoint (/)...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint working!")
            print(f"   Message: {data.get('message', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Version: {data.get('version', 'N/A')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {str(e)}")
    
    print()
    
    # Test 2: Health endpoint (no auth required)
    print("2. Testing health endpoint (/health)...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint working!")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Cache sizes: {data.get('cache_sizes', 'N/A')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {str(e)}")
    
    print()
    
    # Test 3: Main API endpoint (requires auth)
    print("3. Testing main API endpoint (/hackrx/run)...")
    print("   Note: This requires authentication token")
    
    # You can uncomment and modify this section to test with real credentials
    """
    auth_token = "your_auth_token_here"
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    test_data = {
        "documents": "https://example.com/sample.pdf",
        "questions": ["What is this document about?"]
    }
    
    try:
        response = requests.post(
            f"{base_url}/hackrx/run",
            json=test_data,
            headers=headers
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Main API endpoint working!")
            print(f"   Answers: {data.get('answers', 'N/A')}")
        else:
            print(f"❌ Main API endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Main API endpoint error: {str(e)}")
    """
    
    print("   ⚠️  Skipped (requires valid auth token)")
    
    print()
    print("🎉 API testing completed!")
    print("\n📋 Next steps for deployment:")
    print("1. Set your GEMINI_API_KEY environment variable")
    print("2. Set your AUTHORIZE_TOKEN environment variable")
    print("3. Deploy to Hugging Face using the deploy script")
    print("4. Test with real PDF documents and questions")

if __name__ == "__main__":
    # Test local deployment
    test_api_endpoints()
    
    # Uncomment to test Hugging Face deployment
    # test_api_endpoints("https://your-username-your-space-name.hf.space") 