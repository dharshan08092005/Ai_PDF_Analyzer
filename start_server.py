#!/usr/bin/env python3
"""
Optimized PDF Bot API Server Startup Script
"""
import os
import sys
import time
import logging
from pathlib import Path

# Configure FAISS for CPU-only deployment
os.environ['FAISS_NO_AVX2'] = '1'
os.environ['FAISS_NO_GPU'] = '1'
os.environ['FAISS_DISABLE_GPU'] = '1'
os.environ['FAISS_CPU_ONLY'] = '1'
os.environ['FAISS_NO_CUDA'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ['GEMINI_API_KEY', 'AUTHORIZE_TOKEN']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Please set them using:")
        for var in missing_vars:
            logger.info(f"  export {var}=your-value")
        return False
    
    logger.info("✅ Environment variables are set")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import faiss
        import google.generativeai
        import httpx
        import cachetools
        import joblib
        logger.info("✅ All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_faiss_index():
    """Check FAISS index status"""
    faiss_index_path = Path("faiss_index")
    faiss_metadata_path = Path("faiss_metadata.pkl")
    
    if faiss_index_path.exists() and faiss_metadata_path.exists():
        logger.info("✅ FAISS index found")
        return True
    else:
        logger.info("ℹ️  FAISS index will be created on first use")
        return True

def print_performance_info():
    """Print performance optimization information"""
    print("\n" + "="*60)
    print("🚀 OPTIMIZED PDF BOT API SERVER")
    print("="*60)
    print("Performance Optimizations:")
    print("  • FAISS vector database (local, fast)")
    print("  • Enhanced caching system")
    print("  • Optimized thread pools")
    print("  • HTTP/2 enabled")
    print("  • Timeout protection")
    print("  • Simplified AI prompts")
    print("\nExpected Performance:")
    print("  • First request: 15-30 seconds")
    print("  • Cached request: 2-5 seconds")
    print("  • Vector search: <0.1 seconds")
    print("="*60)

def start_server():
    """Start the optimized server"""
    try:
        import uvicorn
        from main import app
        
        print_performance_info()
        
        # Start server with optimized settings
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=1,  # Single worker for better caching
            log_level="info",
            access_log=True,
            loop="asyncio"
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    logger.info("Starting PDF Bot API Server...")
    
    # Pre-flight checks
    if not check_environment():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    check_faiss_index()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main() 