#!/usr/bin/env python3
"""
Deployment configuration for CPU-only FAISS
"""

import os
import sys

def configure_faiss_cpu_only():
    """Configure environment for CPU-only FAISS deployment"""
    
    # Set environment variables to disable GPU and AVX2
    os.environ['FAISS_NO_AVX2'] = '1'
    os.environ['FAISS_NO_GPU'] = '1'
    os.environ['FAISS_DISABLE_GPU'] = '1'
    
    # Additional FAISS CPU-only settings
    os.environ['FAISS_CPU_ONLY'] = '1'
    os.environ['FAISS_NO_CUDA'] = '1'
    
    print("✅ FAISS CPU-only configuration applied")
    print(f"   FAISS_NO_AVX2: {os.environ.get('FAISS_NO_AVX2')}")
    print(f"   FAISS_NO_GPU: {os.environ.get('FAISS_NO_GPU')}")
    print(f"   FAISS_CPU_ONLY: {os.environ.get('FAISS_CPU_ONLY')}")

def verify_faiss_import():
    """Verify FAISS imports correctly in CPU-only mode"""
    try:
        # Import FAISS after setting environment variables
        import faiss
        print(f"✅ FAISS imported successfully: {faiss.__version__}")
        
        # Test basic CPU-only operations
        import numpy as np
        index = faiss.IndexFlatIP(384)
        test_vectors = np.random.rand(5, 384).astype(np.float32)
        faiss.normalize_L2(test_vectors)
        index.add(test_vectors)
        
        print("✅ FAISS CPU-only operations working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAISS import error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuring FAISS for CPU-only deployment...")
    configure_faiss_cpu_only()
    verify_faiss_import() 