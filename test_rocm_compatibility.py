#!/usr/bin/env python3
"""
Test script to verify ROCm/CUDA compatibility for SimpleFold.

This script checks:
1. PyTorch installation and GPU availability
2. CUDA/ROCm device detection
3. Basic tensor operations on GPU
"""

import torch
import sys

def test_rocm_cuda_compatibility():
    print("=" * 80)
    print("SimpleFold ROCm/CUDA Compatibility Test")
    print("=" * 80)
    print()
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Check CUDA/ROCm availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA/ROCm available: {cuda_available}")
    
    if not cuda_available:
        print()
        print("❌ No GPU detected!")
        print()
        print("If you have an AMD GPU with ROCm installed, ensure:")
        print("1. ROCm is properly installed: rocm-smi")
        print("2. PyTorch with ROCm support is installed:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7")
        print()
        sys.exit(1)
    
    print()
    
    # Get device information
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")
    print()
    
    for i in range(device_count):
        print(f"GPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        
        # Get memory info
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  Memory Allocated: {mem_allocated:.2f} GB")
        print(f"  Memory Reserved: {mem_reserved:.2f} GB")
        print()
    
    # Test basic tensor operations
    print("Testing basic GPU operations...")
    device = torch.device("cuda")
    
    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        z = torch.matmul(x, y)
        
        # Move to CPU
        z_cpu = z.cpu()
        
        print("✅ Basic GPU operations successful!")
        print(f"   Created tensors on {device}")
        print(f"   Performed matrix multiplication")
        print(f"   Result shape: {z.shape}")
        print()
        
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        sys.exit(1)
    
    # Check if this is ROCm or CUDA
    print("=" * 80)
    if "rocm" in torch.__version__.lower() or "hip" in str(torch.version.cuda).lower():
        print("✅ ROCm-enabled PyTorch detected!")
        print("   Your AMD GPU is ready for SimpleFold inference.")
    else:
        print("✅ CUDA-enabled PyTorch detected!")
        print("   Your NVIDIA GPU is ready for SimpleFold inference.")
    print("=" * 80)
    print()
    
    # Show device usage in code
    print("In SimpleFold code, GPU detection works automatically:")
    print('  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")')
    print(f'  # This resolves to: {device}')
    print()
    print("Both CUDA and ROCm use the same PyTorch API, so no code changes needed!")
    print()

if __name__ == "__main__":
    test_rocm_cuda_compatibility()
