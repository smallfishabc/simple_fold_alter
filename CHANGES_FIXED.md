# Changes Fixed - ESM Model and ROCm Compatibility

## Issues Addressed

### 1. ✅ Default ESM Model Changed to esm2_3B
**Previous**: ESM model was set to `esm2_150M` for lower memory usage  
**Fixed**: Changed back to `esm2_3B` (highest quality, as intended)

**Files Modified**:
- [src/simplefold/inference.py](src/simplefold/inference.py#L199) - `initialize_esm_model()`
  - Changed from `esm2_150M` to `esm2_3B`
  - Updated MLX specs: 36 layers, 2560 dim, 40 attention heads
  
- [src/simplefold/wrapper.py](src/simplefold/wrapper.py#L241) - `initialize_esm_model()`
  - Changed from `esm2_150M` to `esm2_3B`
  - Updated MLX specs: 36 layers, 2560 dim, 40 attention heads

- [precompute_esm_embeddings.py](precompute_esm_embeddings.py) 
  - Default changed from `esm2_150M` to `esm2_3B`

### 2. ✅ ROCm Compatibility Verified and Documented

**Status**: Code was already ROCm-compatible! PyTorch abstracts CUDA/ROCm differences.

**How it works**:
```python
# This code works for BOTH CUDA and ROCm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

PyTorch with ROCm uses the same CUDA API:
- `torch.cuda.is_available()` → Returns `True` for AMD ROCm GPUs
- `torch.device("cuda")` → Works for both NVIDIA and AMD
- All tensor operations → Identical API for both backends

**New Testing Tool**:
- Created [test_rocm_compatibility.py](test_rocm_compatibility.py) to verify GPU setup
- Tests GPU detection, memory, and basic operations
- Identifies whether ROCm or CUDA is being used

**Documentation Updated**:
- [AMD_INFERENCE_GUIDE.md](AMD_INFERENCE_GUIDE.md) - Clarified ROCm compatibility
- [QUICKSTART.md](QUICKSTART.md) - Added GPU testing instructions

## ESM Model Specifications

| Model | Layers | Embed Dim | Attention Heads | Parameters |
|-------|--------|-----------|-----------------|------------|
| esm2_150M | 30 | 640 | 20 | 150M |
| esm2_650M | 33 | 1280 | 20 | 650M |
| **esm2_3B** | **36** | **2560** | **40** | **3B** |

**Default is now esm2_3B** for best quality.

## Testing

### Test ROCm/CUDA Compatibility
```bash
python test_rocm_compatibility.py
```

Expected output for ROCm:
```
PyTorch version: 2.x.x+rocm5.7
CUDA/ROCm available: True
Number of GPUs: 1
GPU 0:
  Name: AMD Radeon RX 7900 XTX (or similar)
  ...
✅ ROCm-enabled PyTorch detected!
```

### Verify ESM Model Loading
```bash
# Quick check
python -c "from src.simplefold.utils.esm_utils import esm_registry; print('ESM models:', list(esm_registry.keys()))"
```

## Summary

1. ✅ **ESM Model**: Now defaults to `esm2_3B` (highest quality)
2. ✅ **ROCm Support**: Works out-of-the-box with PyTorch ROCm
3. ✅ **Testing**: New script to verify GPU compatibility
4. ✅ **Documentation**: Updated guides with ROCm details

No additional code changes needed - PyTorch handles CUDA/ROCm abstraction perfectly!
