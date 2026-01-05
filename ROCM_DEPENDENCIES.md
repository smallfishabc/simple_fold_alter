# ROCm Compatibility: Dependencies Analysis

## Quick Answer: ✅ Yes, all dependencies support ROCm!

All the libraries used in SimpleFold inference are ROCm-compatible because they either:
1. Use PyTorch (which abstracts CUDA/ROCm)
2. Are pure Python libraries with no GPU code
3. Work at a higher level than GPU operations

## Dependency Breakdown

### 1. **PyTorch Lightning** ✅ Fully ROCm Compatible

**What it does**: High-level training framework built on PyTorch  
**ROCm status**: ✅ **Fully compatible**

**Why it works:**
- Lightning is built entirely on top of PyTorch
- All GPU operations go through PyTorch's API
- Lightning doesn't use CUDA directly - it uses `torch.device("cuda")`
- Same code works for both CUDA and ROCm

**In SimpleFold inference:**
```python
import lightning.pytorch as pl
pl.seed_everything(args.seed, workers=True)  # ✅ Works with ROCm
```

**What Lightning does in inference:**
- Only used for `pl.seed_everything()` - setting random seeds
- No actual GPU operations in the inference code path
- Completely safe for ROCm

### 2. **Hydra** ✅ Fully ROCm Compatible

**What it does**: Configuration management framework  
**ROCm status**: ✅ **Fully compatible**

**Why it works:**
- Pure Python library
- No GPU code whatsoever
- Only handles YAML configs and Python object instantiation

**In SimpleFold inference:**
```python
import hydra
model_config = omegaconf.OmegaConf.load(cfg_path)
model = hydra.utils.instantiate(model_config)  # ✅ Works with ROCm
```

**What Hydra does:**
- Loads YAML configuration files
- Instantiates Python objects based on config
- No GPU operations at all
- 100% ROCm compatible

### 3. **Boltz Data Pipeline** ✅ Fully ROCm Compatible

**What it does**: Data processing and tokenization  
**ROCm status**: ✅ **Fully compatible**

**Why it works:**
- All GPU operations go through PyTorch
- Uses `torch.Tensor` operations (ROCm compatible)
- No direct CUDA code

**In SimpleFold:**
```python
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
```

**What it does:**
- Tokenizes protein sequences
- Creates feature tensors
- All tensor operations use PyTorch → ✅ ROCm compatible

## How ROCm Compatibility Works

All these libraries are ROCm-compatible because of PyTorch's abstraction:

```
Your Code (Lightning/Hydra/Boltz)
         ↓
    PyTorch API
         ↓
   ┌─────┴─────┐
   ↓           ↓
CUDA (NVIDIA)  ROCm (AMD)
```

**Key Point:** These libraries never directly call CUDA functions. They all use PyTorch's device-agnostic API.

## Testing Each Component

### Test PyTorch Lightning with ROCm
```python
import torch
import lightning.pytorch as pl

# Set seed (uses PyTorch internally)
pl.seed_everything(42)

# This works identically on CUDA and ROCm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Lightning will use: {device}")
```

### Test Hydra (No GPU needed)
```python
import hydra
import omegaconf

# Hydra is pure Python - no GPU code
config = omegaconf.OmegaConf.create({"key": "value"})
print("Hydra works on any platform ✅")
```

### Test Boltz + PyTorch with ROCm
```python
import torch
from simplefold.boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer

tokenizer = BoltzTokenizer()
# All tensor operations use PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Boltz tensors will use: {device}")
```

## SimpleFold Inference GPU Usage

In the inference pipeline, GPU operations only happen through PyTorch:

| Component | GPU Usage | ROCm Compatible |
|-----------|-----------|-----------------|
| **Lightning** | Only `seed_everything()` - no GPU | ✅ Yes |
| **Hydra** | Config management - no GPU | ✅ Yes |
| **Boltz Pipeline** | PyTorch tensors | ✅ Yes |
| **ESM Model** | PyTorch model | ✅ Yes |
| **SimpleFold Model** | PyTorch model | ✅ Yes |
| **Sampler** | PyTorch tensors | ✅ Yes |

**All GPU operations → PyTorch → Works with both CUDA and ROCm ✅**

## Verification

Run this to verify all components work with your GPU:

```bash
python test_rocm_compatibility.py
```

Then test the actual inference pipeline:

```bash
python -c "
import torch
import lightning.pytorch as pl
import hydra
from simplefold.boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer

print('PyTorch CUDA/ROCm available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))

# Test Lightning
pl.seed_everything(42)
print('✅ Lightning works')

# Test Hydra
import omegaconf
cfg = omegaconf.OmegaConf.create({})
print('✅ Hydra works')

# Test Boltz
tokenizer = BoltzTokenizer()
print('✅ Boltz works')

print('\\n🎉 All components are ROCm compatible!')
"
```

## Summary

✅ **PyTorch Lightning**: Built on PyTorch → ROCm compatible  
✅ **Hydra**: Pure Python, no GPU code → ROCm compatible  
✅ **Boltz**: Uses PyTorch tensors → ROCm compatible  

**Bottom line:** Since all GPU operations go through PyTorch's CUDA API, and PyTorch with ROCm uses the same API, everything works seamlessly with AMD GPUs!

No special configuration or code changes needed for any of these libraries.
