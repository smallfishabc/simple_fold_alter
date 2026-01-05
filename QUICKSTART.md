# SimpleFold AMD GPU Quick Start

## 🚀 Quick Commands

### 1. Simple Inference (downloads weights once, reuses them)
```bash
python run_inference_amd.py \
    --fasta_path input.fasta \
    --output_dir results
```

### 2. Pre-compute ESM Embeddings (one-time, saves time on repeated runs)
```bash
python precompute_esm_embeddings.py \
    --fasta_path input.fasta \
    --output_dir esm_embeddings
```

### 3. Inference with Pre-computed Embeddings (fastest)
```bash
python run_inference_amd.py \
    --fasta_path input.fasta \
    --output_dir results \
    --use_precomputed_esm \
    --esm_embed_dir esm_embeddings
```

### 4. Batch Processing Multiple Proteins
```bash
python run_inference_amd.py \
    --fasta_path proteins_dir/ \
    --output_dir results \
    --batch_size 4 \
    --use_precomputed_esm \
    --esm_embed_dir esm_embeddings
```

## 📋 Key Features

✅ **AMD GPU Support**: Works with AMD ROCm out-of-the-box (no code changes needed)  
✅ **No Repeated Downloads**: Checkpoints cached after first download  
✅ **Pre-computed Embeddings**: Compute ESM2-3B embeddings once, reuse forever  
✅ **Batch Processing**: Process multiple structures efficiently  
✅ **Minimal Changes**: Only inference code modified, training untouched  

## 🧪 Test Your GPU

```bash
# Test ROCm/CUDA compatibility
python test_rocm_compatibility.py

# Test all dependencies (Lightning, Hydra, Boltz)
python test_all_dependencies.py
```  

## 🔧 Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--fasta_path` | Input FASTA file/directory | **required** |
| `--output_dir` | Output directory | `./results` |
| `--simplefold_model` | Model size (100M/360M/700M/1.1B/1.6B/3B) | `simplefold_1.1B` |
| `--ckpt_dir` | Checkpoint storage directory | `./artifacts` |
| `--batch_size` | Batch size for processing | `1` |
| `--use_precomputed_esm` | Use pre-computed embeddings | `False` |
| `--esm_embed_dir` | Pre-computed embedding directory | `None` |
| `--nsample_per_protein` | Samples to generate | `5` |
| `--num_steps` | Diffusion steps | `50` |
| `--plddt` | Compute confidence scores | `False` |

## 💾 Disk Space Requirements

| Model | Checkpoint Size |
|-------|----------------|
| simplefold_100M | ~400 MB |
| simplefold_360M | ~1.4 GB |
| simplefold_700M | ~2.8 GB |
| simplefold_1.1B | ~4.4 GB |
| simplefold_1.6B | ~6.4 GB |
| simplefold_3B | ~12 GB |

**Note**: Checkpoints downloaded once to `--ckpt_dir` and reused.

## 🐛 Quick Fixes

**Out of Memory?**
```bash
# Use smaller model or reduce batch size
python run_inference_amd.py ... --simplefold_model simplefold_100M --batch_size 1
```

**ESM embeddings not found?**
```bash
# Pre-compute them first
python precompute_esm_embeddings.py --fasta_path input.fasta --output_dir esm_embeddings
```

**Check AMD GPU status:**
```bash
rocm-smi
python test_rocm_compatibility.py
```

## 📊 Performance Tips

1. **First-time users**: Start with default settings
2. **Repeated inference**: Pre-compute ESM embeddings
3. **Large datasets**: Use batch processing with `--batch_size 4`
4. **Limited memory**: Use `simplefold_100M` or `simplefold_360M`
5. **Best quality**: Use `simplefold_3B` with `--plddt`
