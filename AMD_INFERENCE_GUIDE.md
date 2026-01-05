# SimpleFold AMD GPU Inference Guide

This document describes the modifications made to support AMD GPU (ROCm), batch inference, and pre-computed ESM embeddings for inference-only use cases.

## Changes Summary

### 1. AMD GPU (ROCm) Support
- **File**: `src/simplefold/inference.py`
- **Changes**: PyTorch's CUDA API works seamlessly with AMD ROCm
  - `torch.cuda.is_available()` returns `True` for both NVIDIA CUDA and AMD ROCm
  - `torch.device("cuda")` works for both GPU types
  - No special code changes needed - PyTorch abstracts the hardware differences
- **Testing**: Use `python test_rocm_compatibility.py` to verify your GPU setup

### 2. Checkpoint Download Prevention
- **Files**: 
  - `src/simplefold/inference.py` - `initialize_folding_model()`, `initialize_plddt_module()`
- **Changes**: 
  - Added existence checks before downloading checkpoints
  - Prints status messages indicating if using existing or downloading new checkpoints
  - Prevents repeated downloads on subsequent runs

### 3. Pre-computed ESM Embeddings Support
- **Files**:
  - `src/simplefold/inference.py` - `initialize_esm_model()`, `predict_structures_from_fastas()`
  - Default ESM model is **esm2_3B** (highest quality)
  - `src/simplefold/processor/protein_processor.py` - `preprocess_inference()`
- **Changes**:
  - Added `--use_precomputed_esm` flag to skip ESM model loading
  - Added `--esm_embed_dir` parameter to specify embedding directory
  - Loads embeddings from `.pt` files if available
  - Falls back to on-the-fly computation if embedding files not found

### 4. Batch Processing Support
- **File**: `src/simplefold/inference.py` - `predict_structures_from_fastas()`
- **Changes**:
  - Added `--batch_size` parameter to control batch processing
  - Structures are now collected and processed in configurable batches
  - Progress information shows batch number and files being processed

## Usage

### Basic Inference with AMD GPU

```bash
python run_inference_amd.py \
    --fasta_path input.fasta \
    --output_dir results \
    --simplefold_model simplefold_1.1B
```

### Pre-compute ESM Embeddings (One-time Setup)

This step is optional but recommended for repeated inference runs:

```bash
python precompute_esm_embeddings.py \
    --fasta_path input.fasta \
    --output_dir ./esm_embeddings \
    --esm_model esm2_150M
```

This creates `.pt` files containing ESM embeddings that can be reused.

### Inference with Pre-computed Embeddings

```bash
python run_inference_amd.py \
    --fasta_path input.fasta \
    --output_dir results \
    --use_precomputed_esm \
    --esm_embed_dir ./esm_embeddings \
    --simplefold_model simplefold_1.1B
```

### Batch Inference

Process multiple structures in batches (useful for large datasets):

```bash
python run_inference_amd.py \
    --fasta_path input_dir/ \
    --output_dir results \
    --batch_size 4 \
    --use_precomputed_esm \
    --esm_embed_dir ./esm_embeddings
```

### Full Example with All Options

```bash
python run_inference_amd.py \
    --fasta_path input_dir/ \
    --output_dir results \
    --simplefold_model simplefold_1.1B \
    --ckpt_dir ./checkpoints \
    --batch_size 4 \
    --use_precomputed_esm \
    --esm_embed_dir ./esm_embeddings \
    --nsample_per_protein 5 \
    --num_steps 50 \
    --plddt \
    --output_format cif \
    --seed 42
```

## New Command-Line Arguments

### `run_inference_amd.py`

- `--batch_size`: Number of structures to process in each batch (default: 1)
- `--use_precomputed_esm`: Use pre-computed ESM embeddings
- `--esm_embed_dir`: Directory containing pre-computed ESM embeddings (.pt files)
- All original arguments are preserved

### `precompute_esm_embeddings.py`

- `--fasta_path`: Path to input FASTA file or directory (required)
- `--output_dir`: Directory to save embeddings (default: ./esm_embeddings)
- `--esm_model`: ESM2 model to use (choices: esm2_150M, esm2_650M, esm2_3B)
- `--temp_dir`: Temporary directory for processing (default: ./temp_esm)
- `--device`: Device to use (default: cuda if available)

## Performance Tips

1. **Pre-compute ESM embeddings**: This is the most time-consuming step. Compute once, reuse many times.

2. **Use appropriate batch size**: 
   - Start with `--batch_size 1` for testing
   - Increase gradually based on GPU memory (e.g., 2, 4, 8)
   - Monitor GPU memory usage with `rocm-smi` (AMD) or `nvidia-smi` (NVIDIA)

3. **Model selection**:
   - `simplefold_100M`: Fastest, lowest memory
   - `simplefold_1.1B`: Good balance
   - `simplefold_3B`: Best quality, highest memory requirements

4. **Store checkpoints locally**: First run downloads checkpoints to `--ckpt_dir`. Subsequent runs reuse them automatically.

## AMD GPU Setup (ROCm)
### Verify ROCm Installation

```bash
# Check ROCm installation
rocm-smi

# Test GPU compatibility
python test_rocm_compatibility.py
```

The test script will show:
- PyTorch version and ROCm/CUDA detection
- GPU device information
- Memory availability
- Basic tensor operations

### Install PyTorch with ROCm

If PyTorch with ROCm support is not installed:

```bash
# For ROCm 5.7 (check your ROCm version with: rocm-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# For ROCm 6.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### Why PyTorch Works with Both CUDA and ROCm

PyTorch uses the same API for both NVIDIA CUDA and AMD ROCm:
- `torch.cuda.is_available()` - Detects both GPU types
- `torch.device("cuda")` - Works for both
- All CUDA operations have ROCm equivalents

This means SimpleFold code requires **zero modifications** for AMD GPUs! install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

## File Structure

After pre-computing embeddings:

```
project/
├── esm_embeddings/           # Pre-computed ESM embeddings
│   ├── protein1_esm.pt
│   ├── protein2_esm.pt
│   └── ...
├── checkpoints/              # Model checkpoints (downloaded once)
│   ├── simplefold_1.1B.ckpt
│   ├── plddt.ckpt
│   └── simplefold_1.6B.ckpt
├── results/                  # Inference outputs
│   └── predictions_simplefold_1.1B/
│       ├── protein1_sampled_0.cif
│       ├── protein1_sampled_1.cif
│       └── ...
└── input/                    # Input FASTA files
    ├── protein1.fasta
    ├── protein2.fasta
    └── ...
```

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `--batch_size` to 1
2. Use a smaller model (e.g., `simplefold_100M`)
3. Reduce `--nsample_per_protein`
4. Close other GPU applications

### Embeddings Not Found

If you see "Warning: ESM embedding file not found", either:
1. Run `precompute_esm_embeddings.py` first
2. Remove `--use_precomputed_esm` flag to compute on-the-fly
3. Check that embedding filenames match structure filenames

### Slow First Run

The first run downloads model checkpoints (~1-10 GB depending on model). Subsequent runs reuse cached checkpoints.

## Minimal Changes Philosophy

These modifications follow the "minimal changes for inference" principle:
- No training code modified
- Original functionality preserved
- New features are opt-in via command-line flags
- Backward compatible with existing workflows
