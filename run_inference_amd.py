#!/usr/bin/env python3
"""
Enhanced inference script for AMD GPU with batch processing and pre-computed ESM embeddings.

Usage examples:

1. Basic inference with AMD GPU (ROCm):
   python run_inference_amd.py --fasta_path input.fasta --output_dir results

2. With pre-computed ESM embeddings:
   python run_inference_amd.py --fasta_path input.fasta --output_dir results \
       --use_precomputed_esm --esm_embed_dir ./esm_embeddings

3. With batch processing:
   python run_inference_amd.py --fasta_path input_dir/ --output_dir results \
       --batch_size 4

4. Full example with all options:
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
       --plddt
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simplefold.inference import predict_structures_from_fastas


def parse_args():
    parser = argparse.ArgumentParser(
        description="SimpleFold inference with AMD GPU support, batch processing, and pre-computed ESM embeddings"
    )
    
    # Required arguments
    parser.add_argument(
        "--fasta_path",
        type=str,
        required=True,
        help="Path to input FASTA file or directory containing FASTA files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save predictions (default: ./results)"
    )
    
    # Model arguments
    parser.add_argument(
        "--simplefold_model",
        type=str,
        default="simplefold_1.1B",
        choices=[
            "simplefold_100M",
            "simplefold_360M",
            "simplefold_700M",
            "simplefold_1.1B",
            "simplefold_1.6B",
            "simplefold_3B",
        ],
        help="SimpleFold model to use (default: simplefold_1.1B)"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./artifacts",
        help="Directory to store/load model checkpoints (default: ./artifacts)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["torch", "mlx"],
        help="Backend to use: torch (for CUDA/ROCm) or mlx (for Apple Silicon) (default: torch)"
    )
    
    # ESM embedding arguments
    parser.add_argument(
        "--use_precomputed_esm",
        action="store_true",
        help="Use pre-computed ESM embeddings instead of computing them on-the-fly"
    )
    parser.add_argument(
        "--esm_embed_dir",
        type=str,
        default=None,
        help="Directory containing pre-computed ESM embeddings (*.pt files)"
    )
    
    # Batch processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of structures to process in each batch (default: 1)"
    )
    
    # Sampling arguments
    parser.add_argument(
        "--nsample_per_protein",
        type=int,
        default=5,
        help="Number of samples to generate per protein (default: 5)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of diffusion steps (default: 50)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature parameter for sampling (default: 1.0)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_format",
        type=str,
        default="cif",
        choices=["cif", "pdb"],
        help="Output structure format (default: cif)"
    )
    parser.add_argument(
        "--plddt",
        action="store_true",
        help="Compute pLDDT confidence scores"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate arguments
    if args.use_precomputed_esm and not args.esm_embed_dir:
        print("Warning: --use_precomputed_esm is set but --esm_embed_dir is not provided.")
        print("Will fall back to computing ESM embeddings on-the-fly if embedding files are not found.")
    
    # Display configuration
    print("=" * 80)
    print("SimpleFold Inference Configuration")
    print("=" * 80)
    print(f"Input FASTA: {args.fasta_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.simplefold_model}")
    print(f"Backend: {args.backend}")
    print(f"Checkpoint directory: {args.ckpt_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Use pre-computed ESM: {args.use_precomputed_esm}")
    if args.use_precomputed_esm:
        print(f"ESM embedding directory: {args.esm_embed_dir}")
    print(f"Samples per protein: {args.nsample_per_protein}")
    print(f"Diffusion steps: {args.num_steps}")
    print(f"Compute pLDDT: {args.plddt}")
    print(f"Output format: {args.output_format}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    print()
    
    # Check for AMD GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA/ROCm device detected: {torch.cuda.get_device_name(0)}")
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
        else:
            print("No CUDA/ROCm device detected. Running on CPU.")
    except Exception as e:
        print(f"Warning: Could not check GPU availability: {e}")
    
    print()
    
    # Run inference
    try:
        predict_structures_from_fastas(args)
        print("\n" + "=" * 80)
        print("Inference completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
