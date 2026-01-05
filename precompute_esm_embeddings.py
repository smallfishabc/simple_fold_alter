#!/usr/bin/env python3
"""
Utility script to pre-compute ESM2 embeddings for proteins.
This allows you to compute embeddings once and reuse them for multiple inference runs.

Usage:
    python precompute_esm_embeddings.py \
        --fasta_path input.fasta \
        --output_dir ./esm_embeddings \
        --esm_model esm2_3B

The script will save embeddings as .pt files that can be loaded during inference.
"""

import argparse
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simplefold.utils.esm_utils import esm_registry, _af2_to_esm
from simplefold.utils.fasta_utils import process_fastas, download_fasta_utilities, check_fasta_inputs
from simplefold.boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from simplefold.boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from simplefold.processor.protein_processor import ProteinDataProcessor
from simplefold.utils.datamodule_utils import process_one_inference_structure


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute ESM2 embeddings for protein structures"
    )
    
    parser.add_argument(
        "--fasta_path",
        type=str,
        required=True,
        help="Path to input FASTA file or directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./esm_embeddings",
        help="Directory to save pre-computed embeddings (default: ./esm_embeddings)"
    )
    parser.add_argument(
        "--esm_model",
        type=str,
        default="esm2_3B",
        choices=["esm2_150M", "esm2_650M", "esm2_3B"],
        help="ESM2 model to use (default: esm2_3B)"
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="./temp_esm",
        help="Temporary directory for processing (default: ./temp_esm)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    cache = temp_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ESM2 Embedding Pre-computation")
    print("=" * 80)
    print(f"Input FASTA: {args.fasta_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"ESM model: {args.esm_model}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()
    
    # Initialize ESM model
    print(f"Loading {args.esm_model} model...")
    device = torch.device(args.device)
    esm_model, esm_dict = esm_registry[args.esm_model]()
    esm_model = esm_model.to(device)
    esm_model.eval()
    af2_to_esm = _af2_to_esm(esm_dict).to(device)
    print(f"Model loaded on {device}")
    print()
    
    # Initialize tokenizer, featurizer, and processor
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    processor = ProteinDataProcessor(
        device=device,
        scale=16.0,
        ref_scale=5.0,
        multiplicity=1,
        inference_multiplicity=1,
        backend="torch",
    )
    
    # Process FASTA files
    print("Processing FASTA files...")
    download_fasta_utilities(cache)
    data = check_fasta_inputs(Path(args.fasta_path))
    if not data:
        raise ValueError("No valid input files found.")
    
    process_fastas(
        data=data,
        out_dir=temp_dir,
        ccd_path=cache / "ccd.pkl",
    )
    print()
    
    # Process each structure and compute embeddings
    struct_files = list((temp_dir / "structures").glob("*.npz"))
    print(f"Found {len(struct_files)} structures to process")
    print()
    
    for idx, struct_file in enumerate(struct_files, 1):
        record_file = temp_dir / "records" / f"{struct_file.stem}.json"
        
        print(f"[{idx}/{len(struct_files)}] Processing {struct_file.stem}...")
        
        # Prepare the target protein data
        batch, structure, record = process_one_inference_structure(
            struct_file, record_file,
            tokenizer, featurizer, processor,
            esm_model, esm_dict, af2_to_esm,
        )
        
        # Extract ESM embeddings
        esm_s = batch['esm_s'].cpu()
        
        # Save embeddings
        output_path = output_dir / f"{struct_file.stem}_esm.pt"
        torch.save(esm_s, output_path)
        print(f"  Saved embeddings to {output_path}")
        print(f"  Embedding shape: {esm_s.shape}")
        print()
    
    print("=" * 80)
    print("Pre-computation completed successfully!")
    print(f"Embeddings saved to: {args.output_dir}")
    print(f"Total files processed: {len(struct_files)}")
    print()
    print("To use these embeddings during inference, run:")
    print(f"  python run_inference_amd.py \\")
    print(f"      --fasta_path {args.fasta_path} \\")
    print(f"      --use_precomputed_esm \\")
    print(f"      --esm_embed_dir {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
