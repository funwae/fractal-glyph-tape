#!/usr/bin/env python3
"""Run compression and reconstruction experiments."""

import argparse
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import evaluate_compression_and_reconstruction


def run_experiment(
    original_file: Path,
    reconstructed_file: Path,
    fgt_sequences_bytes: int,
    fgt_tables_bytes: int,
    output_dir: Path,
    use_bertscore: bool = False,
):
    """
    Run compression experiment and save results.

    Args:
        original_file: Path to original corpus
        reconstructed_file: Path to reconstructed corpus
        fgt_sequences_bytes: Bytes used for FGT sequences
        fgt_tables_bytes: Bytes used for lookup tables
        output_dir: Directory to save results
        use_bertscore: Whether to compute BERTScore
    """
    print(f"\n{'='*60}")
    print(f"Running Compression Experiment")
    print(f"{'='*60}")
    print(f"Original file: {original_file}")
    print(f"Reconstructed file: {reconstructed_file}")
    print(f"FGT sequences bytes: {fgt_sequences_bytes:,}")
    print(f"FGT tables bytes: {fgt_tables_bytes:,}")
    print(f"{'='*60}\n")

    # Run evaluation
    results = evaluate_compression_and_reconstruction(
        original_file,
        reconstructed_file,
        fgt_sequences_bytes,
        fgt_tables_bytes,
        use_bertscore=use_bertscore,
    )

    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print("\nCompression Metrics:")
    print(f"  Dataset: {results['dataset_name']}")
    print(f"  Sentence count: {results['sentence_count']:,}")
    print(f"  Raw bytes: {results['compression']['raw_bytes']:,}")
    print(f"  FGT bytes (total): {results['compression']['fgt_bytes_total']:,}")
    print(f"  FGT bytes (sequences): {results['compression']['fgt_bytes_sequences']:,}")
    print(f"  FGT bytes (tables): {results['compression']['fgt_bytes_tables']:,}")
    print(f"  Compression ratio: {results['compression']['compression_ratio']:.3f}x")
    print(f"  Compression ratio (sequences only): {results['compression']['compression_ratio_sequences']:.3f}x")
    print(f"  Compression percentage: {results['compression']['compression_percentage']:.1f}%")

    print("\nReconstruction Metrics:")
    print(f"  BLEU: {results['reconstruction']['bleu']:.4f}")
    print(f"  ROUGE-1 F1: {results['reconstruction']['rouge1_f1']:.4f}")
    print(f"  ROUGE-2 F1: {results['reconstruction']['rouge2_f1']:.4f}")
    print(f"  ROUGE-L F1: {results['reconstruction']['rougeL_f1']:.4f}")

    if use_bertscore:
        print(f"  BERTScore F1: {results['reconstruction']['bertscore_f1']:.4f}")

    print("="*60 + "\n")

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{results['dataset_name']}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run compression and reconstruction experiments"
    )

    parser.add_argument(
        "original_file",
        type=Path,
        help="Path to original corpus file"
    )

    parser.add_argument(
        "reconstructed_file",
        type=Path,
        help="Path to reconstructed corpus file"
    )

    parser.add_argument(
        "--fgt-sequences-bytes",
        type=int,
        required=True,
        help="Bytes used for FGT sequences"
    )

    parser.add_argument(
        "--fgt-tables-bytes",
        type=int,
        required=True,
        help="Bytes used for lookup tables"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments"),
        help="Directory to save results (default: results/experiments)"
    )

    parser.add_argument(
        "--use-bertscore",
        action="store_true",
        help="Compute BERTScore (slow)"
    )

    args = parser.parse_args()

    # Run experiment
    run_experiment(
        args.original_file,
        args.reconstructed_file,
        args.fgt_sequences_bytes,
        args.fgt_tables_bytes,
        args.output_dir,
        args.use_bertscore,
    )


if __name__ == "__main__":
    main()
