#!/usr/bin/env python3
"""Generate plots and tables from experiment results."""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_experiment_results(results_dir: Path) -> List[Dict]:
    """
    Load all experiment results from directory.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        List of experiment results
    """
    results_dir = Path(results_dir)
    results = []

    for result_file in results_dir.glob("*_results.json"):
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            results.append(result)

    return results


def create_compression_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create compression metrics table.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with compression metrics
    """
    rows = []

    for result in results:
        comp = result['compression']
        row = {
            'Dataset': result['dataset_name'],
            'Sentences': result['sentence_count'],
            'Raw (MB)': comp['raw_bytes'] / (1024 * 1024),
            'FGT Total (MB)': comp['fgt_bytes_total'] / (1024 * 1024),
            'FGT Sequences (MB)': comp['fgt_bytes_sequences'] / (1024 * 1024),
            'Compression Ratio': comp['compression_ratio'],
            'Compression %': comp['compression_percentage'],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def create_reconstruction_table(results: List[Dict]) -> pd.DataFrame:
    """
    Create reconstruction quality table.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with reconstruction metrics
    """
    rows = []

    for result in results:
        recon = result['reconstruction']
        row = {
            'Dataset': result['dataset_name'],
            'BLEU': recon['bleu'],
            'ROUGE-1': recon['rouge1_f1'],
            'ROUGE-2': recon['rouge2_f1'],
            'ROUGE-L': recon['rougeL_f1'],
            'BERTScore': recon.get('bertscore_f1', 0.0),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_compression_ratios(results: List[Dict], output_file: Path):
    """
    Create bar plot of compression ratios.

    Args:
        results: List of experiment results
        output_file: Path to save plot
    """
    datasets = [r['dataset_name'] for r in results]
    ratios = [r['compression']['compression_ratio'] for r in results]
    ratios_seq = [r['compression']['compression_ratio_sequences'] for r in results]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, ratios, width, label='Total (incl. tables)', alpha=0.8)
    bars2 = ax.bar(x + width/2, ratios_seq, width, label='Sequences only', alpha=0.8)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Compression Ratios by Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}x',
                ha='center',
                va='bottom',
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_reconstruction_quality(results: List[Dict], output_file: Path):
    """
    Create grouped bar plot of reconstruction metrics.

    Args:
        results: List of experiment results
        output_file: Path to save plot
    """
    datasets = [r['dataset_name'] for r in results]
    metrics = {
        'BLEU': [r['reconstruction']['bleu'] for r in results],
        'ROUGE-1': [r['reconstruction']['rouge1_f1'] for r in results],
        'ROUGE-2': [r['reconstruction']['rouge2_f1'] for r in results],
        'ROUGE-L': [r['reconstruction']['rougeL_f1'] for r in results],
    }

    x = np.arange(len(datasets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = width * (i - len(metrics)/2 + 0.5)
        ax.bar(x + offset, values, width, label=metric_name, alpha=0.8)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Reconstruction Quality Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_compression_vs_quality(results: List[Dict], output_file: Path):
    """
    Create scatter plot of compression ratio vs reconstruction quality.

    Args:
        results: List of experiment results
        output_file: Path to save plot
    """
    compression_ratios = [r['compression']['compression_ratio'] for r in results]
    bleu_scores = [r['reconstruction']['bleu'] for r in results]
    datasets = [r['dataset_name'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        compression_ratios,
        bleu_scores,
        s=200,
        alpha=0.6,
        c=range(len(results)),
        cmap='viridis'
    )

    # Add labels for each point
    for i, dataset in enumerate(datasets):
        ax.annotate(
            dataset,
            (compression_ratios[i], bleu_scores[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )

    ax.set_xlabel('Compression Ratio', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title('Compression Ratio vs Reconstruction Quality', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def generate_all_outputs(results_dir: Path, output_dir: Path):
    """
    Generate all tables and plots from experiment results.

    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save outputs
    """
    # Load results
    results = load_experiment_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded {len(results)} experiment results")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tables
    print("\nGenerating tables...")

    compression_table = create_compression_table(results)
    compression_table_file = output_dir / "compression_metrics.csv"
    compression_table.to_csv(compression_table_file, index=False, float_format='%.4f')
    print(f"Saved table: {compression_table_file}")
    print("\nCompression Metrics:")
    print(compression_table.to_string(index=False))

    reconstruction_table = create_reconstruction_table(results)
    reconstruction_table_file = output_dir / "reconstruction_metrics.csv"
    reconstruction_table.to_csv(reconstruction_table_file, index=False, float_format='%.4f')
    print(f"\nSaved table: {reconstruction_table_file}")
    print("\nReconstruction Metrics:")
    print(reconstruction_table.to_string(index=False))

    # Generate plots
    print("\nGenerating plots...")

    plot_compression_ratios(results, output_dir / "compression_ratios.png")
    plot_reconstruction_quality(results, output_dir / "reconstruction_quality.png")

    if len(results) > 1:
        plot_compression_vs_quality(results, output_dir / "compression_vs_quality.png")

    print(f"\nAll outputs saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate plots and tables from experiment results"
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiments"),
        help="Directory containing result JSON files (default: results/experiments)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory to save outputs (default: results/plots)"
    )

    args = parser.parse_args()

    generate_all_outputs(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
