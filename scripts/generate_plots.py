#!/usr/bin/env python3
"""
Generate publication-quality plots for FGT analysis.

Creates:
- Cluster size distribution
- Fractal address depth distribution
- Embedding space visualization (2D projection)
- Coherence score heatmap
"""

import sys
import json
import argparse
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def plot_cluster_size_distribution(cluster_sizes: List[int], output_dir: Path):
    """Plot distribution of cluster sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(cluster_sizes, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Cluster Size (number of phrases)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Cluster Size Distribution')
    axes[0].axvline(np.mean(cluster_sizes), color='red', linestyle='--', label=f'Mean: {np.mean(cluster_sizes):.1f}')
    axes[0].axvline(np.median(cluster_sizes), color='green', linestyle='--', label=f'Median: {np.median(cluster_sizes):.1f}')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(cluster_sizes, vert=True)
    axes[1].set_ylabel('Cluster Size')
    axes[1].set_title('Cluster Size Box Plot')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "cluster_size_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()


def plot_fractal_map_2d(coordinates: np.ndarray, cluster_sizes: List[int], output_dir: Path):
    """Plot 2D fractal map with cluster sizes."""
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=cluster_sizes,
        s=np.sqrt(cluster_sizes) * 5,
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Semantic Phrase Space - Fractal Map\n(size = cluster size, color = cluster size)')
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster Size (phrases)')

    plt.tight_layout()
    plot_path = output_dir / "fractal_map_2d.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()


def plot_address_depth_distribution(addresses: List[str], output_dir: Path):
    """Plot distribution of fractal address depths."""
    depths = [len(addr.split('-')) for addr in addresses]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(depths, bins=range(min(depths), max(depths) + 2), color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Fractal Address Depth')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Fractal Address Depth Distribution')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "address_depth_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()


def plot_glyph_length_distribution(glyph_strings: List[str], output_dir: Path):
    """Plot distribution of glyph string lengths."""
    lengths = [len(g) for g in glyph_strings]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(lengths, bins=range(1, max(lengths) + 2), color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Glyph String Length (characters)')
    ax.set_ylabel('Number of Glyphs')
    ax.set_title('Glyph String Length Distribution')
    ax.grid(axis='y', alpha=0.3)

    # Add statistics
    ax.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.2f}')
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / "glyph_length_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()


def plot_summary_dashboard(stats: Dict[str, Any], output_dir: Path):
    """Create a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Fractal Glyph Tape - Analysis Dashboard', fontsize=18, fontweight='bold')

    # Metric cards
    metrics = [
        ("Total Clusters", stats['n_clusters'], (0, 0)),
        ("Total Phrases", stats['n_phrases'], (0, 1)),
        ("Avg Cluster Size", f"{stats['avg_cluster_size']:.1f}", (0, 2)),
        ("Embedding Dim", stats['embedding_dim'], (1, 0)),
        ("Projection Method", stats['projection_method'], (1, 1)),
        ("Fractal Depth", stats['fractal_depth'], (1, 2)),
    ]

    for title, value, pos in metrics:
        ax = fig.add_subplot(gs[pos])
        ax.axis('off')
        ax.text(0.5, 0.5, str(value), ha='center', va='center', fontsize=32, fontweight='bold')
        ax.text(0.5, 0.2, title, ha='center', va='center', fontsize=14, color='gray')
        ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='gray', linewidth=2))

    # Statistics table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    table_data = [
        ["Metric", "Value"],
        ["Min Cluster Size", f"{stats['min_cluster_size']}"],
        ["Max Cluster Size", f"{stats['max_cluster_size']}"],
        ["Median Cluster Size", f"{stats['median_cluster_size']:.1f}"],
        ["Std Cluster Size", f"{stats['std_cluster_size']:.1f}"],
        ["Total Glyphs", f"{stats['n_clusters']}"],
    ]

    table = ax_table.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plot_path = output_dir / "summary_dashboard.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {plot_path}")
    plt.close()


def generate_all_plots(tape_db_path: str, output_dir: str):
    """Generate all analysis plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("GENERATING ANALYSIS PLOTS")
    logger.info("=" * 80)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # Connect to database
    logger.info(f"Loading data from {tape_db_path}...")
    conn = sqlite3.connect(tape_db_path)
    cursor = conn.cursor()

    # Get cluster sizes
    cursor.execute("SELECT size FROM clusters")
    cluster_sizes = [row[0] for row in cursor.fetchall()]

    # Get coordinates
    cursor.execute("SELECT x_coord, y_coord FROM addresses")
    coords_data = cursor.fetchall()
    coordinates = np.array(coords_data)

    # Get fractal addresses
    cursor.execute("SELECT fractal_address FROM addresses")
    addresses = [row[0] for row in cursor.fetchall()]

    # Get glyph strings
    cursor.execute("SELECT glyph_string FROM glyphs")
    glyph_strings = [row[0] for row in cursor.fetchall()]

    conn.close()

    # Calculate statistics
    stats = {
        'n_clusters': len(cluster_sizes),
        'n_phrases': sum(cluster_sizes),
        'avg_cluster_size': np.mean(cluster_sizes),
        'median_cluster_size': np.median(cluster_sizes),
        'min_cluster_size': min(cluster_sizes),
        'max_cluster_size': max(cluster_sizes),
        'std_cluster_size': np.std(cluster_sizes),
        'embedding_dim': 384,
        'projection_method': 'UMAP',
        'fractal_depth': 10,
    }

    logger.info(f"Clusters: {stats['n_clusters']}")
    logger.info(f"Phrases: {stats['n_phrases']}")
    logger.info(f"Avg cluster size: {stats['avg_cluster_size']:.1f}")

    # Generate plots
    logger.info("\nGenerating plots...")

    logger.info("1. Summary dashboard...")
    plot_summary_dashboard(stats, output_dir)

    logger.info("2. Cluster size distribution...")
    plot_cluster_size_distribution(cluster_sizes, output_dir)

    logger.info("3. Fractal map 2D...")
    plot_fractal_map_2d(coordinates, cluster_sizes, output_dir)

    logger.info("4. Address depth distribution...")
    plot_address_depth_distribution(addresses, output_dir)

    logger.info("5. Glyph length distribution...")
    plot_glyph_length_distribution(glyph_strings, output_dir)

    # Save statistics
    stats_file = output_dir / "statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nSaved statistics to {stats_file}")

    logger.info("\n" + "=" * 80)
    logger.info("ALL PLOTS GENERATED")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate FGT analysis plots")
    parser.add_argument("--tape", default="tape/v1/tape_index.db", help="Path to tape database")
    parser.add_argument("--output", default="results/plots", help="Output directory for plots")
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    generate_all_plots(args.tape, args.output)


if __name__ == "__main__":
    main()
