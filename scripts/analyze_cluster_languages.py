#!/usr/bin/env python3
"""Analyze language distribution in clusters."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embed.multilingual import MultilingualClusterAnalyzer


def plot_language_distribution(analysis_results, output_file):
    """
    Plot global language distribution.

    Args:
        analysis_results: Analysis results dictionary
        output_file: Path to save plot
    """
    lang_dist = analysis_results["global_language_distribution"]

    # Sort by count
    langs = list(lang_dist.keys())
    counts = [lang_dist[lang] for lang in langs]

    # Sort
    sorted_indices = np.argsort(counts)[::-1]
    langs = [langs[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = sns.color_palette("viridis", len(langs))
    bars = ax.bar(langs, counts, color=colors, alpha=0.8)

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Number of Phrases", fontsize=12)
    ax.set_title("Global Language Distribution Across Clusters", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved language distribution plot: {output_file}")
    plt.close()


def plot_entropy_histogram(analysis_results, output_file):
    """
    Plot histogram of cluster language entropy.

    Args:
        analysis_results: Analysis results dictionary
        output_file: Path to save plot
    """
    entropies = [
        cluster["entropy"]
        for cluster in analysis_results["cluster_stats"]
        if cluster["total_phrases"] > 0
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(entropies, bins=50, color="skyblue", edgecolor="black", alpha=0.7)

    ax.set_xlabel("Language Entropy", fontsize=12)
    ax.set_ylabel("Number of Clusters", fontsize=12)
    ax.set_title("Distribution of Language Entropy Across Clusters", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add statistics
    mean_entropy = np.mean(entropies)
    median_entropy = np.median(entropies)
    ax.axvline(mean_entropy, color="red", linestyle="--", label=f"Mean: {mean_entropy:.3f}")
    ax.axvline(median_entropy, color="green", linestyle="--", label=f"Median: {median_entropy:.3f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved entropy histogram: {output_file}")
    plt.close()


def plot_multilingual_clusters(analysis_results, output_file):
    """
    Plot pie chart of multilingual vs monolingual clusters.

    Args:
        analysis_results: Analysis results dictionary
        output_file: Path to save plot
    """
    multilingual = analysis_results["multilingual_clusters"]
    total = analysis_results["total_clusters"]
    monolingual = total - multilingual

    fig, ax = plt.subplots(figsize=(8, 8))

    sizes = [multilingual, monolingual]
    labels = [
        f"Multilingual\n({multilingual} clusters)",
        f"Monolingual\n({monolingual} clusters)",
    ]
    colors = ["#ff9999", "#66b3ff"]
    explode = (0.1, 0)

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
        textprops={"fontsize": 11},
    )

    ax.set_title(
        f"Multilingual vs Monolingual Clusters\n({analysis_results['multilingual_percentage']:.1f}% multilingual)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved multilingual clusters plot: {output_file}")
    plt.close()


def analyze_tape_languages(tape_dir: Path, output_dir: Path):
    """
    Analyze language distribution in a tape.

    Args:
        tape_dir: Path to tape directory
        output_dir: Directory to save outputs
    """
    print(f"\n{'='*60}")
    print(f"Cluster Language Analysis")
    print(f"{'='*60}")
    print(f"Tape: {tape_dir}")
    print(f"{'='*60}\n")

    # Load cluster metadata
    metadata_file = tape_dir / "clusters" / "metadata.json"

    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        return

    with open(metadata_file, "r", encoding="utf-8") as f:
        cluster_metadata = json.load(f)

    print(f"Loaded metadata for {len(cluster_metadata)} clusters\n")

    # Analyze
    print("Analyzing language distribution...")
    analyzer = MultilingualClusterAnalyzer(cluster_metadata)
    results = analyzer.analyze_all_clusters()

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total clusters: {results['total_clusters']}")
    print(f"Multilingual clusters: {results['multilingual_clusters']}")
    print(f"Multilingual percentage: {results['multilingual_percentage']:.1f}%")
    print(f"Average entropy: {results['average_entropy']:.3f}")

    print(f"\nLanguage distribution:")
    lang_dist = results["global_language_distribution"]
    for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"  {lang}: {count:,} phrases")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    output_json = output_dir / f"language_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_json}")

    # Generate plots
    print("\nGenerating visualizations...")

    plot_language_distribution(results, output_dir / "language_distribution.png")
    plot_entropy_histogram(results, output_dir / "entropy_histogram.png")
    plot_multilingual_clusters(results, output_dir / "multilingual_clusters.png")

    print(f"\nAll outputs saved to: {output_dir}")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze cluster language distribution")

    parser.add_argument(
        "--tape-dir",
        type=Path,
        default=Path("tape/v1"),
        help="Path to tape directory (default: tape/v1)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/language_analysis"),
        help="Directory to save outputs (default: results/language_analysis)",
    )

    args = parser.parse_args()

    analyze_tape_languages(args.tape_dir, args.output_dir)


if __name__ == "__main__":
    main()
