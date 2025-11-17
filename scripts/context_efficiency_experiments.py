#!/usr/bin/env python3
"""
Context efficiency experiments for FGT.

Measures:
- Token count reduction with hybrid encoding
- Context window utilization
- Semantic preservation
- Compression vs quality trade-offs
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

from tokenizer import HybridTokenizer
from embed import PhraseEmbedder


def load_test_documents(phrases_file: str, num_docs: int = 100) -> List[str]:
    """
    Load test documents (concatenated phrases).

    Args:
        phrases_file: Path to phrases file
        num_docs: Number of documents to create

    Returns:
        List of document texts
    """
    logger.info(f"Loading test documents from {phrases_file}...")

    phrases = []
    with open(phrases_file, "r") as f:
        for line in f:
            data = json.loads(line)
            phrases.append(data["text"])

    # Create documents by concatenating ~10-20 phrases each
    documents = []
    phrases_per_doc = len(phrases) // num_docs

    for i in range(num_docs):
        start_idx = i * phrases_per_doc
        end_idx = start_idx + phrases_per_doc
        doc_phrases = phrases[start_idx:end_idx]
        doc_text = " ".join(doc_phrases)
        documents.append(doc_text)

    logger.info(f"Created {len(documents)} test documents")
    logger.info(f"Avg document length: {np.mean([len(d) for d in documents]):.0f} chars")

    return documents


def measure_token_reduction(
    documents: List[str],
    base_tokenizer_name: str,
    hybrid_tokenizer: HybridTokenizer
) -> Dict[str, Any]:
    """
    Measure token count reduction with hybrid encoding.

    Args:
        documents: List of documents
        base_tokenizer_name: Name of base tokenizer
        hybrid_tokenizer: Hybrid tokenizer instance

    Returns:
        Dict with results
    """
    logger.info("Measuring token reduction...")

    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)

    regular_counts = []
    hybrid_counts = []
    reduction_ratios = []
    glyph_percentages = []

    for doc in documents:
        # Regular tokenization
        regular_tokens = base_tokenizer.encode(doc)
        regular_count = len(regular_tokens)

        # Hybrid tokenization
        hybrid_result = hybrid_tokenizer.encode_hybrid(doc, return_details=True)
        hybrid_count = len(hybrid_result["token_ids"])
        glyph_encoded = hybrid_result["glyph_encoded"]
        total_phrases = hybrid_result["total_phrases"]

        # Calculate metrics
        reduction = (regular_count - hybrid_count) / regular_count if regular_count > 0 else 0
        glyph_pct = glyph_encoded / total_phrases if total_phrases > 0 else 0

        regular_counts.append(regular_count)
        hybrid_counts.append(hybrid_count)
        reduction_ratios.append(reduction)
        glyph_percentages.append(glyph_pct)

    results = {
        "avg_regular_tokens": np.mean(regular_counts),
        "avg_hybrid_tokens": np.mean(hybrid_counts),
        "avg_reduction_ratio": np.mean(reduction_ratios),
        "avg_glyph_percentage": np.mean(glyph_percentages),
        "median_reduction": np.median(reduction_ratios),
        "std_reduction": np.std(reduction_ratios),
        "max_reduction": np.max(reduction_ratios),
        "min_reduction": np.min(reduction_ratios),
        "total_regular_tokens": sum(regular_counts),
        "total_hybrid_tokens": sum(hybrid_counts),
    }

    logger.info("Token reduction results:")
    logger.info(f"  Avg regular tokens: {results['avg_regular_tokens']:.1f}")
    logger.info(f"  Avg hybrid tokens: {results['avg_hybrid_tokens']:.1f}")
    logger.info(f"  Avg reduction: {results['avg_reduction_ratio']:.1%}")
    logger.info(f"  Avg glyph usage: {results['avg_glyph_percentage']:.1%}")

    return results


def measure_semantic_preservation(
    documents: List[str],
    hybrid_tokenizer: HybridTokenizer,
    embedder: PhraseEmbedder,
    sample_size: int = 50
) -> Dict[str, Any]:
    """
    Measure semantic preservation after hybrid encoding.

    Args:
        documents: List of documents
        hybrid_tokenizer: Hybrid tokenizer
        embedder: Phrase embedder
        sample_size: Number of documents to sample

    Returns:
        Dict with preservation metrics
    """
    logger.info("Measuring semantic preservation...")

    # Sample documents
    if len(documents) > sample_size:
        import random
        sampled_docs = random.sample(documents, sample_size)
    else:
        sampled_docs = documents

    similarities = []

    for doc in sampled_docs:
        # Original embedding
        original_emb = embedder._embed_batch([doc])[0]

        # Hybrid encode and decode
        token_ids = hybrid_tokenizer.encode_hybrid(doc)
        decoded_text = hybrid_tokenizer.decode_hybrid_with_expansion(token_ids)

        # Decoded embedding
        decoded_emb = embedder._embed_batch([decoded_text])[0]

        # Cosine similarity
        cosine_sim = np.dot(original_emb, decoded_emb) / (
            np.linalg.norm(original_emb) * np.linalg.norm(decoded_emb) + 1e-10
        )

        similarities.append(cosine_sim)

    results = {
        "avg_similarity": np.mean(similarities),
        "median_similarity": np.median(similarities),
        "std_similarity": np.std(similarities),
        "min_similarity": np.min(similarities),
        "max_similarity": np.max(similarities),
    }

    logger.info("Semantic preservation results:")
    logger.info(f"  Avg cosine similarity: {results['avg_similarity']:.3f}")
    logger.info(f"  Median similarity: {results['median_similarity']:.3f}")

    return results


def plot_results(
    token_results: Dict[str, Any],
    semantic_results: Dict[str, Any],
    output_dir: Path
):
    """Generate visualization plots."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Token reduction bar chart
    methods = ["Regular\nTokens", "Hybrid\nTokens"]
    counts = [
        token_results["avg_regular_tokens"],
        token_results["avg_hybrid_tokens"]
    ]

    axes[0, 0].bar(methods, counts, color=["#1f77b4", "#2ca02c"])
    axes[0, 0].set_ylabel("Avg Tokens per Document")
    axes[0, 0].set_title("Token Count Comparison")
    axes[0, 0].set_ylim(0, max(counts) * 1.2)

    # Add reduction percentage
    reduction_pct = token_results["avg_reduction_ratio"] * 100
    axes[0, 0].text(
        0.5, max(counts) * 1.1,
        f"{reduction_pct:.1f}% reduction",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="red"
    )

    # 2. Context window utilization
    context_window = 2048  # Typical GPT-2 context
    regular_docs_per_window = context_window / token_results["avg_regular_tokens"]
    hybrid_docs_per_window = context_window / token_results["avg_hybrid_tokens"]

    methods2 = ["Regular", "Hybrid"]
    docs_per_window = [regular_docs_per_window, hybrid_docs_per_window]

    axes[0, 1].bar(methods2, docs_per_window, color=["#1f77b4", "#2ca02c"])
    axes[0, 1].set_ylabel("Documents per Context Window")
    axes[0, 1].set_title(f"Context Efficiency (2048 tokens)")
    axes[0, 1].set_ylim(0, max(docs_per_window) * 1.2)

    improvement = ((hybrid_docs_per_window / regular_docs_per_window) - 1) * 100
    axes[0, 1].text(
        0.5, max(docs_per_window) * 1.1,
        f"+{improvement:.1f}% capacity",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="green"
    )

    # 3. Semantic similarity distribution
    axes[1, 0].hist(
        [semantic_results["avg_similarity"]],
        bins=1,
        color="steelblue",
        edgecolor="black",
        alpha=0.7
    )
    axes[1, 0].axvline(
        semantic_results["avg_similarity"],
        color="red",
        linestyle="--",
        label=f"Avg: {semantic_results['avg_similarity']:.3f}"
    )
    axes[1, 0].set_xlabel("Cosine Similarity")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Semantic Preservation")
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 1)

    # 4. Summary statistics table
    axes[1, 1].axis("off")
    table_data = [
        ["Metric", "Value"],
        ["Avg Token Reduction", f"{token_results['avg_reduction_ratio']:.1%}"],
        ["Avg Glyph Usage", f"{token_results['avg_glyph_percentage']:.1%}"],
        ["Context Capacity Gain", f"+{improvement:.1f}%"],
        ["Semantic Similarity", f"{semantic_results['avg_similarity']:.3f}"],
        ["Regular Tokens (avg)", f"{token_results['avg_regular_tokens']:.0f}"],
        ["Hybrid Tokens (avg)", f"{token_results['avg_hybrid_tokens']:.0f}"],
    ]

    table = axes[1, 1].table(
        cellText=table_data,
        cellLoc="left",
        loc="center",
        colWidths=[0.6, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    axes[1, 1].set_title("Summary Statistics", pad=20)

    plt.tight_layout()
    plot_path = output_dir / "context_efficiency.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot: {plot_path}")
    plt.close()


def run_experiments(
    phrases_file: str,
    tape_db_path: str,
    output_dir: str,
    num_docs: int = 100,
    base_tokenizer: str = "gpt2"
):
    """Run complete context efficiency experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("CONTEXT EFFICIENCY EXPERIMENTS")
    logger.info("=" * 80)

    # Load test documents
    documents = load_test_documents(phrases_file, num_docs)

    # Initialize hybrid tokenizer
    logger.info("\nInitializing hybrid tokenizer...")
    hybrid_tokenizer = HybridTokenizer(
        base_tokenizer=base_tokenizer,
        tape_db_path=tape_db_path,
        similarity_threshold=0.75
    )

    # Initialize embedder
    logger.info("Initializing embedder...")
    embedder = PhraseEmbedder({
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu"
    })

    # Experiment 1: Token reduction
    logger.info("\n" + "=" * 80)
    logger.info("Experiment 1: Token Count Reduction")
    logger.info("=" * 80)
    token_results = measure_token_reduction(documents, base_tokenizer, hybrid_tokenizer)

    # Experiment 2: Semantic preservation
    logger.info("\n" + "=" * 80)
    logger.info("Experiment 2: Semantic Preservation")
    logger.info("=" * 80)
    semantic_results = measure_semantic_preservation(
        documents, hybrid_tokenizer, embedder, sample_size=min(50, num_docs)
    )

    # Save results
    results = {
        "token_reduction": token_results,
        "semantic_preservation": semantic_results,
        "config": {
            "num_docs": num_docs,
            "base_tokenizer": base_tokenizer,
            "tape_db_path": tape_db_path,
        }
    }

    results_file = output_dir / "context_efficiency_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {results_file}")

    # Generate plots
    logger.info("\nGenerating plots...")
    plot_results(token_results, semantic_results, output_dir)

    # Generate report
    report_path = output_dir / "context_efficiency_report.md"
    with open(report_path, "w") as f:
        f.write("# Context Efficiency Experiments - FGT\n\n")
        f.write(f"**Documents tested:** {num_docs}\n")
        f.write(f"**Base tokenizer:** {base_tokenizer}\n\n")

        f.write("## Token Reduction Results\n\n")
        f.write(f"- **Average reduction:** {token_results['avg_reduction_ratio']:.1%}\n")
        f.write(f"- **Glyph usage:** {token_results['avg_glyph_percentage']:.1%}\n")
        f.write(f"- **Regular tokens (avg):** {token_results['avg_regular_tokens']:.1f}\n")
        f.write(f"- **Hybrid tokens (avg):** {token_results['avg_hybrid_tokens']:.1f}\n\n")

        context_window = 2048
        regular_cap = context_window / token_results["avg_regular_tokens"]
        hybrid_cap = context_window / token_results["avg_hybrid_tokens"]
        improvement = ((hybrid_cap / regular_cap) - 1) * 100

        f.write("## Context Window Efficiency\n\n")
        f.write(f"For a {context_window}-token context window:\n")
        f.write(f"- **Regular encoding:** {regular_cap:.1f} documents\n")
        f.write(f"- **Hybrid encoding:** {hybrid_cap:.1f} documents\n")
        f.write(f"- **Improvement:** +{improvement:.1f}%\n\n")

        f.write("## Semantic Preservation\n\n")
        f.write(f"- **Average similarity:** {semantic_results['avg_similarity']:.3f}\n")
        f.write(f"- **Median similarity:** {semantic_results['median_similarity']:.3f}\n")
        f.write(f"- **Min similarity:** {semantic_results['min_similarity']:.3f}\n\n")

        f.write("## Visualization\n\n")
        f.write("![Context Efficiency](context_efficiency.png)\n\n")

    logger.info(f"Saved report to {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENTS COMPLETE!")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run context efficiency experiments")
    parser.add_argument("--phrases", default="data/phrases.jsonl", help="Phrases file")
    parser.add_argument("--tape", default="tape/v1/tape_index.db", help="Tape database")
    parser.add_argument("--output", default="results/context_efficiency", help="Output directory")
    parser.add_argument("--num-docs", type=int, default=100, help="Number of test documents")
    parser.add_argument("--tokenizer", default="gpt2", help="Base tokenizer")
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    run_experiments(
        args.phrases,
        args.tape,
        args.output,
        args.num_docs,
        args.tokenizer
    )


if __name__ == "__main__":
    main()
