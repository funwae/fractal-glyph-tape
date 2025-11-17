#!/usr/bin/env python3
"""Run cross-lingual retrieval experiments."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cluster.crosslingual import CrossLingualClusterer
from src.embed.multilingual import MultilingualEmbedder, LanguageDetector


def compute_retrieval_metrics(
    relevant_indices: List[int],
    retrieved_indices: List[int],
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """
    Compute retrieval metrics.

    Args:
        relevant_indices: Indices of relevant documents
        retrieved_indices: Indices of retrieved documents (ranked)
        k_values: Values of k for Recall@k

    Returns:
        Dictionary with metrics
    """
    relevant_set = set(relevant_indices)
    metrics = {}

    # Recall@k
    for k in k_values:
        retrieved_k = set(retrieved_indices[:k])
        recall = len(relevant_set & retrieved_k) / len(relevant_set) if relevant_set else 0
        metrics[f"recall@{k}"] = recall

    # Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for rank, idx in enumerate(retrieved_indices, 1):
        if idx in relevant_set:
            mrr = 1.0 / rank
            break
    metrics["mrr"] = mrr

    # Average Precision (AP)
    ap = 0.0
    relevant_count = 0
    for rank, idx in enumerate(retrieved_indices, 1):
        if idx in relevant_set:
            relevant_count += 1
            precision_at_rank = relevant_count / rank
            ap += precision_at_rank

    ap = ap / len(relevant_set) if relevant_set else 0
    metrics["average_precision"] = ap

    return metrics


def run_crosslingual_retrieval_experiment(
    query_phrases: List[str],
    query_languages: List[str],
    document_phrases: List[str],
    document_languages: List[str],
    relevance_labels: List[List[int]],  # For each query, list of relevant doc indices
    tape_dir: Path,
    output_dir: Path,
    use_glyphs: bool = True,
):
    """
    Run cross-lingual retrieval experiment.

    Compares retrieval using:
    1. Baseline: Direct embedding similarity
    2. FGT: Glyph-based retrieval

    Args:
        query_phrases: List of query phrases
        query_languages: Languages of queries
        document_phrases: List of document phrases
        document_languages: Languages of documents
        relevance_labels: Ground truth relevance (list of relevant doc indices per query)
        tape_dir: Path to tape with cross-lingual clusters
        output_dir: Directory to save results
        use_glyphs: Whether to use glyph-based retrieval
    """
    print(f"\n{'='*60}")
    print(f"Cross-Lingual Retrieval Experiment")
    print(f"{'='*60}")
    print(f"Queries: {len(query_phrases)}")
    print(f"Documents: {len(document_phrases)}")
    print(f"Tape: {tape_dir}")
    print(f"Use glyphs: {use_glyphs}")
    print(f"{'='*60}\n")

    # Load components
    embedder = MultilingualEmbedder()

    if use_glyphs:
        # Load cross-lingual clusterer
        print("Loading cross-lingual clusterer...")
        clusterer = CrossLingualClusterer.load(
            tape_dir / "clusters",
            embedder=embedder
        )

    # Embed queries and documents
    print("Embedding queries...")
    query_embeddings = embedder.embed(query_phrases, show_progress=True)

    print("Embedding documents...")
    doc_embeddings = embedder.embed(document_phrases, show_progress=True)

    # Results storage
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_queries": len(query_phrases),
        "num_documents": len(document_phrases),
        "use_glyphs": use_glyphs,
        "query_results": [],
    }

    # Run retrieval for each query
    print("\nRunning retrieval...")

    baseline_metrics_all = []
    glyph_metrics_all = [] if use_glyphs else None

    for query_idx, (query, query_emb, query_lang) in enumerate(
        zip(query_phrases, query_embeddings, query_languages)
    ):
        print(f"\nQuery {query_idx + 1}/{len(query_phrases)}")
        print(f"  Text: '{query}'")
        print(f"  Language: {query_lang}")

        relevant_docs = relevance_labels[query_idx]
        print(f"  Relevant docs: {len(relevant_docs)}")

        # Baseline: Direct embedding similarity
        similarities = np.dot(doc_embeddings, query_emb)
        baseline_ranking = np.argsort(-similarities)  # Descending order

        baseline_metrics = compute_retrieval_metrics(relevant_docs, baseline_ranking.tolist())
        baseline_metrics_all.append(baseline_metrics)

        print(f"  Baseline Recall@10: {baseline_metrics['recall@10']:.3f}")

        query_result = {
            "query_idx": query_idx,
            "query": query,
            "query_language": query_lang,
            "relevant_docs": relevant_docs,
            "baseline_metrics": baseline_metrics,
        }

        # Glyph-based retrieval (if enabled)
        if use_glyphs:
            # Get query cluster
            query_label = clusterer.predict([query])[0]
            query_cluster_id = str(query_label)

            # Get document clusters
            doc_labels = clusterer.predict(document_phrases)

            # Rank documents by cluster match + embedding similarity
            glyph_scores = []
            for doc_idx, (doc_label, doc_emb) in enumerate(zip(doc_labels, doc_embeddings)):
                # Cluster match bonus
                cluster_match = 1.0 if doc_label == query_label else 0.0

                # Embedding similarity
                emb_sim = np.dot(doc_emb, query_emb)

                # Combined score (weighted)
                score = 0.7 * cluster_match + 0.3 * emb_sim
                glyph_scores.append(score)

            glyph_ranking = np.argsort(-np.array(glyph_scores))

            glyph_metrics = compute_retrieval_metrics(relevant_docs, glyph_ranking.tolist())
            glyph_metrics_all.append(glyph_metrics)

            print(f"  Glyph Recall@10: {glyph_metrics['recall@10']:.3f}")
            print(f"  Improvement: {(glyph_metrics['recall@10'] - baseline_metrics['recall@10'])*100:+.1f}%")

            query_result["glyph_metrics"] = glyph_metrics
            query_result["query_cluster"] = query_cluster_id

        results["query_results"].append(query_result)

    # Compute average metrics
    print(f"\n{'='*60}")
    print("Average Metrics")
    print(f"{'='*60}")

    avg_baseline = {}
    for metric_name in baseline_metrics_all[0].keys():
        avg_baseline[metric_name] = np.mean([m[metric_name] for m in baseline_metrics_all])

    print("\nBaseline:")
    for metric, value in avg_baseline.items():
        print(f"  {metric}: {value:.4f}")

    results["average_baseline_metrics"] = avg_baseline

    if use_glyphs and glyph_metrics_all:
        avg_glyph = {}
        for metric_name in glyph_metrics_all[0].keys():
            avg_glyph[metric_name] = np.mean([m[metric_name] for m in glyph_metrics_all])

        print("\nGlyph-based:")
        for metric, value in avg_glyph.items():
            improvement = (value - avg_baseline[metric]) / avg_baseline[metric] * 100 if avg_baseline[metric] > 0 else 0
            print(f"  {metric}: {value:.4f} ({improvement:+.1f}%)")

        results["average_glyph_metrics"] = avg_glyph

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"crosslingual_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run cross-lingual retrieval experiments"
    )

    parser.add_argument(
        "--tape-dir",
        type=Path,
        default=Path("tape/v1"),
        help="Path to tape directory with cross-lingual clusters"
    )

    parser.add_argument(
        "--query-file",
        type=Path,
        required=True,
        help="File with query phrases (one per line)"
    )

    parser.add_argument(
        "--doc-file",
        type=Path,
        required=True,
        help="File with document phrases (one per line)"
    )

    parser.add_argument(
        "--relevance-file",
        type=Path,
        required=True,
        help="JSON file with relevance labels (list of lists)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/crosslingual_retrieval"),
        help="Directory to save results"
    )

    parser.add_argument(
        "--no-glyphs",
        action="store_true",
        help="Run baseline only (no glyph-based retrieval)"
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")

    with open(args.query_file, "r", encoding="utf-8") as f:
        query_phrases = [line.strip() for line in f if line.strip()]

    with open(args.doc_file, "r", encoding="utf-8") as f:
        document_phrases = [line.strip() for line in f if line.strip()]

    with open(args.relevance_file, "r", encoding="utf-8") as f:
        relevance_labels = json.load(f)

    # Detect languages
    detector = LanguageDetector()
    query_languages = detector.detect_batch(query_phrases)
    document_languages = detector.detect_batch(document_phrases)

    # Run experiment
    run_crosslingual_retrieval_experiment(
        query_phrases=query_phrases,
        query_languages=query_languages,
        document_phrases=document_phrases,
        document_languages=document_languages,
        relevance_labels=relevance_labels,
        tape_dir=args.tape_dir,
        output_dir=args.output_dir,
        use_glyphs=not args.no_glyphs,
    )


if __name__ == "__main__":
    main()
