"""Evaluation metrics for FGT system."""

import json
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
from loguru import logger
from sklearn.metrics import silhouette_score


def compute_compression_ratio(
    original_text: str,
    glyph_encoded_text: str
) -> float:
    """
    Compute compression ratio.

    Args:
        original_text: Original text
        glyph_encoded_text: Glyph-encoded text

    Returns:
        Compression ratio (original_size / compressed_size)
    """
    original_size = len(original_text.encode('utf-8'))
    glyph_size = len(glyph_encoded_text.encode('utf-8'))

    if glyph_size == 0:
        return 0.0

    ratio = original_size / glyph_size
    return ratio


def compute_cluster_coherence(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 10000
) -> float:
    """
    Compute cluster coherence using silhouette score.

    Args:
        embeddings: Phrase embeddings
        labels: Cluster labels
        sample_size: Maximum sample size for efficiency

    Returns:
        Silhouette score (range: -1 to 1, higher is better)
    """
    logger.info("Computing cluster coherence...")

    # Use a sample for efficiency if dataset is large
    if len(embeddings) > sample_size:
        logger.info(f"Using sample of {sample_size} embeddings for coherence computation")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        sample_labels = labels[indices]
    else:
        sample_embeddings = embeddings
        sample_labels = labels

    score = silhouette_score(sample_embeddings, sample_labels, sample_size=sample_size)
    logger.info(f"Silhouette score: {score:.4f}")

    return float(score)


def compute_reconstruction_quality(
    original_embeddings: np.ndarray,
    reconstructed_embeddings: np.ndarray
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.

    Args:
        original_embeddings: Original phrase embeddings
        reconstructed_embeddings: Reconstructed phrase embeddings

    Returns:
        Dictionary of quality metrics
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Cosine similarity
    similarities = []
    for i in range(len(original_embeddings)):
        sim = cosine_similarity(
            original_embeddings[i:i+1],
            reconstructed_embeddings[i:i+1]
        )[0, 0]
        similarities.append(sim)

    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    # Euclidean distance
    distances = np.linalg.norm(original_embeddings - reconstructed_embeddings, axis=1)
    avg_distance = np.mean(distances)

    return {
        "avg_cosine_similarity": float(avg_similarity),
        "std_cosine_similarity": float(std_similarity),
        "avg_euclidean_distance": float(avg_distance),
    }


def evaluate_tape(
    tape_db_path: str,
    phrases_file: str,
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on the tape.

    Args:
        tape_db_path: Path to tape database
        phrases_file: Path to phrases JSONL file
        embeddings: Phrase embeddings
        labels: Cluster labels

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Running tape evaluation...")

    results = {}

    # 1. Cluster coherence
    coherence = compute_cluster_coherence(embeddings, labels)
    results["cluster_coherence"] = coherence

    # 2. Cluster statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    results["cluster_stats"] = {
        "n_clusters": len(unique_labels),
        "avg_cluster_size": float(counts.mean()),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "std_cluster_size": float(counts.std()),
    }

    # 3. Glyph distribution
    # Count unique glyphs would require loading the tape
    # This is a placeholder for now

    logger.info("Evaluation complete!")
    logger.info(f"Results: {json.dumps(results, indent=2)}")

    return results


def save_evaluation_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved evaluation results to {output_path}")
