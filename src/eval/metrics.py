"""Evaluation metrics for Fractal Glyph Tape experiments."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def compute_compression_ratio(
    raw_bytes: int,
    fgt_bytes_sequences: int,
    fgt_bytes_tables: int,
) -> Dict[str, float]:
    """
    Compute compression ratio metrics.

    Args:
        raw_bytes: Total bytes in raw text representation
        fgt_bytes_sequences: Bytes for FGT sequences
        fgt_bytes_tables: Bytes for lookup tables

    Returns:
        Dictionary with compression metrics
    """
    fgt_bytes_total = fgt_bytes_sequences + fgt_bytes_tables

    return {
        "raw_bytes": raw_bytes,
        "fgt_bytes_total": fgt_bytes_total,
        "fgt_bytes_sequences": fgt_bytes_sequences,
        "fgt_bytes_tables": fgt_bytes_tables,
        "compression_ratio": raw_bytes / fgt_bytes_total if fgt_bytes_total > 0 else 0,
        "compression_ratio_sequences": raw_bytes / fgt_bytes_sequences if fgt_bytes_sequences > 0 else 0,
        "compression_percentage": (1 - fgt_bytes_total / raw_bytes) * 100 if raw_bytes > 0 else 0,
    }


def compute_cluster_coherence(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute cluster coherence using average cosine similarity within clusters.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: Cluster labels for each sample

    Returns:
        Average coherence score
    """
    from sklearn.metrics.pairwise import cosine_similarity

    coherence_scores = []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = embeddings[mask]

        if len(cluster_embeddings) < 2:
            continue

        # Compute pairwise cosine similarities
        sim_matrix = cosine_similarity(cluster_embeddings)

        # Average of upper triangle (excluding diagonal)
        n = len(cluster_embeddings)
        triu_indices = np.triu_indices(n, k=1)
        avg_similarity = sim_matrix[triu_indices].mean()

        coherence_scores.append(avg_similarity)

    return float(np.mean(coherence_scores)) if coherence_scores else 0.0


def compute_bleu_score(
    references: List[str],
    hypotheses: List[str],
    max_n: int = 4,
) -> float:
    """
    Compute BLEU score for reconstruction quality.

    Args:
        references: Original text
        hypotheses: Reconstructed text
        max_n: Maximum n-gram size

    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        import nltk

        # Download required data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        # Tokenize
        references_tokenized = [[ref.split()] for ref in references]
        hypotheses_tokenized = [hyp.split() for hyp in hypotheses]

        # Use smoothing to handle edge cases
        smoothing = SmoothingFunction().method1

        score = corpus_bleu(
            references_tokenized,
            hypotheses_tokenized,
            smoothing_function=smoothing,
        )

        return float(score)

    except ImportError:
        print("Warning: nltk not available for BLEU computation", file=sys.stderr)
        return 0.0


def compute_rouge_scores(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE scores for reconstruction quality.

    Args:
        references: Original text
        hypotheses: Reconstructed text

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref, hyp)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            'rouge1_f1': float(np.mean(rouge1_scores)),
            'rouge2_f1': float(np.mean(rouge2_scores)),
            'rougeL_f1': float(np.mean(rougeL_scores)),
        }

    except ImportError:
        print("Warning: rouge-score not available. Install with: pip install rouge-score", file=sys.stderr)
        return {
            'rouge1_f1': 0.0,
            'rouge2_f1': 0.0,
            'rougeL_f1': 0.0,
        }


def compute_bertscore(
    references: List[str],
    hypotheses: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Compute BERTScore for semantic similarity.

    Args:
        references: Original text
        hypotheses: Reconstructed text
        model_type: Model to use for BERTScore
        batch_size: Batch size for processing

    Returns:
        Dictionary with precision, recall, F1 scores
    """
    try:
        from bert_score import score

        # Compute BERTScore
        P, R, F1 = score(
            hypotheses,
            references,
            model_type=model_type,
            batch_size=batch_size,
            verbose=False,
        )

        return {
            'bertscore_precision': float(P.mean()),
            'bertscore_recall': float(R.mean()),
            'bertscore_f1': float(F1.mean()),
        }

    except ImportError:
        print("Warning: bert-score not available. Install with: pip install bert-score", file=sys.stderr)
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0,
        }
    except Exception as e:
        print(f"Warning: BERTScore computation failed: {e}", file=sys.stderr)
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0,
        }


def compute_reconstruction_metrics(
    references: List[str],
    hypotheses: List[str],
    use_bertscore: bool = False,
) -> Dict[str, float]:
    """
    Compute all reconstruction quality metrics.

    Args:
        references: Original text
        hypotheses: Reconstructed text
        use_bertscore: Whether to compute BERTScore (slow)

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # BLEU
    metrics['bleu'] = compute_bleu_score(references, hypotheses)

    # ROUGE
    rouge_scores = compute_rouge_scores(references, hypotheses)
    metrics.update(rouge_scores)

    # BERTScore (optional, as it's slow)
    if use_bertscore:
        bert_scores = compute_bertscore(references, hypotheses)
        metrics.update(bert_scores)
    else:
        metrics.update({
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0,
        })

    return metrics


def load_corpus(file_path: Path) -> Tuple[List[str], int]:
    """
    Load corpus from file.

    Args:
        file_path: Path to corpus file

    Returns:
        Tuple of (sentences, total_bytes)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into sentences (simple approach)
    sentences = [s.strip() for s in content.split('\n') if s.strip()]

    total_bytes = len(content.encode('utf-8'))

    return sentences, total_bytes


def evaluate_compression_and_reconstruction(
    original_file: Path,
    reconstructed_file: Path,
    fgt_sequences_bytes: int,
    fgt_tables_bytes: int,
    use_bertscore: bool = False,
) -> Dict[str, any]:
    """
    Comprehensive evaluation of compression and reconstruction.

    Args:
        original_file: Path to original corpus
        reconstructed_file: Path to reconstructed corpus
        fgt_sequences_bytes: Bytes used for FGT sequences
        fgt_tables_bytes: Bytes used for lookup tables
        use_bertscore: Whether to compute BERTScore

    Returns:
        Dictionary with all metrics
    """
    # Load data
    original_sentences, raw_bytes = load_corpus(original_file)
    reconstructed_sentences, _ = load_corpus(reconstructed_file)

    # Ensure same length
    min_len = min(len(original_sentences), len(reconstructed_sentences))
    original_sentences = original_sentences[:min_len]
    reconstructed_sentences = reconstructed_sentences[:min_len]

    # Compression metrics
    compression_metrics = compute_compression_ratio(
        raw_bytes,
        fgt_sequences_bytes,
        fgt_tables_bytes,
    )

    # Reconstruction metrics
    reconstruction_metrics = compute_reconstruction_metrics(
        original_sentences,
        reconstructed_sentences,
        use_bertscore=use_bertscore,
    )

    # Combine results
    results = {
        'dataset_name': original_file.stem,
        'sentence_count': len(original_sentences),
        'compression': compression_metrics,
        'reconstruction': reconstruction_metrics,
    }

    return results


if __name__ == "__main__":
    # Example usage
    print("Metrics module loaded successfully")
    print("\nAvailable functions:")
    print("  - compute_compression_ratio()")
    print("  - compute_cluster_coherence()")
    print("  - compute_bleu_score()")
    print("  - compute_rouge_scores()")
    print("  - compute_bertscore()")
    print("  - compute_reconstruction_metrics()")
    print("  - evaluate_compression_and_reconstruction()")
