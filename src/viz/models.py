"""Data models for visualization API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ClusterSummary(BaseModel):
    """Summary information for a cluster."""

    cluster_id: str = Field(..., description="Unique cluster identifier")
    glyph: str = Field(..., description="Glyph character assigned to this cluster")
    size: int = Field(..., description="Number of phrases in this cluster")
    language: str = Field(..., description="Primary language of cluster")
    frequency: int = Field(..., description="Total frequency of phrases in cluster")
    representative_phrase: str = Field(..., description="Most representative phrase")


class ClusterInfo(ClusterSummary):
    """Detailed information about a cluster."""

    example_phrases: List[str] = Field(
        default_factory=list,
        description="Example phrases from this cluster"
    )
    embedding_centroid: List[float] = Field(
        default_factory=list,
        description="Centroid of phrase embeddings"
    )
    coherence_score: float = Field(
        default=0.0,
        description="Cluster coherence metric"
    )


class LayoutPoint(BaseModel):
    """2D coordinate for cluster visualization."""

    cluster_id: str = Field(..., description="Cluster identifier")
    x: float = Field(..., description="X coordinate in 2D layout")
    y: float = Field(..., description="Y coordinate in 2D layout")
    glyph: str = Field(..., description="Glyph character")
    language: str = Field(..., description="Primary language")
    frequency: int = Field(..., description="Total frequency")


class CompressionMetrics(BaseModel):
    """Metrics for compression experiments."""

    dataset_name: str = Field(..., description="Name of the dataset")
    raw_bytes_total: int = Field(..., description="Total bytes in raw text")
    raw_bytes_per_sentence: float = Field(..., description="Average bytes per sentence (raw)")
    fgt_bytes_total: int = Field(..., description="Total bytes with FGT (including tables)")
    fgt_bytes_sequences: int = Field(..., description="Bytes for sequences only")
    fgt_bytes_tables: int = Field(..., description="Bytes for lookup tables")
    compression_ratio: float = Field(..., description="Overall compression ratio")
    compression_ratio_sequences: float = Field(..., description="Compression ratio for sequences only")
    sentence_count: int = Field(..., description="Number of sentences")


class ReconstructionMetrics(BaseModel):
    """Metrics for reconstruction quality."""

    dataset_name: str = Field(..., description="Name of the dataset")
    bleu_score: float = Field(..., description="BLEU score")
    rouge_1_f1: float = Field(..., description="ROUGE-1 F1 score")
    rouge_2_f1: float = Field(..., description="ROUGE-2 F1 score")
    rouge_l_f1: float = Field(..., description="ROUGE-L F1 score")
    bertscore_f1: float = Field(..., description="BERTScore F1")
    sample_count: int = Field(..., description="Number of samples evaluated")


class ExperimentResult(BaseModel):
    """Combined experiment results."""

    compression: CompressionMetrics
    reconstruction: ReconstructionMetrics
    config: dict = Field(default_factory=dict, description="Experiment configuration")
