"""Embedding module for converting phrases to dense vectors."""

from .embedder import PhraseEmbedder
from .multilingual import (
    LanguageDetector,
    MultilingualClusterAnalyzer,
    MultilingualEmbedder,
)

__all__ = [
    "PhraseEmbedder",
    "MultilingualEmbedder",
    "LanguageDetector",
    "MultilingualClusterAnalyzer",
]
