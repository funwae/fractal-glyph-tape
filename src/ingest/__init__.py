"""Data ingestion module for extracting phrases from corpora."""

from .reader import CorpusReader
from .segmenter import Segmenter
from .phrases import PhraseExtractor

__all__ = ["CorpusReader", "Segmenter", "PhraseExtractor"]
