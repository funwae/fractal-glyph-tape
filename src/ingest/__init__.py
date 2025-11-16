"""Data ingestion module for extracting phrases from corpora."""

from .reader import CorpusReader
from .segmenter import Segmenter
from .phrases import PhraseExtractor
from .language_detector import LanguageDetector

__all__ = ["CorpusReader", "Segmenter", "PhraseExtractor", "LanguageDetector"]
