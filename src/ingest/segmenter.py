"""Sentence segmentation for phrase extraction."""

import re
from typing import List
from loguru import logger


class Segmenter:
    """Segment text into sentences/phrases."""

    def __init__(self, method: str = "nltk", language: str = "english"):
        """
        Initialize segmenter.

        Args:
            method: Segmentation method ('nltk', 'spacy', or 'regex')
            language: Language for segmentation
        """
        self.method = method
        self.language = language

        if method == "nltk":
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    logger.info("Downloading NLTK punkt tokenizer...")
                    nltk.download('punkt', quiet=True)
                self.tokenizer = nltk.sent_tokenize
            except ImportError:
                logger.warning("NLTK not available, falling back to regex")
                self.method = "regex"

        elif method == "spacy":
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                logger.warning("spaCy not available or model not installed, falling back to regex")
                self.method = "regex"

    def segment(self, text: str) -> List[str]:
        """
        Segment text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []

        if self.method == "nltk":
            return self._segment_nltk(text)
        elif self.method == "spacy":
            return self._segment_spacy(text)
        else:
            return self._segment_regex(text)

    def _segment_nltk(self, text: str) -> List[str]:
        """Segment using NLTK."""
        import nltk
        sentences = nltk.sent_tokenize(text, language=self.language)
        return [s.strip() for s in sentences if s.strip()]

    def _segment_spacy(self, text: str) -> List[str]:
        """Segment using spaCy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _segment_regex(self, text: str) -> List[str]:
        """Simple regex-based segmentation (fallback)."""
        # Split on sentence-ending punctuation followed by whitespace
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def count_tokens(self, text: str) -> int:
        """
        Count approximate token count.

        Args:
            text: Input text

        Returns:
            Token count
        """
        # Simple whitespace-based tokenization for filtering
        return len(text.split())
