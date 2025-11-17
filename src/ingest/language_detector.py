"""Language detection for multilingual phrase processing."""

import re
from typing import Optional
from loguru import logger


class LanguageDetector:
    """Detect language of text phrases."""

    def __init__(self, method: str = "simple"):
        """
        Initialize language detector.

        Args:
            method: Detection method ('simple', 'langdetect', or 'langid')
        """
        self.method = method

        if method == "langdetect":
            try:
                import langdetect
                self.detector = langdetect
                logger.info("Using langdetect for language detection")
            except ImportError:
                logger.warning("langdetect not available, falling back to simple")
                self.method = "simple"

        elif method == "langid":
            try:
                import langid
                self.detector = langid
                logger.info("Using langid for language detection")
            except ImportError:
                logger.warning("langid not available, falling back to simple")
                self.method = "simple"

    def detect(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            ISO 639-1 language code (e.g., 'en', 'es', 'zh')
        """
        if not text or not text.strip():
            return "unknown"

        if self.method == "langdetect":
            return self._detect_langdetect(text)
        elif self.method == "langid":
            return self._detect_langid(text)
        else:
            return self._detect_simple(text)

    def _detect_langdetect(self, text: str) -> str:
        """Detect using langdetect library."""
        try:
            lang = self.detector.detect(text)
            return lang
        except:
            return self._detect_simple(text)

    def _detect_langid(self, text: str) -> str:
        """Detect using langid library."""
        try:
            lang, confidence = self.detector.classify(text)
            return lang
        except:
            return self._detect_simple(text)

    def _detect_simple(self, text: str) -> str:
        """
        Simple pattern-based language detection.

        Uses character ranges to identify languages:
        - Chinese: CJK characters
        - Spanish: Spanish-specific characters and patterns
        - English: Default if no other matches
        """
        # Chinese: Check for CJK characters
        if self._has_cjk(text):
            return "zh"

        # Spanish: Check for Spanish-specific characters
        if self._has_spanish_chars(text):
            return "es"

        # Default to English
        return "en"

    def _has_cjk(self, text: str) -> bool:
        """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
        # CJK Unified Ideographs: U+4E00 to U+9FFF
        cjk_pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(cjk_pattern.search(text))

    def _has_spanish_chars(self, text: str) -> bool:
        """Check for Spanish-specific characters."""
        # Spanish has: ñ, á, é, í, ó, ú, ü, ¿, ¡
        spanish_chars = ['ñ', 'Ñ', 'á', 'é', 'í', 'ó', 'ú', 'ü', '¿', '¡']
        return any(char in text for char in spanish_chars)

    def detect_with_confidence(self, text: str) -> tuple[str, float]:
        """
        Detect language with confidence score.

        Args:
            text: Input text

        Returns:
            (language_code, confidence) tuple
        """
        if self.method == "langdetect":
            try:
                from langdetect import detect_langs
                langs = detect_langs(text)
                if langs:
                    return langs[0].lang, langs[0].prob
            except:
                pass

        elif self.method == "langid":
            try:
                import langid
                lang, confidence = langid.classify(text)
                return lang, confidence
            except:
                pass

        # Simple method doesn't provide confidence
        lang = self.detect(text)
        confidence = 1.0 if lang != "unknown" else 0.0
        return lang, confidence
