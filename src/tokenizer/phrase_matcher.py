"""Phrase matcher for detecting phrase spans in text."""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PhraseSpan:
    """Represents a detected phrase span in text."""

    start_char: int
    end_char: int
    cluster_id: str
    confidence: float
    original_text: str


class PhraseMatcher:
    """
    Detects spans in text that correspond to phrase families.

    Uses exact string matching with normalization for the initial prototype.
    Can be extended with approximate matching later.
    """

    def __init__(
        self,
        cluster_metadata: Dict,
        min_confidence: float = 0.9,
        max_span_length: int = 6,
        allow_overlaps: bool = False,
    ):
        """
        Initialize phrase matcher.

        Args:
            cluster_metadata: Dictionary mapping cluster_id → metadata
            min_confidence: Minimum confidence threshold for matches
            max_span_length: Maximum number of words in a phrase
            allow_overlaps: Whether to allow overlapping phrase spans
        """
        self.cluster_metadata = cluster_metadata
        self.min_confidence = min_confidence
        self.max_span_length = max_span_length
        self.allow_overlaps = allow_overlaps

        # Build phrase index
        self._build_phrase_index()

    def _build_phrase_index(self):
        """Build index for fast phrase lookups."""
        # Map normalized phrase → cluster_id
        self.phrase_to_cluster: Dict[str, str] = {}

        # Map cluster_id → list of phrases
        self.cluster_to_phrases: Dict[str, List[str]] = defaultdict(list)

        for cluster_id, meta in self.cluster_metadata.items():
            # Add representative phrase
            rep_phrase = meta.get("representative_phrase", "")
            if rep_phrase:
                normalized = self._normalize_text(rep_phrase)
                self.phrase_to_cluster[normalized] = cluster_id
                self.cluster_to_phrases[cluster_id].append(rep_phrase)

            # Add example phrases
            examples = meta.get("example_phrases", [])
            for phrase in examples:
                normalized = self._normalize_text(phrase)
                self.phrase_to_cluster[normalized] = cluster_id
                if phrase not in self.cluster_to_phrases[cluster_id]:
                    self.cluster_to_phrases[cluster_id].append(phrase)

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for matching.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Remove punctuation at boundaries (but keep internal punctuation)
        text = text.strip('.,!?;:')

        return text

    def _split_into_ngrams(
        self,
        text: str,
        max_n: int
    ) -> List[Tuple[int, int, str]]:
        """
        Split text into n-grams.

        Args:
            text: Input text
            max_n: Maximum n-gram size

        Returns:
            List of (start_char, end_char, ngram_text) tuples
        """
        # Simple word-based n-grams
        words = text.split()
        ngrams = []

        for i in range(len(words)):
            for n in range(1, min(max_n + 1, len(words) - i + 1)):
                # Calculate character positions
                start_idx = len(' '.join(words[:i]))
                if i > 0:
                    start_idx += 1  # Account for space

                end_idx = start_idx + len(' '.join(words[i:i+n]))

                ngram_text = ' '.join(words[i:i+n])
                ngrams.append((start_idx, end_idx, ngram_text))

        return ngrams

    def match_phrases(self, text: str) -> List[PhraseSpan]:
        """
        Find phrase spans in text.

        Args:
            text: Input text to search

        Returns:
            List of detected phrase spans
        """
        spans = []

        # Generate n-grams
        ngrams = self._split_into_ngrams(text, self.max_span_length)

        # Check each n-gram for matches
        for start_char, end_char, ngram_text in ngrams:
            normalized = self._normalize_text(ngram_text)

            cluster_id = self.phrase_to_cluster.get(normalized)
            if cluster_id:
                # Found a match
                span = PhraseSpan(
                    start_char=start_char,
                    end_char=end_char,
                    cluster_id=cluster_id,
                    confidence=1.0,  # Exact match
                    original_text=ngram_text,
                )
                spans.append(span)

        # Remove overlaps if not allowed
        if not self.allow_overlaps:
            spans = self._remove_overlaps(spans)

        # Filter by confidence
        spans = [s for s in spans if s.confidence >= self.min_confidence]

        return spans

    def _remove_overlaps(self, spans: List[PhraseSpan]) -> List[PhraseSpan]:
        """
        Remove overlapping spans, keeping longer/higher confidence ones.

        Args:
            spans: List of phrase spans

        Returns:
            Non-overlapping spans
        """
        if not spans:
            return []

        # Sort by length (descending), then confidence (descending)
        spans = sorted(
            spans,
            key=lambda s: (s.end_char - s.start_char, s.confidence),
            reverse=True
        )

        # Greedy selection of non-overlapping spans
        selected = []
        for span in spans:
            # Check if it overlaps with any selected span
            overlaps = False
            for selected_span in selected:
                if (
                    span.start_char < selected_span.end_char and
                    span.end_char > selected_span.start_char
                ):
                    overlaps = True
                    break

            if not overlaps:
                selected.append(span)

        # Sort by position
        selected = sorted(selected, key=lambda s: s.start_char)

        return selected

    def get_cluster_phrases(self, cluster_id: str) -> List[str]:
        """
        Get all phrases for a cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            List of phrases
        """
        return self.cluster_to_phrases.get(cluster_id, [])

    def __len__(self) -> int:
        """Return number of indexed phrases."""
        return len(self.phrase_to_cluster)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PhraseMatcher("
            f"phrases={len(self.phrase_to_cluster)}, "
            f"clusters={len(self.cluster_to_phrases)}, "
            f"max_span={self.max_span_length})"
        )


class ApproximatePhraseMatcher(PhraseMatcher):
    """
    Phrase matcher with approximate/embedding-based matching.

    This is a placeholder for future approximate matching functionality.
    """

    def __init__(
        self,
        cluster_metadata: Dict,
        embedding_model: Optional[str] = None,
        similarity_threshold: float = 0.85,
        **kwargs
    ):
        """
        Initialize approximate matcher.

        Args:
            cluster_metadata: Cluster metadata
            embedding_model: Name of embedding model to use
            similarity_threshold: Minimum similarity for match
            **kwargs: Additional arguments for PhraseMatcher
        """
        super().__init__(cluster_metadata, **kwargs)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

        # TODO: Load embedding model and build index
        # self._build_embedding_index()

    def match_phrases(self, text: str) -> List[PhraseSpan]:
        """
        Find phrase spans using approximate matching.

        Args:
            text: Input text

        Returns:
            List of phrase spans
        """
        # For now, fall back to exact matching
        # TODO: Implement embedding-based approximate matching
        return super().match_phrases(text)


if __name__ == "__main__":
    # Example usage
    print("Phrase Matcher Example")
    print("=" * 60)

    # Create matcher with mock data
    mock_metadata = {
        "cluster_0": {
            "representative_phrase": "Can you send me that file?",
            "example_phrases": [
                "Can you send me that file?",
                "Mind emailing the document?",
                "Please share that file with me",
                "send me that file",
            ],
        },
        "cluster_1": {
            "representative_phrase": "Thank you very much",
            "example_phrases": [
                "Thank you very much",
                "Thanks a lot",
                "Many thanks",
                "thank you",
            ],
        },
        "cluster_2": {
            "representative_phrase": "I don't understand",
            "example_phrases": [
                "I don't understand",
                "I'm confused",
                "I don't get it",
            ],
        },
    }

    matcher = PhraseMatcher(
        cluster_metadata=mock_metadata,
        min_confidence=0.9,
        max_span_length=8,
        allow_overlaps=False,
    )

    print(f"\nMatcher: {matcher}")

    # Test matching
    test_texts = [
        "Can you send me that file? Thank you!",
        "I'm confused about this. Can someone help?",
        "Please share that file with me. Thanks a lot!",
    ]

    for text in test_texts:
        print(f"\nText: {text}")
        spans = matcher.match_phrases(text)
        print(f"Found {len(spans)} phrase spans:")
        for span in spans:
            print(f"  [{span.start_char}:{span.end_char}] "
                  f"'{span.original_text}' → cluster_{span.cluster_id} "
                  f"(conf: {span.confidence:.2f})")
