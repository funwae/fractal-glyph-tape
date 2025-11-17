"""Phrase extraction and filtering."""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Set
from tqdm import tqdm
from loguru import logger

from .reader import CorpusReader
from .segmenter import Segmenter


class PhraseExtractor:
    """Extract and filter phrases from documents."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize phrase extractor.

        Args:
            config: Configuration dictionary with ingestion settings
        """
        self.config = config
        self.min_tokens = config.get("min_tokens", 3)
        self.max_tokens = config.get("max_tokens", 50)
        self.min_length = config.get("filters", {}).get("min_length", 10)
        self.max_length = config.get("filters", {}).get("max_length", 500)
        self.remove_duplicates = config.get("filters", {}).get("remove_duplicates", True)
        self.target_count = config.get("target_phrase_count", 100000)

        # Initialize components
        segmenter_config = config.get("segmenter", {})
        self.reader = CorpusReader(config["input_path"])
        self.segmenter = Segmenter(
            method=segmenter_config.get("method", "nltk"),
            language=segmenter_config.get("language", "english")
        )

        self.seen_hashes: Set[str] = set()

    def extract_phrases(self) -> None:
        """
        Extract phrases from corpus and save to output file.
        """
        output_path = Path(self.config["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        phrase_count = 0
        doc_count = 0
        skipped_count = 0

        logger.info("Starting phrase extraction...")
        logger.info(f"Target: {self.target_count} phrases")

        with open(output_path, "w", encoding="utf-8") as out_file:
            for doc in tqdm(self.reader.read_documents(), desc="Processing documents"):
                doc_count += 1
                text = doc["text"]

                # Segment into sentences
                sentences = self.segmenter.segment(text)

                for sentence in sentences:
                    # Apply filters
                    if not self._should_include(sentence):
                        skipped_count += 1
                        continue

                    # Check for duplicates
                    if self.remove_duplicates:
                        sentence_hash = self._hash_text(sentence)
                        if sentence_hash in self.seen_hashes:
                            skipped_count += 1
                            continue
                        self.seen_hashes.add(sentence_hash)

                    # Create phrase record
                    phrase_record = {
                        "phrase_id": f"{phrase_count:08d}",
                        "text": sentence,
                        "lang": "en",  # TODO: Add language detection
                        "doc_id": doc["doc_id"],
                        "metadata": doc.get("metadata", {}),
                    }

                    # Write to output
                    out_file.write(json.dumps(phrase_record) + "\n")
                    phrase_count += 1

                    # Check if we've reached target
                    if phrase_count >= self.target_count:
                        logger.info(f"Reached target count of {self.target_count} phrases")
                        break

                if phrase_count >= self.target_count:
                    break

        logger.info("Phrase extraction complete!")
        logger.info(f"Documents processed: {doc_count}")
        logger.info(f"Phrases extracted: {phrase_count}")
        logger.info(f"Phrases skipped: {skipped_count}")
        logger.info(f"Output saved to: {output_path}")

    def _should_include(self, text: str) -> bool:
        """
        Check if a phrase should be included based on filters.

        Args:
            text: Phrase text

        Returns:
            True if phrase should be included
        """
        # Length filters
        if len(text) < self.min_length or len(text) > self.max_length:
            return False

        # Token count filters
        token_count = self.segmenter.count_tokens(text)
        if token_count < self.min_tokens or token_count > self.max_tokens:
            return False

        # Basic quality filters
        # Skip if too many special characters
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
        if alpha_ratio < 0.7:
            return False

        return True

    def _hash_text(self, text: str) -> str:
        """
        Create hash of text for duplicate detection.

        Args:
            text: Input text

        Returns:
            MD5 hash string
        """
        # Normalize: lowercase and remove extra whitespace
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()
