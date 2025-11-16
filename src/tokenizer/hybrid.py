"""Hybrid tokenizer that combines regular tokens with FGT glyph tokens."""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer


class HybridTokenizer:
    """
    Hybrid tokenizer that encodes text as a mix of regular tokens and glyph tokens.

    For each phrase in the input:
    1. Embed the phrase
    2. Find nearest cluster in tape
    3. If similarity > threshold, use glyph token
    4. Otherwise, use regular tokens
    """

    # Special token markers
    GLYPH_START = "<GLYPH>"
    GLYPH_END = "</GLYPH>"

    def __init__(
        self,
        base_tokenizer: str = "gpt2",
        tape_db_path: Optional[str] = None,
        embedder_config: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.75,
        max_phrase_length: int = 100,
    ):
        """
        Initialize hybrid tokenizer.

        Args:
            base_tokenizer: Name or path of base tokenizer (e.g., 'gpt2', 'bert-base-uncased')
            tape_db_path: Path to tape database for glyph lookups
            embedder_config: Configuration for phrase embedder
            similarity_threshold: Minimum similarity to use glyph (0-1)
            max_phrase_length: Maximum phrase length to attempt glyph encoding
        """
        logger.info(f"Initializing HybridTokenizer with base: {base_tokenizer}")

        # Load base tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)

        # Add special tokens
        special_tokens = {"additional_special_tokens": [self.GLYPH_START, self.GLYPH_END]}
        num_added = self.base_tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added} special tokens")

        # Get special token IDs
        self.glyph_start_id = self.base_tokenizer.convert_tokens_to_ids(self.GLYPH_START)
        self.glyph_end_id = self.base_tokenizer.convert_tokens_to_ids(self.GLYPH_END)

        # Configuration
        self.tape_db_path = tape_db_path
        self.embedder_config = embedder_config or {}
        self.similarity_threshold = similarity_threshold
        self.max_phrase_length = max_phrase_length

        # Lazy-loaded components
        self._embedder = None
        self._tape_index = None
        self._centroids = None
        self._glyph_mapping = None

        logger.info("HybridTokenizer initialized")

    def _load_embedder(self):
        """Lazy-load the phrase embedder."""
        if self._embedder is None:
            from embed import PhraseEmbedder

            # Use default config if not provided
            if not self.embedder_config:
                self.embedder_config = {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "cpu"
                }

            self._embedder = PhraseEmbedder(self.embedder_config)
            logger.info("Loaded phrase embedder")

        return self._embedder

    def _load_tape_index(self):
        """Lazy-load the tape index."""
        if self._tape_index is None:
            import sqlite3

            if not self.tape_db_path or not Path(self.tape_db_path).exists():
                raise ValueError(f"Tape database not found: {self.tape_db_path}")

            conn = sqlite3.connect(self.tape_db_path)
            cursor = conn.cursor()

            # Load all centroids and glyph mappings
            cursor.execute("SELECT cluster_id, centroid FROM clusters ORDER BY cluster_id")
            rows = cursor.fetchall()

            centroids = []
            cluster_ids = []
            for cluster_id, centroid_blob in rows:
                # Convert blob back to numpy array
                centroid = np.frombuffer(centroid_blob, dtype=np.float32)
                centroids.append(centroid)
                cluster_ids.append(cluster_id)

            self._centroids = np.array(centroids)

            # Load glyph mapping
            cursor.execute("SELECT cluster_id, glyph_string FROM glyphs ORDER BY cluster_id")
            glyph_rows = cursor.fetchall()
            self._glyph_mapping = {cluster_id: glyph for cluster_id, glyph in glyph_rows}

            conn.close()

            logger.info(f"Loaded {len(centroids)} cluster centroids")

        return self._centroids, self._glyph_mapping

    def _find_nearest_cluster(self, embedding: np.ndarray) -> Tuple[int, float]:
        """
        Find nearest cluster for an embedding.

        Args:
            embedding: Phrase embedding vector

        Returns:
            (cluster_id, similarity) tuple
        """
        centroids, _ = self._load_tape_index()

        # Compute cosine similarity to all centroids
        # Normalize vectors
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)

        # Cosine similarity
        similarities = np.dot(centroids_norm, embedding_norm)

        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        return int(best_idx), float(best_similarity)

    def _encode_as_glyph(self, glyph_string: str) -> List[int]:
        """
        Encode a glyph string as token IDs.

        Format: [GLYPH_START, glyph_char_tokens..., GLYPH_END]

        Args:
            glyph_string: Glyph string (Mandarin characters)

        Returns:
            List of token IDs
        """
        # Tokenize the glyph string itself
        glyph_tokens = self.base_tokenizer.encode(
            glyph_string,
            add_special_tokens=False
        )

        # Wrap in special markers
        return [self.glyph_start_id] + glyph_tokens + [self.glyph_end_id]

    def encode_hybrid(
        self,
        text: str,
        segment_phrases: bool = True,
        return_details: bool = False
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Encode text as hybrid tokens (mix of regular and glyph tokens).

        Args:
            text: Input text
            segment_phrases: Whether to segment into phrases first
            return_details: If True, return detailed info about encoding

        Returns:
            List of token IDs, or dict with details if return_details=True
        """
        if segment_phrases:
            # Segment into phrases
            from ingest import Segmenter
            segmenter = Segmenter(method="regex")
            phrases = segmenter.segment(text)
        else:
            # Treat entire text as one phrase
            phrases = [text]

        embedder = self._load_embedder()
        token_ids = []
        details = {
            "total_phrases": len(phrases),
            "glyph_encoded": 0,
            "regular_encoded": 0,
            "encoding_decisions": []
        }

        for phrase in phrases:
            # Skip very short or very long phrases
            if len(phrase) < 10 or len(phrase) > self.max_phrase_length:
                # Use regular tokens
                phrase_tokens = self.base_tokenizer.encode(phrase, add_special_tokens=False)
                token_ids.extend(phrase_tokens)
                details["regular_encoded"] += 1
                details["encoding_decisions"].append({
                    "phrase": phrase[:50],
                    "method": "regular",
                    "reason": "length_filter"
                })
                continue

            # Embed the phrase
            embedding = embedder._embed_batch([phrase])[0]

            # Find nearest cluster
            cluster_id, similarity = self._find_nearest_cluster(embedding)

            # Decide: glyph or regular tokens?
            if similarity >= self.similarity_threshold:
                # Use glyph encoding
                _, glyph_mapping = self._load_tape_index()
                glyph_string = glyph_mapping.get(cluster_id, "")

                if glyph_string:
                    glyph_tokens = self._encode_as_glyph(glyph_string)
                    token_ids.extend(glyph_tokens)
                    details["glyph_encoded"] += 1
                    details["encoding_decisions"].append({
                        "phrase": phrase[:50],
                        "method": "glyph",
                        "glyph": glyph_string,
                        "similarity": similarity,
                        "cluster_id": cluster_id
                    })
                else:
                    # Fallback to regular
                    phrase_tokens = self.base_tokenizer.encode(phrase, add_special_tokens=False)
                    token_ids.extend(phrase_tokens)
                    details["regular_encoded"] += 1
                    details["encoding_decisions"].append({
                        "phrase": phrase[:50],
                        "method": "regular",
                        "reason": "no_glyph_found"
                    })
            else:
                # Use regular tokens
                phrase_tokens = self.base_tokenizer.encode(phrase, add_special_tokens=False)
                token_ids.extend(phrase_tokens)
                details["regular_encoded"] += 1
                details["encoding_decisions"].append({
                    "phrase": phrase[:50],
                    "method": "regular",
                    "similarity": similarity,
                    "reason": "low_similarity"
                })

        if return_details:
            details["token_ids"] = token_ids
            details["token_count"] = len(token_ids)
            return details
        else:
            return token_ids

    def decode_hybrid(self, token_ids: List[int]) -> str:
        """
        Decode hybrid token IDs back to text.

        Args:
            token_ids: List of token IDs (mix of regular and glyph tokens)

        Returns:
            Decoded text
        """
        # Simple approach: decode directly
        # The base tokenizer will handle glyph markers as tokens
        text = self.base_tokenizer.decode(token_ids, skip_special_tokens=False)

        # Could enhance this to expand glyphs back to representative phrases
        # For now, just return the decoded text with glyph markers visible

        return text

    def decode_hybrid_with_expansion(self, token_ids: List[int]) -> str:
        """
        Decode hybrid tokens and expand glyphs to representative phrases.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text with glyphs expanded
        """
        from tape import TapeStorage

        # Decode to get raw text with markers
        text = self.base_tokenizer.decode(token_ids, skip_special_tokens=False)

        # Find glyph sections
        import re
        glyph_pattern = re.compile(rf"{self.GLYPH_START}(.*?){self.GLYPH_END}")

        def replace_glyph(match):
            glyph_text = match.group(1).strip()

            # Look up representative phrase for this glyph
            try:
                with TapeStorage(self.tape_db_path) as storage:
                    storage.connect()
                    cluster_info = storage.get_cluster_by_glyph(glyph_text)

                    if cluster_info and cluster_info.get("metadata"):
                        metadata = cluster_info["metadata"]
                        if isinstance(metadata, str):
                            metadata = eval(metadata)
                        examples = metadata.get("examples", [])
                        if examples:
                            return examples[0].get("text", glyph_text)

                    return glyph_text
            except:
                return glyph_text

        # Replace all glyphs with representative phrases
        expanded_text = glyph_pattern.sub(replace_glyph, text)

        return expanded_text

    def get_vocab_size(self) -> int:
        """Get vocabulary size (including special tokens)."""
        return len(self.base_tokenizer)

    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        self.base_tokenizer.save_pretrained(save_directory)
        logger.info(f"Saved tokenizer to {save_directory}")

    def __len__(self):
        """Return vocabulary size."""
        return self.get_vocab_size()
