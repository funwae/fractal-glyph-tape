"""Hybrid tokenizer combining standard tokenization with FGT glyph tokens."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizer

from ..glyph.codec import GlyphCodec
from .phrase_matcher import PhraseMatcher, PhraseSpan


@dataclass
class TokenMetadata:
    """Metadata for a token."""

    token_type: str  # "raw" or "glyph"
    cluster_id: Optional[str] = None
    glyph_char: Optional[str] = None
    original_span: Optional[Tuple[int, int]] = None


@dataclass
class TextSegment:
    """Segment of text (raw or glyph)."""

    start_char: int
    end_char: int
    text: str
    segment_type: str  # "raw" or "glyph"
    cluster_id: Optional[str] = None


class HybridTokenizer:
    """
    Hybrid tokenizer that combines standard tokenization with FGT glyphs.

    This tokenizer:
    1. Detects phrase spans in input text
    2. Replaces detected spans with glyph characters
    3. Tokenizes the resulting mixed text
    4. Tracks metadata about which tokens are glyphs
    """

    def __init__(
        self,
        base_tokenizer: Union[str, PreTrainedTokenizer],
        glyph_codec: GlyphCodec,
        phrase_matcher: PhraseMatcher,
        use_glyph_markers: bool = False,
        glyph_marker_start: str = "<GLYPH>",
        glyph_marker_end: str = "</GLYPH>",
    ):
        """
        Initialize hybrid tokenizer.

        Args:
            base_tokenizer: HuggingFace tokenizer name or instance
            glyph_codec: Glyph codec for cluster ↔ character conversion
            phrase_matcher: Phrase matcher for detecting spans
            use_glyph_markers: Whether to wrap glyphs in special markers
            glyph_marker_start: Start marker for glyphs
            glyph_marker_end: End marker for glyphs
        """
        # Load base tokenizer
        if isinstance(base_tokenizer, str):
            self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
        else:
            self.base_tokenizer = base_tokenizer

        self.glyph_codec = glyph_codec
        self.phrase_matcher = phrase_matcher
        self.use_glyph_markers = use_glyph_markers
        self.glyph_marker_start = glyph_marker_start
        self.glyph_marker_end = glyph_marker_end

        # Add special tokens if using markers
        if use_glyph_markers:
            self._add_special_tokens()

    def _add_special_tokens(self):
        """Add glyph marker tokens to vocabulary."""
        special_tokens = {
            "additional_special_tokens": [
                self.glyph_marker_start,
                self.glyph_marker_end,
            ]
        }

        num_added = self.base_tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            print(f"Added {num_added} special tokens to vocabulary")

    def _segment_text_with_spans(
        self,
        text: str,
        spans: List[PhraseSpan],
    ) -> List[TextSegment]:
        """
        Segment text into raw and glyph segments.

        Args:
            text: Input text
            spans: Detected phrase spans

        Returns:
            List of text segments
        """
        segments = []
        current_pos = 0

        for span in spans:
            # Add raw segment before this span (if any)
            if current_pos < span.start_char:
                segments.append(TextSegment(
                    start_char=current_pos,
                    end_char=span.start_char,
                    text=text[current_pos:span.start_char],
                    segment_type="raw",
                ))

            # Add glyph segment
            glyph_char = self.glyph_codec.cluster_to_glyph(span.cluster_id)

            if self.use_glyph_markers:
                glyph_text = f"{self.glyph_marker_start}{glyph_char}{self.glyph_marker_end}"
            else:
                glyph_text = glyph_char

            segments.append(TextSegment(
                start_char=span.start_char,
                end_char=span.end_char,
                text=glyph_text,
                segment_type="glyph",
                cluster_id=span.cluster_id,
            ))

            current_pos = span.end_char

        # Add final raw segment (if any)
        if current_pos < len(text):
            segments.append(TextSegment(
                start_char=current_pos,
                end_char=len(text),
                text=text[current_pos:],
                segment_type="raw",
            ))

        return segments

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_metadata: bool = True,
        **kwargs
    ) -> Dict:
        """
        Encode text with glyph insertion.

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            return_metadata: Whether to return token metadata
            **kwargs: Additional arguments for base tokenizer

        Returns:
            Dictionary with:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - metadata: Token metadata (if return_metadata=True)
                - glyph_count: Number of glyph tokens
        """
        # Detect phrase spans
        spans = self.phrase_matcher.match_phrases(text)

        # Segment text
        segments = self._segment_text_with_spans(text, spans)

        # Encode segments
        all_token_ids = []
        all_metadata = []

        for segment in segments:
            # Tokenize segment
            segment_ids = self.base_tokenizer.encode(
                segment.text,
                add_special_tokens=False,
            )

            all_token_ids.extend(segment_ids)

            # Track metadata
            for token_id in segment_ids:
                meta = TokenMetadata(
                    token_type=segment.segment_type,
                    cluster_id=segment.cluster_id,
                    glyph_char=self.glyph_codec.cluster_to_glyph(segment.cluster_id)
                        if segment.cluster_id else None,
                    original_span=(segment.start_char, segment.end_char),
                )
                all_metadata.append(meta)

        # Add special tokens if requested
        if add_special_tokens:
            # Add BOS/EOS using base tokenizer's logic
            encoded = self.base_tokenizer.encode(
                "",  # Empty to get just special tokens
                add_special_tokens=True,
            )

            # Extract special tokens (usually just [BOS] and [EOS])
            # This is a simplified approach; might need adjustment per tokenizer
            if len(encoded) > 0:
                # Prepend BOS metadata
                all_metadata.insert(0, TokenMetadata(token_type="special"))
                all_token_ids.insert(0, encoded[0])

                if len(encoded) > 1:
                    # Append EOS metadata
                    all_metadata.append(TokenMetadata(token_type="special"))
                    all_token_ids.append(encoded[-1])

        # Create attention mask
        attention_mask = [1] * len(all_token_ids)

        # Count glyphs
        glyph_count = sum(1 for m in all_metadata if m.token_type == "glyph")

        result = {
            "input_ids": all_token_ids,
            "attention_mask": attention_mask,
            "glyph_count": glyph_count,
        }

        if return_metadata:
            result["metadata"] = all_metadata

        return result

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        expand_glyphs: bool = True,
        glyph_expansion_mode: str = "representative",
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            expand_glyphs: Whether to expand glyphs to phrases
            glyph_expansion_mode: How to expand glyphs ('representative', 'first_example', 'random')

        Returns:
            Decoded text
        """
        # Decode to raw text
        text = self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        if not expand_glyphs:
            return text

        # Find and expand glyph characters
        glyphs = self.glyph_codec.extract_glyphs(text)

        if not glyphs:
            return text

        # Replace glyphs with expanded text (in reverse to maintain positions)
        for start, end, glyph_char in reversed(glyphs):
            expanded = self.glyph_codec.decode_glyph(glyph_char, mode=glyph_expansion_mode)
            text = text[:start] + expanded + text[end:]

        # Clean up glyph markers if present
        if self.use_glyph_markers:
            text = text.replace(self.glyph_marker_start, "")
            text = text.replace(self.glyph_marker_end, "")

        return text

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        return_metadata: bool = False,
        **kwargs
    ) -> Dict:
        """
        Batch encode multiple texts.

        Args:
            texts: List of input texts
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            return_metadata: Whether to return token metadata
            **kwargs: Additional arguments

        Returns:
            Dictionary with batched encodings
        """
        # Encode each text
        encoded_batch = [
            self.encode(text, return_metadata=return_metadata, **kwargs)
            for text in texts
        ]

        # Find max length if padding
        if padding:
            if max_length is None:
                max_length = max(len(enc["input_ids"]) for enc in encoded_batch)

            # Pad sequences
            for enc in encoded_batch:
                padding_length = max_length - len(enc["input_ids"])
                if padding_length > 0:
                    enc["input_ids"].extend([self.base_tokenizer.pad_token_id] * padding_length)
                    enc["attention_mask"].extend([0] * padding_length)
                    if return_metadata:
                        enc["metadata"].extend([TokenMetadata(token_type="padding")] * padding_length)

        # Collate into tensors
        result = {
            "input_ids": [enc["input_ids"] for enc in encoded_batch],
            "attention_mask": [enc["attention_mask"] for enc in encoded_batch],
            "glyph_count": [enc["glyph_count"] for enc in encoded_batch],
        }

        if return_metadata:
            result["metadata"] = [enc["metadata"] for enc in encoded_batch]

        return result

    @classmethod
    def from_tape(
        cls,
        tape_dir: Path,
        base_tokenizer: str = "gpt2",
        **kwargs
    ) -> "HybridTokenizer":
        """
        Create hybrid tokenizer from a tape directory.

        Args:
            tape_dir: Path to tape directory
            base_tokenizer: Base tokenizer name
            **kwargs: Additional arguments for HybridTokenizer

        Returns:
            Initialized HybridTokenizer
        """
        # Load codec
        glyph_codec = GlyphCodec.from_tape(tape_dir)

        # Load phrase matcher
        import json
        metadata_file = Path(tape_dir) / "clusters" / "metadata.json"
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        phrase_matcher = PhraseMatcher(cluster_metadata=metadata)

        return cls(
            base_tokenizer=base_tokenizer,
            glyph_codec=glyph_codec,
            phrase_matcher=phrase_matcher,
            **kwargs
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HybridTokenizer("
            f"base={type(self.base_tokenizer).__name__}, "
            f"glyphs={len(self.glyph_codec)}, "
            f"phrases={len(self.phrase_matcher)})"
        )


if __name__ == "__main__":
    # Example usage
    print("Hybrid Tokenizer Example")
    print("=" * 60)

    # This example uses mock data - in practice, use from_tape()
    print("\nNote: This is a demo with mock data.")
    print("In practice, use: HybridTokenizer.from_tape('tape/v1')")
