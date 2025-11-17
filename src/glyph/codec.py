"""Glyph codec for encoding and decoding cluster IDs to glyph characters."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class GlyphCodec:
    """
    Converts between cluster IDs and glyph characters.

    This codec manages the mapping from:
    - Cluster IDs (internal identifiers) ↔ Glyph IDs (numeric indices)
    - Glyph IDs ↔ Unicode characters (visual glyphs)

    The glyph alphabet is based on a predefined Unicode range, typically
    using Chinese characters (U+4E00 - U+9FFF) for visual distinctiveness.
    """

    def __init__(
        self,
        cluster_metadata: Optional[Dict] = None,
        glyph_range_start: int = 0x4E00,  # CJK Unified Ideographs start
        glyph_range_end: int = 0x9FFF,    # CJK Unified Ideographs end
    ):
        """
        Initialize the glyph codec.

        Args:
            cluster_metadata: Dictionary mapping cluster_id → metadata
            glyph_range_start: Start of Unicode range for glyphs
            glyph_range_end: End of Unicode range for glyphs
        """
        self.cluster_metadata = cluster_metadata or {}
        self.glyph_range_start = glyph_range_start
        self.glyph_range_end = glyph_range_end

        # Maximum glyph ID based on available characters
        self.max_glyph_id = glyph_range_end - glyph_range_start

        # Build bidirectional mappings
        self.cluster_to_glyph_id: Dict[str, int] = {}
        self.glyph_id_to_cluster: Dict[int, str] = {}

        self._initialize_mappings()

    def _initialize_mappings(self):
        """Initialize cluster ↔ glyph_id mappings from metadata."""
        if not self.cluster_metadata:
            return

        for cluster_id, meta in self.cluster_metadata.items():
            # Check if metadata already has a glyph assigned
            if "glyph" in meta:
                glyph_char = meta["glyph"]
                glyph_id = self.unicode_to_glyph_id(glyph_char)
            else:
                # Assign sequential glyph IDs
                glyph_id = len(self.cluster_to_glyph_id)

            self.cluster_to_glyph_id[cluster_id] = glyph_id
            self.glyph_id_to_cluster[glyph_id] = cluster_id

    @classmethod
    def from_tape(cls, tape_dir: Path) -> "GlyphCodec":
        """
        Create codec from a tape directory.

        Args:
            tape_dir: Path to tape directory containing cluster metadata

        Returns:
            Initialized GlyphCodec
        """
        tape_dir = Path(tape_dir)
        metadata_file = tape_dir / "clusters" / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Cluster metadata not found: {metadata_file}")

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return cls(cluster_metadata=metadata)

    def glyph_id_to_unicode(self, glyph_id: int) -> str:
        """
        Convert glyph ID to Unicode character(s).

        Args:
            glyph_id: Numeric glyph identifier

        Returns:
            Unicode character(s) representing the glyph

        Raises:
            ValueError: If glyph_id is out of range
        """
        if glyph_id < 0 or glyph_id > self.max_glyph_id:
            raise ValueError(f"Glyph ID {glyph_id} out of range [0, {self.max_glyph_id}]")

        # Simple encoding: direct mapping to Unicode codepoint
        codepoint = self.glyph_range_start + glyph_id
        return chr(codepoint)

    def unicode_to_glyph_id(self, glyph_char: str) -> int:
        """
        Convert Unicode character to glyph ID.

        Args:
            glyph_char: Unicode character (single char)

        Returns:
            Numeric glyph identifier

        Raises:
            ValueError: If character is not in glyph range
        """
        if len(glyph_char) != 1:
            raise ValueError(f"Expected single character, got: {glyph_char}")

        codepoint = ord(glyph_char)

        if codepoint < self.glyph_range_start or codepoint > self.glyph_range_end:
            raise ValueError(
                f"Character '{glyph_char}' (U+{codepoint:04X}) not in glyph range "
                f"[U+{self.glyph_range_start:04X}, U+{self.glyph_range_end:04X}]"
            )

        return codepoint - self.glyph_range_start

    def cluster_to_glyph(self, cluster_id: str) -> str:
        """
        Get glyph character for a cluster ID.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Unicode glyph character

        Raises:
            KeyError: If cluster_id not found
        """
        glyph_id = self.cluster_to_glyph_id.get(cluster_id)

        if glyph_id is None:
            raise KeyError(f"Cluster ID not found: {cluster_id}")

        return self.glyph_id_to_unicode(glyph_id)

    def glyph_to_cluster(self, glyph_char: str) -> str:
        """
        Get cluster ID from glyph character.

        Args:
            glyph_char: Unicode glyph character

        Returns:
            Cluster identifier

        Raises:
            ValueError: If character is invalid
            KeyError: If no cluster assigned to this glyph
        """
        glyph_id = self.unicode_to_glyph_id(glyph_char)
        cluster_id = self.glyph_id_to_cluster.get(glyph_id)

        if cluster_id is None:
            raise KeyError(f"No cluster assigned to glyph '{glyph_char}' (ID: {glyph_id})")

        return cluster_id

    def get_cluster_info(self, cluster_id: str) -> Dict:
        """
        Get metadata for a cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Cluster metadata dictionary
        """
        return self.cluster_metadata.get(cluster_id, {})

    def get_representative_phrase(self, cluster_id: str) -> str:
        """
        Get representative phrase for a cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Representative phrase or empty string
        """
        meta = self.get_cluster_info(cluster_id)
        return meta.get("representative_phrase", "")

    def encode_text_span(self, cluster_id: str) -> Tuple[str, Dict]:
        """
        Encode a text span as a glyph with metadata.

        Args:
            cluster_id: Cluster identifier for the span

        Returns:
            Tuple of (glyph_character, metadata)
        """
        glyph = self.cluster_to_glyph(cluster_id)
        meta = {
            "cluster_id": cluster_id,
            "glyph": glyph,
            "type": "glyph",
        }
        return glyph, meta

    def decode_glyph(self, glyph_char: str, mode: str = "representative") -> str:
        """
        Decode glyph character to text.

        Args:
            glyph_char: Unicode glyph character
            mode: Decoding mode - 'representative', 'first_example', or 'random'

        Returns:
            Decoded text

        Raises:
            ValueError: If invalid mode or glyph
        """
        cluster_id = self.glyph_to_cluster(glyph_char)
        meta = self.get_cluster_info(cluster_id)

        if mode == "representative":
            return meta.get("representative_phrase", glyph_char)

        elif mode == "first_example":
            examples = meta.get("example_phrases", [])
            return examples[0] if examples else meta.get("representative_phrase", glyph_char)

        elif mode == "random":
            examples = meta.get("example_phrases", [])
            if examples:
                idx = np.random.randint(0, len(examples))
                return examples[idx]
            return meta.get("representative_phrase", glyph_char)

        else:
            raise ValueError(f"Unknown decoding mode: {mode}")

    def is_glyph_char(self, char: str) -> bool:
        """
        Check if a character is a valid glyph.

        Args:
            char: Character to check

        Returns:
            True if character is in glyph range
        """
        if len(char) != 1:
            return False

        codepoint = ord(char)
        return self.glyph_range_start <= codepoint <= self.glyph_range_end

    def extract_glyphs(self, text: str) -> list[Tuple[int, int, str]]:
        """
        Extract glyph characters from text.

        Args:
            text: Text potentially containing glyphs

        Returns:
            List of (start_pos, end_pos, glyph_char) tuples
        """
        glyphs = []

        for i, char in enumerate(text):
            if self.is_glyph_char(char):
                try:
                    # Verify it's a valid cluster glyph
                    self.glyph_to_cluster(char)
                    glyphs.append((i, i + 1, char))
                except KeyError:
                    # Glyph in range but not assigned to a cluster
                    pass

        return glyphs

    def __len__(self) -> int:
        """Return number of cluster-glyph mappings."""
        return len(self.cluster_to_glyph_id)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GlyphCodec("
            f"clusters={len(self.cluster_to_glyph_id)}, "
            f"range=U+{self.glyph_range_start:04X}-U+{self.glyph_range_end:04X})"
        )


if __name__ == "__main__":
    # Example usage
    print("Glyph Codec Example")
    print("=" * 60)

    # Create codec with mock data
    mock_metadata = {
        "cluster_0": {
            "glyph": "谷",
            "representative_phrase": "Can you send me that file?",
            "example_phrases": [
                "Can you send me that file?",
                "Mind emailing the document?",
                "Please share that file with me",
            ],
        },
        "cluster_1": {
            "glyph": "阜",
            "representative_phrase": "Thank you very much",
            "example_phrases": [
                "Thank you very much",
                "Thanks a lot",
                "Many thanks",
            ],
        },
    }

    codec = GlyphCodec(cluster_metadata=mock_metadata)

    print(f"\nCodec: {codec}")
    print(f"\nCluster → Glyph:")
    print(f"  cluster_0 → {codec.cluster_to_glyph('cluster_0')}")
    print(f"  cluster_1 → {codec.cluster_to_glyph('cluster_1')}")

    print(f"\nGlyph → Cluster:")
    print(f"  谷 → {codec.glyph_to_cluster('谷')}")
    print(f"  阜 → {codec.glyph_to_cluster('阜')}")

    print(f"\nGlyph → Text:")
    print(f"  谷 → '{codec.decode_glyph('谷')}'")
    print(f"  阜 → '{codec.decode_glyph('阜')}'")

    print(f"\nExtract glyphs from text:")
    text = "Hello 谷 world 阜"
    glyphs = codec.extract_glyphs(text)
    print(f"  Text: {text}")
    print(f"  Glyphs: {glyphs}")
