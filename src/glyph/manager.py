"""Glyph ID manager for encoding/decoding cluster IDs as Mandarin character strings."""

import msgpack
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger


class GlyphManager:
    """Manage glyph IDs using Mandarin characters as a symbol alphabet."""

    def __init__(self, config: Dict):
        """
        Initialize glyph manager.

        Args:
            config: Configuration dictionary with glyph settings
        """
        self.config = config
        self.alphabet_path = config.get("alphabet_path", "src/glyph/mandarin_alphabet.txt")
        self.max_glyph_length = config.get("max_glyph_length", 4)

        # Load or generate alphabet
        self.alphabet = self._load_alphabet()
        self.base = len(self.alphabet)

        # Create reverse mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}

        # Mappings
        self.cluster_to_glyph: Dict[int, str] = {}
        self.glyph_to_cluster: Dict[str, int] = {}

        logger.info(f"Glyph manager initialized with {self.base} characters")
        logger.info(f"Max addressable clusters: {self.base ** self.max_glyph_length:,}")

    def _load_alphabet(self) -> List[str]:
        """
        Load Mandarin character alphabet.

        Returns:
            List of characters
        """
        alphabet_path = Path(self.alphabet_path)

        if alphabet_path.exists():
            logger.info(f"Loading alphabet from {alphabet_path}")
            with open(alphabet_path, "r", encoding="utf-8") as f:
                chars = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(chars)} characters")
            return chars
        else:
            # Generate default alphabet
            logger.info("Generating default alphabet")
            return self._generate_default_alphabet()

    def _generate_default_alphabet(self) -> List[str]:
        """
        Generate default Mandarin character alphabet.

        Returns:
            List of characters
        """
        # Use common, visually distinct Mandarin characters
        # Unicode ranges for CJK unified ideographs
        # We'll select a subset of ~3000 high-frequency characters

        chars = []
        # Start with basic CJK ideographs
        start = 0x4E00  # Start of CJK Unified Ideographs
        default_size = self.config.get("default_alphabet_size", 3000)

        for i in range(default_size):
            char = chr(start + i)
            chars.append(char)

        logger.info(f"Generated alphabet with {len(chars)} characters")

        # Save for future use
        alphabet_path = Path(self.alphabet_path)
        alphabet_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alphabet_path, "w", encoding="utf-8") as f:
            for char in chars:
                f.write(char + "\n")
        logger.info(f"Saved alphabet to {alphabet_path}")

        return chars

    def assign_glyphs(self, n_clusters: int) -> None:
        """
        Assign glyph IDs to clusters.

        Args:
            n_clusters: Number of clusters
        """
        logger.info(f"Assigning glyphs to {n_clusters} clusters...")

        for cluster_id in range(n_clusters):
            glyph_string = self.encode_glyph_id(cluster_id)
            self.cluster_to_glyph[cluster_id] = glyph_string
            self.glyph_to_cluster[glyph_string] = cluster_id

        logger.info("Glyph assignment complete!")
        logger.info(f"Example mappings:")
        for i in range(min(5, n_clusters)):
            logger.info(f"  Cluster {i} → '{self.cluster_to_glyph[i]}'")

    def encode_glyph_id(self, glyph_id: int) -> str:
        """
        Encode integer ID as Mandarin character string.

        Args:
            glyph_id: Integer cluster ID

        Returns:
            Glyph string (1-4 characters)
        """
        if glyph_id < 0:
            raise ValueError("Glyph ID must be non-negative")

        if glyph_id == 0:
            return self.alphabet[0]

        chars = []
        remaining = glyph_id

        while remaining > 0:
            chars.append(self.alphabet[remaining % self.base])
            remaining //= self.base

        # Reverse to get big-endian encoding
        glyph_string = ''.join(reversed(chars))

        if len(glyph_string) > self.max_glyph_length:
            raise ValueError(f"Glyph ID {glyph_id} exceeds max length {self.max_glyph_length}")

        return glyph_string

    def decode_glyph_string(self, glyph_str: str) -> int:
        """
        Decode Mandarin character string to integer ID.

        Args:
            glyph_str: Glyph string

        Returns:
            Integer cluster ID
        """
        glyph_id = 0

        for char in glyph_str:
            if char not in self.char_to_idx:
                raise ValueError(f"Invalid glyph character: {char}")
            glyph_id = glyph_id * self.base + self.char_to_idx[char]

        return glyph_id

    def get_cluster_id(self, glyph_str: str) -> Optional[int]:
        """
        Get cluster ID from glyph string.

        Args:
            glyph_str: Glyph string

        Returns:
            Cluster ID or None if not found
        """
        return self.glyph_to_cluster.get(glyph_str)

    def get_glyph_string(self, cluster_id: int) -> Optional[str]:
        """
        Get glyph string from cluster ID.

        Args:
            cluster_id: Cluster ID

        Returns:
            Glyph string or None if not found
        """
        return self.cluster_to_glyph.get(cluster_id)

    def save_mapping(self, output_path: str) -> None:
        """
        Save glyph mapping to file.

        Args:
            output_path: Path to save mapping
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mapping_data = {
            "cluster_to_glyph": self.cluster_to_glyph,
            "glyph_to_cluster": self.glyph_to_cluster,
            "alphabet_size": self.base,
            "max_glyph_length": self.max_glyph_length,
        }

        with open(output_path, "wb") as f:
            msgpack.pack(mapping_data, f)

        logger.info(f"Saved glyph mapping to {output_path}")

    def load_mapping(self, input_path: str) -> None:
        """
        Load glyph mapping from file.

        Args:
            input_path: Path to mapping file
        """
        with open(input_path, "rb") as f:
            mapping_data = msgpack.unpack(f, raw=False)

        self.cluster_to_glyph = mapping_data["cluster_to_glyph"]
        self.glyph_to_cluster = mapping_data["glyph_to_cluster"]

        # Convert string keys back to integers for cluster_to_glyph
        self.cluster_to_glyph = {int(k): v for k, v in self.cluster_to_glyph.items()}

        logger.info(f"Loaded glyph mapping for {len(self.cluster_to_glyph)} clusters")
