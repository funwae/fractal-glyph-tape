"""Phrase embedding using sentence transformers."""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer


class PhraseEmbedder:
    """Embed phrases using sentence transformers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize phrase embedder.

        Args:
            config: Configuration dictionary with embedding settings
        """
        self.config = config
        self.model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = config.get("batch_size", 128)
        self.max_seq_length = config.get("max_seq_length", 256)
        self.shard_size = config.get("shard_size", 10000)

        # Determine device
        device_name = config.get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device_name = "cpu"
        self.device = device_name

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, 'max_seq_length'):
            self.model.max_seq_length = self.max_seq_length

        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed_phrases_from_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Embed all phrases from a JSONL file.

        Args:
            input_path: Path to phrases.jsonl file
            output_path: Directory to save embeddings

        Returns:
            Metadata dictionary with statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # First pass: count phrases
        logger.info("Counting phrases...")
        phrase_count = sum(1 for _ in open(input_path, "r"))
        logger.info(f"Total phrases to embed: {phrase_count}")

        # Read phrases and embed in batches
        phrases = []
        phrase_ids = []
        all_embeddings = []
        shard_num = 0

        logger.info("Starting embedding process...")
        with open(input_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=phrase_count, desc="Embedding phrases"):
                data = json.loads(line)
                phrases.append(data["text"])
                phrase_ids.append(data["phrase_id"])

                # Process in batches
                if len(phrases) >= self.batch_size:
                    embeddings = self._embed_batch(phrases)
                    all_embeddings.append(embeddings)
                    phrases = []

                # Save shard when reaching shard size
                if len(all_embeddings) * self.batch_size >= self.shard_size:
                    self._save_shard(all_embeddings, shard_num, output_path)
                    shard_num += 1
                    all_embeddings = []

            # Process remaining phrases
            if phrases:
                embeddings = self._embed_batch(phrases)
                all_embeddings.append(embeddings)

            # Save final shard
            if all_embeddings:
                self._save_shard(all_embeddings, shard_num, output_path)
                shard_num += 1

        # Save phrase index mapping
        logger.info("Saving phrase index...")
        index_data = {
            "phrase_ids": phrase_ids,
            "total_phrases": phrase_count,
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
            "model_name": self.model_name,
            "num_shards": shard_num,
        }

        with open(output_path / "phrase_index.json", "w") as f:
            json.dump(index_data, f, indent=2)

        logger.info("Embedding complete!")
        logger.info(f"Total embeddings: {phrase_count}")
        logger.info(f"Num shards: {shard_num}")
        logger.info(f"Output directory: {output_path}")

        return index_data

    def _embed_batch(self, phrases: List[str]) -> np.ndarray:
        """
        Embed a batch of phrases.

        Args:
            phrases: List of phrase strings

        Returns:
            NumPy array of embeddings
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                phrases,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return embeddings

    def _save_shard(self, embeddings_list: List[np.ndarray], shard_num: int, output_path: Path) -> None:
        """
        Save embeddings shard to disk.

        Args:
            embeddings_list: List of embedding arrays
            shard_num: Shard number
            output_path: Output directory
        """
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings_list)

        # Save as .npy file
        shard_path = output_path / f"shard_{shard_num:04d}.npy"
        np.save(shard_path, embeddings)
        logger.debug(f"Saved shard {shard_num} with {len(embeddings)} embeddings")

    def load_embeddings(self, embeddings_path: str) -> np.ndarray:
        """
        Load all embeddings from shards.

        Args:
            embeddings_path: Path to embeddings directory

        Returns:
            Concatenated embeddings array
        """
        embeddings_path = Path(embeddings_path)

        # Load index
        with open(embeddings_path / "phrase_index.json", "r") as f:
            index_data = json.load(f)

        # Load all shards
        logger.info(f"Loading {index_data['num_shards']} embedding shards...")
        shards = []
        for i in range(index_data["num_shards"]):
            shard_path = embeddings_path / f"shard_{i:04d}.npy"
            shard = np.load(shard_path)
            shards.append(shard)

        embeddings = np.vstack(shards)
        logger.info(f"Loaded embeddings: {embeddings.shape}")

        return embeddings
