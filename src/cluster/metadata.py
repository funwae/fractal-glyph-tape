"""Extract and manage cluster metadata."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
from loguru import logger


class ClusterMetadata:
    """Extract metadata from phrase clusters."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cluster metadata extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.top_phrases = config.get("metadata", {}).get("top_phrases_per_cluster", 10)
        self.compute_stats = config.get("metadata", {}).get("compute_statistics", True)

        self.metadata = {}

    def extract_metadata(
        self,
        phrases_file: str,
        labels: np.ndarray,
        centroids: np.ndarray,
        embeddings: np.ndarray,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract metadata for all clusters.

        Args:
            phrases_file: Path to phrases.jsonl file
            labels: Cluster labels for each phrase
            centroids: Cluster centroid vectors
            embeddings: Phrase embeddings

        Returns:
            Dictionary mapping cluster_id to metadata
        """
        logger.info("Extracting cluster metadata...")

        # Group phrases by cluster
        cluster_phrases = defaultdict(list)
        phrase_idx = 0

        with open(phrases_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                cluster_id = int(labels[phrase_idx])
                cluster_phrases[cluster_id].append({
                    "phrase_id": data["phrase_id"],
                    "text": data["text"],
                    "embedding_idx": phrase_idx,
                })
                phrase_idx += 1

        # Extract metadata for each cluster
        logger.info(f"Processing {len(cluster_phrases)} clusters...")
        for cluster_id in range(len(centroids)):
            phrases = cluster_phrases.get(cluster_id, [])

            if not phrases:
                # Empty cluster
                self.metadata[cluster_id] = {
                    "cluster_id": cluster_id,
                    "size": 0,
                    "examples": [],
                }
                continue

            # Get example phrases (most representative ones)
            examples = self._get_representative_phrases(
                phrases,
                centroids[cluster_id],
                embeddings,
                limit=self.top_phrases,
            )

            # Compute statistics
            stats = {}
            if self.compute_stats:
                stats = self._compute_cluster_stats(
                    phrases,
                    centroids[cluster_id],
                    embeddings,
                )

            self.metadata[cluster_id] = {
                "cluster_id": cluster_id,
                "size": len(phrases),
                "examples": examples,
                **stats,
            }

        logger.info("Metadata extraction complete!")
        return self.metadata

    def _get_representative_phrases(
        self,
        phrases: List[Dict[str, Any]],
        centroid: np.ndarray,
        embeddings: np.ndarray,
        limit: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Get most representative phrases for a cluster.

        Args:
            phrases: List of phrase dictionaries
            centroid: Cluster centroid vector
            embeddings: All embeddings
            limit: Maximum number of examples

        Returns:
            List of representative phrase examples
        """
        # Compute distances to centroid
        distances = []
        for phrase in phrases:
            emb = embeddings[phrase["embedding_idx"]]
            dist = np.linalg.norm(emb - centroid)
            distances.append((dist, phrase))

        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[0])

        # Return top examples
        return [
            {"phrase_id": phrase["phrase_id"], "text": phrase["text"]}
            for _, phrase in distances[:limit]
        ]

    def _compute_cluster_stats(
        self,
        phrases: List[Dict[str, Any]],
        centroid: np.ndarray,
        embeddings: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute statistics for a cluster.

        Args:
            phrases: List of phrase dictionaries
            centroid: Cluster centroid vector
            embeddings: All embeddings

        Returns:
            Dictionary of statistics
        """
        # Get all embeddings for this cluster
        cluster_embeddings = np.array([
            embeddings[phrase["embedding_idx"]]
            for phrase in phrases
        ])

        # Centroid norm
        centroid_norm = float(np.linalg.norm(centroid))

        # Intra-cluster distance (average distance to centroid)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        avg_distance = float(distances.mean())
        std_distance = float(distances.std())

        # Coherence score (negative of average distance)
        coherence = -avg_distance

        return {
            "centroid_norm": centroid_norm,
            "intra_cluster_distance": avg_distance,
            "distance_std": std_distance,
            "coherence_score": coherence,
        }

    def save_metadata(self, output_path: str) -> None:
        """
        Save metadata to JSON file.

        Args:
            output_path: Path to save metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved cluster metadata to {output_path}")

    def load_metadata(self, input_path: str) -> None:
        """
        Load metadata from JSON file.

        Args:
            input_path: Path to metadata file
        """
        with open(input_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Convert string keys back to integers
        self.metadata = {int(k): v for k, v in self.metadata.items()}
        logger.info(f"Loaded metadata for {len(self.metadata)} clusters")
