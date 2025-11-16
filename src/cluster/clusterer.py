"""Phrase clustering using K-means."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.cluster import MiniBatchKMeans
from loguru import logger


class PhraseClusterer:
    """Cluster phrase embeddings into semantic families."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize phrase clusterer.

        Args:
            config: Configuration dictionary with clustering settings
        """
        self.config = config
        self.method = config.get("method", "minibatch_kmeans")
        self.n_clusters = config.get("n_clusters", 10000)
        self.batch_size = config.get("batch_size", 1000)
        self.max_iter = config.get("max_iter", 100)
        self.n_init = config.get("n_init", 3)
        self.random_state = 42

        self.model = None
        self.labels = None
        self.centroids = None

    def cluster_embeddings(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Cluster embeddings using MiniBatchKMeans.

        Args:
            embeddings: NumPy array of embeddings (n_samples, embedding_dim)

        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Clustering {len(embeddings)} embeddings into {self.n_clusters} clusters...")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")

        if self.method == "minibatch_kmeans":
            self.model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
                verbose=1,
            )
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

        # Fit and predict
        logger.info("Fitting clustering model...")
        self.labels = self.model.fit_predict(embeddings)
        self.centroids = self.model.cluster_centers_

        # Compute statistics
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        logger.info(f"Clustering complete!")
        logger.info(f"Number of clusters: {len(unique_labels)}")
        logger.info(f"Average cluster size: {counts.mean():.1f}")
        logger.info(f"Min cluster size: {counts.min()}")
        logger.info(f"Max cluster size: {counts.max()}")

        return {
            "n_clusters": len(unique_labels),
            "cluster_sizes": counts.tolist(),
            "inertia": float(self.model.inertia_),
        }

    def save_results(self, output_path: str) -> None:
        """
        Save clustering results to disk.

        Args:
            output_path: Directory to save results
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save labels
        labels_path = output_path / "labels.npy"
        np.save(labels_path, self.labels)
        logger.info(f"Saved labels to {labels_path}")

        # Save centroids
        centroids_path = output_path / "centroids.npy"
        np.save(centroids_path, self.centroids)
        logger.info(f"Saved centroids to {centroids_path}")

        # Save metadata
        metadata = {
            "n_clusters": self.n_clusters,
            "method": self.method,
            "n_samples": len(self.labels),
            "embedding_dim": self.centroids.shape[1],
        }
        metadata_path = output_path / "cluster_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

    def load_results(self, input_path: str) -> None:
        """
        Load clustering results from disk.

        Args:
            input_path: Directory containing clustering results
        """
        input_path = Path(input_path)

        # Load labels
        self.labels = np.load(input_path / "labels.npy")
        logger.info(f"Loaded labels: {self.labels.shape}")

        # Load centroids
        self.centroids = np.load(input_path / "centroids.npy")
        logger.info(f"Loaded centroids: {self.centroids.shape}")

        # Load metadata
        with open(input_path / "cluster_info.json", "r") as f:
            metadata = json.load(f)
        self.n_clusters = metadata["n_clusters"]
        logger.info(f"Loaded clustering results for {self.n_clusters} clusters")

    def compute_coherence(self, embeddings: np.ndarray) -> float:
        """
        Compute cluster coherence using silhouette score.

        Args:
            embeddings: NumPy array of embeddings

        Returns:
            Silhouette score (higher is better)
        """
        from sklearn.metrics import silhouette_score

        logger.info("Computing cluster coherence...")
        # Use a sample for efficiency if dataset is large
        if len(embeddings) > 10000:
            logger.info("Using sample of 10k embeddings for coherence computation")
            indices = np.random.choice(len(embeddings), 10000, replace=False)
            sample_embeddings = embeddings[indices]
            sample_labels = self.labels[indices]
        else:
            sample_embeddings = embeddings
            sample_labels = self.labels

        score = silhouette_score(sample_embeddings, sample_labels, sample_size=10000)
        logger.info(f"Silhouette score: {score:.4f}")
        return float(score)
