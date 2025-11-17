"""2D layout computation for cluster visualization."""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class LayoutComputer:
    """Compute 2D layout coordinates for cluster visualization."""

    def __init__(self, method: str = "umap"):
        """
        Initialize layout computer.

        Args:
            method: Layout method - 'umap', 'pca', or 'fractal'
        """
        self.method = method

        if method == "umap" and not HAS_UMAP:
            raise ValueError("UMAP not available. Install umap-learn or use 'pca' method.")

    def compute_layout(
        self,
        embeddings: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Compute 2D layout from high-dimensional embeddings.

        Args:
            embeddings: Array of shape (n_clusters, embedding_dim)
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            random_state: Random seed

        Returns:
            2D coordinates array of shape (n_clusters, 2)
        """
        if self.method == "umap":
            return self._compute_umap_layout(
                embeddings, n_neighbors, min_dist, random_state
            )
        elif self.method == "pca":
            return self._compute_pca_layout(embeddings, random_state)
        elif self.method == "fractal":
            return self._compute_fractal_layout(embeddings)
        else:
            raise ValueError(f"Unknown layout method: {self.method}")

    def _compute_umap_layout(
        self,
        embeddings: np.ndarray,
        n_neighbors: int,
        min_dist: float,
        random_state: int,
    ) -> np.ndarray:
        """Compute layout using UMAP."""
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric="cosine",
        )

        layout = reducer.fit_transform(embeddings)
        return layout

    def _compute_pca_layout(
        self,
        embeddings: np.ndarray,
        random_state: int,
    ) -> np.ndarray:
        """Compute layout using PCA."""
        pca = PCA(n_components=2, random_state=random_state)
        layout = pca.fit_transform(embeddings)
        return layout

    def _compute_fractal_layout(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute fractal triangle layout.

        This places clusters on a triangular grid based on their hierarchical
        relationships, creating a fractal-like visualization.

        For now, we use PCA as a placeholder and scale to triangle.
        TODO: Implement proper hierarchical fractal layout.
        """
        # Use PCA as base
        layout = self._compute_pca_layout(embeddings, random_state=42)

        # Map to equilateral triangle coordinates
        layout = self._map_to_triangle(layout)

        return layout

    def _map_to_triangle(self, layout: np.ndarray) -> np.ndarray:
        """
        Map 2D coordinates to an equilateral triangle.

        Args:
            layout: 2D coordinates

        Returns:
            Coordinates mapped to triangle
        """
        # Normalize to [0, 1]
        x_min, x_max = layout[:, 0].min(), layout[:, 0].max()
        y_min, y_max = layout[:, 1].min(), layout[:, 1].max()

        if x_max > x_min:
            layout[:, 0] = (layout[:, 0] - x_min) / (x_max - x_min)
        else:
            layout[:, 0] = 0.5

        if y_max > y_min:
            layout[:, 1] = (layout[:, 1] - y_min) / (y_max - y_min)
        else:
            layout[:, 1] = 0.5

        # Map to triangle: vertices at (0, 0), (1, 0), (0.5, sqrt(3)/2)
        # Simple approach: scale y to triangle height
        height = np.sqrt(3) / 2
        layout[:, 1] *= height

        # Ensure points stay within triangle bounds
        # For each point, adjust x based on y to stay in triangle
        for i in range(len(layout)):
            x, y = layout[i]
            # Triangle boundaries: left edge x >= y/sqrt(3), right edge x <= 1 - y/sqrt(3)
            left_bound = y / np.sqrt(3)
            right_bound = 1 - y / np.sqrt(3)

            if x < left_bound:
                layout[i, 0] = left_bound
            elif x > right_bound:
                layout[i, 0] = right_bound

        return layout


def compute_and_save_layout(
    clusters_dir: Path,
    method: str = "umap",
    output_file: Optional[Path] = None,
) -> np.ndarray:
    """
    Compute and save 2D layout for clusters.

    Args:
        clusters_dir: Directory containing cluster data
        method: Layout computation method
        output_file: Output file path (default: clusters_dir/layout.npy)

    Returns:
        2D layout coordinates
    """
    clusters_dir = Path(clusters_dir)

    # Load cluster metadata
    metadata_file = clusters_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Cluster metadata not found: {metadata_file}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Extract embeddings
    cluster_ids = sorted(metadata.keys())
    embeddings = []

    for cluster_id in cluster_ids:
        centroid = metadata[cluster_id].get("embedding_centroid", [])
        if not centroid:
            # Use zero vector if no embedding
            centroid = [0.0] * 384  # Default sentence transformer size

        embeddings.append(centroid)

    embeddings = np.array(embeddings)

    # Compute layout
    computer = LayoutComputer(method=method)
    layout = computer.compute_layout(embeddings)

    # Save
    if output_file is None:
        output_file = clusters_dir / "layout.npy"

    np.save(output_file, layout)
    print(f"Layout saved to {output_file}")
    print(f"Shape: {layout.shape}")

    return layout


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.viz.layout <clusters_dir> [method]")
        sys.exit(1)

    clusters_dir = Path(sys.argv[1])
    method = sys.argv[2] if len(sys.argv) > 2 else "umap"

    layout = compute_and_save_layout(clusters_dir, method)
    print(f"Computed layout for {len(layout)} clusters using {method}")
