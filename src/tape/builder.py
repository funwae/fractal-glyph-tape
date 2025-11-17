"""Fractal tape builder - orchestrates dimensionality reduction and tape construction."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .fractal import FractalAddresser
from .storage import TapeStorage


class TapeBuilder:
    """Build fractal tape from cluster centroids."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tape builder.

        Args:
            config: Configuration dictionary with tape settings
        """
        self.config = config
        self.projection_method = config.get("projection_method", "umap")
        self.projection_dims = config.get("projection_dims", 2)
        self.fractal_type = config.get("fractal_type", "triangular")
        self.max_depth = config.get("max_depth", 10)

        # Initialize components
        self.addresser = FractalAddresser(
            fractal_type=self.fractal_type,
            max_depth=self.max_depth
        )

        self.coords_2d = None
        self.addresses = None

    def build_tape(
        self,
        centroids: np.ndarray,
        glyph_mapping: Dict[int, str],
        cluster_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Build fractal tape from cluster centroids.

        Args:
            centroids: Cluster centroid vectors (n_clusters, embedding_dim)
            glyph_mapping: Dict mapping cluster_id to glyph_string
            cluster_metadata: Optional cluster metadata
            output_path: Output directory (uses config if None)

        Returns:
            Path to tape database
        """
        if output_path is None:
            output_path = self.config["output_path"]

        logger.info("Building fractal tape...")
        logger.info(f"Centroids shape: {centroids.shape}")
        logger.info(f"Projection method: {self.projection_method}")

        # Step 1: Project to 2D
        self.coords_2d = self._project_to_2d(centroids)

        # Step 2: Normalize coordinates to [0, 1]
        coords_norm = self._normalize_coords(self.coords_2d)

        # Step 3: Assign fractal addresses
        self.addresses = self.addresser.batch_assign_addresses(coords_norm)

        # Step 4: Build storage database
        db_path = Path(output_path) / "tape_index.db"
        self._build_database(
            db_path,
            centroids,
            coords_norm,
            glyph_mapping,
            cluster_metadata
        )

        logger.info(f"Tape built successfully: {db_path}")
        return str(db_path)

    def _project_to_2d(self, centroids: np.ndarray) -> np.ndarray:
        """
        Project high-dimensional centroids to 2D.

        Args:
            centroids: Centroid vectors

        Returns:
            2D coordinates
        """
        logger.info(f"Projecting {len(centroids)} centroids to 2D using {self.projection_method}...")

        if self.projection_method == "umap":
            coords_2d = self._project_umap(centroids)
        elif self.projection_method == "pca":
            coords_2d = self._project_pca(centroids)
        else:
            raise ValueError(f"Unsupported projection method: {self.projection_method}")

        logger.info(f"Projection complete: {coords_2d.shape}")
        return coords_2d

    def _project_umap(self, centroids: np.ndarray) -> np.ndarray:
        """Project using UMAP."""
        import umap

        umap_config = self.config.get("umap", {})
        reducer = umap.UMAP(
            n_components=self.projection_dims,
            n_neighbors=umap_config.get("n_neighbors", 15),
            min_dist=umap_config.get("min_dist", 0.1),
            metric=umap_config.get("metric", "cosine"),
            random_state=umap_config.get("random_state", 42),
            verbose=True
        )

        coords_2d = reducer.fit_transform(centroids)
        return coords_2d

    def _project_pca(self, centroids: np.ndarray) -> np.ndarray:
        """Project using PCA."""
        from sklearn.decomposition import PCA

        pca_config = self.config.get("pca", {})
        reducer = PCA(
            n_components=self.projection_dims,
            random_state=pca_config.get("random_state", 42)
        )

        coords_2d = reducer.fit_transform(centroids)
        logger.info(f"PCA explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
        return coords_2d

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates to [0, 1] range.

        Args:
            coords: Input coordinates

        Returns:
            Normalized coordinates
        """
        logger.info("Normalizing coordinates to [0, 1]...")

        # Min-max normalization
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)

        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1

        coords_norm = (coords - min_vals) / ranges

        logger.info(f"Normalized coords range: [{coords_norm.min():.3f}, {coords_norm.max():.3f}]")
        return coords_norm

    def _build_database(
        self,
        db_path: Path,
        centroids: np.ndarray,
        coords_norm: np.ndarray,
        glyph_mapping: Dict[int, str],
        cluster_metadata: Optional[Dict[int, Dict[str, Any]]]
    ) -> None:
        """
        Build SQLite database with tape data.

        Args:
            db_path: Path to database file
            centroids: Cluster centroids
            coords_norm: Normalized 2D coordinates
            glyph_mapping: Cluster to glyph mapping
            cluster_metadata: Cluster metadata
        """
        logger.info("Building tape database...")

        with TapeStorage(str(db_path)) as storage:
            storage.create_schema()

            n_clusters = len(centroids)

            # Prepare batch data
            glyph_data = []
            address_data = []
            cluster_data = []

            for cluster_id in range(n_clusters):
                glyph_string = glyph_mapping.get(cluster_id, "")
                x, y = coords_norm[cluster_id]
                fractal_address = self.addresses[cluster_id]

                # Glyph mapping
                glyph_data.append((cluster_id, glyph_string, cluster_id))

                # Address mapping
                address_data.append((cluster_id, fractal_address, float(x), float(y)))

                # Cluster data
                metadata = cluster_metadata.get(cluster_id, {}) if cluster_metadata else {}
                size = metadata.get("size", 0)
                centroid_blob = centroids[cluster_id].tobytes()
                metadata_json = str(metadata) if metadata else None

                cluster_data.append((cluster_id, size, centroid_blob, metadata_json))

            # Batch insert
            logger.info("Inserting glyph mappings...")
            storage.batch_insert_glyphs(glyph_data)

            logger.info("Inserting fractal addresses...")
            storage.batch_insert_addresses(address_data)

            logger.info("Inserting cluster metadata...")
            storage.batch_insert_clusters(cluster_data)

        logger.info("Database build complete!")
