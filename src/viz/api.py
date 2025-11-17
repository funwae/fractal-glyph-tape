"""FastAPI backend for Fractal Glyph Tape visualizer."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models import ClusterInfo, ClusterSummary, LayoutPoint


class TapeVisualizerAPI:
    """API for serving fractal tape visualization data."""

    def __init__(self, tape_dir: Path):
        """
        Initialize the visualizer API.

        Args:
            tape_dir: Path to the tape directory containing cluster data
        """
        self.tape_dir = Path(tape_dir)
        self.clusters_dir = self.tape_dir / "clusters"
        self.layout_file = self.clusters_dir / "layout.npy"

        # Load data
        self._load_cluster_metadata()
        self._load_layout()

    def _load_cluster_metadata(self):
        """Load cluster metadata from disk."""
        metadata_file = self.clusters_dir / "metadata.json"

        if not metadata_file.exists():
            self.cluster_metadata = {}
            return

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.cluster_metadata = json.load(f)

    def _load_layout(self):
        """Load 2D layout coordinates."""
        if not self.layout_file.exists():
            self.layout = None
            return

        self.layout = np.load(self.layout_file)

    def get_clusters_summary(
        self,
        language: Optional[str] = None,
        min_frequency: Optional[int] = None,
    ) -> List[ClusterSummary]:
        """
        Get summary of all clusters.

        Args:
            language: Filter by language (e.g., 'en', 'zh')
            min_frequency: Minimum frequency threshold

        Returns:
            List of cluster summaries
        """
        summaries = []

        for cluster_id, meta in self.cluster_metadata.items():
            # Apply filters
            if language and meta.get("language") != language:
                continue

            if min_frequency and meta.get("frequency", 0) < min_frequency:
                continue

            summary = ClusterSummary(
                cluster_id=cluster_id,
                glyph=meta.get("glyph", ""),
                size=meta.get("size", 0),
                language=meta.get("language", "unknown"),
                frequency=meta.get("frequency", 0),
                representative_phrase=meta.get("representative_phrase", ""),
            )
            summaries.append(summary)

        return summaries

    def get_cluster_detail(self, cluster_id: str) -> ClusterInfo:
        """
        Get detailed information about a specific cluster.

        Args:
            cluster_id: The cluster identifier

        Returns:
            Detailed cluster information

        Raises:
            HTTPException: If cluster not found
        """
        if cluster_id not in self.cluster_metadata:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        meta = self.cluster_metadata[cluster_id]

        return ClusterInfo(
            cluster_id=cluster_id,
            glyph=meta.get("glyph", ""),
            size=meta.get("size", 0),
            language=meta.get("language", "unknown"),
            frequency=meta.get("frequency", 0),
            representative_phrase=meta.get("representative_phrase", ""),
            example_phrases=meta.get("example_phrases", []),
            embedding_centroid=meta.get("embedding_centroid", []),
            coherence_score=meta.get("coherence_score", 0.0),
        )

    def get_glyph_cluster(self, glyph: str) -> ClusterInfo:
        """
        Lookup cluster by glyph.

        Args:
            glyph: The glyph character/string

        Returns:
            Cluster information

        Raises:
            HTTPException: If glyph not found
        """
        # Find cluster with matching glyph
        for cluster_id, meta in self.cluster_metadata.items():
            if meta.get("glyph") == glyph:
                return self.get_cluster_detail(cluster_id)

        raise HTTPException(status_code=404, detail=f"Glyph '{glyph}' not found")

    def get_layout(self) -> List[LayoutPoint]:
        """
        Get 2D layout coordinates for all clusters.

        Returns:
            List of layout points with coordinates and cluster info

        Raises:
            HTTPException: If layout not computed
        """
        if self.layout is None:
            raise HTTPException(
                status_code=503,
                detail="Layout not computed. Run layout computation first."
            )

        points = []
        for idx, cluster_id in enumerate(sorted(self.cluster_metadata.keys())):
            meta = self.cluster_metadata[cluster_id]

            point = LayoutPoint(
                cluster_id=cluster_id,
                x=float(self.layout[idx, 0]),
                y=float(self.layout[idx, 1]),
                glyph=meta.get("glyph", ""),
                language=meta.get("language", "unknown"),
                frequency=meta.get("frequency", 0),
            )
            points.append(point)

        return points


def create_app(tape_dir: str = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        tape_dir: Path to tape directory (default: from TAPE_DIR env or 'tape/v1')

    Returns:
        Configured FastAPI app
    """
    import os

    if tape_dir is None:
        tape_dir = os.environ.get("TAPE_DIR", "tape/v1")

    app = FastAPI(
        title="Fractal Glyph Tape Visualizer",
        description="API for exploring fractal tape clusters",
        version="0.1.0",
    )

    # Enable CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize API handler
    api_handler = TapeVisualizerAPI(tape_dir)

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "Fractal Glyph Tape Visualizer",
            "version": "0.1.0",
            "endpoints": [
                "/clusters",
                "/cluster/{id}",
                "/glyph/{glyph}",
                "/layout",
            ],
        }

    @app.get("/clusters", response_model=List[ClusterSummary])
    async def get_clusters(
        language: Optional[str] = None,
        min_frequency: Optional[int] = None,
    ):
        """Get list of cluster summaries."""
        return api_handler.get_clusters_summary(language, min_frequency)

    @app.get("/cluster/{cluster_id}", response_model=ClusterInfo)
    async def get_cluster(cluster_id: str):
        """Get detailed information about a cluster."""
        return api_handler.get_cluster_detail(cluster_id)

    @app.get("/glyph/{glyph}", response_model=ClusterInfo)
    async def get_glyph(glyph: str):
        """Lookup cluster by glyph."""
        return api_handler.get_glyph_cluster(glyph)

    @app.get("/layout", response_model=List[LayoutPoint])
    async def get_layout():
        """Get 2D layout coordinates for visualization."""
        return api_handler.get_layout()

    return app


# For running with uvicorn
app = create_app()
