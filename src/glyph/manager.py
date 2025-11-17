"""Glyph manager for allocating and managing glyph IDs."""

from typing import Dict, Optional


class GlyphManager:
    """
    Manages glyph ID allocation and assignment.

    This is a placeholder for future glyph management functionality.
    For now, glyph management is handled by GlyphCodec.
    """

    def __init__(self):
        """Initialize glyph manager."""
        pass

    def allocate_glyph(self, cluster_id: str) -> int:
        """
        Allocate a glyph ID for a cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Allocated glyph ID
        """
        raise NotImplementedError("Use GlyphCodec for glyph management")
