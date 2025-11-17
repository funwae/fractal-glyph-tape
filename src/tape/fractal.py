"""Fractal addressing for mapping 2D coordinates to hierarchical addresses."""

import numpy as np
from typing import Tuple, List, Optional
from loguru import logger


class FractalAddresser:
    """Assign fractal addresses to 2D coordinates using triangular subdivision."""

    def __init__(self, fractal_type: str = "triangular", max_depth: int = 10):
        """
        Initialize fractal addresser.

        Args:
            fractal_type: Type of fractal subdivision (currently only 'triangular')
            max_depth: Maximum depth of fractal subdivision
        """
        self.fractal_type = fractal_type
        self.max_depth = max_depth

        if fractal_type != "triangular":
            raise ValueError("Only 'triangular' fractal type is currently supported")

    def assign_address(self, x: float, y: float, depth: Optional[int] = None) -> str:
        """
        Assign fractal address to a point in [0,1] × [0,1].

        Args:
            x: X coordinate (normalized to [0,1])
            y: Y coordinate (normalized to [0,1])
            depth: Subdivision depth (uses max_depth if None)

        Returns:
            Fractal address string (e.g., "L-R-C-L")
        """
        if depth is None:
            depth = self.max_depth

        if not (0 <= x <= 1 and 0 <= y <= 1):
            # Clip to valid range
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)

        address_parts = []

        # Current triangle vertices (normalized coordinates)
        v0 = np.array([0.0, 0.0])  # Bottom-left
        v1 = np.array([1.0, 0.0])  # Bottom-right
        v2 = np.array([0.5, 1.0])  # Top

        point = np.array([x, y])

        for _ in range(depth):
            # Subdivide triangle into 4 sub-triangles
            # Calculate midpoints
            m01 = (v0 + v1) / 2  # Bottom edge midpoint
            m12 = (v1 + v2) / 2  # Right edge midpoint
            m20 = (v2 + v0) / 2  # Left edge midpoint

            # Determine which sub-triangle contains the point
            # Use barycentric coordinates

            # Sub-triangle L (left): v0, m01, m20
            if self._point_in_triangle(point, v0, m01, m20):
                address_parts.append("L")
                v0, v1, v2 = v0, m01, m20

            # Sub-triangle R (right): m01, v1, m12
            elif self._point_in_triangle(point, m01, v1, m12):
                address_parts.append("R")
                v0, v1, v2 = m01, v1, m12

            # Sub-triangle T (top): m20, m12, v2
            elif self._point_in_triangle(point, m20, m12, v2):
                address_parts.append("T")
                v0, v1, v2 = m20, m12, v2

            # Sub-triangle C (center): m01, m12, m20
            else:
                address_parts.append("C")
                v0, v1, v2 = m01, m12, m20

        return "-".join(address_parts)

    def _point_in_triangle(
        self,
        p: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> bool:
        """
        Check if point p is inside triangle defined by v0, v1, v2.

        Uses barycentric coordinates.

        Args:
            p: Point to test
            v0, v1, v2: Triangle vertices

        Returns:
            True if point is inside triangle
        """
        # Compute barycentric coordinates
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p - v0

        # Solve for barycentric coordinates
        dot00 = np.dot(v0v2, v0v2)
        dot01 = np.dot(v0v2, v0v1)
        dot02 = np.dot(v0v2, v0p)
        dot11 = np.dot(v0v1, v0v1)
        dot12 = np.dot(v0v1, v0p)

        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        return (u >= 0) and (v >= 0) and (u + v <= 1)

    def batch_assign_addresses(
        self,
        coords: np.ndarray,
        depth: Optional[int] = None
    ) -> List[str]:
        """
        Assign fractal addresses to multiple points.

        Args:
            coords: Array of shape (n, 2) with normalized coordinates
            depth: Subdivision depth

        Returns:
            List of fractal address strings
        """
        logger.info(f"Assigning fractal addresses to {len(coords)} points...")

        addresses = []
        for i, (x, y) in enumerate(coords):
            address = self.assign_address(x, y, depth)
            addresses.append(address)

            if (i + 1) % 1000 == 0:
                logger.debug(f"Processed {i + 1}/{len(coords)} addresses")

        logger.info("Fractal address assignment complete!")
        return addresses
