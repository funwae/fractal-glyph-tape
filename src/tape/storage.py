"""Storage backend for fractal tape using SQLite."""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class TapeStorage:
    """SQLite-based storage for fractal tape data."""

    def __init__(self, db_path: str):
        """
        Initialize tape storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """Connect to database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        logger.info(f"Connected to database: {self.db_path}")

    def create_schema(self) -> None:
        """Create database schema."""
        logger.info("Creating database schema...")

        # Glyphs table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS glyphs (
                glyph_id INTEGER PRIMARY KEY,
                glyph_string TEXT NOT NULL UNIQUE,
                cluster_id INTEGER NOT NULL,
                UNIQUE(cluster_id)
            )
        """)

        # Addresses table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS addresses (
                cluster_id INTEGER PRIMARY KEY,
                fractal_address TEXT NOT NULL,
                x_coord REAL NOT NULL,
                y_coord REAL NOT NULL
            )
        """)

        # Clusters table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id INTEGER PRIMARY KEY,
                size INTEGER NOT NULL,
                centroid BLOB NOT NULL,
                metadata TEXT
            )
        """)

        # Create indexes
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_glyph_string ON glyphs(glyph_string)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_fractal_address ON addresses(fractal_address)")

        self.conn.commit()
        logger.info("Schema created successfully")

    def insert_glyph(self, glyph_id: int, glyph_string: str, cluster_id: int) -> None:
        """Insert glyph mapping."""
        self.cursor.execute(
            "INSERT OR REPLACE INTO glyphs (glyph_id, glyph_string, cluster_id) VALUES (?, ?, ?)",
            (glyph_id, glyph_string, cluster_id)
        )

    def insert_address(
        self,
        cluster_id: int,
        fractal_address: str,
        x_coord: float,
        y_coord: float
    ) -> None:
        """Insert fractal address."""
        self.cursor.execute(
            "INSERT OR REPLACE INTO addresses (cluster_id, fractal_address, x_coord, y_coord) VALUES (?, ?, ?, ?)",
            (cluster_id, fractal_address, x_coord, y_coord)
        )

    def insert_cluster(
        self,
        cluster_id: int,
        size: int,
        centroid: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert cluster metadata."""
        centroid_blob = centroid.tobytes()
        metadata_json = json.dumps(metadata) if metadata else None

        self.cursor.execute(
            "INSERT OR REPLACE INTO clusters (cluster_id, size, centroid, metadata) VALUES (?, ?, ?, ?)",
            (cluster_id, size, centroid_blob, metadata_json)
        )

    def batch_insert_glyphs(self, glyph_data: List[Tuple[int, str, int]]) -> None:
        """Batch insert glyph mappings."""
        self.cursor.executemany(
            "INSERT OR REPLACE INTO glyphs (glyph_id, glyph_string, cluster_id) VALUES (?, ?, ?)",
            glyph_data
        )

    def batch_insert_addresses(self, address_data: List[Tuple[int, str, float, float]]) -> None:
        """Batch insert addresses."""
        self.cursor.executemany(
            "INSERT OR REPLACE INTO addresses (cluster_id, fractal_address, x_coord, y_coord) VALUES (?, ?, ?, ?)",
            address_data
        )

    def batch_insert_clusters(self, cluster_data: List[Tuple]) -> None:
        """Batch insert clusters."""
        self.cursor.executemany(
            "INSERT OR REPLACE INTO clusters (cluster_id, size, centroid, metadata) VALUES (?, ?, ?, ?)",
            cluster_data
        )

    def get_cluster_by_glyph(self, glyph_string: str) -> Optional[Dict[str, Any]]:
        """Get cluster info by glyph string."""
        self.cursor.execute(
            """
            SELECT c.cluster_id, c.size, c.centroid, c.metadata, a.fractal_address, a.x_coord, a.y_coord
            FROM glyphs g
            JOIN clusters c ON g.cluster_id = c.cluster_id
            JOIN addresses a ON c.cluster_id = a.cluster_id
            WHERE g.glyph_string = ?
            """,
            (glyph_string,)
        )

        row = self.cursor.fetchone()
        if not row:
            return None

        cluster_id, size, centroid_blob, metadata_json, fractal_address, x_coord, y_coord = row

        return {
            "cluster_id": cluster_id,
            "size": size,
            "fractal_address": fractal_address,
            "coords": (x_coord, y_coord),
            "metadata": json.loads(metadata_json) if metadata_json else {},
        }

    def get_glyph_by_cluster(self, cluster_id: int) -> Optional[str]:
        """Get glyph string by cluster ID."""
        self.cursor.execute("SELECT glyph_string FROM glyphs WHERE cluster_id = ?", (cluster_id,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def get_all_coords(self) -> List[Tuple[int, float, float, str]]:
        """Get all cluster coordinates and glyph strings."""
        self.cursor.execute(
            """
            SELECT a.cluster_id, a.x_coord, a.y_coord, g.glyph_string
            FROM addresses a
            JOIN glyphs g ON a.cluster_id = g.cluster_id
            ORDER BY a.cluster_id
            """
        )
        return self.cursor.fetchall()

    def commit(self) -> None:
        """Commit changes."""
        self.conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.commit()
        self.close()
