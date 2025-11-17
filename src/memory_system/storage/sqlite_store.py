"""
SQLite-backed memory storage with fractal indexing
"""
import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from ..models import MemoryEntry, FractalAddress, Glyph


class SQLiteMemoryStore:
    """
    SQLite-backed persistent storage for fractal memory entries.

    Features:
    - Fractal address indexing for hierarchical queries
    - Actor-based partitioning
    - Full-text search on content
    - Temporal queries
    """

    def __init__(self, db_path: str = "data/fgms_memory.db"):
        """Initialize SQLite store."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()

        # Main memory entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                entry_id TEXT PRIMARY KEY,
                actor_id TEXT NOT NULL,
                world TEXT NOT NULL,
                region TEXT NOT NULL,
                tri_path TEXT NOT NULL,
                depth INTEGER NOT NULL,
                time_slice TEXT NOT NULL,
                address_full TEXT NOT NULL,
                text TEXT NOT NULL,
                glyphs TEXT,
                embedding TEXT,
                tags TEXT,
                source TEXT DEFAULT 'user',
                created_at TEXT NOT NULL,
                token_estimate INTEGER DEFAULT 0
            )
        """)

        # Indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_actor_id ON memory_entries(actor_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_world_region ON memory_entries(world, region)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tri_path ON memory_entries(tri_path)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_address_full ON memory_entries(address_full)
        """)

        # Full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                entry_id UNINDEXED,
                text,
                tags
            )
        """)

        self.conn.commit()

    def write(self, entry: MemoryEntry) -> str:
        """
        Write a memory entry to storage.

        Args:
            entry: MemoryEntry to store

        Returns:
            entry_id of the stored entry
        """
        cursor = self.conn.cursor()

        # Generate ID if not provided
        if not entry.entry_id:
            entry.entry_id = str(uuid.uuid4())

        # Serialize complex fields
        glyphs_json = json.dumps([g.to_dict() for g in entry.glyphs])
        embedding_json = json.dumps(entry.embedding) if entry.embedding else None
        tags_json = json.dumps(entry.tags)

        # Insert into main table
        cursor.execute("""
            INSERT OR REPLACE INTO memory_entries
            (entry_id, actor_id, world, region, tri_path, depth, time_slice,
             address_full, text, glyphs, embedding, tags, source, created_at, token_estimate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.entry_id,
            entry.actor_id,
            entry.address.world,
            entry.address.region,
            entry.address.tri_path,
            entry.address.depth,
            entry.address.time_slice.isoformat(),
            entry.address.to_string(),
            entry.text,
            glyphs_json,
            embedding_json,
            tags_json,
            entry.source,
            entry.created_at.isoformat(),
            int(entry.token_estimate)
        ))

        # Insert into FTS table
        cursor.execute("""
            INSERT OR REPLACE INTO memory_fts (entry_id, text, tags)
            VALUES (?, ?, ?)
        """, (entry.entry_id, entry.text, ' '.join(entry.tags)))

        self.conn.commit()
        return entry.entry_id

    def read(
        self,
        actor_id: Optional[str] = None,
        world: Optional[str] = None,
        region: Optional[str] = None,
        tri_path_prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryEntry]:
        """
        Read memory entries with filtering.

        Args:
            actor_id: Filter by actor
            world: Filter by world
            region: Filter by region
            tri_path_prefix: Filter by tri_path prefix (for hierarchical queries)
            tags: Filter by tags (any match)
            limit: Maximum number of results
            offset: Skip first N results

        Returns:
            List of matching memory entries
        """
        cursor = self.conn.cursor()

        # Build query
        conditions = []
        params = []

        if actor_id:
            conditions.append("actor_id = ?")
            params.append(actor_id)

        if world:
            conditions.append("world = ?")
            params.append(world)

        if region:
            conditions.append("region = ?")
            params.append(region)

        if tri_path_prefix:
            conditions.append("tri_path LIKE ?")
            params.append(f"{tri_path_prefix}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT * FROM memory_entries
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert rows to MemoryEntry objects
        entries = []
        for row in rows:
            entry = self._row_to_entry(dict(row))

            # Filter by tags if specified
            if tags and not any(tag in entry.tags for tag in tags):
                continue

            entries.append(entry)

        return entries

    def search(
        self,
        query: str,
        actor_id: Optional[str] = None,
        limit: int = 50
    ) -> List[MemoryEntry]:
        """
        Full-text search across memory entries.

        Args:
            query: Search query
            actor_id: Optional actor filter
            limit: Maximum results

        Returns:
            List of matching entries
        """
        cursor = self.conn.cursor()

        # FTS query
        if actor_id:
            cursor.execute("""
                SELECT m.* FROM memory_fts f
                JOIN memory_entries m ON f.entry_id = m.entry_id
                WHERE memory_fts MATCH ? AND m.actor_id = ?
                ORDER BY rank
                LIMIT ?
            """, (query, actor_id, limit))
        else:
            cursor.execute("""
                SELECT m.* FROM memory_fts f
                JOIN memory_entries m ON f.entry_id = m.entry_id
                WHERE memory_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))

        rows = cursor.fetchall()
        return [self._row_to_entry(dict(row)) for row in rows]

    def get_by_address(self, address: FractalAddress) -> Optional[MemoryEntry]:
        """Get memory entry by exact address."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM memory_entries
            WHERE address_full = ?
        """, (address.to_string(),))

        row = cursor.fetchone()
        return self._row_to_entry(dict(row)) if row else None

    def get_stats(self, actor_id: Optional[str] = None) -> Dict[str, Any]:
        """Get storage statistics."""
        cursor = self.conn.cursor()

        if actor_id:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT world) as unique_worlds,
                    COUNT(DISTINCT region) as unique_regions,
                    SUM(token_estimate) as total_tokens
                FROM memory_entries
                WHERE actor_id = ?
            """, (actor_id,))
        else:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT actor_id) as unique_actors,
                    COUNT(DISTINCT world) as unique_worlds,
                    COUNT(DISTINCT region) as unique_regions,
                    SUM(token_estimate) as total_tokens
                FROM memory_entries
            """)

        row = cursor.fetchone()
        return dict(row) if row else {}

    def _row_to_entry(self, row: dict) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        address = FractalAddress(
            world=row["world"],
            region=row["region"],
            tri_path=row["tri_path"],
            depth=row["depth"],
            time_slice=datetime.fromisoformat(row["time_slice"])
        )

        glyphs_data = json.loads(row["glyphs"]) if row["glyphs"] else []
        glyphs = [Glyph.from_dict(g) for g in glyphs_data]

        embedding = json.loads(row["embedding"]) if row["embedding"] else None
        tags = json.loads(row["tags"]) if row["tags"] else []

        return MemoryEntry(
            entry_id=row["entry_id"],
            actor_id=row["actor_id"],
            address=address,
            text=row["text"],
            glyphs=glyphs,
            embedding=embedding,
            tags=tags,
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            token_estimate=row["token_estimate"]
        )

    def close(self):
        """Close database connection."""
        self.conn.close()
