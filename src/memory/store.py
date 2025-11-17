"""Memory storage and indexing for FGMS."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .addresses import address_from_str, address_to_str
from .models import FractalAddress, MemoryRecord, MemorySpan


class MemoryStore:
    """Abstract base class for memory storage.

    Provides persistence and indexing for MemoryRecord objects.
    """

    def save_record(self, record: MemoryRecord) -> None:
        """Save a memory record.

        Args:
            record: MemoryRecord to save
        """
        raise NotImplementedError

    def get_records_for_actor(
        self, actor_id: str, region: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterable[MemoryRecord]:
        """Get all records for an actor.

        Args:
            actor_id: Actor/user ID
            region: Optional region filter
            limit: Optional maximum number of records to return

        Returns:
            Iterable of MemoryRecord objects
        """
        raise NotImplementedError

    def get_spans_by_address(
        self, addresses: List[FractalAddress]
    ) -> List[Tuple[FractalAddress, MemorySpan]]:
        """Get spans at specific addresses.

        Args:
            addresses: List of FractalAddress to retrieve

        Returns:
            List of (address, span) tuples
        """
        raise NotImplementedError

    def list_regions(self, actor_id: str) -> List[str]:
        """List all regions for an actor.

        Args:
            actor_id: Actor/user ID

        Returns:
            List of region strings
        """
        raise NotImplementedError

    def list_addresses(
        self, actor_id: str, region: str, limit: int = 100
    ) -> List[FractalAddress]:
        """List addresses for an actor in a region.

        Args:
            actor_id: Actor/user ID
            region: Region to query
            limit: Maximum number of addresses to return

        Returns:
            List of FractalAddress objects
        """
        raise NotImplementedError

    def get_region_stats(self, actor_id: str, region: str) -> dict:
        """Get statistics for a region.

        Args:
            actor_id: Actor/user ID
            region: Region to query

        Returns:
            Dict with stats (record_count, span_count, first_timestamp, last_timestamp)
        """
        raise NotImplementedError


class SQLiteMemoryStore(MemoryStore):
    """SQLite-based memory store implementation.

    Schema:
        records: id, actor_id, created_at, world, region, tags, raw_text, source, extra
        spans: record_id, address, glyph_ids, text, language, meta

    This provides a simple file-based storage suitable for development
    and moderate workloads.
    """

    def __init__(self, db_path: str = "memory.db"):
        """Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Create records table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    actor_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    world TEXT NOT NULL,
                    region TEXT NOT NULL,
                    tags TEXT,
                    raw_text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    extra TEXT
                )
            """
            )

            # Create spans table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS spans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    address TEXT NOT NULL,
                    glyph_ids TEXT NOT NULL,
                    text TEXT NOT NULL,
                    language TEXT NOT NULL,
                    meta TEXT,
                    FOREIGN KEY (record_id) REFERENCES records(id)
                )
            """
            )

            # Create indices
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_actor ON records(actor_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_region ON records(region)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_actor_region ON records(actor_id, region)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_spans_record ON spans(record_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_spans_address ON spans(address)")

            conn.commit()
        finally:
            conn.close()

    def save_record(self, record: MemoryRecord) -> None:
        """Save a memory record to SQLite.

        Args:
            record: MemoryRecord to save
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Insert record
            cursor.execute(
                """
                INSERT OR REPLACE INTO records
                (id, actor_id, created_at, world, region, tags, raw_text, source, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.id,
                    record.actor_id,
                    record.created_at.isoformat(),
                    record.world,
                    record.region,
                    json.dumps(record.tags),
                    record.raw_text,
                    record.source,
                    json.dumps(record.extra),
                ),
            )

            # Insert spans
            for span in record.spans:
                cursor.execute(
                    """
                    INSERT INTO spans
                    (record_id, address, glyph_ids, text, language, meta)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.id,
                        address_to_str(span.address),
                        json.dumps(span.glyph_ids),
                        span.text,
                        span.language,
                        json.dumps(span.meta),
                    ),
                )

            conn.commit()
        finally:
            conn.close()

    def get_records_for_actor(
        self, actor_id: str, region: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterable[MemoryRecord]:
        """Get records for an actor from SQLite.

        Args:
            actor_id: Actor/user ID
            region: Optional region filter
            limit: Optional maximum number of records

        Yields:
            MemoryRecord objects
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Build query
            if region:
                query = """
                    SELECT id, actor_id, created_at, world, region, tags, raw_text, source, extra
                    FROM records
                    WHERE actor_id = ? AND region = ?
                    ORDER BY created_at DESC
                """
                params = [actor_id, region]
            else:
                query = """
                    SELECT id, actor_id, created_at, world, region, tags, raw_text, source, extra
                    FROM records
                    WHERE actor_id = ?
                    ORDER BY created_at DESC
                """
                params = [actor_id]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)

            for row in cursor.fetchall():
                # Fetch spans for this record
                cursor_spans = conn.cursor()
                cursor_spans.execute(
                    """
                    SELECT address, glyph_ids, text, language, meta
                    FROM spans
                    WHERE record_id = ?
                """,
                    (row[0],),
                )

                spans = []
                for span_row in cursor_spans.fetchall():
                    spans.append(
                        MemorySpan(
                            address=address_from_str(span_row[0]),
                            glyph_ids=json.loads(span_row[1]),
                            text=span_row[2],
                            language=span_row[3],
                            meta=json.loads(span_row[4]),
                        )
                    )

                yield MemoryRecord(
                    id=row[0],
                    actor_id=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    world=row[3],
                    region=row[4],
                    tags=json.loads(row[5]),
                    raw_text=row[6],
                    source=row[7],
                    spans=spans,
                    extra=json.loads(row[8]),
                )
        finally:
            conn.close()

    def get_spans_by_address(
        self, addresses: List[FractalAddress]
    ) -> List[Tuple[FractalAddress, MemorySpan]]:
        """Get spans at specific addresses from SQLite.

        Args:
            addresses: List of FractalAddress to retrieve

        Returns:
            List of (address, span) tuples
        """
        if not addresses:
            return []

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            results = []
            address_strs = [address_to_str(addr) for addr in addresses]

            # Query spans
            placeholders = ",".join("?" * len(address_strs))
            cursor.execute(
                f"""
                SELECT address, glyph_ids, text, language, meta
                FROM spans
                WHERE address IN ({placeholders})
            """,
                address_strs,
            )

            for row in cursor.fetchall():
                addr = address_from_str(row[0])
                span = MemorySpan(
                    address=addr,
                    glyph_ids=json.loads(row[1]),
                    text=row[2],
                    language=row[3],
                    meta=json.loads(row[4]),
                )
                results.append((addr, span))

            return results
        finally:
            conn.close()

    def list_regions(self, actor_id: str) -> List[str]:
        """List regions for an actor from SQLite.

        Args:
            actor_id: Actor/user ID

        Returns:
            List of region strings
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT region
                FROM records
                WHERE actor_id = ?
                ORDER BY region
            """,
                (actor_id,),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def list_addresses(
        self, actor_id: str, region: str, limit: int = 100
    ) -> List[FractalAddress]:
        """List addresses for an actor in a region from SQLite.

        Args:
            actor_id: Actor/user ID
            region: Region to query
            limit: Maximum number of addresses

        Returns:
            List of FractalAddress objects
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT spans.address
                FROM spans
                JOIN records ON spans.record_id = records.id
                WHERE records.actor_id = ? AND records.region = ?
                ORDER BY spans.address DESC
                LIMIT ?
            """,
                (actor_id, region, limit),
            )
            return [address_from_str(row[0]) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_region_stats(self, actor_id: str, region: str) -> dict:
        """Get statistics for a region from SQLite.

        Args:
            actor_id: Actor/user ID
            region: Region to query

        Returns:
            Dict with stats
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get record count and timestamps
            cursor.execute(
                """
                SELECT COUNT(*), MIN(created_at), MAX(created_at)
                FROM records
                WHERE actor_id = ? AND region = ?
            """,
                (actor_id, region),
            )
            row = cursor.fetchone()
            record_count = row[0] or 0
            first_ts = datetime.fromisoformat(row[1]) if row[1] else None
            last_ts = datetime.fromisoformat(row[2]) if row[2] else None

            # Get span count
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM spans
                JOIN records ON spans.record_id = records.id
                WHERE records.actor_id = ? AND records.region = ?
            """,
                (actor_id, region),
            )
            span_count = cursor.fetchone()[0] or 0

            return {
                "record_count": record_count,
                "span_count": span_count,
                "first_timestamp": first_ts,
                "last_timestamp": last_ts,
            }
        finally:
            conn.close()


class InMemoryStore(MemoryStore):
    """In-memory store for testing."""

    def __init__(self):
        """Initialize in-memory store."""
        self.records: List[MemoryRecord] = []

    def save_record(self, record: MemoryRecord) -> None:
        """Save record to memory."""
        # Remove existing record with same ID
        self.records = [r for r in self.records if r.id != record.id]
        self.records.append(record)

    def get_records_for_actor(
        self, actor_id: str, region: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterable[MemoryRecord]:
        """Get records from memory."""
        results = [r for r in self.records if r.actor_id == actor_id]
        if region:
            results = [r for r in results if r.region == region]
        results.sort(key=lambda r: r.created_at, reverse=True)
        if limit:
            results = results[:limit]
        return results

    def get_spans_by_address(
        self, addresses: List[FractalAddress]
    ) -> List[Tuple[FractalAddress, MemorySpan]]:
        """Get spans by address from memory."""
        address_strs = {address_to_str(addr) for addr in addresses}
        results = []
        for record in self.records:
            for span in record.spans:
                if address_to_str(span.address) in address_strs:
                    results.append((span.address, span))
        return results

    def list_regions(self, actor_id: str) -> List[str]:
        """List regions from memory."""
        regions = set()
        for record in self.records:
            if record.actor_id == actor_id:
                regions.add(record.region)
        return sorted(regions)

    def list_addresses(
        self, actor_id: str, region: str, limit: int = 100
    ) -> List[FractalAddress]:
        """List addresses from memory."""
        addresses = []
        for record in self.records:
            if record.actor_id == actor_id and record.region == region:
                for span in record.spans:
                    addresses.append(span.address)
        return addresses[:limit]

    def get_region_stats(self, actor_id: str, region: str) -> dict:
        """Get region stats from memory."""
        records = [r for r in self.records if r.actor_id == actor_id and r.region == region]
        if not records:
            return {
                "record_count": 0,
                "span_count": 0,
                "first_timestamp": None,
                "last_timestamp": None,
            }

        span_count = sum(len(r.spans) for r in records)
        timestamps = [r.created_at for r in records]

        return {
            "record_count": len(records),
            "span_count": span_count,
            "first_timestamp": min(timestamps),
            "last_timestamp": max(timestamps),
        }
