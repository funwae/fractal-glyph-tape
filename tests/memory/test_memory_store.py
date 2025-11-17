"""Unit tests for MemoryStore implementations."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from src.memory.store import InMemoryStore, SQLiteMemoryStore
from src.memory.models import MemoryRecord, MemorySpan
from src.memory.addresses import create_address


@pytest.fixture
def sample_record():
    """Create a sample memory record for testing."""
    addr1 = create_address("default", "agent:test", [1, 2, 3], depth=2, time_slice=1)
    addr2 = create_address("default", "agent:test", [4, 5], depth=2, time_slice=2)

    span1 = MemorySpan(
        address=addr1,
        glyph_ids=["谷阜"],
        text="First test span",
        language="en",
    )

    span2 = MemorySpan(
        address=addr2,
        glyph_ids=["嶽岭"],
        text="Second test span",
        language="en",
    )

    return MemoryRecord(
        id="test-record-1",
        actor_id="test_user",
        created_at=datetime.utcnow(),
        world="default",
        region="agent:test",
        spans=[span1, span2],
        tags=["test", "sample"],
        raw_text="First test span. Second test span.",
        source="user",
        extra={"test_key": "test_value"},
    )


class TestInMemoryStore:
    """Tests for InMemoryStore implementation."""

    def test_save_and_retrieve_record(self, sample_record):
        """Test saving and retrieving a record."""
        store = InMemoryStore()
        store.save_record(sample_record)

        records = list(store.get_records_for_actor("test_user"))
        assert len(records) == 1
        assert records[0].id == sample_record.id
        assert records[0].actor_id == sample_record.actor_id

    def test_save_duplicate_id_replaces(self, sample_record):
        """Test that saving a record with duplicate ID replaces it."""
        store = InMemoryStore()
        store.save_record(sample_record)

        # Create modified record with same ID
        modified = MemoryRecord(
            id=sample_record.id,
            actor_id=sample_record.actor_id,
            created_at=datetime.utcnow(),
            world=sample_record.world,
            region=sample_record.region,
            spans=[],
            tags=["modified"],
            raw_text="Modified text",
            source="assistant",
        )
        store.save_record(modified)

        records = list(store.get_records_for_actor("test_user"))
        assert len(records) == 1
        assert records[0].tags == ["modified"]

    def test_get_records_by_region(self, sample_record):
        """Test filtering records by region."""
        store = InMemoryStore()
        store.save_record(sample_record)

        # Add record in different region
        other_record = MemoryRecord(
            id="test-record-2",
            actor_id="test_user",
            created_at=datetime.utcnow(),
            world="default",
            region="other:region",
            spans=[],
            tags=[],
            raw_text="Other region",
            source="user",
        )
        store.save_record(other_record)

        # Get records for specific region
        records = list(store.get_records_for_actor("test_user", region="agent:test"))
        assert len(records) == 1
        assert records[0].region == "agent:test"

    def test_get_spans_by_address(self, sample_record):
        """Test retrieving spans by address."""
        store = InMemoryStore()
        store.save_record(sample_record)

        addresses = [span.address for span in sample_record.spans]
        spans = store.get_spans_by_address(addresses)

        assert len(spans) == 2
        assert all(isinstance(addr, type(addresses[0])) for addr, _ in spans)

    def test_list_regions(self, sample_record):
        """Test listing regions for an actor."""
        store = InMemoryStore()
        store.save_record(sample_record)

        # Add another record in different region
        other_record = MemoryRecord(
            id="test-record-2",
            actor_id="test_user",
            created_at=datetime.utcnow(),
            world="default",
            region="other:region",
            spans=[],
            tags=[],
            raw_text="Other",
            source="user",
        )
        store.save_record(other_record)

        regions = store.list_regions("test_user")
        assert len(regions) == 2
        assert "agent:test" in regions
        assert "other:region" in regions

    def test_list_addresses(self, sample_record):
        """Test listing addresses for an actor in a region."""
        store = InMemoryStore()
        store.save_record(sample_record)

        addresses = store.list_addresses("test_user", "agent:test", limit=10)
        assert len(addresses) == 2

    def test_get_region_stats(self, sample_record):
        """Test getting statistics for a region."""
        store = InMemoryStore()
        store.save_record(sample_record)

        stats = store.get_region_stats("test_user", "agent:test")
        assert stats["record_count"] == 1
        assert stats["span_count"] == 2
        assert stats["first_timestamp"] is not None
        assert stats["last_timestamp"] is not None

    def test_get_region_stats_empty(self):
        """Test stats for nonexistent region."""
        store = InMemoryStore()
        stats = store.get_region_stats("nobody", "nowhere")
        assert stats["record_count"] == 0
        assert stats["span_count"] == 0


class TestSQLiteMemoryStore:
    """Tests for SQLiteMemoryStore implementation."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        Path(db_path).unlink(missing_ok=True)

    def test_save_and_retrieve_record(self, temp_db, sample_record):
        """Test saving and retrieving a record."""
        store = SQLiteMemoryStore(temp_db)
        store.save_record(sample_record)

        records = list(store.get_records_for_actor("test_user"))
        assert len(records) == 1
        assert records[0].id == sample_record.id
        assert records[0].raw_text == sample_record.raw_text

        # Check spans were saved
        assert len(records[0].spans) == 2

    def test_persistence_across_instances(self, temp_db, sample_record):
        """Test that data persists across store instances."""
        # Save with first instance
        store1 = SQLiteMemoryStore(temp_db)
        store1.save_record(sample_record)

        # Retrieve with second instance
        store2 = SQLiteMemoryStore(temp_db)
        records = list(store2.get_records_for_actor("test_user"))
        assert len(records) == 1
        assert records[0].id == sample_record.id

    def test_get_records_by_region(self, temp_db, sample_record):
        """Test filtering records by region."""
        store = SQLiteMemoryStore(temp_db)
        store.save_record(sample_record)

        # Add record in different region
        other_record = MemoryRecord(
            id="test-record-2",
            actor_id="test_user",
            created_at=datetime.utcnow(),
            world="default",
            region="other:region",
            spans=[],
            tags=[],
            raw_text="Other region",
            source="user",
        )
        store.save_record(other_record)

        # Get records for specific region
        records = list(store.get_records_for_actor("test_user", region="agent:test"))
        assert len(records) == 1
        assert records[0].region == "agent:test"

    def test_get_spans_by_address(self, temp_db, sample_record):
        """Test retrieving spans by address."""
        store = SQLiteMemoryStore(temp_db)
        store.save_record(sample_record)

        addresses = [span.address for span in sample_record.spans]
        spans = store.get_spans_by_address(addresses)

        assert len(spans) == 2

    def test_list_regions(self, temp_db, sample_record):
        """Test listing regions for an actor."""
        store = SQLiteMemoryStore(temp_db)
        store.save_record(sample_record)

        regions = store.list_regions("test_user")
        assert len(regions) >= 1
        assert "agent:test" in regions

    def test_get_region_stats(self, temp_db, sample_record):
        """Test getting statistics for a region."""
        store = SQLiteMemoryStore(temp_db)
        store.save_record(sample_record)

        stats = store.get_region_stats("test_user", "agent:test")
        assert stats["record_count"] == 1
        assert stats["span_count"] == 2

    def test_limit_parameter(self, temp_db, sample_record):
        """Test that limit parameter works correctly."""
        store = SQLiteMemoryStore(temp_db)

        # Save multiple records
        for i in range(5):
            record = MemoryRecord(
                id=f"test-record-{i}",
                actor_id="test_user",
                created_at=datetime.utcnow(),
                world="default",
                region="agent:test",
                spans=[],
                tags=[],
                raw_text=f"Record {i}",
                source="user",
            )
            store.save_record(record)

        # Retrieve with limit
        records = list(store.get_records_for_actor("test_user", limit=3))
        assert len(records) == 3
