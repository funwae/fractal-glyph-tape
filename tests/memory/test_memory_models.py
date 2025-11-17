"""Unit tests for memory data models and addressing."""

import pytest
from datetime import datetime

from src.memory.models import FractalAddress, MemorySpan, MemoryRecord
from src.memory.addresses import (
    address_to_str,
    address_from_str,
    validate_address_str,
    create_address,
    parse_region_from_actor,
    addresses_equal,
    increment_time_slice,
    with_depth,
)


class TestFractalAddress:
    """Tests for FractalAddress dataclass."""

    def test_create_valid_address(self):
        """Test creating a valid fractal address."""
        addr = FractalAddress(
            world="default",
            region="agent:hayden",
            tri_path=[1, 2, 3],
            depth=2,
            time_slice=5,
        )
        assert addr.world == "default"
        assert addr.region == "agent:hayden"
        assert addr.tri_path == [1, 2, 3]
        assert addr.depth == 2
        assert addr.time_slice == 5

    def test_empty_world_raises_error(self):
        """Test that empty world raises ValueError."""
        with pytest.raises(ValueError, match="world cannot be empty"):
            FractalAddress(
                world="",
                region="agent:hayden",
                tri_path=[],
                depth=0,
                time_slice=0,
            )

    def test_empty_region_raises_error(self):
        """Test that empty region raises ValueError."""
        with pytest.raises(ValueError, match="region cannot be empty"):
            FractalAddress(
                world="default",
                region="",
                tri_path=[],
                depth=0,
                time_slice=0,
            )

    def test_negative_depth_raises_error(self):
        """Test that negative depth raises ValueError."""
        with pytest.raises(ValueError, match="depth must be non-negative"):
            FractalAddress(
                world="default",
                region="agent:hayden",
                tri_path=[],
                depth=-1,
                time_slice=0,
            )

    def test_address_is_frozen(self):
        """Test that FractalAddress is immutable."""
        addr = create_address("default", "agent:hayden", [1, 2, 3])
        with pytest.raises(Exception):  # FrozenInstanceError in dataclasses
            addr.depth = 5  # type: ignore


class TestAddressSerialization:
    """Tests for address serialization/deserialization."""

    def test_address_to_str_basic(self):
        """Test basic address serialization."""
        addr = create_address("default", "agent:hayden", [1, 2, 3], depth=2, time_slice=17)
        result = address_to_str(addr)
        assert result == "default/agent:hayden#1,2,3@d2t17"

    def test_address_to_str_empty_path(self):
        """Test serialization with empty tri_path."""
        addr = create_address("default", "agent:hayden", [], depth=0, time_slice=0)
        result = address_to_str(addr)
        assert result == "default/agent:hayden#0@d0t0"

    def test_address_from_str_basic(self):
        """Test basic address deserialization."""
        addr = address_from_str("default/agent:hayden#1,2,3@d2t17")
        assert addr.world == "default"
        assert addr.region == "agent:hayden"
        assert addr.tri_path == [1, 2, 3]
        assert addr.depth == 2
        assert addr.time_slice == 17

    def test_address_from_str_empty_path(self):
        """Test deserialization with empty path."""
        addr = address_from_str("default/agent:hayden#0@d0t0")
        assert addr.tri_path == []

    def test_address_from_str_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid address format"):
            address_from_str("invalid")

    def test_address_roundtrip(self):
        """Test that serialization roundtrips correctly."""
        original = create_address("earthcloud", "support-logs", [5, 7, 3], depth=1, time_slice=42)
        serialized = address_to_str(original)
        deserialized = address_from_str(serialized)
        assert addresses_equal(original, deserialized)

    def test_validate_address_str_valid(self):
        """Test validation of valid address string."""
        assert validate_address_str("default/agent:hayden#1,2,3@d2t17")
        assert validate_address_str("world/region#0@d0t0")

    def test_validate_address_str_invalid(self):
        """Test validation of invalid address string."""
        assert not validate_address_str("invalid")
        assert not validate_address_str("world/region")
        assert not validate_address_str("")


class TestAddressUtilities:
    """Tests for address utility functions."""

    def test_parse_region_from_actor(self):
        """Test generating region from actor ID."""
        assert parse_region_from_actor("hayden") == "agent:hayden"
        assert parse_region_from_actor("user123", "user") == "user:user123"

    def test_addresses_equal_true(self):
        """Test equality check for equal addresses."""
        addr1 = create_address("default", "agent:hayden", [1, 2, 3], depth=2, time_slice=5)
        addr2 = create_address("default", "agent:hayden", [1, 2, 3], depth=2, time_slice=5)
        assert addresses_equal(addr1, addr2)

    def test_addresses_equal_false(self):
        """Test equality check for different addresses."""
        addr1 = create_address("default", "agent:hayden", [1, 2, 3], depth=2, time_slice=5)
        addr2 = create_address("default", "agent:hayden", [1, 2, 3], depth=2, time_slice=6)
        assert not addresses_equal(addr1, addr2)

    def test_increment_time_slice(self):
        """Test incrementing time slice."""
        addr1 = create_address("default", "agent:hayden", [1, 2, 3], time_slice=5)
        addr2 = increment_time_slice(addr1)
        assert addr2.time_slice == 6
        assert addr2.world == addr1.world
        assert addr2.region == addr1.region

    def test_with_depth(self):
        """Test creating address with different depth."""
        addr1 = create_address("default", "agent:hayden", [1, 2, 3], depth=0)
        addr2 = with_depth(addr1, 2)
        assert addr2.depth == 2
        assert addr2.world == addr1.world
        assert addr2.region == addr1.region


class TestMemorySpan:
    """Tests for MemorySpan dataclass."""

    def test_create_valid_span(self):
        """Test creating a valid memory span."""
        addr = create_address("default", "agent:hayden", [1, 2, 3])
        span = MemorySpan(
            address=addr,
            glyph_ids=["谷阜", "嶽岭"],
            text="This is a test span",
            language="en",
            meta={"cluster_id": "123"},
        )
        assert span.address == addr
        assert span.glyph_ids == ["谷阜", "嶽岭"]
        assert span.text == "This is a test span"
        assert span.language == "en"
        assert span.meta["cluster_id"] == "123"

    def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        addr = create_address("default", "agent:hayden", [1, 2, 3])
        with pytest.raises(ValueError, match="text cannot be empty"):
            MemorySpan(
                address=addr,
                glyph_ids=["谷阜"],
                text="   ",
                language="en",
            )

    def test_empty_language_raises_error(self):
        """Test that empty language raises ValueError."""
        addr = create_address("default", "agent:hayden", [1, 2, 3])
        with pytest.raises(ValueError, match="language cannot be empty"):
            MemorySpan(
                address=addr,
                glyph_ids=["谷阜"],
                text="Test",
                language="",
            )


class TestMemoryRecord:
    """Tests for MemoryRecord dataclass."""

    def test_create_valid_record(self):
        """Test creating a valid memory record."""
        addr = create_address("default", "agent:hayden", [1, 2, 3])
        span = MemorySpan(
            address=addr,
            glyph_ids=["谷阜"],
            text="Test span",
            language="en",
        )
        record = MemoryRecord(
            id="test-id",
            actor_id="hayden",
            created_at=datetime.utcnow(),
            world="default",
            region="agent:hayden",
            spans=[span],
            tags=["test"],
            raw_text="Test raw text",
            source="user",
        )
        assert record.id == "test-id"
        assert record.actor_id == "hayden"
        assert len(record.spans) == 1
        assert record.source == "user"

    def test_empty_id_raises_error(self):
        """Test that empty ID raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            MemoryRecord(
                id="",
                actor_id="hayden",
                created_at=datetime.utcnow(),
                world="default",
                region="agent:hayden",
                spans=[],
                tags=[],
                raw_text="Test",
                source="user",
            )

    def test_empty_actor_id_raises_error(self):
        """Test that empty actor_id raises ValueError."""
        with pytest.raises(ValueError, match="actor_id cannot be empty"):
            MemoryRecord(
                id="test-id",
                actor_id="",
                created_at=datetime.utcnow(),
                world="default",
                region="agent:hayden",
                spans=[],
                tags=[],
                raw_text="Test",
                source="user",
            )

    def test_empty_raw_text_raises_error(self):
        """Test that empty raw_text raises ValueError."""
        with pytest.raises(ValueError, match="raw_text cannot be empty"):
            MemoryRecord(
                id="test-id",
                actor_id="hayden",
                created_at=datetime.utcnow(),
                world="default",
                region="agent:hayden",
                spans=[],
                tags=[],
                raw_text="   ",
                source="user",
            )
