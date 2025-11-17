"""Data models for the Fractal Glyph Memory Service (FGMS)."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Core Data Models (dataclasses for internal use)
# ============================================================================


@dataclass(frozen=True)
class FractalAddress:
    """Logical address in fractal memory space.

    Attributes:
        world: Logical namespace / tenant (e.g., "earthcloud", "default")
        region: Domain/topic (e.g., "hayden-agent", "support-logs")
        tri_path: Path in triangular fractal space (FGT-level)
        depth: Z-depth (summary=0, deeper detail > 0)
        time_slice: Increasing index within region
    """

    world: str
    region: str
    tri_path: List[int]
    depth: int
    time_slice: int

    def __post_init__(self):
        """Validate address components."""
        if not self.world:
            raise ValueError("world cannot be empty")
        if not self.region:
            raise ValueError("region cannot be empty")
        if self.depth < 0:
            raise ValueError("depth must be non-negative")
        if self.time_slice < 0:
            raise ValueError("time_slice must be non-negative")


@dataclass
class MemorySpan:
    """Represents a contiguous span of text mapped to glyphs.

    Attributes:
        address: Fractal address for this span
        glyph_ids: List of glyph IDs (e.g., ["谷阜", "嶽岭"])
        text: Raw or paraphrased span text
        language: Language code (e.g., "en", "zh")
        meta: Arbitrary metadata (tags, source, etc.)
    """

    address: FractalAddress
    glyph_ids: List[str]
    text: str
    language: str
    meta: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate span data."""
        if not self.text.strip():
            raise ValueError("text cannot be empty")
        if not self.language:
            raise ValueError("language cannot be empty")


@dataclass
class MemoryRecord:
    """Represents one "write" operation (e.g., one message, one event).

    Attributes:
        id: Unique record identifier
        actor_id: ID of the actor/user
        created_at: Timestamp of record creation
        world: World/namespace for this record
        region: Region/topic for this record
        spans: List of memory spans in this record
        tags: List of tags for categorization
        raw_text: Original unprocessed text
        source: Source of the record ("user", "assistant", "system")
        extra: Arbitrary extra metadata
    """

    id: str
    actor_id: str
    created_at: datetime
    world: str
    region: str
    spans: List[MemorySpan]
    tags: List[str]
    raw_text: str
    source: Literal["user", "assistant", "system"]
    extra: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate record data."""
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.actor_id:
            raise ValueError("actor_id cannot be empty")
        if not self.raw_text.strip():
            raise ValueError("raw_text cannot be empty")


# ============================================================================
# API Models (Pydantic for FastAPI)
# ============================================================================


class MemoryWriteRequest(BaseModel):
    """Request model for writing memory."""

    actor_id: str = Field(..., description="ID of the actor/user")
    text: str = Field(..., description="Text to write to memory")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags")
    region: Optional[str] = Field(default=None, description="Optional region override")
    source: Literal["user", "assistant", "system"] = Field(
        default="user", description="Source of the memory"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "actor_id": "hayden",
                "text": "I want to implement a memory service for my agent",
                "tags": ["task", "planning"],
                "source": "user",
            }
        }


class MemoryWriteResponse(BaseModel):
    """Response model for write operations."""

    status: Literal["ok", "error"] = Field(..., description="Operation status")
    world: str = Field(..., description="World/namespace used")
    region: str = Field(..., description="Region used")
    addresses: List[str] = Field(..., description="Serialized fractal addresses created")
    glyph_density: float = Field(
        ..., description="Ratio of glyph tokens to total tokens", ge=0.0, le=1.0
    )
    error: Optional[str] = Field(default=None, description="Error message if status=error")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "world": "default",
                "region": "agent:hayden",
                "addresses": ["default/agent:hayden#123@d2t5", "default/agent:hayden#124@d2t5"],
                "glyph_density": 0.42,
            }
        }


class MemoryReadFocus(BaseModel):
    """Focus parameters for memory read operations."""

    region: Optional[str] = Field(default=None, description="Specific region to focus on")
    max_depth: Optional[int] = Field(default=None, description="Maximum depth to retrieve", ge=0)

    class Config:
        json_schema_extra = {"example": {"region": "agent:hayden", "max_depth": 2}}


class MemoryReadRequest(BaseModel):
    """Request model for reading memory."""

    actor_id: str = Field(..., description="ID of the actor/user")
    query: str = Field(..., description="Query text for retrieval")
    focus: Optional[MemoryReadFocus] = Field(
        default=None, description="Optional focus parameters"
    )
    token_budget: int = Field(
        default=2048, description="Maximum tokens for context", ge=1, le=32768
    )
    mode: Literal["glyph", "text", "mixed"] = Field(
        default="mixed", description="Context representation mode"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "actor_id": "hayden",
                "query": "What did we discuss about the memory service?",
                "token_budget": 2048,
                "mode": "mixed",
            }
        }


class MemoryContextItem(BaseModel):
    """A single item in the memory context."""

    address: str = Field(..., description="Serialized fractal address")
    glyphs: List[str] = Field(..., description="List of glyph IDs")
    summary: Optional[str] = Field(default=None, description="Optional summary text")
    excerpt: Optional[str] = Field(default=None, description="Optional text excerpt")
    score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "address": "default/agent:hayden#123@d2t5",
                "glyphs": ["谷阜", "嶽岭"],
                "summary": "User discussed implementing a memory service",
                "score": 0.89,
            }
        }


class MemoryReadResponse(BaseModel):
    """Response model for read operations."""

    status: Literal["ok", "error"] = Field(..., description="Operation status")
    world: str = Field(..., description="World/namespace used")
    region: str = Field(..., description="Region used")
    mode: str = Field(..., description="Mode used (glyph/text/mixed)")
    context: List[MemoryContextItem] = Field(..., description="Context items retrieved")
    token_estimate: int = Field(..., description="Estimated token count", ge=0)
    error: Optional[str] = Field(default=None, description="Error message if status=error")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "world": "default",
                "region": "agent:hayden",
                "mode": "mixed",
                "context": [
                    {
                        "address": "default/agent:hayden#123@d2t5",
                        "glyphs": ["谷阜", "嶽岭"],
                        "summary": "User discussed memory service",
                        "score": 0.89,
                    }
                ],
                "token_estimate": 512,
            }
        }


class RegionInfo(BaseModel):
    """Information about a memory region."""

    region: str = Field(..., description="Region identifier")
    record_count: int = Field(..., description="Number of records", ge=0)
    span_count: int = Field(..., description="Number of spans", ge=0)
    first_timestamp: Optional[datetime] = Field(
        default=None, description="Timestamp of first record"
    )
    last_timestamp: Optional[datetime] = Field(
        default=None, description="Timestamp of last record"
    )


class RegionsListResponse(BaseModel):
    """Response model for listing regions."""

    status: Literal["ok", "error"] = Field(..., description="Operation status")
    actor_id: str = Field(..., description="Actor ID")
    regions: List[RegionInfo] = Field(..., description="List of regions")
    error: Optional[str] = Field(default=None, description="Error message if status=error")


class AddressInfo(BaseModel):
    """Information about a specific address."""

    address: str = Field(..., description="Serialized fractal address")
    span_count: int = Field(..., description="Number of spans at this address", ge=0)
    created_at: datetime = Field(..., description="Timestamp of creation")


class AddressesListResponse(BaseModel):
    """Response model for listing addresses."""

    status: Literal["ok", "error"] = Field(..., description="Operation status")
    actor_id: str = Field(..., description="Actor ID")
    region: str = Field(..., description="Region")
    addresses: List[AddressInfo] = Field(..., description="List of addresses")
    error: Optional[str] = Field(default=None, description="Error message if status=error")
