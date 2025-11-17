"""
Data models for Fractal Glyph Memory System
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Fractal Address(BaseModel):
    """
    Fractal address for hierarchical memory location.

    Format: world/region#tri_path@dDtT
    Example: earth/tech#102012@d3t17
    """
    world: str = Field(..., description="World/domain identifier")
    region: str = Field(..., description="Region within world")
    tri_path: str = Field(..., description="Ternary path (base-3 fractal coordinate)")
    depth: int = Field(..., description="Depth in fractal hierarchy")
    time_slice: datetime = Field(..., description="Temporal coordinate")

    def to_string(self) -> str:
        """Convert to string representation."""
        time_str = f"t{self.time_slice.hour * 60 + self.time_slice.minute}"
        return f"{self.world}/{self.region}#{self.tri_path}@d{self.depth}{time_str}"

    @classmethod
    def from_string(cls, address_str: str) -> "FractalAddress":
        """Parse from string representation."""
        # Simple parser for format: world/region#tri_path@dDtT
        parts = address_str.split('/')
        world = parts[0]

        region_parts = parts[1].split('#')
        region = region_parts[0]

        path_parts = region_parts[1].split('@')
        tri_path = path_parts[0]

        coords = path_parts[1]
        depth = int(coords.split('t')[0][1:])  # Extract D from dD
        time_val = int(coords.split('t')[1])  # Extract T from tT

        # Reconstruct datetime from time_val (minutes since midnight)
        hour = time_val // 60
        minute = time_val % 60
        time_slice = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)

        return cls(
            world=world,
            region=region,
            tri_path=tri_path,
            depth=depth,
            time_slice=time_slice
        )


class Glyph(BaseModel):
    """
    Glyph representation of a semantic cluster.

    Glyphs are compressed representations of phrase families.
    """
    glyph_id: int = Field(..., description="Unique glyph identifier")
    glyph_str: str = Field(..., description="Unicode glyph character(s)")
    cluster_id: int = Field(..., description="Semantic cluster ID")
    frequency: int = Field(default=1, description="Usage frequency")
    semantic_summary: str = Field(default="", description="Human-readable summary")


class MemoryEntry(BaseModel):
    """
    A single memory entry in the fractal memory system.
    """
    entry_id: str = Field(default="", description="Unique entry identifier")
    actor_id: str = Field(..., description="Actor/agent identifier")
    address: FractalAddress = Field(..., description="Fractal address")
    text: str = Field(..., description="Memory text content")
    glyphs: List[Glyph] = Field(default_factory=list, description="Glyph representations")
    embedding: Optional[List[float]] = Field(default=None, description="Optional vector embedding")
    tags: List[str] = Field(default_factory=list, description="Tags for filtering")
    source: str = Field(default="user", description="Source of memory")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    token_estimate: int = Field(default=0, description="Estimated token count")

    class Config:
        arbitrary_types_allowed = True
