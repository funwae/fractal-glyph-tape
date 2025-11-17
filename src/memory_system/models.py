"""
Data models for Fractal Glyph Memory System
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class FractalAddress:
    """
    A fractal address in the memory space.

    Format: world/region#tri_path@depth.time_slice
    Example: earth/tech#012120@d6t1234567890
    """
    world: str
    region: str
    tri_path: str
    depth: int
    time_slice: datetime

    def to_string(self) -> str:
        """Convert to string format."""
        timestamp_str = f"t{int(self.time_slice.timestamp())}"
        return f"{self.world}/{self.region}#{self.tri_path}@d{self.depth}{timestamp_str}"

    @staticmethod
    def from_string(address_str: str) -> 'FractalAddress':
        """Parse from string format."""
        # Format: world/region#tri_path@d{depth}t{timestamp}
        parts = address_str.split('/')
        world = parts[0]

        region_and_rest = parts[1].split('#')
        region = region_and_rest[0]

        tri_path_and_rest = region_and_rest[1].split('@')
        tri_path = tri_path_and_rest[0]

        depth_and_time = tri_path_and_rest[1]
        depth = int(depth_and_time.split('d')[1].split('t')[0])
        timestamp = int(depth_and_time.split('t')[1])

        return FractalAddress(
            world=world,
            region=region,
            tri_path=tri_path,
            depth=depth,
            time_slice=datetime.fromtimestamp(timestamp)
        )


@dataclass
class Glyph:
    """
    A glyph representing a compressed semantic unit.
    """
    glyph_id: int
    glyph_str: str
    cluster_id: int
    frequency: int
    semantic_summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "glyph_id": self.glyph_id,
            "glyph_str": self.glyph_str,
            "cluster_id": self.cluster_id,
            "frequency": self.frequency,
            "semantic_summary": self.semantic_summary
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Glyph':
        """Create from dictionary."""
        return Glyph(
            glyph_id=data["glyph_id"],
            glyph_str=data["glyph_str"],
            cluster_id=data["cluster_id"],
            frequency=data["frequency"],
            semantic_summary=data["semantic_summary"]
        )


@dataclass
class MemoryEntry:
    """
    A single memory entry with fractal address and glyph encoding.
    """
    entry_id: str
    actor_id: str
    address: FractalAddress
    text: str
    glyphs: List[Glyph]
    tags: List[str] = field(default_factory=list)
    source: str = "user"
    created_at: datetime = field(default_factory=datetime.now)
    token_estimate: int = 0
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "entry_id": self.entry_id,
            "actor_id": self.actor_id,
            "address": self.address.to_string(),
            "world": self.address.world,
            "region": self.address.region,
            "tri_path": self.address.tri_path,
            "depth": self.address.depth,
            "time_slice": self.address.time_slice.isoformat(),
            "text": self.text,
            "glyphs": [g.to_dict() for g in self.glyphs],
            "tags": self.tags,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "token_estimate": self.token_estimate,
            "embedding": self.embedding
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary."""
        return MemoryEntry(
            entry_id=data["entry_id"],
            actor_id=data["actor_id"],
            address=FractalAddress(
                world=data["world"],
                region=data["region"],
                tri_path=data["tri_path"],
                depth=data["depth"],
                time_slice=datetime.fromisoformat(data["time_slice"])
            ),
            text=data["text"],
            glyphs=[Glyph.from_dict(g) for g in data["glyphs"]],
            tags=data.get("tags", []),
            source=data.get("source", "user"),
            created_at=datetime.fromisoformat(data["created_at"]),
            token_estimate=data.get("token_estimate", 0),
            embedding=data.get("embedding")
        )
