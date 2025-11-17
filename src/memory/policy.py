"""Memory policy and foveation logic for FGMS."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .addresses import create_address, parse_region_from_actor
from .models import FractalAddress


@dataclass
class MemoryPolicyConfig:
    """Configuration for memory policy behavior.

    Attributes:
        default_world: Default world/namespace to use
        default_depth: Default depth for new writes
        shallow_budget_ratio: Ratio of token budget for shallow depth (0-1)
        deep_budget_ratio: Ratio of token budget for deep detail
        max_tri_path_length: Maximum length of tri_path
        summary_depth: Depth level considered "summary"
        detail_depth: Depth level considered "detail"
    """

    default_world: str = "default"
    default_depth: int = 2
    shallow_budget_ratio: float = 0.3
    deep_budget_ratio: float = 0.7
    max_tri_path_length: int = 10
    summary_depth: int = 0
    detail_depth: int = 2

    def __post_init__(self):
        """Validate configuration."""
        if not (0 <= self.shallow_budget_ratio <= 1):
            raise ValueError("shallow_budget_ratio must be between 0 and 1")
        if not (0 <= self.deep_budget_ratio <= 1):
            raise ValueError("deep_budget_ratio must be between 0 and 1")
        if abs((self.shallow_budget_ratio + self.deep_budget_ratio) - 1.0) > 0.01:
            raise ValueError("shallow + deep budget ratios should sum to ~1.0")


@dataclass
class ClusterInfo:
    """Information about a cluster for address assignment.

    Attributes:
        cluster_id: Unique cluster identifier
        centroid: Cluster centroid in embedding space
        size: Number of members in cluster
        tri_path: Triangular fractal path for this cluster
    """

    cluster_id: int
    centroid: np.ndarray
    size: int
    tri_path: List[int] = field(default_factory=list)


class MemoryPolicy:
    """Policy for assigning addresses and selecting context for reads.

    This class implements:
    - Address assignment on write (cluster -> tri_path mapping)
    - Foveated selection on read (relevance-based depth allocation)
    """

    def __init__(self, config: Optional[MemoryPolicyConfig] = None):
        """Initialize memory policy.

        Args:
            config: Policy configuration (uses defaults if None)
        """
        self.config = config or MemoryPolicyConfig()
        # Track cluster -> tri_path mappings
        self._cluster_to_tri_path: Dict[int, List[int]] = {}
        # Track time_slice counters per region
        self._region_time_slices: Dict[str, int] = {}

    def assign_address(
        self,
        world: str,
        region: str,
        cluster_id: int,
        depth: Optional[int] = None,
        cluster_info: Optional[ClusterInfo] = None,
    ) -> FractalAddress:
        """Assign a fractal address for a given cluster.

        Args:
            world: World/namespace
            region: Region/topic
            cluster_id: Cluster ID for the span
            depth: Depth level (uses default if None)
            cluster_info: Optional cluster information

        Returns:
            FractalAddress for this span

        Note:
            The tri_path is determined by:
            1. If cluster_info is provided and has tri_path, use that
            2. If cluster was seen before, reuse its tri_path
            3. Otherwise, generate a new tri_path from cluster_id
        """
        # Determine tri_path
        if cluster_info and cluster_info.tri_path:
            tri_path = cluster_info.tri_path
        elif cluster_id in self._cluster_to_tri_path:
            tri_path = self._cluster_to_tri_path[cluster_id]
        else:
            # Generate tri_path from cluster_id (simple hash-based approach)
            tri_path = self._generate_tri_path(cluster_id)
            self._cluster_to_tri_path[cluster_id] = tri_path

        # Determine depth
        if depth is None:
            depth = self.config.default_depth

        # Get and increment time_slice for this region
        region_key = f"{world}/{region}"
        time_slice = self._region_time_slices.get(region_key, 0)
        self._region_time_slices[region_key] = time_slice + 1

        return create_address(
            world=world,
            region=region,
            tri_path=tri_path,
            depth=depth,
            time_slice=time_slice,
        )

    def _generate_tri_path(self, cluster_id: int) -> List[int]:
        """Generate a tri_path from cluster_id.

        This is a simple deterministic mapping. In a full implementation,
        you would use the actual fractal triangular layout from FGT.

        Args:
            cluster_id: Cluster identifier

        Returns:
            List of integers representing the tri_path
        """
        # Simple approach: use cluster_id digits as path components
        # Limited by max_tri_path_length
        path = []
        n = abs(cluster_id)
        max_len = self.config.max_tri_path_length

        if n == 0:
            return [0]

        while n > 0 and len(path) < max_len:
            path.append(n % 10)
            n //= 10

        return path or [0]

    def select_addresses(
        self,
        actor_id: str,
        world: str,
        region: str,
        query_embedding: np.ndarray,
        candidate_addresses: List[Tuple[FractalAddress, float]],
        token_budget: int,
        max_depth: Optional[int] = None,
    ) -> List[Tuple[FractalAddress, float]]:
        """Select addresses for a read operation using foveation.

        Args:
            actor_id: Actor/user ID
            world: World/namespace
            region: Region to search
            query_embedding: Embedded query for similarity scoring
            candidate_addresses: List of (address, score) tuples
            token_budget: Maximum tokens to allocate
            max_depth: Optional maximum depth constraint

        Returns:
            List of selected (address, score) tuples, sorted by score descending

        Note:
            Foveation strategy:
            - Allocate shallow_budget_ratio to depth 0-1 (broad coverage)
            - Allocate deep_budget_ratio to depth 2+ (focused detail)
            - Prioritize by score within each depth tier
        """
        if not candidate_addresses:
            return []

        # Filter by max_depth if specified
        if max_depth is not None:
            candidate_addresses = [
                (addr, score) for addr, score in candidate_addresses if addr.depth <= max_depth
            ]

        # Separate into shallow and deep candidates
        shallow = [
            (addr, score)
            for addr, score in candidate_addresses
            if addr.depth <= self.config.summary_depth
        ]
        deep = [
            (addr, score)
            for addr, score in candidate_addresses
            if addr.depth > self.config.summary_depth
        ]

        # Sort by score descending
        shallow.sort(key=lambda x: x[1], reverse=True)
        deep.sort(key=lambda x: x[1], reverse=True)

        # Allocate budget
        shallow_budget = int(token_budget * self.config.shallow_budget_ratio)
        deep_budget = int(token_budget * self.config.deep_budget_ratio)

        # Select addresses within budget
        # (Simplified: assume each address costs roughly equal tokens)
        # In a real implementation, you'd track actual token costs per address
        selected = []

        # Estimate tokens per address (rough heuristic)
        avg_tokens_per_addr = 100  # Adjustable based on actual data

        shallow_count = min(len(shallow), shallow_budget // avg_tokens_per_addr)
        deep_count = min(len(deep), deep_budget // avg_tokens_per_addr)

        selected.extend(shallow[:shallow_count])
        selected.extend(deep[:deep_count])

        # Sort final selection by score
        selected.sort(key=lambda x: x[1], reverse=True)

        return selected

    def resolve_world_and_region(
        self, actor_id: str, region: Optional[str] = None, world: Optional[str] = None
    ) -> Tuple[str, str]:
        """Resolve world and region from actor_id and optional overrides.

        Args:
            actor_id: Actor/user ID
            region: Optional region override
            world: Optional world override

        Returns:
            Tuple of (world, region)
        """
        resolved_world = world or self.config.default_world
        resolved_region = region or parse_region_from_actor(actor_id)
        return resolved_world, resolved_region

    def get_default_depth(self) -> int:
        """Get the default depth for new writes.

        Returns:
            Default depth value
        """
        return self.config.default_depth

    def update_cluster_mapping(self, cluster_id: int, tri_path: List[int]) -> None:
        """Update the cluster -> tri_path mapping.

        Args:
            cluster_id: Cluster identifier
            tri_path: Triangular fractal path
        """
        self._cluster_to_tri_path[cluster_id] = tri_path

    def get_cluster_tri_path(self, cluster_id: int) -> Optional[List[int]]:
        """Get the tri_path for a cluster if known.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Tri_path list or None if not known
        """
        return self._cluster_to_tri_path.get(cluster_id)


class SimpleFoveationPolicy(MemoryPolicy):
    """Simplified foveation policy for testing and demos.

    This policy uses basic heuristics and doesn't require actual
    embeddings or clustering to work.
    """

    def select_addresses(
        self,
        actor_id: str,
        world: str,
        region: str,
        query_embedding: np.ndarray,
        candidate_addresses: List[Tuple[FractalAddress, float]],
        token_budget: int,
        max_depth: Optional[int] = None,
    ) -> List[Tuple[FractalAddress, float]]:
        """Simplified address selection.

        Just takes top-K by score up to budget, without depth-based foveation.

        Args:
            actor_id: Actor/user ID
            world: World/namespace
            region: Region to search
            query_embedding: Embedded query (unused in simple policy)
            candidate_addresses: List of (address, score) tuples
            token_budget: Maximum tokens to allocate
            max_depth: Optional maximum depth constraint

        Returns:
            List of selected (address, score) tuples
        """
        if not candidate_addresses:
            return []

        # Filter by max_depth if specified
        if max_depth is not None:
            candidate_addresses = [
                (addr, score) for addr, score in candidate_addresses if addr.depth <= max_depth
            ]

        # Sort by score
        sorted_candidates = sorted(candidate_addresses, key=lambda x: x[1], reverse=True)

        # Simple selection: take top K addresses up to budget
        avg_tokens_per_addr = 100
        max_count = token_budget // avg_tokens_per_addr
        return sorted_candidates[:max_count]
