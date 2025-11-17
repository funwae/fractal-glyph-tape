"""
Foveation engine - orchestrates memory retrieval with policies
"""
from typing import List, Optional, Dict, Any
from ..models import MemoryEntry
from ..storage import SQLiteMemoryStore
from .policies import FoveationPolicy, MixedPolicy, RecentPolicy, RelevantPolicy


class FoveationEngine:
    """
    Orchestrates memory retrieval using foveation policies.

    The engine:
    1. Retrieves candidate memories from storage
    2. Applies a policy to select and order memories
    3. Returns memories that fit within token budget
    """

    def __init__(self, store: SQLiteMemoryStore):
        """
        Initialize foveation engine.

        Args:
            store: Memory storage backend
        """
        self.store = store
        self.policies = {
            "recent": RecentPolicy(),
            "relevant": RelevantPolicy(),
            "mixed": MixedPolicy(recent_weight=0.3, relevant_weight=0.7)
        }

    def retrieve(
        self,
        actor_id: str,
        query: Optional[str] = None,
        token_budget: int = 2048,
        mode: str = "mixed",
        world: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_candidates: int = 500
    ) -> Dict[str, Any]:
        """
        Retrieve memories using foveation policy.

        Args:
            actor_id: Actor whose memories to retrieve
            query: Optional query for relevance ranking
            token_budget: Maximum tokens to include
            mode: Policy mode ("recent", "relevant", "mixed")
            world: Optional world filter
            region: Optional region filter
            tags: Optional tag filters
            max_candidates: Maximum candidates to consider

        Returns:
            Dictionary with:
                - memories: Selected memory entries
                - addresses: List of fractal addresses
                - glyphs: List of unique glyphs used
                - token_estimate: Total tokens used
                - policy: Policy used
        """
        # Get policy
        policy = self.policies.get(mode, self.policies["mixed"])

        # Retrieve candidate memories
        if query:
            # Use full-text search
            candidates = self.store.search(
                query=query,
                actor_id=actor_id,
                limit=max_candidates
            )
        else:
            # Use filtered read
            candidates = self.store.read(
                actor_id=actor_id,
                world=world,
                region=region,
                tags=tags,
                limit=max_candidates
            )

        # Apply policy to select memories within budget
        selected_memories = policy.select_memories(
            memories=candidates,
            query=query,
            token_budget=token_budget
        )

        # Extract addresses and glyphs
        addresses = [m.address.to_string() for m in selected_memories]

        # Collect unique glyphs
        glyph_map = {}
        for memory in selected_memories:
            for glyph in memory.glyphs:
                if glyph.glyph_id not in glyph_map:
                    glyph_map[glyph.glyph_id] = glyph

        glyphs = [g.to_dict() for g in glyph_map.values()]

        # Calculate total tokens
        total_tokens = sum(m.token_estimate for m in selected_memories)

        return {
            "memories": [m.to_dict() for m in selected_memories],
            "addresses": addresses,
            "glyphs": glyphs,
            "token_estimate": int(total_tokens),
            "policy": mode,
            "candidates_considered": len(candidates),
            "memories_selected": len(selected_memories)
        }
