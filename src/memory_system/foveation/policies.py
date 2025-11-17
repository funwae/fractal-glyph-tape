"""
Foveation policies for memory retrieval

Policies determine how to prioritize and select memories under token budgets.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from ..models import MemoryEntry


class FoveationPolicy(ABC):
    """
    Abstract base class for foveation policies.

    A policy determines which memories to include in the context
    and in what order, given a token budget.
    """

    @abstractmethod
    def select_memories(
        self,
        memories: List[MemoryEntry],
        query: Optional[str],
        token_budget: int
    ) -> List[MemoryEntry]:
        """
        Select and order memories according to the policy.

        Args:
            memories: Available memory entries
            query: Optional query for relevance ranking
            token_budget: Maximum tokens to use

        Returns:
            Ordered list of selected memories
        """
        pass


class RecentPolicy(FoveationPolicy):
    """Select most recent memories first."""

    def select_memories(
        self,
        memories: List[MemoryEntry],
        query: Optional[str],
        token_budget: int
    ) -> List[MemoryEntry]:
        """Select most recent memories within budget."""
        # Sort by creation time (newest first)
        sorted_memories = sorted(
            memories,
            key=lambda m: m.created_at,
            reverse=True
        )

        # Select within budget
        selected = []
        tokens_used = 0

        for memory in sorted_memories:
            if tokens_used + memory.token_estimate <= token_budget:
                selected.append(memory)
                tokens_used += memory.token_estimate
            else:
                break

        return selected


class RelevantPolicy(FoveationPolicy):
    """Select memories by semantic relevance to query."""

    def select_memories(
        self,
        memories: List[MemoryEntry],
        query: Optional[str],
        token_budget: int
    ) -> List[MemoryEntry]:
        """Select most relevant memories within budget."""
        if not query:
            # Fallback to recent if no query
            return RecentPolicy().select_memories(memories, query, token_budget)

        # Simple keyword-based relevance (in production, use embeddings)
        query_terms = set(query.lower().split())

        def relevance_score(memory: MemoryEntry) -> float:
            text_terms = set(memory.text.lower().split())
            tag_terms = set(tag.lower() for tag in memory.tags)
            all_terms = text_terms | tag_terms

            # Jaccard similarity
            intersection = query_terms & all_terms
            union = query_terms | all_terms
            return len(intersection) / len(union) if union else 0.0

        # Sort by relevance
        sorted_memories = sorted(
            memories,
            key=relevance_score,
            reverse=True
        )

        # Select within budget
        selected = []
        tokens_used = 0

        for memory in sorted_memories:
            if tokens_used + memory.token_estimate <= token_budget:
                selected.append(memory)
                tokens_used += memory.token_estimate
            else:
                break

        return selected


class MixedPolicy(FoveationPolicy):
    """Blend recent and relevant memories."""

    def __init__(self, recent_weight: float = 0.3, relevant_weight: float = 0.7):
        """
        Initialize mixed policy.

        Args:
            recent_weight: Weight for recency (0-1)
            relevant_weight: Weight for relevance (0-1)
        """
        self.recent_weight = recent_weight
        self.relevant_weight = relevant_weight

        # Normalize weights
        total = recent_weight + relevant_weight
        self.recent_weight /= total
        self.relevant_weight /= total

    def select_memories(
        self,
        memories: List[MemoryEntry],
        query: Optional[str],
        token_budget: int
    ) -> List[MemoryEntry]:
        """Select mixed recent + relevant memories within budget."""
        if not query:
            return RecentPolicy().select_memories(memories, query, token_budget)

        # Calculate recency scores (normalize to 0-1)
        if not memories:
            return []

        max_timestamp = max(m.created_at.timestamp() for m in memories)
        min_timestamp = min(m.created_at.timestamp() for m in memories)
        timestamp_range = max_timestamp - min_timestamp or 1

        def recency_score(memory: MemoryEntry) -> float:
            return (memory.created_at.timestamp() - min_timestamp) / timestamp_range

        # Calculate relevance scores
        query_terms = set(query.lower().split())

        def relevance_score(memory: MemoryEntry) -> float:
            text_terms = set(memory.text.lower().split())
            tag_terms = set(tag.lower() for tag in memory.tags)
            all_terms = text_terms | tag_terms

            intersection = query_terms & all_terms
            union = query_terms | all_terms
            return len(intersection) / len(union) if union else 0.0

        # Combined score
        def combined_score(memory: MemoryEntry) -> float:
            recency = recency_score(memory)
            relevance = relevance_score(memory)
            return (self.recent_weight * recency +
                    self.relevant_weight * relevance)

        # Sort by combined score
        sorted_memories = sorted(
            memories,
            key=combined_score,
            reverse=True
        )

        # Select within budget
        selected = []
        tokens_used = 0

        for memory in sorted_memories:
            if tokens_used + memory.token_estimate <= token_budget:
                selected.append(memory)
                tokens_used += memory.token_estimate
            else:
                break

        return selected


class FoveatedPolicy(FoveationPolicy):
    """
    FGT-FOVEATED policy from Phase 5 benchmark.

    Allocates budget across three zones:
    - 30% to early turns (first 1-3 turns)
    - 30% to relevant turns (semantic match to query)
    - 40% to recent turns (last N turns)

    This policy achieved +46.7pp accuracy improvement over naive truncation
    at 256-token budgets in Phase 5 benchmarks.
    """

    def __init__(
        self,
        early_weight: float = 0.30,
        relevant_weight: float = 0.30,
        recent_weight: float = 0.40
    ):
        """
        Initialize foveated policy.

        Args:
            early_weight: Fraction of budget for early turns (default 0.30)
            relevant_weight: Fraction for relevant turns (default 0.30)
            recent_weight: Fraction for recent turns (default 0.40)
        """
        self.early_weight = early_weight
        self.relevant_weight = relevant_weight
        self.recent_weight = recent_weight

    def select_memories(
        self,
        memories: List[MemoryEntry],
        query: Optional[str],
        token_budget: int
    ) -> List[MemoryEntry]:
        """
        Select memories using three-zone allocation.

        1. Allocate sub-budgets to each zone
        2. Select memories for each zone independently
        3. Merge and deduplicate
        4. Return in chronological order
        """
        if not memories:
            return []

        # Sort memories by creation time
        sorted_by_time = sorted(memories, key=lambda m: m.created_at)

        # Zone budgets
        early_budget = int(token_budget * self.early_weight)
        relevant_budget = int(token_budget * self.relevant_weight)
        recent_budget = int(token_budget * self.recent_weight)

        # Zone 1: EARLY (first 1-3 turns)
        early_memories = self._select_early(sorted_by_time, early_budget)

        # Zone 2: RELEVANT (semantic match to query)
        relevant_memories = self._select_relevant(memories, query, relevant_budget)

        # Zone 3: RECENT (last N turns)
        recent_memories = self._select_recent(sorted_by_time, recent_budget)

        # Merge and deduplicate
        selected = self._merge_zones(early_memories, relevant_memories, recent_memories)

        # Respect total budget (in case of overlap reducing actual token usage)
        final = []
        tokens_used = 0
        for memory in sorted(selected, key=lambda m: m.created_at):
            if tokens_used + memory.token_estimate <= token_budget:
                final.append(memory)
                tokens_used += memory.token_estimate

        return final

    def _select_early(
        self,
        sorted_memories: List[MemoryEntry],
        budget: int
    ) -> List[MemoryEntry]:
        """Select first 1-3 turns within budget."""
        selected = []
        tokens_used = 0

        for memory in sorted_memories[:3]:  # First 3 turns max
            if tokens_used + memory.token_estimate <= budget:
                selected.append(memory)
                tokens_used += memory.token_estimate

        return selected

    def _select_relevant(
        self,
        memories: List[MemoryEntry],
        query: Optional[str],
        budget: int
    ) -> List[MemoryEntry]:
        """Select semantically relevant memories within budget."""
        if not query:
            return []

        # Score by keyword overlap (TODO: use embeddings in future)
        query_terms = set(query.lower().split())

        def relevance_score(memory: MemoryEntry) -> float:
            text_terms = set(memory.text.lower().split())
            tag_terms = set(tag.lower() for tag in memory.tags)
            all_terms = text_terms | tag_terms

            intersection = query_terms & all_terms
            union = query_terms | all_terms
            return len(intersection) / len(union) if union else 0.0

        sorted_by_relevance = sorted(
            memories,
            key=relevance_score,
            reverse=True
        )

        selected = []
        tokens_used = 0

        for memory in sorted_by_relevance:
            if tokens_used + memory.token_estimate <= budget:
                selected.append(memory)
                tokens_used += memory.token_estimate

        return selected

    def _select_recent(
        self,
        sorted_memories: List[MemoryEntry],
        budget: int
    ) -> List[MemoryEntry]:
        """Select most recent memories within budget."""
        selected = []
        tokens_used = 0

        for memory in reversed(sorted_memories):
            if tokens_used + memory.token_estimate <= budget:
                selected.insert(0, memory)  # Insert at front to maintain chronological order
                tokens_used += memory.token_estimate

        return selected

    def _merge_zones(
        self,
        early: List[MemoryEntry],
        relevant: List[MemoryEntry],
        recent: List[MemoryEntry]
    ) -> List[MemoryEntry]:
        """Merge three zones, deduplicating by entry_id."""
        seen = set()
        merged = []

        for memory in early + relevant + recent:
            if memory.entry_id not in seen:
                seen.add(memory.entry_id)
                merged.append(memory)

        return merged
