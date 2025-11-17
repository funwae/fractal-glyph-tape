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
