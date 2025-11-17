"""Memory service orchestration for FGMS."""

import hashlib
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from .addresses import address_to_str
from .models import (
    FractalAddress,
    MemoryContextItem,
    MemoryReadRequest,
    MemoryReadResponse,
    MemoryRecord,
    MemorySpan,
    MemoryWriteRequest,
    MemoryWriteResponse,
    RegionInfo,
    RegionsListResponse,
    AddressInfo,
    AddressesListResponse,
)
from .policy import MemoryPolicy
from .store import MemoryStore


class FGTAdapter:
    """Adapter interface for FGT components.

    This provides a simplified interface to FGT's phrase extraction,
    clustering, and glyph mapping functionality.
    """

    def __init__(self):
        """Initialize FGT adapter."""
        # In a full implementation, this would initialize:
        # - Embedding model
        # - Phrase extractor
        # - Clusterer
        # - Glyph manager
        pass

    def extract_and_encode(
        self, text: str, language: str = "en"
    ) -> List[Tuple[str, List[str], int]]:
        """Extract phrases and map to glyphs.

        Args:
            text: Input text
            language: Language code

        Returns:
            List of (span_text, glyph_ids, cluster_id) tuples

        Note:
            This is a placeholder implementation. In a full system, this would:
            1. Extract phrases using phrase matcher
            2. Embed phrases
            3. Assign to clusters
            4. Map clusters to glyph IDs
        """
        # Simple placeholder: treat sentences as spans
        sentences = text.split(". ")
        results = []

        for i, sent in enumerate(sentences):
            if not sent.strip():
                continue

            # Generate deterministic cluster_id and glyphs
            cluster_id = abs(hash(sent.lower())) % 1000

            # Simple glyph mapping (placeholder)
            glyph_ids = self._generate_glyphs(sent, cluster_id)

            results.append((sent.strip(), glyph_ids, cluster_id))

        return results

    def _generate_glyphs(self, text: str, cluster_id: int) -> List[str]:
        """Generate placeholder glyph IDs.

        Args:
            text: Text to generate glyphs for
            cluster_id: Cluster identifier

        Returns:
            List of glyph IDs (Mandarin characters)
        """
        # Use a deterministic hash to select glyphs
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Map to Mandarin character range (simplified)
        # Using common characters from the glyph set
        glyphs_pool = [
            "谷阜",
            "嶽岭",
            "峰岭",
            "崖岚",
            "峦峪",
            "岱岫",
            "巒嶂",
            "崦崧",
        ]
        idx = int(text_hash[:8], 16) % len(glyphs_pool)
        return [glyphs_pool[idx]]

    def embed_query(self, query: str, language: str = "en") -> np.ndarray:
        """Embed a query string.

        Args:
            query: Query text
            language: Language code

        Returns:
            Embedding vector

        Note:
            Placeholder implementation returns random vector.
        """
        # Deterministic "embedding" based on query hash
        query_hash = hashlib.md5(query.encode()).digest()
        np.random.seed(int.from_bytes(query_hash[:4], "big"))
        return np.random.randn(384).astype(np.float32)

    def estimate_tokens(self, text: str, glyphs: List[str]) -> int:
        """Estimate token count for text/glyphs.

        Args:
            text: Text content
            glyphs: Glyph IDs

        Returns:
            Estimated token count
        """
        # Simple heuristic: 1 word ~= 1.3 tokens, 1 glyph ~= 1 token
        word_count = len(text.split())
        glyph_count = len(glyphs)
        return int(word_count * 1.3 + glyph_count)


class MemoryService:
    """Core memory service orchestration.

    Handles write and read operations, coordinating between:
    - MemoryStore (persistence)
    - MemoryPolicy (address assignment & foveation)
    - FGTAdapter (phrase extraction & glyph mapping)
    """

    def __init__(
        self,
        store: MemoryStore,
        policy: MemoryPolicy,
        fgt_adapter: Optional[FGTAdapter] = None,
    ):
        """Initialize memory service.

        Args:
            store: MemoryStore instance
            policy: MemoryPolicy instance
            fgt_adapter: Optional FGTAdapter instance (creates default if None)
        """
        self.store = store
        self.policy = policy
        self.fgt = fgt_adapter or FGTAdapter()

    async def handle_write(self, req: MemoryWriteRequest) -> MemoryWriteResponse:
        """Handle a memory write request.

        Args:
            req: MemoryWriteRequest

        Returns:
            MemoryWriteResponse
        """
        try:
            # Resolve world and region
            world, region = self.policy.resolve_world_and_region(req.actor_id, req.region)

            # Extract phrases and map to glyphs
            language = "en"  # Could detect from text
            encoded_spans = self.fgt.extract_and_encode(req.text, language)

            # Create memory spans with addresses
            spans = []
            addresses = []
            total_tokens = 0
            glyph_tokens = 0

            for span_text, glyph_ids, cluster_id in encoded_spans:
                # Assign address
                address = self.policy.assign_address(
                    world=world,
                    region=region,
                    cluster_id=cluster_id,
                )

                # Create span
                span = MemorySpan(
                    address=address,
                    glyph_ids=glyph_ids,
                    text=span_text,
                    language=language,
                    meta={"cluster_id": str(cluster_id)},
                )
                spans.append(span)
                addresses.append(address_to_str(address))

                # Track tokens
                span_tokens = self.fgt.estimate_tokens(span_text, glyph_ids)
                total_tokens += span_tokens
                glyph_tokens += len(glyph_ids)

            # Calculate glyph density
            glyph_density = glyph_tokens / max(total_tokens, 1)

            # Create memory record
            record = MemoryRecord(
                id=str(uuid.uuid4()),
                actor_id=req.actor_id,
                created_at=datetime.utcnow(),
                world=world,
                region=region,
                spans=spans,
                tags=req.tags or [],
                raw_text=req.text,
                source=req.source,
                extra={},
            )

            # Save to store
            self.store.save_record(record)

            return MemoryWriteResponse(
                status="ok",
                world=world,
                region=region,
                addresses=addresses,
                glyph_density=glyph_density,
            )

        except Exception as e:
            return MemoryWriteResponse(
                status="error",
                world="",
                region="",
                addresses=[],
                glyph_density=0.0,
                error=str(e),
            )

    async def handle_read(self, req: MemoryReadRequest) -> MemoryReadResponse:
        """Handle a memory read request.

        Args:
            req: MemoryReadRequest

        Returns:
            MemoryReadResponse
        """
        try:
            # Resolve world and region
            world, region = self.policy.resolve_world_and_region(
                req.actor_id,
                req.focus.region if req.focus else None,
            )

            # Embed query
            query_embedding = self.fgt.embed_query(req.query)

            # Get recent records for this actor/region
            records = list(
                self.store.get_records_for_actor(req.actor_id, region, limit=1000)
            )

            # Extract candidate addresses with scores
            candidate_addresses = self._score_addresses(
                records, query_embedding, req.focus.max_depth if req.focus else None
            )

            # Use policy to select addresses
            selected = self.policy.select_addresses(
                actor_id=req.actor_id,
                world=world,
                region=region,
                query_embedding=query_embedding,
                candidate_addresses=candidate_addresses,
                token_budget=req.token_budget,
                max_depth=req.focus.max_depth if req.focus else None,
            )

            # Retrieve spans for selected addresses
            selected_addrs = [addr for addr, score in selected]
            spans_with_addrs = self.store.get_spans_by_address(selected_addrs)

            # Build context items
            context = self._build_context_items(
                spans_with_addrs,
                selected,
                req.mode,
            )

            # Estimate total tokens
            token_estimate = sum(
                self.fgt.estimate_tokens(item.excerpt or item.summary or "", item.glyphs)
                for item in context
            )

            return MemoryReadResponse(
                status="ok",
                world=world,
                region=region,
                mode=req.mode,
                context=context,
                token_estimate=token_estimate,
            )

        except Exception as e:
            return MemoryReadResponse(
                status="error",
                world="",
                region="",
                mode=req.mode,
                context=[],
                token_estimate=0,
                error=str(e),
            )

    def _score_addresses(
        self,
        records: List[MemoryRecord],
        query_embedding: np.ndarray,
        max_depth: Optional[int],
    ) -> List[Tuple[FractalAddress, float]]:
        """Score addresses by relevance to query.

        Args:
            records: List of memory records
            query_embedding: Query embedding vector
            max_depth: Optional maximum depth filter

        Returns:
            List of (address, score) tuples
        """
        scored = []

        for record in records:
            for span in record.spans:
                # Skip if exceeds max_depth
                if max_depth is not None and span.address.depth > max_depth:
                    continue

                # Generate span embedding (placeholder)
                span_embedding = self.fgt.embed_query(span.text)

                # Compute cosine similarity
                score = float(
                    np.dot(query_embedding, span_embedding)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(span_embedding) + 1e-9)
                )

                scored.append((span.address, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _build_context_items(
        self,
        spans_with_addrs: List[Tuple[FractalAddress, MemorySpan]],
        selected_scored: List[Tuple[FractalAddress, float]],
        mode: str,
    ) -> List[MemoryContextItem]:
        """Build context items from spans.

        Args:
            spans_with_addrs: List of (address, span) tuples
            selected_scored: List of (address, score) tuples
            mode: Context mode (glyph/text/mixed)

        Returns:
            List of MemoryContextItem
        """
        # Create address -> span mapping
        addr_to_span = {address_to_str(addr): span for addr, span in spans_with_addrs}

        # Create address -> score mapping
        addr_to_score = {address_to_str(addr): score for addr, score in selected_scored}

        items = []

        for addr_str, span in addr_to_span.items():
            score = addr_to_score.get(addr_str, 0.5)

            if mode == "glyph":
                # Glyph-only mode: minimal text
                items.append(
                    MemoryContextItem(
                        address=addr_str,
                        glyphs=span.glyph_ids,
                        summary=None,
                        excerpt=None,
                        score=score,
                    )
                )
            elif mode == "text":
                # Text mode: full text, no glyphs
                items.append(
                    MemoryContextItem(
                        address=addr_str,
                        glyphs=[],
                        summary=None,
                        excerpt=span.text,
                        score=score,
                    )
                )
            else:  # mixed
                # Mixed mode: glyphs + summary/excerpt
                items.append(
                    MemoryContextItem(
                        address=addr_str,
                        glyphs=span.glyph_ids,
                        summary=span.text[:100] + "..." if len(span.text) > 100 else span.text,
                        excerpt=span.text if len(span.text) <= 200 else None,
                        score=score,
                    )
                )

        return items

    async def list_regions(self, actor_id: str) -> RegionsListResponse:
        """List regions for an actor.

        Args:
            actor_id: Actor/user ID

        Returns:
            RegionsListResponse
        """
        try:
            regions = self.store.list_regions(actor_id)
            region_infos = []

            for region in regions:
                stats = self.store.get_region_stats(actor_id, region)
                region_infos.append(
                    RegionInfo(
                        region=region,
                        record_count=stats["record_count"],
                        span_count=stats["span_count"],
                        first_timestamp=stats["first_timestamp"],
                        last_timestamp=stats["last_timestamp"],
                    )
                )

            return RegionsListResponse(
                status="ok",
                actor_id=actor_id,
                regions=region_infos,
            )

        except Exception as e:
            return RegionsListResponse(
                status="error",
                actor_id=actor_id,
                regions=[],
                error=str(e),
            )

    async def list_addresses(
        self, actor_id: str, region: str, limit: int = 100
    ) -> AddressesListResponse:
        """List addresses for an actor in a region.

        Args:
            actor_id: Actor/user ID
            region: Region to query
            limit: Maximum number of addresses

        Returns:
            AddressesListResponse
        """
        try:
            addresses = self.store.list_addresses(actor_id, region, limit)

            # Get span information for each address
            addr_infos = []
            spans_by_addr = self.store.get_spans_by_address(addresses)

            # Count spans per address
            span_counts = {}
            for addr, _ in spans_by_addr:
                addr_str = address_to_str(addr)
                span_counts[addr_str] = span_counts.get(addr_str, 0) + 1

            # Get timestamps (use current time as placeholder)
            for addr in addresses:
                addr_str = address_to_str(addr)
                addr_infos.append(
                    AddressInfo(
                        address=addr_str,
                        span_count=span_counts.get(addr_str, 0),
                        created_at=datetime.utcnow(),
                    )
                )

            return AddressesListResponse(
                status="ok",
                actor_id=actor_id,
                region=region,
                addresses=addr_infos,
            )

        except Exception as e:
            return AddressesListResponse(
                status="error",
                actor_id=actor_id,
                region=region,
                addresses=[],
                error=str(e),
            )


def create_memory_service(
    store: Optional[MemoryStore] = None,
    policy: Optional[MemoryPolicy] = None,
    fgt_adapter: Optional[FGTAdapter] = None,
) -> MemoryService:
    """Factory function to create a MemoryService with defaults.

    Args:
        store: Optional MemoryStore (creates SQLite store if None)
        policy: Optional MemoryPolicy (creates default if None)
        fgt_adapter: Optional FGTAdapter (creates default if None)

    Returns:
        MemoryService instance
    """
    from .policy import MemoryPolicyConfig
    from .store import SQLiteMemoryStore

    if store is None:
        store = SQLiteMemoryStore()

    if policy is None:
        policy = MemoryPolicy(MemoryPolicyConfig())

    if fgt_adapter is None:
        fgt_adapter = FGTAdapter()

    return MemoryService(store=store, policy=policy, fgt_adapter=fgt_adapter)
