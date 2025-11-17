"""In-process client for Fractal Glyph Memory Service."""

from typing import List, Optional

from .models import (
    AddressesListResponse,
    MemoryReadRequest,
    MemoryReadResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
    RegionsListResponse,
)
from .service import MemoryService


class MemoryClient:
    """In-process client for interacting with memory service.

    This provides a thin wrapper around MemoryService for use in Python code
    without going through HTTP.

    Example:
        ```python
        from src.memory.client import MemoryClient
        from src.memory.service import create_memory_service

        service = create_memory_service()
        client = MemoryClient(service)

        # Write memory
        response = await client.write(
            actor_id="hayden",
            text="I want to build an agent with memory",
            tags=["task", "planning"],
        )

        # Read memory
        context = await client.read(
            actor_id="hayden",
            query="What did I want to build?",
            token_budget=2048,
        )
        ```
    """

    def __init__(self, service: MemoryService):
        """Initialize memory client.

        Args:
            service: MemoryService instance to use
        """
        self.service = service

    async def write(
        self,
        actor_id: str,
        text: str,
        tags: Optional[List[str]] = None,
        region: Optional[str] = None,
        source: str = "user",
    ) -> MemoryWriteResponse:
        """Write a new memory record.

        Args:
            actor_id: Actor/user ID
            text: Text to write to memory
            tags: Optional list of tags
            region: Optional region override
            source: Source of the memory ("user", "assistant", "system")

        Returns:
            MemoryWriteResponse with status and addresses

        Example:
            ```python
            response = await client.write(
                actor_id="hayden",
                text="I need to implement Phase 4 of FGT",
                tags=["task"],
            )
            print(f"Created addresses: {response.addresses}")
            ```
        """
        req = MemoryWriteRequest(
            actor_id=actor_id,
            text=text,
            tags=tags,
            region=region,
            source=source,  # type: ignore
        )
        return await self.service.handle_write(req)

    async def read(
        self,
        actor_id: str,
        query: str,
        token_budget: int = 2048,
        mode: str = "mixed",
        region: Optional[str] = None,
        max_depth: Optional[int] = None,
    ) -> MemoryReadResponse:
        """Read memory context for a query.

        Args:
            actor_id: Actor/user ID
            query: Query text for retrieval
            token_budget: Maximum tokens for context (default: 2048)
            mode: Context mode - "glyph", "text", or "mixed" (default: "mixed")
            region: Optional region to focus on
            max_depth: Optional maximum depth to retrieve

        Returns:
            MemoryReadResponse with context items

        Example:
            ```python
            response = await client.read(
                actor_id="hayden",
                query="What tasks do I need to complete?",
                token_budget=1024,
                mode="mixed",
            )
            for item in response.context:
                print(f"{item.address}: {item.summary}")
            ```
        """
        from .models import MemoryReadFocus

        focus = None
        if region is not None or max_depth is not None:
            focus = MemoryReadFocus(region=region, max_depth=max_depth)

        req = MemoryReadRequest(
            actor_id=actor_id,
            query=query,
            token_budget=token_budget,
            mode=mode,  # type: ignore
            focus=focus,
        )
        return await self.service.handle_read(req)

    async def list_regions(self, actor_id: str) -> RegionsListResponse:
        """List all memory regions for an actor.

        Args:
            actor_id: Actor/user ID

        Returns:
            RegionsListResponse with region information

        Example:
            ```python
            response = await client.list_regions("hayden")
            for region_info in response.regions:
                print(f"{region_info.region}: {region_info.record_count} records")
            ```
        """
        return await self.service.list_regions(actor_id)

    async def list_addresses(
        self, actor_id: str, region: str, limit: int = 100
    ) -> AddressesListResponse:
        """List memory addresses for an actor in a region.

        Args:
            actor_id: Actor/user ID
            region: Region to query
            limit: Maximum number of addresses (default: 100)

        Returns:
            AddressesListResponse with address information

        Example:
            ```python
            response = await client.list_addresses("hayden", "agent:hayden", limit=50)
            for addr_info in response.addresses:
                print(f"{addr_info.address}: {addr_info.span_count} spans")
            ```
        """
        return await self.service.list_addresses(actor_id, region, limit)


class SyncMemoryClient:
    """Synchronous wrapper for MemoryClient.

    This provides a blocking interface for use in non-async contexts.

    Example:
        ```python
        from src.memory.client import SyncMemoryClient
        from src.memory.service import create_memory_service

        service = create_memory_service()
        client = SyncMemoryClient(service)

        # Write memory (blocking)
        response = client.write(
            actor_id="hayden",
            text="I want to build an agent with memory",
        )
        ```
    """

    def __init__(self, service: MemoryService):
        """Initialize synchronous memory client.

        Args:
            service: MemoryService instance to use
        """
        self._async_client = MemoryClient(service)

    def write(
        self,
        actor_id: str,
        text: str,
        tags: Optional[List[str]] = None,
        region: Optional[str] = None,
        source: str = "user",
    ) -> MemoryWriteResponse:
        """Write a new memory record (blocking).

        See MemoryClient.write for details.
        """
        import asyncio

        return asyncio.run(
            self._async_client.write(
                actor_id=actor_id,
                text=text,
                tags=tags,
                region=region,
                source=source,
            )
        )

    def read(
        self,
        actor_id: str,
        query: str,
        token_budget: int = 2048,
        mode: str = "mixed",
        region: Optional[str] = None,
        max_depth: Optional[int] = None,
    ) -> MemoryReadResponse:
        """Read memory context for a query (blocking).

        See MemoryClient.read for details.
        """
        import asyncio

        return asyncio.run(
            self._async_client.read(
                actor_id=actor_id,
                query=query,
                token_budget=token_budget,
                mode=mode,
                region=region,
                max_depth=max_depth,
            )
        )

    def list_regions(self, actor_id: str) -> RegionsListResponse:
        """List all memory regions for an actor (blocking).

        See MemoryClient.list_regions for details.
        """
        import asyncio

        return asyncio.run(self._async_client.list_regions(actor_id))

    def list_addresses(
        self, actor_id: str, region: str, limit: int = 100
    ) -> AddressesListResponse:
        """List memory addresses for an actor in a region (blocking).

        See MemoryClient.list_addresses for details.
        """
        import asyncio

        return asyncio.run(self._async_client.list_addresses(actor_id, region, limit))
