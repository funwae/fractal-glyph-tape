"""FastAPI endpoints for Fractal Glyph Memory Service."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from .models import (
    AddressesListResponse,
    MemoryReadRequest,
    MemoryReadResponse,
    MemoryWriteRequest,
    MemoryWriteResponse,
    RegionsListResponse,
)
from .service import MemoryService, create_memory_service

# Global service instance (will be initialized on startup)
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Dependency injection for memory service.

    Returns:
        MemoryService instance

    Raises:
        HTTPException: If service is not initialized
    """
    global _memory_service
    if _memory_service is None:
        _memory_service = create_memory_service()
    return _memory_service


def set_memory_service(service: MemoryService) -> None:
    """Set the global memory service instance.

    Args:
        service: MemoryService to use
    """
    global _memory_service
    _memory_service = service


# Create router
router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.post("/write", response_model=MemoryWriteResponse)
async def write_memory(
    req: MemoryWriteRequest,
    service: MemoryService = Depends(get_memory_service),
) -> MemoryWriteResponse:
    """Write a new memory record.

    This endpoint appends a new memory record for an actor, performing:
    1. Phrase extraction and glyph mapping
    2. Address assignment
    3. Persistence to memory store

    Args:
        req: MemoryWriteRequest containing text and metadata
        service: MemoryService dependency

    Returns:
        MemoryWriteResponse with status and created addresses

    Example:
        ```bash
        curl -X POST http://localhost:8001/api/memory/write \\
          -H "Content-Type: application/json" \\
          -d '{
            "actor_id": "hayden",
            "text": "I want to build a memory service",
            "tags": ["task"],
            "source": "user"
          }'
        ```
    """
    response = await service.handle_write(req)

    if response.status == "error":
        raise HTTPException(status_code=500, detail=response.error)

    return response


@router.post("/read", response_model=MemoryReadResponse)
async def read_memory(
    req: MemoryReadRequest,
    service: MemoryService = Depends(get_memory_service),
) -> MemoryReadResponse:
    """Read memory context for a query.

    This endpoint retrieves foveated, glyph-aware context for an actor's query:
    1. Embeds the query
    2. Scores candidate memory addresses by relevance
    3. Applies foveation policy to select addresses within token budget
    4. Returns context items with glyphs, summaries, and/or excerpts

    Args:
        req: MemoryReadRequest containing query and parameters
        service: MemoryService dependency

    Returns:
        MemoryReadResponse with context items and token estimate

    Example:
        ```bash
        curl -X POST http://localhost:8001/api/memory/read \\
          -H "Content-Type: application/json" \\
          -d '{
            "actor_id": "hayden",
            "query": "What did I say about the memory service?",
            "token_budget": 2048,
            "mode": "mixed"
          }'
        ```
    """
    response = await service.handle_read(req)

    if response.status == "error":
        raise HTTPException(status_code=500, detail=response.error)

    return response


@router.get("/regions", response_model=RegionsListResponse)
async def list_regions(
    actor_id: str = Query(..., description="Actor/user ID"),
    service: MemoryService = Depends(get_memory_service),
) -> RegionsListResponse:
    """List all memory regions for an actor.

    Args:
        actor_id: Actor/user ID
        service: MemoryService dependency

    Returns:
        RegionsListResponse with region information

    Example:
        ```bash
        curl http://localhost:8001/api/memory/regions?actor_id=hayden
        ```
    """
    response = await service.list_regions(actor_id)

    if response.status == "error":
        raise HTTPException(status_code=500, detail=response.error)

    return response


@router.get("/addresses", response_model=AddressesListResponse)
async def list_addresses(
    actor_id: str = Query(..., description="Actor/user ID"),
    region: str = Query(..., description="Region to query"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of addresses"),
    service: MemoryService = Depends(get_memory_service),
) -> AddressesListResponse:
    """List memory addresses for an actor in a region.

    Args:
        actor_id: Actor/user ID
        region: Region to query
        limit: Maximum number of addresses to return
        service: MemoryService dependency

    Returns:
        AddressesListResponse with address information

    Example:
        ```bash
        curl "http://localhost:8001/api/memory/addresses?actor_id=hayden&region=agent:hayden&limit=50"
        ```
    """
    response = await service.list_addresses(actor_id, region, limit)

    if response.status == "error":
        raise HTTPException(status_code=500, detail=response.error)

    return response


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Dict with status information

    Example:
        ```bash
        curl http://localhost:8001/api/memory/health
        ```
    """
    return {
        "status": "ok",
        "service": "Fractal Glyph Memory Service",
        "version": "0.1.0",
    }


# Convenience function to create FastAPI app with memory router
def create_memory_api(service: Optional[MemoryService] = None):
    """Create a FastAPI application with memory endpoints.

    Args:
        service: Optional MemoryService instance (creates default if None)

    Returns:
        FastAPI application

    Example:
        ```python
        from fastapi import FastAPI
        from src.memory.api import router, set_memory_service
        from src.memory.service import create_memory_service

        app = FastAPI()
        app.include_router(router)

        # Optional: configure service
        service = create_memory_service()
        set_memory_service(service)
        ```
    """
    from fastapi import FastAPI

    app = FastAPI(
        title="Fractal Glyph Memory Service",
        description="Agent Memory OS with fractal-addressable phrase memory",
        version="0.1.0",
    )

    if service is not None:
        set_memory_service(service)

    app.include_router(router)

    return app
