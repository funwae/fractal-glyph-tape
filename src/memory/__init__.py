"""Fractal Glyph Memory Service (FGMS) - Phase 4.

This module provides an Agent Memory OS with fractal-addressable phrase memory:

- **Data Models**: FractalAddress, MemorySpan, MemoryRecord, API request/response models
- **Addressing**: Logical addressing schema with fractal coordinates
- **Policy**: Memory policy and foveation logic for read/write operations
- **Store**: Persistence layer (SQLite, in-memory)
- **Service**: Core orchestration connecting all components
- **API**: FastAPI endpoints for HTTP access
- **Client**: In-process Python client for direct access

Basic Usage:

    from src.memory import create_memory_service, MemoryClient

    # Create service with defaults
    service = create_memory_service()
    client = MemoryClient(service)

    # Write memory
    response = await client.write(
        actor_id="hayden",
        text="I want to build a memory service",
        tags=["task"],
    )

    # Read memory
    context = await client.read(
        actor_id="hayden",
        query="What did I want to build?",
        token_budget=2048,
        mode="mixed",
    )

HTTP API Usage:

    from src.memory.api import create_memory_api
    import uvicorn

    app = create_memory_api()
    uvicorn.run(app, host="0.0.0.0", port=8001)

For more details, see:
- docs/210-phase-4-agent-memory-and-api.md
- docs/211-agent-memory-eval-and-test-plan.md
"""

# Core data models
from .models import (
    AddressesListResponse,
    AddressInfo,
    FractalAddress,
    MemoryContextItem,
    MemoryReadFocus,
    MemoryReadRequest,
    MemoryReadResponse,
    MemoryRecord,
    MemorySpan,
    MemoryWriteRequest,
    MemoryWriteResponse,
    RegionInfo,
    RegionsListResponse,
)

# Addressing utilities
from .addresses import (
    address_from_str,
    address_to_str,
    addresses_equal,
    create_address,
    get_address_depth_level,
    get_address_region,
    increment_time_slice,
    parse_region_from_actor,
    validate_address_str,
    with_depth,
)

# Policy and foveation
from .policy import (
    ClusterInfo,
    MemoryPolicy,
    MemoryPolicyConfig,
    SimpleFoveationPolicy,
)

# Storage
from .store import InMemoryStore, MemoryStore, SQLiteMemoryStore

# Service
from .service import FGTAdapter, MemoryService, create_memory_service

# API
from .api import create_memory_api, get_memory_service, router, set_memory_service

# Client
from .client import MemoryClient, SyncMemoryClient

__all__ = [
    # Models
    "FractalAddress",
    "MemorySpan",
    "MemoryRecord",
    "MemoryWriteRequest",
    "MemoryWriteResponse",
    "MemoryReadRequest",
    "MemoryReadResponse",
    "MemoryReadFocus",
    "MemoryContextItem",
    "RegionInfo",
    "RegionsListResponse",
    "AddressInfo",
    "AddressesListResponse",
    # Addressing
    "address_to_str",
    "address_from_str",
    "validate_address_str",
    "create_address",
    "parse_region_from_actor",
    "addresses_equal",
    "get_address_depth_level",
    "get_address_region",
    "increment_time_slice",
    "with_depth",
    # Policy
    "MemoryPolicy",
    "MemoryPolicyConfig",
    "SimpleFoveationPolicy",
    "ClusterInfo",
    # Store
    "MemoryStore",
    "SQLiteMemoryStore",
    "InMemoryStore",
    # Service
    "MemoryService",
    "FGTAdapter",
    "create_memory_service",
    # API
    "router",
    "create_memory_api",
    "get_memory_service",
    "set_memory_service",
    # Client
    "MemoryClient",
    "SyncMemoryClient",
]

__version__ = "0.1.0"
