# Phase 4 — Agent Memory OS & API (Fractal Glyph Memory Service)

**Status:** Implemented
**Depends on:** Phases 1–3 complete (FGT core, viz, metrics, LLM adapter, multilingual pipeline)

---

## 0. Executive Summary

Phase 4 turns Fractal Glyph Tape from a research pipeline into a **Memory OS for agents and assistants**:

- A **Fractal Glyph Memory Service (FGMS)** exposes:
  - `POST /memory/write` — append new text into the tape.
  - `POST /memory/read` — retrieve foveated, glyph-aware context.
- It wraps existing Phase-1/2/3 modules:
  - ingest → embed → cluster → glyph codec → hybrid tokenizer → LLM adapter → viz.
- It introduces a **stable internal data model**:
  - `MemoryRecord`, `MemorySpan`, `MemoryRegion`, `FractalAddress` (logical).
- It is designed to be **agent-facing**, not researcher-only:
  - simple API for any chat loop or tool agent,
  - clear configuration for memory policy and token budgets.

This document specifies:

1. Scope & assumptions
2. Directory layout & ownership
3. Data model
4. API (HTTP + in-process Python)
5. Read/write flows
6. Configuration & policies
7. Logging & observability hooks

---

## 1. Scope & Assumptions

### 1.1 In scope

- A single **Memory Service** that:
  - runs in the same Python environment as FGT,
  - exposes HTTP endpoints (FastAPI),
  - can also be called directly from Python.
- Integrates **only** text for Phase 4:
  - phrase extraction, clustering, glyph mapping, hybrid tokenization.
- Returns **glyph-aware context** suitable for:
  - LLM prompts,
  - debugging UI,
  - downstream experiments.

### 1.2 Out of scope (for Phase 4)

- Direct integration with:
  - fractal-video-tape (FVT),
  - glyph-core,
  - glyph-drive-3d,
  - TGO event streaming.
  (These are Phase 5+ and covered in other docs.)
- Fine-tuning LLMs (Phase 4 uses LLM adapter only; training is Phase 5).

### 1.3 Assumptions

- Existing modules:
  - `src/glyph/` — glyph codec (cluster ↔ glyph IDs).
  - `src/tokenizer/` — hybrid tokenizer, phrase matcher.
  - `src/embed/` — embedding models (monolingual/multilingual).
  - `src/cluster/` — clustering & tape-building logic.
  - `src/llm_adapter/` — context construction for LLMs.
  - `src/viz/` + `src/eval/` — visualization & metric utilities.
- Persistence:
  - You already persist necessary cluster/tape state to disk or DB as part of Phases 1–3.
  - Phase 4 adds a **lightweight metadata store** keyed by `actor_id`, regions, and addresses.

---

## 2. Implementation Status

### 2.1 Core Components

✅ **Implemented:**
- Data models (`src/memory/models.py`)
- Addressing system (`src/memory/addresses.py`)
- Memory policy & foveation (`src/memory/policy.py`)
- Memory store (SQLite & in-memory) (`src/memory/store.py`)
- Memory service orchestration (`src/memory/service.py`)
- FastAPI endpoints (`src/memory/api.py`)
- In-process client (`src/memory/client.py`)

### 2.2 API Endpoints

All endpoints are live at `/api/memory/`:

- `POST /api/memory/write` — Write new memory
- `POST /api/memory/read` — Read foveated context
- `GET /api/memory/regions` — List regions
- `GET /api/memory/addresses` — List addresses
- `GET /api/memory/health` — Health check

### 2.3 Memory Console UI

✅ **Implemented:**
- Chat panel with agent interaction
- Context visualization
- Timeline view
- Glyph cluster explorer
- Address inspector

Access at: `/memory-console`

---

## 3. Quick Start

### 3.1 Start the Memory Server

```bash
python scripts/run_memory_server.py --host 0.0.0.0 --port 8001
```

### 3.2 Start the Web UI

```bash
cd web
npm install
npm run dev
```

Navigate to: `http://localhost:3000/memory-console`

### 3.3 Python Client Usage

```python
from src.memory import create_memory_service, MemoryClient

# Create service
service = create_memory_service()
client = MemoryClient(service)

# Write memory
await client.write(
    actor_id="hayden",
    text="I want to build an agent with memory",
    tags=["task"],
)

# Read memory
response = await client.read(
    actor_id="hayden",
    query="What did I want to build?",
    token_budget=2048,
)

for item in response.context:
    print(f"{item.address}: {item.summary}")
```

---

## 4. Configuration

### 4.1 Environment Variables

Create a `.env` file:

```env
MEMORY_API_URL=http://localhost:8001
DATABASE_PATH=memory.db
```

### 4.2 Memory Policy

Configure in `src/memory/policy.py`:

```python
config = MemoryPolicyConfig(
    default_world="default",
    default_depth=2,
    shallow_budget_ratio=0.3,
    deep_budget_ratio=0.7,
)
```

---

## 5. Testing

### 5.1 Run Unit Tests

```bash
pytest tests/memory/
```

### 5.2 Debug Memory Store

```bash
python scripts/debug_memory_dump.py --actor-id hayden --show-spans
```

---

## 6. Architecture

### 6.1 Data Flow

```
User Input
    ↓
Memory Write API
    ↓
FGT Adapter (phrase extraction + glyph mapping)
    ↓
Memory Policy (address assignment)
    ↓
Memory Store (persistence)

Query
    ↓
Memory Read API
    ↓
FGT Adapter (embed query)
    ↓
Memory Store (retrieve candidates)
    ↓
Memory Policy (foveation)
    ↓
Context Items
```

### 6.2 Components

- **Models**: Core data structures
- **Addresses**: Fractal addressing logic
- **Policy**: Address assignment & foveation
- **Store**: Persistence (SQLite/in-memory)
- **Service**: Orchestration layer
- **API**: FastAPI HTTP interface
- **Client**: In-process Python client

---

## 7. Next Steps

- Integrate with real FGT components (clustering, embeddings)
- Add fine-grained glyph extraction
- Implement advanced foveation strategies
- Add performance monitoring & metrics
- Phase 5: Integration with glyph-core and TGO
