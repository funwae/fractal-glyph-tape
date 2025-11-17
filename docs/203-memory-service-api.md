# Fractal Glyph Memory Service — API Specification

This document specifies the external API for FGMS.

Base path: `/api/memory`

All examples use JSON over HTTP; gRPC is an optional future addition.

---

## 1. Write API

### 1.1 `POST /api/memory/write`

Ingests new textual memory for an actor.

**Request body:**

```json
{
  "actor_id": "user_123",
  "text": "Today I finalized the fractal glyph tape integration.",
  "tags": ["work", "project:glyphd"],
  "region": "hayden-agent"
}
```

**Response:**

```json
{
  "status": "ok",
  "world": "earthcloud",
  "region": "hayden-agent",
  "addresses": [
    "earthcloud/hayden-agent#573@d1t102",
    "earthcloud/hayden-agent#132@d2t103"
  ],
  "glyph_summary": {
    "total_phrases": 7,
    "glyph_density": 0.64
  }
}
```

**Error responses:**

* `400` invalid payload.
* `500` internal error (embedding/cluster failure, etc).

---

## 2. Read API

### 2.1 `POST /api/memory/read`

Retrieves memory context for an actor using foveated semantics.

**Request body:**

```json
{
  "actor_id": "user_123",
  "query": "What did I do on the fractal tape project last week?",
  "focus": {
    "region": "hayden-agent",
    "max_depth": 3
  },
  "token_budget": 2048,
  "mode": "mixed"
}
```

**Response:**

```json
{
  "status": "ok",
  "world": "earthcloud",
  "region": "hayden-agent",
  "mode": "mixed",
  "context": [
    {
      "address": "earthcloud/hayden-agent#573@d0t89",
      "glyphs": ["谷阜"],
      "summary": "High-level: working on Fractal Glyph Tape integration with existing fractal repos."
    },
    {
      "address": "earthcloud/hayden-agent#573@d2t102",
      "glyphs": ["谷阜", "嶽岭"],
      "excerpt": "Worked on z-depth foveation and common address spec for phase-2 integration."
    }
  ],
  "token_estimate": 1832
}
```

---

## 3. Introspection API

### 3.1 `GET /api/memory/regions?actor_id=...`

List regions for an actor.

**Response:**

```json
{
  "actor_id": "user_123",
  "regions": ["hayden-agent", "support-logs", "personal-journal"]
}
```

### 3.2 `GET /api/memory/addresses?actor_id=...&region=...&limit=...`

List known addresses in a region (debug/admin).

---

## 4. TGO Events (logical)

FGMS emits internal events to TGO; this is not public API but must be stable:

```json
{
  "type": "memory.write",
  "timestamp": "2025-11-17T10:12:34Z",
  "actor_id": "user_123",
  "addresses": ["earthcloud/hayden-agent#573@d1t102"],
  "stats": {
    "text_chars": 108,
    "glyph_density": 0.64
  }
}
```

```json
{
  "type": "memory.read",
  "timestamp": "2025-11-17T10:13:01Z",
  "actor_id": "user_123",
  "query": "What did I do on the fractal tape project last week?",
  "addresses": [
    "earthcloud/hayden-agent#573@d0t89",
    "earthcloud/hayden-agent#573@d2t102"
  ],
  "token_budget": 2048,
  "token_estimate": 1832
}
```

TGO uses these for visualization and experiment tracking.
