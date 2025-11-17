# Fractal Glyph Memory Service (FGMS) — Architecture

FGMS is the **OS layer** over Fractal Glyph Memory:

> A network service that exposes read/write APIs for agents, apps, and experiments,
> and orchestrates FGT, FVT, viewers, and TGO.

## 1. Responsibilities

- Accept `write` requests:
  - ingest text (and later video),
  - update FGT tape and associated stores.
- Accept `read` requests:
  - perform foveated queries,
  - return compressed/expanded context.
- Manage **address assignment** (via FGT).
- Emit **events** to TGO for observability and experiments.
- Provide **introspection APIs** (what's stored where?).

## 2. High-level component diagram

```text
Client (agent/app)
   ↓ HTTP / gRPC
[ FGMS API ]
   ↓
[ Memory Orchestrator ]
   ├─ FGT Core (phrase tape)
   ├─ FVT Core (video tape) [optional Phase C]
   ├─ Address Mapper (@glyph/common-address)
   ├─ Storage (DB / object store)
   └─ TGO Adapter (events → temporal-glyph-operator)
```

## 3. Technology choices

* **Backend language:** Python or TypeScript (both viable).

  * Recommendation: **Python FGMS** for tight reuse of FGT code paths.
* **API:** FastAPI (Python) or equivalent.
* **Storage:**

  * FGT tape files + SQLite/Parquet (as defined in FGT docs).
  * Additional tables for:

    * `actor_id` / `world` / `region` mapping,
    * per-address metadata (tags, timestamps).

## 4. Core flows

### 4.1 Write flow (text)

**Endpoint:** `POST /memory/write`

Inputs:

* `actor_id: string`
* `text: string`
* `tags?: string[]`
* `region?: string` (optional override)

Steps:

1. Resolve `world` & `region`:

   * `world = config.tenant_for_actor(actor_id)`
   * `region = input.region || config.default_region(actor_id)`

2. Phrase extraction:

   * Use FGT ingest to split `text` into phrases.

3. Embedding & clustering:

   * Use existing FGT embedding and clustering pipeline (online or batch).
   * Determine phrase families / cluster IDs.

4. Address assignment:

   * For each cluster:

     * compute / look up `tri_path` via FGT.
     * determine `depth` and `time_slice` via policy (z-depth/time semantics).

5. Storage:

   * Persist:

     * FGT glyph-coded sequence,
     * mapping from `actor_id` + `text` + metadata → list of `FractalAddress` entries.

6. Event emission:

   * Emit `memory.write` event to TGO, including:

     * addresses,
     * text length, glyph density,
     * tags.

7. Response:

   * Return:

     * list of `FractalAddress` used,
     * glyph-coded representation (optional).

### 4.2 Read flow (foveated query)

**Endpoint:** `POST /memory/read`

Inputs:

* `actor_id: string`
* `query: string`
* `focus?: { region?: string; max_depth?: number }`
* `token_budget?: number`
* `mode?: "glyph" | "text" | "mixed"`

Steps:

1. Resolve `world` & `region` (as in write).
2. Convert `query` to embedding (or use LLM).
3. Search:

   * find relevant phrase families / clusters for this actor & region.
   * use FGT + addresses to gather candidate `FractalAddress` entries.
4. Foveated selection:

   * apply token/byte budget:

     * prioritize:

       * shallow depth across broader region for overview,
       * deeper depth in the most relevant region(s).
5. Reconstruction / packaging:

   * Depending on `mode`:

     * `glyph`: return glyph-coded context only.
     * `text`: expand glyphs into sample phrases / summaries.
     * `mixed`: combine both (glyphs + human-readable paraphrases).
6. Event emission:

   * Emit `memory.read` event to TGO with:

     * query,
     * addresses touched,
     * depth usage,
     * size of response.
7. Response:

   * `context`: text/glyph/mixed,
   * `addresses`: list of addresses used,
   * `debug`: optional metrics.

## 5. Internal services

Logical modules inside FGMS:

* `AddressPolicyService`

  * Encapsulates how depths and regions are chosen.
* `FGTAdapter`

  * Thin wrapper around existing FGT codebase.
* `StorageService`

  * Knows how to read/write FGT artifacts plus metadata.
* `TGOAdapter`

  * Encodes memory events as TGO frames.

Each module should be unit-testable and documented separately.
