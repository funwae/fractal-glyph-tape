# Implementation Plan — Fractal Glyph Memory Stack

This document provides a concrete plan for implementing the integrated stack.

## Phase A — Common Address & FGMS Core

**Goal:** Shared address spec + basic Memory Service with FGT.

### Tasks

1. Implement `@glyph/common-address`
   - String/JSON encoding/decoding.
   - Types and helpers.
2. Implement FGT–address mapping
   - `FGTMapper` in the FGT repo.
   - Store `FractalAddress` alongside FGT cluster data.
3. Build FGMS:
   - FastAPI (or equivalent) app.
   - Implement:
     - `/memory/write` (text only),
     - `/memory/read` (glyph+text mixed).
   - Connect to existing FGT codepaths.

**Success criteria:**

- Can write and read memory for a single actor.
- `FractalAddress` consistently generated and returned.

---

## Phase B — Agent Memory Vertical + Viewers

**Goal:** Demonstrate agent memory use case end-to-end.

### Tasks

1. Implement agent-client integration
   - Minimal chat/agent loop that uses FGMS.
2. Integrate glyph-core viewer
   - Accept `FractalAddress` and highlight coil segment.
3. Integrate glyph-drive-3d viewer
   - Accept `FractalAddress` and highlight Császár region.
4. Build "Memory Console" UI
   - Show:
     - actor list,
     - regions,
     - addresses,
     - buttons to open viewers.

**Success criteria:**

- Working demo:
  - Chat with agent,
  - See memory evolving in viewers,
  - Retrieve long-horizon context via FGMS.

---

## Phase C — TGO Integration

**Goal:** Observe and measure memory.

### Tasks

1. Implement `TGOAdapter` in FGMS
   - Emit `memory.write` / `memory.read` events.
2. Extend TGO
   - New panels for:
     - memory metrics,
     - strategy comparison.
3. Run experiments
   - Compare at least two memory strategies.

**Success criteria:**

- Visual graphs of:
  - memory growth,
  - depth usage,
  - token budgets.

---

## Phase D — Multimodal Experience Tape

**Goal:** Integrate FVT and glyph-drive-3d for multimodal experiences.

### Tasks

1. Implement `FVTMapper`
   - Map `FractalAddress` ↔ FVT coords.
2. Extend FGMS to accept video references
   - Minimal `POST /memory/write-video` or combined payload.
3. Extend viewers
   - glyph-drive-3d + FVT player integration.
4. Build a small experience demo
   - Text + video for one session.

**Success criteria:**

- Click on a region in the 3D drive and see both:
  - text in that area,
  - corresponding video segments.

---

## Phase E — Glyph-aware LLM Experiment (Optional but Powerful)

**Goal:** Prove that glyph-coded context can outperform raw context under fixed budgets.

### Tasks

1. Prepare dataset
   - Use FGMS to generate glyph-coded histories for a single domain.
2. Fine-tune a small model:
   - Baseline: raw text only.
   - FGM model: uses glyph-coded context.
3. Evaluate:
   - retrieval QA,
   - summarization,
   - long-context tasks.

**Success criteria:**

- At least one clear regime where glyph-coded context
  performs better at the same token budget.

---

This plan, combined with the previous FGT docs and these new integration specs, is sufficient for an implementation-focused model (e.g., Claude Code) to wire the whole system together without further architectural invention.
