# Fractal Glyph Memory — Integrated Stack Overview

This document ties together:

- **Fractal Glyph Tape (FGT)** – phrase-level semantic tape with glyph IDs and fractal addresses.
- **glyph-core** – coil/torus-based fractal substrate and visualizer.
- **fractal-video-tape (FVT)** – fractal video codec / tape.
- **glyph-drive-3d** – Császár-polyhedron-based 3D drive visualization.
- **temporal-glyph-operator (TGO)** – temporal observer, metrics, replay, and experiment lab.

We define a **Fractal Glyph Memory Stack**:

> A multi-scale, z-depth, multimodal memory OS where:
> - *what* is stored = semantic phrase/video/event families (glyphs),
> - *where* it lives = fractal address space (triangles, coil, Császár),
> - *how* it behaves = foveated read/write, temporal evolution, measurable via TGO.

## 1. Design goals

1. **Unified address space**
   All systems (FGT, glyph-core, FVT, glyph-drive-3d, TGO) share a common logical address format.

2. **Foveated memory**
   Read and write operations operate on:
   - a **region of interest**,
   - configurable **depth / detail levels**,
   - under a **token/byte budget**.

3. **Multimodal storage**
   Text, video, and (later) other modalities can coexist at the same addresses.

4. **Agent-ready Memory Service**
   A single **Fractal Glyph Memory Service (FGMS)** exposes read/write APIs for agents, apps, and pipelines.

5. **Instrumentation & replay**
   All memory operations emit events into TGO for:
   - observability,
   - experiments,
   - strategy comparison.

## 2. Key components

- **Common Address Layer (`@glyph/common-address`)**
  - Library that defines the canonical address structure and mapping helpers.

- **FGT Core**
  - Phrase extraction, embeddings, clustering, glyph assignment, fractal (triangle) addressing, z-depth semantics.

- **Memory Service**
  - Backend that:
    - accepts `write` and `read` requests,
    - routes them to FGT + FVT + storage,
    - exposes agent-friendly APIs.

- **Viewers**
  - glyph-core viewer (coil/torus).
  - glyph-drive-3d (Császár 3D drive).
  - FVT player.
  - FGT fractal map / web UI.

- **TGO Integration**
  - Hooks from Memory Service and viewers into TGO as a "behavior microscope."

## 3. Phases

- **Phase A — Shared Address & Memory Service**
  - Implement common address spec.
  - Build FGMS with FGT integration.
  - Basic glyph-core / glyph-drive visualization.

- **Phase B — Agent Memory Vertical**
  - Implement agent memory use case end-to-end.
  - Integrate TGO to observe memory strategies.

- **Phase C — Multimodal Experience Tape**
  - Integrate FVT and glyph-drive-3d.
  - Text + video + time on one tape.

Details in subsequent docs.
