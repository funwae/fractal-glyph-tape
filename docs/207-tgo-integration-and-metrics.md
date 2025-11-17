# TGO Integration & Metrics for Fractal Glyph Memory

This document explains how **temporal-glyph-operator (TGO)** integrates with Fractal Glyph Memory.

## 1. Role of TGO

TGO acts as:

- an **observer** of memory operations,
- a **metrics engine** for strategies (e.g., how aggressively to compress),
- a **replay tool** for seeing how memory evolved.

## 2. Event model

FGMS must emit:

- `memory.write` events,
- `memory.read` events,

as described in `203-memory-service-api.md`.

TGO ingests these as **frames** and computes:

- time series of:
  - memory size,
  - glyph density,
  - read/write frequencies,
  - average depth used.

## 3. Strategy comparison

Multiple "memory strategies" can be defined in FGMS:

- Strategy A:
  - shallow depths, conservative clustering.
- Strategy B:
  - deeper depths, aggressive compression.

TGO:

- runs experiments by tagging events with strategy IDs,
- compares:
  - performance metrics (retrieval accuracy, LLM eval),
  - efficiency metrics (tokens, bytes, latencies).

## 4. Visualization

In the TGO UI, add views:

- **Memory Evolution Timeline**
  - shows over time:
    - addresses added,
    - regions growing,
    - glyph usage patterns.

- **Depth Utilization Chart**
  - shows depth levels hit by reads/writes.

- **Region Focus Map**
  - which regions get the most memory traffic.

## 5. Data exports

TGO should support:

- exporting metrics as CSV/JSON for:
  - research papers,
  - cost modeling,
  - debugging.

## 6. Implementation notes

- Implement a small `TGOAdapter` package:
  - `log_memory_write(...)`
  - `log_memory_read(...)`
- Configure FGMS to call it synchronously (or via queue) on each operation.
