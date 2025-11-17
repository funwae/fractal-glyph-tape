# Viewers Integration: glyph-core, glyph-drive-3d, FGT Map, FVT Player

This document describes how viewers connect to the shared Fractal Glyph Memory.

## 1. Requirements

- All viewers must be able to:
  - accept a `FractalAddress` or list of addresses,
  - highlight corresponding regions,
  - optionally fetch content via FGMS.

## 2. glyph-core (coil/torus viewer)

### 2.1 Input contract

Add support for:

- Query parameter or message:
  - `address=<serialized-address-string>`

Internally:

- Use `CoilMapper.toCoilCoords(addr)` to compute `(L,U,V,D)`.
- Highlight:
  - the coil segment corresponding to that address,
  - with optional blinking / color.

### 2.2 FGMS link

Optionally:

- On click:
  - send selected address back to FGMS (via UI API),
  - FGMS returns:
    - text summary,
    - stats,
    - links to other viewers.

## 3. glyph-drive-3d (Császár viewer)

### 3.1 Input contract

- Accept address via:
  - query param,
  - WebSocket message (if already using real-time updates).

Use `CsaszarMapper.toCsaszarCoords(...)`:

- Determine:
  - `faceId`,
  - subdiv path.

Highlight corresponding face / area and show tooltip:

- glyphs,
- counts,
- etc.

## 4. FGT Map (2D triangle view)

- Already uses `tri_path` and FGT's own address layout.
- Should accept `FractalAddress` and:
  - zoom to `tri_path`,
  - display available depths/time slices.

## 5. FVT Player

- Accept `FractalAddress`:
  - Look up video segments via `FVTMapper`.
- Use:
  - `time_slice` to select temporal window,
  - `tri_path` / `tilePath` to select spatial region.

## 6. Orchestration

Optionally, create a small **"memory console"** frontend:

- It:
  - calls FGMS APIs,
  - coordinates all viewers,
  - lets user:
    - select an actor/region,
    - pick addresses from lists,
    - open them in 2D/3D/coil/video views.
