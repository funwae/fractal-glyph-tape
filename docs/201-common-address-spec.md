# Common Address Specification (`@glyph/common-address`)

This document defines the canonical address format shared by:

- Fractal Glyph Tape (FGT)
- glyph-core
- fractal-video-tape (FVT)
- glyph-drive-3d
- temporal-glyph-operator (TGO)
- Fractal Glyph Memory Service (FGMS)

## 1. Logical address structure

We define:

```text
FractalAddress = {
  world: string,        // logical namespace / tenant / persona
  region: string,       // domain or topic (e.g. "agent:chat", "project:funwae")
  tri_path: int[],      // base-3 path in triangle space (FGT-style)
  depth: int,           // z-depth within that triangle (semantic/time layer)
  time_slice: int,      // optional finer-grained temporal index
}
```

Semantics:

* `world`

  * Top-level namespace (user, tenant, environment).
* `region`

  * Logical grouping (memory domain, project, modality stream).
* `tri_path`

  * Multi-scale position in the triangular fractal (FGT tape).
  * Each `tri_path[i] ∈ {0,1,2}` corresponds to child triangle at level `i`.
* `depth`

  * Z-depth within that triangle:

    * `0` = highest-level summary / most abstract.
    * Increasing depth = more detailed / specific / raw.
* `time_slice`

  * Optional index for ordering multiple entries at same spatial position and depth.

## 2. Canonical serialization

Addresses are serialized as a compact string:

```text
<world>/<region>#<tri_path_str>@d<depth>t<time_slice>
```

Where:

* `tri_path_str` is a base-3 integer formed from tri_path digits:

```
tri_path_id = sum_{i=0}^{L-1} tri_path[i] * 3^i
```

Example:

```text
"earthcloud/hayden-agent#573@d2t17"
```

### 2.1. JSON representation

```json
{
  "world": "earthcloud",
  "region": "hayden-agent",
  "tri_path": [2, 1, 0, 2],
  "depth": 2,
  "time_slice": 17
}
```

Both string and JSON forms must be supported by `@glyph/common-address`.

## 3. Mapping to subsystem coordinates

We define pure interfaces (implementations live in the respective repos):

```ts
// Pseudo-TypeScript interfaces

type FractalAddress = {
  world: string;
  region: string;
  tri_path: number[];
  depth: number;
  time_slice: number;
};

interface FGTMapper {
  toFGTCell(addr: FractalAddress): {
    tape_version: number;
    level: number;
    cell_id: number;
    depth: number;
  };
  fromFGTCell(tape_version: number, level: number, cell_id: number, depth: number): FractalAddress;
}

interface CoilMapper {
  toCoilCoords(addr: FractalAddress): {
    L: number;
    U: number;
    V: number;
    D: number;
  };
}

interface CsaszarMapper {
  toCsaszarCoords(addr: FractalAddress): {
    faceId: string;
    subdivisionPath: number[];
  };
}

interface FVTMapper {
  toFVTCoords(addr: FractalAddress): {
    streamId: string;
    frameIndex: number;
    tilePath: number[];
  };
}
```

Each subsystem:

* implements its mapper using its existing coordinate systems,
* treats `FractalAddress` as the "master key."

## 4. Address assignment policy

FGMS is responsible for:

* choosing `world` and `region` based on:

  * actor ID, tenant ID, use case (e.g. `world="earthcloud"`, `region="support-logs"`).
* determining `tri_path` via FGT's projection + clustering.
* incrementing `depth` / `time_slice`:

  * deeper detail → higher `depth`,
  * new events in same region → increment `time_slice`.

So:

* **FGT** controls spatial layout (`tri_path`).
* **FGMS policy** controls depth/time semantics.

## 5. Library deliverables

Package: `@glyph/common-address` (language-agnostic; TS & Python bindings recommended):

* Types:

  * `FractalAddress`
* Constructors:

  * `createAddress(world, region, tri_path, depth, time_slice)`
* Parsers:

  * `parseAddressString(s: string) -> FractalAddress`
  * `formatAddress(addr: FractalAddress) -> string`
* Mappers (pure interfaces, actual impl in subsystems):

  * `FGTMapper`, `CoilMapper`, `CsaszarMapper`, `FVTMapper`

These definitions must be stable and versioned.
