# Fractal Addressing Specification

This document specifies the **fractal address space** used by Fractal Glyph Tape (FGT) to organize phrase-family glyph IDs.

Each phrase family has:

- A **glyph ID** (sequence of Mandarin characters).

- A **fractal address** in a structured index space.

- Optional **multi-scale coordinates** for zooming and navigation.

---

## 1. Design goals

Fractal addressing must:

1. Provide a **stable coordinate** for each phrase family.

2. Support **multi-scale structure**:

   - Coarse regions for broad semantic categories.

   - Fine positions for specific phrase families.

3. Preserve **locality**:

   - Similar phrases → nearby addresses.

4. Be **efficient to store**:

   - Address representation must be compact.

5. Be **invertible**:

   - Address → index in storage.

   - Index → address.

---

## 2. Address format

We represent addresses as fixed-length tuples of small integers, plus an optional level indicator.

### 2.1 Canonical logical form

A fractal address is:

\[
A = (L, a_0, a_1, ..., a_{d-1})
\]

Where:

- \( L \) = **level** (non-negative integer).

- \( d \) = **address dimension** (e.g., 2 or 3).

- \( a_i \) = integer coordinate at level \(L\), bounded by a per-level maximum.

For a 2D Sierpiński-like layout, we can use:

- \( d = 2 \)

- Each level subdivides the space into 3 child cells (triangular decomposition).

### 2.2 Compact encoded form

Implementation-level representation:

- **Packed integer**:

  - Encode the path through the fractal recursion as a base-3 or base-4 integer.

- **Binary format**:

  - First bits: level \(L\)

  - Remaining bits: interleaved child indices.

Example (conceptual):

```text
[L][c1][c2][c3]...[cL]
```

Where:

* `ci` ∈ {0,1,2} for a ternary branching factor (Sierpiński-like).

---

## 3. Fractal scheme (Sierpiński-like)

We use a **recursive triangular partition**:

1. Start with a single root triangle (level 0).

2. Each triangle is split into 3 child triangles (level 1).

3. Each child splits again into 3 children (level 2), and so on.

Each **phrase family** is assigned to a leaf triangle at a given level.

### 3.1 Child index semantics

For each parent triangle, we define:

* Child 0: left sub-triangle

* Child 1: right sub-triangle

* Child 2: top sub-triangle

The sequence of child indices from root to leaf defines the **address path**.

Example path: `(L=3, c1=2, c2=0, c3=1)`

---

## 4. Mapping from semantic space to fractal addresses

### 4.1 Semantic coordinates

We assume each phrase family has an **embedding vector** \( v \in \mathbb{R}^n \) (centroid of its phrases).

We reduce \( v \) to 2D (for visualization and layout) using:

* PCA, t-SNE, UMAP, or a learned projection \( f: \mathbb{R}^n \to \mathbb{R}^2 \).

Let:

\[
p = f(v) = (x, y)
\]

### 4.2 Normalization

We normalize all projected points into a unit triangle:

\[
T = \{(x, y) | x \ge 0, y \ge 0, x + y \le 1\}
\]

Using an affine transformation that maps the bounding box of all projections into T.

### 4.3 Recursive subdivision

Given a point \( p \in T \), we determine its address path by:

1. At level 1:

   * Determine which of the 3 child sub-triangles contains \( p \).

   * Record `c1`.

2. Map \( p \) into the coordinate system of that child triangle.

3. Repeat until:

   * We reach desired depth `L_max`, or

   * We run out of phrase families to distinguish.

This produces the path `(c1, c2, ..., cL)`.

---

## 5. Address assignment algorithm

We want phrase families with similar embeddings to have similar addresses.

Algorithm sketch:

1. Compute embeddings and 2D projections for all phrase families.

2. Normalize points into unit triangle T.

3. Choose a maximum recursion depth `L_max` (e.g., 10–16, depending on scale).

4. For each phrase family:

   * Compute path `(c1..cL)` as described above.

   * If the leaf cell is **overfull** (too many families):

     * Either:

       * Increase L locally, or

       * Apply a tie-breaking rule (e.g., local sorting, slight perturbation).

5. Store:

   * The final address for each family.

   * A mapping from `(L, path)` to an integer index (cell ID).

---

## 6. Address representation in storage

### 6.1 Integer cell ID

For practical storage, each address path is converted to a single integer `cell_id`:

```text
cell_id = Σ (ci * 3^(i-1)) for i = 1..L
```

Separately store:

* `L` = level.

* `cell_id` = path-derived index.

Alternatively, embed `L` into the high bits of a single 64-bit integer.

### 6.2 Data structures

We maintain:

* `family_id -> (L, cell_id)` map

* `(L, cell_id) -> [family_id...]` map

These can be stored using:

* On-disk key-value store (e.g., SQLite, RocksDB).

* In-memory structures for fast lookup.

---

## 7. Neighborhood queries

The fractal layout allows us to define semantic neighborhoods:

1. **Same cell neighborhood**:

   * All phrase families sharing `(L, cell_id)`.

2. **Parent neighborhood**:

   * All families whose addresses share a prefix `(c1..ck)` for `k < L`.

3. **Adjacent neighborhoods**:

   * Cells at the same level whose triangular coordinates share an edge.

We support queries like:

* "All phrase families within radius `r` of a given family."

* "All families in the same coarse region (same top `k` child indices)."

Implementation can:

* Use prefix search on address paths.

* Or map addresses to 2D coordinates and use spatial indexes.

---

## 8. Stability and versioning

Fractal address assignment depends on:

* Embeddings.

* Projection function `f`.

* Normalization transform.

To preserve stability over time:

1. **Freeze**:

   * The embedding model version.

   * The projection method and its parameters.

2. Keep a **tape version ID**:

   * `tape_version = 1, 2, ...`

3. When any of the above change:

   * Increment `tape_version`.

   * Recompute addresses.

   * Maintain a mapping:

     * `family_id -> {tape_version -> address}`.

This allows:

* Backward compatibility.

* Multiple concurrent views of phrase space.

---

## 9. Address operations API

Minimal API sketch (pseudo-code):

```python
class FractalAddress:
    level: int
    cell_id: int  # packed path

def embed_phrase_family(family) -> np.ndarray:
    ...

def project_to_2d(embedding: np.ndarray) -> tuple[float, float]:
    ...

def normalize_to_triangle(point_2d) -> tuple[float, float]:
    ...

def point_to_address(point_2d_norm, L_max: int) -> FractalAddress:
    ...

def address_to_neighbors(addr: FractalAddress, radius: int) -> list[FractalAddress]:
    ...

def address_prefix(addr: FractalAddress, k: int) -> FractalAddress:
    ...
```

Detailed implementation is specified in `44-fractal-tape-storage-impl.md`.

---

## 10. Invariants

The system should maintain the following invariants:

1. Each phrase family has **at most one** primary address per `tape_version`.

2. Addresses are **unique at full depth**:

   * `(tape_version, level, cell_id, family_id)` uniquely identifies an entry.

3. Neighbor queries return **deterministic results** for a fixed `tape_version`.

4. For any `tape_version`, the projection and subdivision logic are **frozen** and reproducible.

These invariants ensure that:

* Visualizations and demos are stable across runs.

* Results can be reproduced and shared in publications.

