# Fractal Tape Storage Implementation

This document describes how fractal addresses, glyph IDs, and cluster IDs are stored on disk and accessed at runtime.

---

## 1. Core entities

We have three main IDs:

- `cluster_id` — integer index of phrase family.

- `glyph_id` — integer index of glyph code (separate from the Unicode characters).

- `fractal_address` — `(tape_version, level, cell_id)` triple.

Also:

- `family_id` and `cluster_id` may be equivalent in the first prototype.

---

## 2. Storage goals

The tape storage layer must:

1. Support **fast lookup**:

   - `cluster_id -> glyph_id`

   - `glyph_id -> cluster_id`

   - `cluster_id -> fractal_address`

   - `fractal_address -> [cluster_id...]`

2. Support **range and neighborhood queries**:

   - By `cell_id` prefix / level.

   - By tape version.

3. Be **append-friendly**:

   - We may add new clusters over time.

4. Be **portable**:

   - Usable from Python and possibly Rust/Go in the future.

---

## 3. On-disk layout (first prototype)

Use a combination of:

- A **lightweight key-value store** (e.g., SQLite or RocksDB).

- Numpy arrays for dense tables.

### 3.1 Tables

1. `clusters_table.npy` (dense array)

   - Index by `cluster_id`.

   - Columns:

     - `glyph_id` (int32)

     - `tape_version` (int16)

     - `level` (int16)

     - `cell_id` (int64)

2. `glyph_to_cluster.sqlite` (KV index)

   - Keys:

     - `glyph_id` (int)

   - Values:

     - `cluster_id` (int)

3. `address_to_clusters.sqlite` (KV index)

   - Keys:

     - `(tape_version, level, cell_id)` packed into a single integer key.

   - Values:

     - Packed list of `cluster_id`s (e.g., msgpack or JSON).

### 3.2 Key packing

Define:

```python
def pack_address_key(tape_version: int, level: int, cell_id: int) -> int:
    # 16 bits for tape_version, 8 bits for level, 40 bits for cell_id
    return (tape_version << 56) | (level << 48) | (cell_id & ((1 << 48) - 1))
```

This yields a 64-bit integer key.

---

## 4. Write path

The tape construction pipeline will:

1. Read:

   * `cluster_id` list.

   * Computed `glyph_id` for each cluster.

   * Computed fractal addresses `(tape_version, level, cell_id)`.

2. Populate in-memory arrays:

```python
num_clusters = ...
clusters_table = np.zeros((num_clusters,), dtype=[
    ("glyph_id", "int32"),
    ("tape_version", "int16"),
    ("level", "int16"),
    ("cell_id", "int64"),
])
```

3. For each cluster:

```python
clusters_table[cid]["glyph_id"] = glyph_id
clusters_table[cid]["tape_version"] = tape_version
clusters_table[cid]["level"] = level
clusters_table[cid]["cell_id"] = cell_id
```

4. Persist `clusters_table`:

```python
np.save("tape/clusters_table.npy", clusters_table)
```

5. Build KV indexes:

* `glyph_to_cluster.sqlite`:

  * For each cluster:

    * Insert `glyph_id -> cluster_id`.

* `address_to_clusters.sqlite`:

  * For each cluster:

    * Compute `key = pack_address_key(tape_version, level, cell_id)`.

    * Append `cluster_id` to list at this key.

---

## 5. Read path (runtime API)

We expose an API layer that loads these structures and answers queries.

### 5.1 Initialization

```python
class FractalTapeStore:
    def __init__(self, root_dir: str):
        self.clusters_table = np.load(
            os.path.join(root_dir, "clusters_table.npy"), mmap_mode="r"
        )
        self.glyph_db = sqlite3.connect(os.path.join(root_dir, "glyph_to_cluster.sqlite"))
        self.addr_db = sqlite3.connect(os.path.join(root_dir, "address_to_clusters.sqlite"))
```

### 5.2 Basic lookups

#### 5.2.1 Cluster → glyph

```python
def glyph_id_for_cluster(self, cluster_id: int) -> int:
    return int(self.clusters_table[cluster_id]["glyph_id"])
```

#### 5.2.2 Cluster → address

```python
from dataclasses import dataclass

@dataclass
class FractalAddress:
    tape_version: int
    level: int
    cell_id: int

def address_for_cluster(self, cluster_id: int) -> FractalAddress:
    row = self.clusters_table[cluster_id]
    return FractalAddress(
        tape_version=int(row["tape_version"]),
        level=int(row["level"]),
        cell_id=int(row["cell_id"]),
    )
```

#### 5.2.3 Glyph → cluster

```python
def cluster_for_glyph(self, glyph_id: int) -> int | None:
    cur = self.glyph_db.execute(
        "SELECT cluster_id FROM glyph_to_cluster WHERE glyph_id = ?",
        (glyph_id,)
    )
    row = cur.fetchone()
    return row[0] if row else None
```

#### 5.2.4 Address → clusters

```python
import msgpack

def clusters_for_address(self, addr: FractalAddress) -> list[int]:
    key = pack_address_key(addr.tape_version, addr.level, addr.cell_id)
    cur = self.addr_db.execute(
        "SELECT cluster_ids FROM address_to_clusters WHERE key = ?",
        (key,)
    )
    row = cur.fetchone()
    if not row:
        return []
    return msgpack.loads(row[0])
```

---

## 6. Neighborhood support

Neighborhood queries rely on **address arithmetic**, which is defined in `20-fractal-addressing-spec.md`.

Implementation pattern:

1. Get the address `addr` of a cluster.

2. Compute a set of neighboring addresses (same level and/or parent-level).

3. For each neighbor address:

   * Use `clusters_for_address(neighbor)` to get cluster IDs.

Exposed API:

```python
def neighbor_clusters(
    self,
    cluster_id: int,
    radius_levels: int = 1,
) -> list[int]:
    addr = self.address_for_cluster(cluster_id)
    neighbor_addrs = compute_neighbor_addresses(addr, radius_levels)
    results = set()
    for na in neighbor_addrs:
        results.update(self.clusters_for_address(na))
    return list(results)
```

Where `compute_neighbor_addresses` is implemented based on the fractal grid.

---

## 7. Versioning and migrations

Tape storage root directory should include a `meta.json`:

```json
{
  "tape_version": 1,
  "embedding_model": "all-mpnet-base-v2",
  "projection_method": "umap",
  "addressing_scheme": "sierpinski_triangular_v1",
  "created_at": "2025-11-16T00:00:00Z"
}
```

If any of these fields change, a new tape should be created under a new directory:

```text
tape/
  v1/
    meta.json
    clusters_table.npy
    glyph_to_cluster.sqlite
    address_to_clusters.sqlite
  v2/
    ...
```

The runtime can choose which tape version to use:

* Default to latest.

* Or allow explicit selection via config.

---

## 8. Testing

Basic tests to implement:

1. **Round-trip tests**

   * For random `cluster_id`:

     * Fetch glyph and address.

     * Fetch back from glyph and address.

     * Ensure the original cluster appears.

2. **Neighborhood tests**

   * For known clusters placed close in projection space:

     * Ensure they appear as neighbors.

3. **Performance tests**

   * Measure lookup latency for bulk queries.

   * Ensure within acceptable bounds (e.g., microseconds to low milliseconds).

These tests ensure that FGT's storage layer is robust enough for:

* CLI demos.

* Web visualizer backend.

* LLM integration.

