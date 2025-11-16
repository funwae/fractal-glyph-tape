# Storage Layout and Indexes

We define where data lives and how it is indexed.

## 1. Top-level directories

- `data/` – input and processed text.

- `embeddings/` – phrase embedding shards.

- `clusters/` – clustering results.

- `tape/` – glyph/tape structures.

- `configs/` – configuration files.

## 2. Embeddings

- `embeddings/shard_000.npy`, `shard_001.npy`, ...

Metadata:

- `embeddings/meta.json`

Fields:

- `model_name`

- `dim`

- `num_phrases`

- `shard_size`

## 3. Clusters

- `clusters/labels.npy` — `phrase_id` → `cluster_id`.

- `clusters/centroids.npy` — centroids vector array.

- `clusters/metadata.jsonl` — per-cluster info.

- `clusters/config.json` — clustering settings.

## 4. Tape

Under `tape/vX/`:

- `meta.json` — tape version metadata.

- `clusters_table.npy` — dense map `cluster_id` → glyph + address.

- `glyph_to_cluster.sqlite` — glyph → cluster index.

- `address_to_clusters.sqlite` — address → cluster list.

## 5. Phrase index (optional)

- `indexes/phrase_to_cluster.sqlite`

  - Key: `phrase_id`

  - Value: `cluster_id`

Useful for evaluation and debugging.

## 6. Indexing rationale

We use:

- Numpy arrays for dense, sequential IDs (fast).

- SQLite/KV stores for associative lookups with flexible queries.

Design target:

- Load important tables into memory for quick lookups.

- Keep big data (embeddings, corpora) on disk and stream when needed.

