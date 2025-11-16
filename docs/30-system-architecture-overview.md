# System Architecture Overview

FGT is implemented as a set of cooperating components.

## 1. High-level components

1. **Data Ingestion (`src/ingest`)**

   - Reads raw corpora.

   - Extracts phrases and metadata.

2. **Embedding Service (`src/embed`)**

   - GPU-accelerated phrase embedding.

   - Writes shards of embeddings.

3. **Clustering Service (`src/cluster`)**

   - Runs clustering over embeddings.

   - Outputs cluster assignments and metadata.

4. **Glyph Manager (`src/glyph`)**

   - Manages glyph alphabet.

   - Assigns glyph IDs to clusters.

5. **Fractal Tape Builder (`src/tape`)**

   - Projects cluster centroids into 2D.

   - Computes fractal addresses.

   - Builds tape storage.

6. **Hybrid Tokenizer (`src/tokenizer`)**

   - Wraps base tokenizer.

   - Inserts glyph tokens.

7. **LLM Adapter (`src/llm_adapter`)**

   - Utilities for integrating FGT with LLM training/inference.

8. **Evaluation Suite (`src/eval`)**

   - Runs experiments and metrics.

9. **Visualizer Backend (`src/viz`)**

   - Serves data to frontend fractal map UI.

## 2. Data flow

```text
raw corpora
   ↓ ingest
phrases + metadata
   ↓ embed
embeddings
   ↓ cluster
cluster assignments + centroids
   ↓ glyph manager
glyph_id ←→ cluster_id
   ↓ tape builder
fractal addresses + storage
   ↓ tokenizer / eval / viz / llm_adapter
```

## 3. Storage layout

See `32-storage-layout-and-indexes.md` and `44-fractal-tape-storage-impl.md` for details.

Key directories:

* `data/` – raw/processed corpora.

* `embeddings/` – `.npy` shards.

* `clusters/` – labels, centroids, metadata.

* `tape/` – glyph/tape tables.

* `configs/` – YAML/JSON configs.

## 4. Execution model

* Standalone Python services / scripts.

* Pipeline orchestrated by CLI (in `scripts/`), e.g.:

```bash
python scripts/build_tape.py --config configs/demo.yaml
```

Optional:

* Later: Airflow / Prefect / Dagster for more complex orchestrations.

