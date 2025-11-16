# Embedding and Clustering Implementation

This document describes the concrete implementation of:

1. Phrase embedding generation.

2. Phrase-family clustering.

3. Cluster metadata needed for Fractal Glyph Tape (FGT).

---

## 1. Goals

We need an implementation that:

- Produces high-quality **phrase embeddings**.

- Scales to large corpora on a single RTX-class GPU.

- Outputs **stable, reusable clusters** (phrase families).

- Logs enough metadata for:

  - Evaluation.

  - Later re-clustering or refinement.

  - Cross-lingual alignment.

---

## 2. Data flow overview

Pipeline stages:

1. **Phrase extraction**

   - From raw corpus → list of phrase units.

   - See `41-data-ingestion-implementation.md`.

2. **Batch embedding**

   - Use a transformer encoder (e.g., sentence-transformer or custom).

   - Store embeddings on disk (e.g., `.npy` or `HDF5`).

3. **Clustering**

   - Run clustering algorithm over embeddings.

   - Assign a `cluster_id` to each phrase.

4. **Metadata computation**

   - For each cluster:

     - Centroid.

     - Variance / spread.

     - Representative phrases.

     - Language distribution.

5. **Export to cluster index**

   - Save cluster index to disk for downstream steps:

     - Glyph assignment.

     - Fractal addressing.

     - Eval.

---

## 3. Phrase embedding

### 3.1 Model choice

Initial implementation:

- Use a publicly available sentence embedding model, such as:

  - `all-mpnet-base-v2` (SentenceTransformers style), or

  - A multilingual equivalent for cross-lingual experiments.

Rationale:

- Good balance of quality vs speed.

- Easy to run on a single GPU.

Later versions can swap in:

- Domain-specific models.

- Fine-tuned models for your corpora.

### 3.2 Implementation sketch

Python + PyTorch (pseudo-code):

```python
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader

model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

def embed_phrases(phrases: list[str], batch_size: int = 256) -> np.ndarray:
    dl = DataLoader(phrases, batch_size=batch_size, shuffle=False)
    all_embeddings = []
    for batch in dl:
        with torch.no_grad():
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(emb)
    return np.vstack(all_embeddings)
```

### 3.3 Storage

* Store embeddings in chunks to avoid memory blowup.

Example:

```text
embeddings/
  shard_000.npy
  shard_001.npy
  ...
```

And a metadata file:

```json
{
  "model_name": "all-mpnet-base-v2",
  "dim": 768,
  "num_phrases": 1234567,
  "shard_size": 100000
}
```

---

## 4. Clustering

### 4.1 Requirements

Clustering must:

* Handle up to millions of vectors.

* Support incremental or hierarchical clustering.

* Output:

  * `cluster_id` for each phrase.

  * Number of clusters chosen by configuration.

### 4.2 Algorithm options

Initial options:

1. **Mini-batch k-means**

   * Scales well.

   * Simple implementation.

   * Good baseline.

2. **Product quantization + k-means on centroids**

   * For extremely large corpora.

   * Might be overkill for first prototype.

Pick **mini-batch k-means** as the default.

### 4.3 Hyperparameters

Config example (in YAML):

```yaml
clustering:
  algorithm: minibatch_kmeans
  num_clusters: 50000
  batch_size: 10000
  max_iter: 200
  init: k-means++
  random_state: 42
```

### 4.4 Implementation sketch (mini-batch k-means)

```python
from sklearn.cluster import MiniBatchKMeans
import numpy as np

def cluster_embeddings(emb_shards_dir: str, num_clusters: int, batch_size: int):
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=batch_size,
        init="k-means++",
        random_state=42,
        verbose=1,
    )

    # Pass over embeddings for fitting
    for shard_path in list_shards(emb_shards_dir):
        emb = np.load(shard_path)
        kmeans.partial_fit(emb)

    # Second pass to assign cluster labels
    all_labels = []
    for shard_path in list_shards(emb_shards_dir):
        emb = np.load(shard_path)
        labels = kmeans.predict(emb)
        all_labels.append(labels)

    labels_concat = np.concatenate(all_labels, axis=0)
    centroids = kmeans.cluster_centers_

    return labels_concat, centroids
```

### 4.5 Output format

Save:

* `clusters/labels.npy` — `int32` array of size `num_phrases`.

* `clusters/centroids.npy` — `float32` array of shape `(num_clusters, dim)`.

* `clusters/config.json` — hyperparameters and metadata.

Example `config.json`:

```json
{
  "algorithm": "minibatch_kmeans",
  "num_clusters": 50000,
  "dim": 768,
  "embedding_model": "all-mpnet-base-v2",
  "max_iter": 200,
  "random_state": 42
}
```

---

## 5. Cluster metadata

To support FGT and research, compute for each `cluster_id`:

* `num_members` — count of phrases.

* `centroid` — mean embedding (already from k-means).

* `avg_distance` — average L2 distance to centroid.

* `max_distance` — maximum distance to centroid.

* `example_phrases` — up to N representative phrases.

* `language_distribution` — counts per language tag (if available).

### 5.1 Implementation sketch

```python
import numpy as np
from collections import defaultdict, Counter

def compute_cluster_metadata(emb_shards_dir, labels, phrases_meta, num_clusters, dim):
    # phrases_meta: list of dicts [{ "phrase_id": int, "text": str, "lang": "en" }, ...]
    cluster_info = {
        cid: {
            "num_members": 0,
            "sum_embedding": np.zeros(dim, dtype=np.float32),
            "sum_sq_dist": 0.0,
            "max_dist": 0.0,
            "examples": [],
            "lang_counter": Counter(),
        }
        for cid in range(num_clusters)
    }

    phrase_idx = 0
    for shard_path in list_shards(emb_shards_dir):
        emb = np.load(shard_path)
        shard_size = emb.shape[0]
        shard_labels = labels[phrase_idx:phrase_idx+shard_size]

        for i in range(shard_size):
            cid = int(shard_labels[i])
            v = emb[i]
            info = cluster_info[cid]

            info["num_members"] += 1
            info["sum_embedding"] += v

            # Distance to centroid will be refined in a second pass,
            # or we recompute after centroids are known.

            meta = phrases_meta[phrase_idx + i]
            if len(info["examples"]) < 10:
                info["examples"].append(meta["text"])
            info["lang_counter"][meta.get("lang", "unknown")] += 1

        phrase_idx += shard_size

    # Normalize and finalize metadata...
```

Then store cluster metadata as:

```text
clusters/metadata.jsonl
```

Each line:

```json
{
  "cluster_id": 12345,
  "num_members": 987,
  "centroid": [ ... floats ... ],
  "avg_distance": 0.54,
  "max_distance": 1.23,
  "examples": ["...", "..."],
  "language_distribution": { "en": 800, "zh": 120, "other": 67 }
}
```

(For large centroids, store them separately in `.npy` and reference indices.)

---

## 6. Quality checks

Before using clusters for glyph assignment:

1. **Size distribution**

   * Plot histogram of `num_members` per cluster.

   * Detect:

     * Empty clusters.

     * Very large clusters (overly broad).

2. **Distance distribution**

   * Plot histogram of `avg_distance`.

   * Flag clusters with very high variance.

3. **Language composition**

   * Check if clusters are mono-lingual or multi-lingual.

   * For multilingual experiments, ensure cross-lingual clusters exist.

These diagnostics feed into:

* Thresholds for splitting/merging clusters.

* Confidence scores for glyph ID assignment.

---

## 7. Cross-lingual alignment (optional first-phase extension)

To make cross-lingual glyph families:

* Use a **multilingual embedding model** so that:

  * English and Mandarin sentences live in the same embedding space.

* Run clustering on the **combined multilingual corpus**.

* Result:

  * Clusters naturally contain multiple languages.

  * Language distribution metadata shows cross-lingual families.

Later experiments can compare:

* Monolingual clustering + alignment.

* Direct multilingual clustering.

---

## 8. Hooks for downstream stages

Downstream components need:

* `cluster_id -> centroid_embedding`

* `cluster_id -> representative_examples`

* `phrase_id -> cluster_id`

These are used by:

* `43-glyph-id-manager-impl.md` to allocate glyph IDs.

* `20-fractal-addressing-spec.md` to map centroids onto the fractal tape.

* `60-eval-metrics-overview.md` to compute metrics like cluster purity.

Ensure final data layout supports:

* Efficient iteration over clusters.

* Fast lookups from `phrase_id` to `cluster_id`.

