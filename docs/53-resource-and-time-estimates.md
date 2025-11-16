# Resource and Time Estimates

Rough estimates for a single RTX GPU.

## 1. Example: 1M phrases

- Embedding:

  - Batch size 256–512.

  - Time: O(hours) depending on model.

- Clustering:

  - Mini-batch k-means:

    - Time: O(tens of minutes).

- Tape building:

  - Projection + addressing: O(minutes).

Disk usage (ballpark):

- Phrases: ~GB-scale depending on corpus.

- Embeddings: `1M * 768 * 4 bytes ≈ 3 GB`.

- Clusters + tape tables: hundreds of MB.

## 2. Demo-scale (100k phrases)

- Suitable for overnight or shorter runs.

- Good starting point for Phase 0.

## 3. Config knobs

- More clusters => more memory, more compute.

- Larger corpora => scaling linearly in embedding + clustering.

These are approximations; actual numbers logged by scripts.

