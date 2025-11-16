# Online Update Strategy

This defines how to evolve the tape as new data arrives.

## 1. Requirements

- Avoid full rebuild for small updates.

- Keep mapping stable when possible.

- Track tape versions.

## 2. Approaches

### 2.1 Batch updates

- Accumulate new phrases.

- Periodically:

  - Re-embed and cluster full dataset.

  - Build new tape version (`tape_version + 1`).

Pros:

- Clean, reproducible.

Cons:

- Heavy.

### 2.2 Incremental clustering

- Update centroids incrementally.

- Add new clusters for genuinely new motifs.

Requires more complex clustering algorithms.

## 3. Versioning strategy

- Maintain multiple tape versions concurrently.

- Consumers can:

  - Pin to a specific version.

  - Or adopt latest after validation.

## 4. Future work

- Explore streaming clustering algorithms.

- Explore delta tapes: storing only differences between versions.

