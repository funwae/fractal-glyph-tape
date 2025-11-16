# Phrase Clustering Math

This describes the mathematical framing of phrase clustering for FGT.

## 1. Embedding space

Each phrase `x` is mapped to:

- `v = f(x) ∈ ℝ^d`

Where:

- `f` is a transformer-based encoder.

- `d` is embedding dimension (e.g., 768).

Distance measure:

- Typically Euclidean: `‖v_i - v_j‖₂`

- Or cosine dissimilarity.

## 2. Objective

Given a set of vectors `{v_i}`, we aim to partition them into `K` clusters `{C_1..C_K}` minimizing:

\[
J = \sum_{k=1}^K \sum_{v_i \in C_k} \| v_i - \mu_k \|_2^2
\]

Where:

- `μ_k` is centroid of cluster `k`.

This is the standard k-means objective.

## 3. Algorithm

We use **mini-batch k-means** for scalability.

Key equations:

- Centroid update:

\[
\mu_k \leftarrow \mu_k - \eta \cdot \left(\mu_k - \frac{1}{|B_k|} \sum_{v_i \in B_k} v_i \right)
\]

Where:

- `B_k` is the subset of batch assigned to cluster `k`.

- `η` is learning rate (implicitly chosen by implementation).

## 4. Cluster quality metrics

We compute:

- **Within-cluster variance**:

\[
\sigma_k^2 = \frac{1}{|C_k|} \sum_{v_i \in C_k} \| v_i - \mu_k \|_2^2
\]

- **Silhouette score** for sample `i`:

\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

Where:

- `a(i)` = avg distance to points in same cluster.

- `b(i)` = min avg distance to points in other clusters.

These metrics help:

- Decide `K`.

- Identify poor clusters for splitting/merging.

## 5. Cross-lingual clustering

If using multilingual embeddings:

- Phrases from languages `L1` and `L2` live in same space.

- Clusters naturally mix languages.

We can compute **language entropy** per cluster:

\[
H_k = -\sum_{l} p_{k,l} \log p_{k,l}
\]

Where:

- `p_{k,l}` = fraction of phrases in cluster `k` from language `l`.

This describes cross-lingual mixing.

## 6. Incremental updates

To handle new data:

- We can:

  - Run incremental k-means updates.

  - Or train secondary clusters and map them to existing ones.

Details in `52-online-update-strategy.md`.

