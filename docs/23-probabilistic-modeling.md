# Probabilistic Modeling of Phrase Families

Phrase families can be seen as distributions over phrases.

## 1. Phrase family as distribution

For cluster `k`:

- Let `X_k` be random variable over phrases in the family.

- We approximate distribution using:

  - Empirical frequency of observed phrases.

  - Or a parametric model over embedding space.

## 2. Embedding-space view

We can model embeddings `v` in cluster `k` as:

- `v ~ N(μ_k, Σ_k)` (Gaussian approximation).

Where:

- `μ_k` is centroid.

- `Σ_k` is covariance (often approximated as diagonal or scalar).

## 3. Glyph ID semantics

A glyph ID can be interpreted as:

- A pointer to distribution `p_k(x)` (or `p_k(v)`).

When an LLM uses a glyph:

- It is implicitly referring to "sample from this region of phrase space."

## 4. Reconstruction models

To reconstruct text from glyphs:

- We can train a conditional generator:

\[
p(x \mid k, c)
\]

Where:

- `k` is glyph/cluster ID.

- `c` is context.

Model learns:

- To generate phrase-level realizations of the motif specified by `k`.

## 5. Use in training objectives

We can include losses like:

1. **Cluster prediction loss**

   - Predict cluster ID from text:

\[
\mathcal{L}_{\text{cluster}} = -\log p(k \mid x)
\]

2. **Glyph-conditioned generation loss**

   - Given glyph `k`, predict text `x`:

\[
\mathcal{L}_{\text{gen}} = -\log p(x \mid k, c)
\]

These encourage:

- Robust linkage between phrase space and glyph codes.

