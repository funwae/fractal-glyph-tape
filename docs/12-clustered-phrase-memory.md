# Clustered Phrase Memory

The core object of FGT is the **phrase family**: a cluster of semantically similar phrases.

## 1. Phrase units

Phrase units can be:

- Full sentences.

- Clausal units (sub-sentences).

- n-grams.

- Templates (with slots).

We extract phrases using:

- Sentence segmentation.

- Tokenization.

- Optional dependency parsing or heuristic chunking.

## 2. Embedding and similarity

Each phrase is encoded into a vector via:

- A sentence embedding model (e.g., SentenceTransformers).

Similarity:

- Typically cosine distance or Euclidean distance in embedding space.

## 3. Clustering

Clusters are formed via:

- Mini-batch k-means or similar scalable algorithms.

- Each cluster -> `cluster_id`.

We store:

- Centroid embedding.

- Member count.

- Example phrases.

- Language distribution.

## 4. Phrase family properties

A good phrase family:

- Has clear semantic coherence ("ask for help", "apologize", "confirm receipt").

- Has manageable variance (not too broad or too narrow).

- Can be described by a short label (optional).

## 5. Role in FGT

Phrase families are:

- The atomic units to which we assign glyph IDs and fractal addresses.

- The core of compression:

  - Frequent patterns -> shared representation.

- The handles for cross-lingual mapping:

  - Multiple languages in same cluster.

Cluster quality directly affects:

- Compression quality.

- Interpretability.

- LLM performance.

The clustering pipeline is therefore a critical part of the system (see `42-embedding-and-clustering-impl.md` and `22-phrase-clustering-math.md`).

