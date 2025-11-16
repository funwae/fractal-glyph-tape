# Glyph ID Manager Implementation

This defines how we assign glyph IDs to clusters.

## 1. Responsibilities

- Maintain mapping:

  - `cluster_id -> glyph_id`

  - `glyph_id -> cluster_id`

- Use chosen glyph alphabet and encoding scheme.

- Allocate IDs based on cluster importance.

## 2. Cluster importance scoring

Score clusters using:

- `num_members` (frequency).

- Optional heuristics:

  - Centrality in embedding space.

  - Domain-specific weights.

Generate `importance_score` per cluster.

## 3. Allocation algorithm

1. Sort clusters by `importance_score` descending.

2. For sorted list:

   - Assign consecutive `glyph_id`s starting from 0.

3. Convert `glyph_id` to glyph string via base-N encoding.

4. Store:

- `glyphs/cluster_to_glyph.jsonl`

- `glyphs/glyph_to_cluster.jsonl`

## 4. API

Python interface:

```python
class GlyphManager:
    def cluster_to_glyph(self, cluster_id: int) -> int: ...
    def glyph_to_cluster(self, glyph_id: int) -> int: ...
    def glyph_to_unicode(self, glyph_id: int) -> str: ...
    def unicode_to_glyph(self, s: str) -> int: ...
```

## 5. Integration

GlyphManager is used by:

* Tape builder (to store glyph IDs).

* Tokenizer wrapper (to emit glyph strings).

* Visualizer (to display glyphs).

