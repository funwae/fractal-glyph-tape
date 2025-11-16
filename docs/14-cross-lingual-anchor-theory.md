# Cross-Lingual Anchor Theory

FGT is naturally suited for cross-lingual work.

## 1. Multilingual embeddings

If embeddings come from a multilingual model:

- English, Mandarin, etc. phrases that mean similar things are close in vector space.

- Clustering across combined corpora yields **cross-lingual phrase families**.

## 2. Glyph IDs as anchors

When a cluster contains multiple languages:

- The same `glyph_id` becomes a **language-agnostic anchor** for that motif.

- E.g., file-request phrases in English, Chinese, Spanish share one glyph.

## 3. Benefits

- **Cross-lingual retrieval**:

  - Search using glyphs or English phrases.

  - Retrieve matching content in other languages.

- **Translation via phrase families**:

  - Instead of direct text-to-text, go:

    - Source text → glyph(s) → example target phrases.

## 4. Stability vs language drift

Because glyph IDs are:

- External,

- Versioned,

- And stored with cluster metadata,

They provide a stable "coordinate system" even as underlying models evolve.

## 5. Research directions

- How many clusters become cross-lingual under different models?

- How does glyph-based retrieval compare to embedding-only retrieval?

- Can we bootstrap low-resource languages by linking them into existing glyph families?

