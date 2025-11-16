# Related Work and Positioning

FGT relates to several existing ideas:

## 1. Tokenization (BPE, SentencePiece)

- These methods:

  - Operate at character/byte level.

  - Optimize for string compression.

- FGT:

  - Operates at **phrase / semantics** level.

  - Names **families of phrases**, not substrings.

## 2. Vector Quantization (VQ-VAE, etc.)

- VQ methods:

  - Learn discrete codes for embedding vectors.

- FGT:

  - Uses clustering and explicit glyph IDs.

  - Adds a **fractal address** and **cross-lingual framing**.

## 3. Neural semantic hashing

- Neural hashing maps texts to discrete codes.

- FGT:

  - Uses external, interpretable glyph code space.

  - Emphasizes visualization and multi-scale structure.

## 4. Retrieval and RAG

- Vector DBs build indexes over embeddings.

- FGT:

  - Creates a **global phrase map** with stable names.

  - Can sit under RAG systems as a shared phrase substrate.

## 5. Positioning

FGT is best seen as:

> A semantic compression and naming layer for phrase patterns, designed for use with LLMs and cross-lingual systems, with explicit structure and visualizability.

It can coexist with:

- Standard tokenizers.

- Vector DBs.

- Existing LLM infrastructures.

