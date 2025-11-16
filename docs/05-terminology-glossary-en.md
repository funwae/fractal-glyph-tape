# Terminology Glossary (English)

**Fractal Glyph Tape (FGT)**

The overall system: a fractal-addressable phrase memory with glyph IDs.

**Phrase family / cluster**

A group of phrases that are semantically similar (e.g., "ask for a file politely").

**Cluster ID (`cluster_id`)**

Integer identifier of a phrase family produced by clustering.

**Glyph ID (`glyph_id`)**

Integer identifier that is mapped to one or more Mandarin characters. It is the public "glyph code" for a phrase family.

**Glyph string**

The visible sequence of Mandarin characters representing a glyph ID (e.g., `谷阜`).

**Fractal address**

A multi-scale coordinate for a phrase family in the fractal tape, usually represented as `(tape_version, level, cell_id)`.

**Fractal tape**

The overall address space (triangular / fractal layout) where glyph IDs are placed.

**Base tokenizer**

The standard text tokenizer (e.g., BPE, WordPiece) used before adding FGT.

**Hybrid tokenizer**

Tokenizer that outputs a mix of base tokens and glyph tokens.

**Glyph span**

A contiguous fragment of text that can be replaced by a glyph representing a phrase family.

**Tape version (`tape_version`)**

Version tag for a specific FGT build; changes when embeddings, projection, or clustering pipeline changes in ways that affect layout.

**Embedding model**

The transformer encoder that maps phrases to high-dimensional vectors.

**Projection function**

Function that maps embeddings to 2D coordinates to place them on the fractal tape.

**Neighborhood / neighbor clusters**

Phrase families located near each other in fractal address space.

**FGT representation**

The combination of:

- Shared phrase-family tables.

- Glyph-coded sequences.

- Metadata needed for reconstruction.

**Effective context multiplier**

Factor describing how much more semantic content fits into a fixed token budget using glyph codes vs raw text.

