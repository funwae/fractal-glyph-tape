# Fractal Tape Concept

The **fractal tape** is a recursive, multi-scale layout where phrase families are placed and referenced.

## 1. Intuition

Imagine a large triangle. This is the entire phrase space.

- At level 0, it's a single region.

- At level 1, we split into 3 sub-triangles.

- At level 2, each of those splits into 3 again, and so on.

Each leaf triangle at a given depth can hold one or more **phrase families**.

Similar phrase families end up in:

- Nearby triangles.

- Or triangles that share a long prefix in their subdivision path.

This gives us:

- **Zoomable structure**: top levels capture broad categories, deeper levels capture fine distinctions.

- **Addressable geometry**: each leaf has a path like `(2, 0, 1, 2, ...)`.

## 2. Why fractal, not grid

Fractals give us:

- Natural multi-scale semantics: large triangles for "big motifs", smaller ones for specific variants.

- Efficient representation as **paths** (sequence of small integers).

- Compact, visually engaging maps for demos.

A grid would work but lacks the "inherently multi-scale" feel and nice triangle domain mapping for simplex-like embedding.

## 3. Mapping semantic space to tape

Steps:

1. Compute embeddings for each phrase family.

2. Project to 2D (e.g., UMAP or PCA).

3. Normalize points into a triangular domain.

4. Recursively decide which sub-triangle each point falls into.

The resulting **path** is the family's fractal address.

## 4. How LLMs benefit

- Tape provides **structured neighborhoods**:

  - When an LLM accesses one glyph, related glyphs are nearby in address space.

- Retrieval systems can:

  - Expand queries to neighboring addresses to get richer context.

- Visual tools can:

  - Show clusters and relationships on an interactive fractal map.

## 5. Versioning

The tape is:

- Deterministic for a fixed pipeline and corpora.

- Tagged with `tape_version` when pipeline changes.

This allows reproducible research and stable references over time.

