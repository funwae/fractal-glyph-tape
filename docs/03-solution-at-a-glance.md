# Solution at a Glance

Fractal Glyph Tape (FGT) introduces a new layer under/alongside LLMs:

> A **fractal phrase map** where each phrase family gets a short **glyph code** (Mandarin characters as symbols) and a coordinate on a **fractal tape**.

## 1. High-level pipeline

1. **Ingest text corpora**

   - Extract sentences, phrases, n-grams.

   - Attach language and metadata.

2. **Embed phrases**

   - Use a transformer encoder to obtain vectors.

3. **Cluster into phrase families**

   - Run scalable clustering (mini-batch k-means).

   - Each cluster ≈ one phrase family / motif.

4. **Assign glyph IDs**

   - Map each `cluster_id` to a `glyph_id` integer.

   - Encode `glyph_id` as 1–4 Mandarin characters.

5. **Place on fractal tape**

   - Project cluster centroids into 2D.

   - Normalize into a triangle domain.

   - Use recursive triangular subdivision to assign fractal addresses.

6. **Store in tape index**

   - `cluster_id ↔ glyph_id ↔ fractal_address` in compact tables and KV stores.

7. **Hybrid tokenizer**

   - Detect spans of text that belong to known phrase families.

   - Replace them with glyph sequences.

   - Emit mix of raw tokens + glyph tokens.

8. **LLM integration**

   - Models trained/fine-tuned so:

     - They interpret glyph tokens.

     - They can expand glyphs into text.

     - They can use glyphs as internal pointers to phrase families.

## 2. Outcomes

- **Compression:** one family table + glyph-coded sequences << raw text.

- **Context expansion:** same tokens, more semantic content.

- **Cross-lingual:** glyph IDs tie together multiple languages.

- **Visualization:** fractal map UI for exploring phrase space.

## 3. Key design principles

- **Mandarin as glyph library only** – semantics come from clusters, not characters.

- **Fractal layout** – multi-scale, local-structure-preserving addressing.

- **Hybrid, not replacement** – FGT augments existing tokenizers and models.

