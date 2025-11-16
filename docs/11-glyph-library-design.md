# Glyph Library Design

We use **Mandarin characters as a glyph alphabet**, not as language.

## 1. Requirements

The glyph alphabet must:

- Be large (thousands of characters).

- Avoid collision with:

  - "Normal" Chinese text content in corpora (to the extent possible).

- Be compatible with:

  - Unicode.

  - Existing tokenizers.

- Be visually and token-wise compact.

## 2. Character range selection

Strategy:

1. Start from CJK Unified Ideographs / Extension blocks.

2. Exclude:

   - Most common characters used in real Mandarin text.

   - Characters that create tokenizer issues (rare surrogate behavior, etc.).

3. Build a curated list:

- `glyph_alphabet = [char_0, char_1, ..., char_N]`

This list is stored as:

- `glyph_alphabet.json` in the repo.

## 3. Encoding glyph IDs

Each glyph ID is an integer `g` in `[0, G_max)`.

We encode `g` in base-`N` where `N = len(glyph_alphabet)`:

- Represent `g` as digits `d_0, d_1, ..., d_{k-1}` in base N.

- Map each digit `d_i` to `glyph_alphabet[d_i]`.

- Concatenate to form glyph string (1–4 chars, configurable).

See `21-glyph-id-encoding-spec.md` for full details.

## 4. Length distribution

We want:

- Shorter glyphs for more frequent / central phrase families.

- Longer glyphs for rare ones.

Approach:

- Reserve 1-char glyphs for top `K1` clusters.

- 2-char glyphs for next `K2`, etc.

This can be configured based on:

- Cluster frequency.

- Importance scores.

## 5. Tokenizer considerations

To make sure glyph strings behave as single or few tokens:

- Extend base tokenizer vocab with all `glyph_alphabet` characters.

- Ensure each character is an independent token.

Optionally:

- Add composite tokens for common glyph bigrams.

## 6. Alternatives / future tweaks

- Use two disjoint glyph alphabets:

  - One for "core" families.

  - One for experimental or temporary families.

- Encode additional info in glyph length patterns (e.g., length indicates family category).

For now, keep glyph strings purely as IDs without semantics baked into the string itself.

