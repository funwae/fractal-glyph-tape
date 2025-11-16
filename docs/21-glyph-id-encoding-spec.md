# Glyph ID Encoding Specification

This defines how integer `glyph_id`s are encoded as Mandarin glyph strings and back.

## 1. Glyph alphabet

We define:

- `glyph_alphabet`: ordered list of Unicode codepoints used as glyph characters.

- Size: `N = len(glyph_alphabet)`.

Stored in:

- `config/glyph_alphabet.json`

## 2. Integer glyph IDs

- `glyph_id` is a non-negative integer.

- Range: `[0, G_max)`, where `G_max` is determined by cluster count.

## 3. Base-N encoding

We encode `glyph_id` in base `N`:

1. For `g = glyph_id`:

   - Compute digits: `d_0, d_1, ..., d_{k-1}` such that:

     - `g = Σ d_i * N^i`

     - `0 <= d_i < N`.

2. Map each `d_i` to `glyph_alphabet[d_i]`.

3. Concatenate characters to form glyph string.

We constrain `k`:

- `k_min <= k <= k_max` (e.g., 1 to 4).

If `g` would require more than `k_max` digits:

- Increase `N` or adjust mapping (e.g., allocate glyph IDs in lexicographic ranges).

## 4. Length allocation strategy

We want shorter glyphs for more frequent clusters.

Strategy:

1. Estimate cluster frequencies.

2. Rank clusters by frequency.

3. Allocate IDs such that:

   - Top `M1` clusters get 1-char codes.

   - Next `M2` clusters get 2-char codes.

   - etc.

We achieve this by:

- Assigning `glyph_id`s in frequency order.

- Using base-N encoding with a bias that produces shorter strings first.

## 5. Unicode representation

Glyph strings:

- Are stored internally as Unicode strings.

- Must be passed through tokenizers that understand full codepoints.

Caveat:

- Some tooling may need to be tested against unusual CJK characters.

## 6. Decoding

To decode a glyph string:

1. For each character:

   - Find index in `glyph_alphabet` → digit `d_i`.

2. Compute:

\[
g = \sum d_i \cdot N^i
\]

3. `g` is `glyph_id`.

Mapping to cluster:

- Use `glyph_id -> cluster_id` mapping in tape storage.

## 7. Examples

Assume:

- `glyph_alphabet = [家, 山, 水, 火, 木]` (N=5 for example).

- `glyph_id = 7`.

Encoding:

- `7` in base 5 = `2 * 5^1 + 2 * 5^0` → digits `[2, 2]` → `水水`.

(In practice N will be much larger.)

## 8. Config

Config file:

```json
{
  "glyph_alphabet_file": "config/glyph_alphabet.json",
  "min_length": 1,
  "max_length": 4,
  "allocation_strategy": "frequency-biased-baseN"
}
```

