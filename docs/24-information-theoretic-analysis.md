# Information-Theoretic Analysis

We want to understand FGT's compression and capacity in information-theoretic terms.

## 1. Entropy of phrase space

Let `X` be random variable over phrases in corpus.

- Entropy: `H(X)` bits.

If many phrases are near-duplicates or variants of motifs:

- Effective entropy is lower than raw text suggests.

## 2. FGT factorization

FGT factorizes phrase generation into:

- Choice of **phrase family** `K`.

- Choice of **within-family variant** `V`.

So:

\[
H(X) = H(K) + H(V \mid K)
\]

FGT compresses by:

- Explicitly modeling and storing `H(K)` (glyph codes, frequency counts).

- Using more compact representations for `H(V|K)`.

## 3. Compression ratio insight

If we assign:

- Short codes for high-probability families.

- Longer codes or explicit text for rare ones.

We approach something like a **Shannon-style code** over cluster IDs.

## 4. Capacity of glyph space

If we have:

- `M` possible glyph codes (given length and alphabet).

Then we can name up to `M` phrase families.

We can compare:

- `log2(M)` bits of ID capacity vs estimated `H(K)`.

Goal:

- Glyph code capacity comfortably exceeds actual entropy we need for phrase families.

## 5. Context window utilization

One glyph token encoding a high-entropy phrase family can replace many tokens of explicit text.

- Effective bits per token go up.

- So context window carries more information.

We can derive:

\[
\text{EffectiveContextMultiplier} \approx \frac{\text{bits-per-token-with-FGT}}{\text{bits-per-token-raw}}
\]

Measured empirically (see `62-context-window-efficiency-experiments.md`).

