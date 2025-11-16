# Open Questions and Risks

## 1. Cluster quality

- How well do clusters align with human intuitions?

- How sensitive are results to embedding model choice and clustering hyperparameters?

## 2. Glyph matching in the wild

- Can we reliably detect phrase spans in arbitrary text?

- Risk of mis-glyphification leading to semantic drift.

## 3. LLM behavior

- Will models actually use glyph tokens meaningfully?

- Risk of overfitting to quirks of tape structure.

## 4. Cross-lingual issues

- Do multilingual embeddings produce enough cross-lingual clusters?

- Are there language-specific artifacts that confuse clustering?

## 5. Privacy and compliance

- When compressing real user data/logs:

  - Need to handle privacy and deletion requests.

  - FGT does not inherently anonymize data.

## 6. Engineering complexity

- Managing tape versions.

- Keeping architecture maintainable as experiments grow.

Each risk should be revisited after Phase 0–2 experiments.

