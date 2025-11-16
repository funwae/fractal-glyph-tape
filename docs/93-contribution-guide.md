# Contribution Guide

## 1. Code style

- Python:

  - PEP8 where practical.

  - Type hints for public APIs.

- Use `black` or similar formatter.

## 2. Directory structure

- Keep modules under `src/` with clear boundaries (`ingest`, `embed`, etc.).

- Avoid monolithic scripts; prefer composable functions.

## 3. Issues and feature requests

- Open an issue with:

  - Problem description.

  - Proposed solution.

  - Expected impact on experiments.

## 4. Adding experiments

- Place experiment configs under `configs/`.

- Document new experiments in `docs/6x-*.md` as needed.

## 5. Tests

- Add unit tests for:

  - Tape storage.

  - Glyph manager.

  - Tokenizer wrapper.

## 6. Documentation

- Keep docs in sync with implementation.

- If you change an interface, update the corresponding `.md` spec.

