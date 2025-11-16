# Web Visualizer Specification

We provide a web UI for exploring the fractal tape.

## 1. Goals

- Show phrase families as points on a fractal map.

- Allow interactive inspection of glyphs and examples.

## 2. Backend

- **FastAPI** service.

- Endpoints:

  - `/clusters`: list of cluster summaries.

  - `/cluster/{id}`: detailed metadata including example phrases.

  - `/glyph/{glyph}`: cluster lookup.

  - `/layout`: 2D coordinates for plotting.

## 3. Frontend

- React/Next.js app.

- Panels:

  - **Map panel**:

    - Display scatterplot of clusters on triangle.

    - Color by language or frequency.

  - **Details panel**:

    - Show glyph, examples, stats on hover/click.

## 4. Data preparation

- Precompute 2D layout coordinates and store in `clusters/layout.npy`.

- Backend streams them to frontend.

## 5. Interaction

- Hover a point → show cluster summary.

- Click → lock selection in side panel.

- Filters:

  - Language, frequency thresholds, tape version.

