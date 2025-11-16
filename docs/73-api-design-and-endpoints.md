# API Design and Endpoints

This defines programmatic access to FGT.

## 1. REST API

Base path: `/api/fgt`

Endpoints:

- `POST /encode`

  - Input: text.

  - Output: glyph-coded representation + metadata.

- `POST /decode`

  - Input: glyph-coded text.

  - Output: reconstructed text.

- `GET /glyph/{glyph}`

  - Details of glyph/cluster.

- `GET /cluster/{cluster_id}`

  - Cluster metadata.

- `GET /layout`

  - Coordinates for visualization.

## 2. Response formats

Use JSON for structured responses.

Example `/encode` response:

```json
{
  "input": "Can you send me that file?",
  "encoded": "谷阜",
  "spans": [
    {
      "start": 0,
      "end": 28,
      "cluster_id": 12345,
      "glyph": "谷阜"
    }
  ]
}
```

## 3. Auth and rate limits

* For public demos:

  * API keys or per-IP rate limiting.

* For internal use:

  * Wider limits.

Implementation via FastAPI + standard middlewares.

