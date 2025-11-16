# Tech Stack and Dependencies

## 1. Target environment

- OS: Linux (Ubuntu recommended)

- GPU: NVIDIA RTX-class with CUDA

- Python: 3.10+ (configurable)

## 2. Core libraries

- **PyTorch** – GPU-accelerated tensors and transformer models.

- **SentenceTransformers** – pre-trained sentence embedding models.

- **scikit-learn** – clustering (MiniBatchKMeans).

- **NumPy** – numerical arrays.

- **SQLite** – lightweight databases.

- **msgpack / JSON** – serialization.

Optional:

- **UMAP / PCA** – dimensionality reduction.

- **Faiss / Annoy** – approximate nearest neighbor search.

- **FastAPI** – visualize/serve tape data.

- **React/Next.js** – front-end visualizer.

## 3. Python environment

Example `requirements.txt`:

```text
torch
sentence-transformers
scikit-learn
numpy
umap-learn
fastapi
uvicorn
msgpack
sqlalchemy
pydantic
```

## 4. GPU configuration

* CUDA toolkit installed.

* `torch.cuda.is_available()` must be `True`.

* Batch sizes configurable based on GPU memory.

## 5. Project structure

* `src/` – core modules.

* `scripts/` – pipeline orchestration.

* `configs/` – YAML configs per experiment.

