"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_phrases():
    """Sample phrases for testing."""
    return [
        "Can you send me that file?",
        "Could you email me the document?",
        "Please share the file with me.",
        "Hello, how are you?",
        "Hi there!",
        "Good morning!",
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    # 6 phrases x 384 dimensions (typical for all-MiniLM-L6-v2)
    np.random.seed(42)
    return np.random.randn(6, 384).astype(np.float32)


@pytest.fixture
def sample_alphabet():
    """Sample Mandarin character alphabet."""
    return ["谷", "阜", "堯", "奚", "齊", "沐", "燁", "璟"]


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    return data_dir


@pytest.fixture
def temp_config(tmp_path):
    """Create temporary config file."""
    config = {
        "project_name": "test_project",
        "random_seed": 42,
        "ingest": {
            "input_path": str(tmp_path / "data" / "raw"),
            "output_path": str(tmp_path / "data" / "phrases.jsonl"),
        },
    }
    return config
