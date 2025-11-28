"""Theme discovery utilities using sentence embeddings + clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ClusteringConfig:
    """Configuration for UMAP + HDBSCAN clustering."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    min_cluster_size: int = 10


def embed_sentences(texts: Iterable[str]) -> np.ndarray:
    """Placeholder embedding function."""
    # TODO: Replace with sentence-transformers embedding generation.
    text_list = list(texts)
    return np.zeros((len(text_list), 768))


def cluster_embeddings(embeddings: np.ndarray, config: ClusteringConfig) -> np.ndarray:
    """Placeholder clustering function."""
    # TODO: Integrate UMAP + HDBSCAN pipeline here.
    return np.zeros(len(embeddings), dtype=int)