"""
embeddings.py
Embeddings utilities - fine wrapper over sentence-transformers
Separately maintained to avoid dependency on sentence-transformers in the main memory module.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    
    return float(
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9)
    )

def load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load the sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as e:
        raise ImportError(
            f"Failed to load the model '{model_name}'. Ensure 'sentence-transformers' is installed."
        ) from e

def encode(model, text: str) -> Optional[np.ndarray]:
    """Encode the text into an embedding vector using the provided model."""
    if not model:
        raise ValueError("Model is not loaded. Call load_model() first.")
    return model.encode(text, convert_to_numpy=True).astype(np.float32)
