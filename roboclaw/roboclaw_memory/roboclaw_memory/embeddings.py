"""Utilitários de embedding."""
from __future__ import annotations
from typing import Optional
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

def load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception: return None

def encode(model, text: str) -> Optional[np.ndarray]:
    if model is None: return None
    return model.encode(text, convert_to_numpy=True).astype(np.float32)
