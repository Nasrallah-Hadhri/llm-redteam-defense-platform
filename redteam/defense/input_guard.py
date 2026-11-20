"""Input guard — classifies an incoming prompt as adversarial or benign."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


class InputGuard:
    """Blocks adversarial prompts before they reach the target model.

    Uses a sentence-transformers embedding + scikit-learn classifier
    trained by train_guard.py.
    """

    def __init__(self, embedder: Any, classifier: Any) -> None:
        self.embedder = embedder
        self.classifier = classifier

    def blocks(self, text: str) -> bool:
        """Return True if the prompt is classified as adversarial (class 1)."""
        embedding = self.embedder.encode([text])
        prediction = self.classifier.predict(embedding)
        return bool(prediction[0])

    def predict_proba(self, text: str) -> float:
        """Return the estimated probability of the prompt being adversarial (class 1)."""
        embedding = self.embedder.encode([text])
        proba = self.classifier.predict_proba(embedding)
        return float(proba[0][1])

    @classmethod
    def load(
        cls,
        model_path: str | Path = "data/input_guard.pkl",
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> InputGuard:
        """Load a trained InputGuard from disk."""
        from sentence_transformers import SentenceTransformer  # deferred import

        embedder = SentenceTransformer(embedder_name)
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        return cls(embedder, clf)
