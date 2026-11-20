"""Output guard — classifies a target response as policy-violating or safe."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


class OutputGuard:
    """Blocks policy-violating responses before the judge or user receives them.

    Same architecture as InputGuard but trained on model response texts
    and judge verdicts as ground truth.
    """

    def __init__(self, embedder: Any, classifier: Any) -> None:
        self.embedder = embedder
        self.classifier = classifier

    def blocks(self, text: str) -> bool:
        """Return True if the response is classified as policy-violating (class 1)."""
        embedding = self.embedder.encode([text])
        prediction = self.classifier.predict(embedding)
        return bool(prediction[0])

    def predict_proba(self, text: str) -> float:
        """Return the estimated probability of the response being policy-violating (class 1)."""
        embedding = self.embedder.encode([text])
        proba = self.classifier.predict_proba(embedding)
        return float(proba[0][1])

    @classmethod
    def load(
        cls,
        model_path: str | Path = "data/output_guard.pkl",
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> OutputGuard:
        """Load a trained OutputGuard from disk."""
        from sentence_transformers import SentenceTransformer  # deferred import

        embedder = SentenceTransformer(embedder_name)
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        return cls(embedder, clf)
