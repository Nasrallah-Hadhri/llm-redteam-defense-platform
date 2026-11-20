"""Training script for input and output defense guards.

Pulls attack data from the SQLite database (guard_active=0 runs only),
combines with the benign prompts seed set, trains LogisticRegression
classifiers for both prompt (input) and response (output) screening,
evaluates precision, recall, and false-positive rate on benign inputs,
and serializes the models to disk.

Usage:
    python -m redteam.defense.train_guard [--db data/redteam.db]
                                          [--benign data/benign_prompts.jsonl]
                                          [--out-dir data/]
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import redteam.db as db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_benign_texts(jsonl_path: str | Path) -> list[str]:
    """Load benign prompt/response strings from a JSONL file."""
    path = Path(jsonl_path)
    if not path.exists():
        logger.warning("Benign prompts file not found at %s", path)
        return []
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("prompt") or ""
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue
    return texts


def train_classifier(texts: list[str], labels: list[int], embedder: Any) -> Any:
    """Embed texts and train a LogisticRegression classifier. Returns fitted model."""
    logger.info("Generating sentence embeddings for %d texts...", len(texts))
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)

    # If all labels are of a single class, train_test_split stratify will fail
    unique_classes = set(labels)
    if len(unique_classes) < 2:
        logger.warning("Only 1 class found in labels (%s). Fitting dummy/fallback model.", unique_classes)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(embeddings, labels)
        return clf

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    logger.info(
        "Classification report (held-out test set):\n%s",
        classification_report(y_test, y_pred, target_names=["benign", "adversarial"], zero_division=0),
    )
    return clf


def compute_fpr_on_benign(embedder: Any, clf: Any, benign_texts: list[str]) -> float:
    """Compute false-positive rate: fraction of benign prompts classified as adversarial."""
    if not benign_texts:
        return 0.0
    embeddings = embedder.encode(benign_texts, show_progress_bar=False)
    preds = clf.predict(embeddings)
    fpr = float(np.sum(preds == 1) / len(preds))
    logger.info(
        "False-Positive Rate on benign set: %.2f%% (%d / %d benign blocked)",
        fpr * 100.0,
        int(np.sum(preds == 1)),
        len(preds),
    )
    return fpr


def main() -> None:
    parser = argparse.ArgumentParser(description="Train input and output defense guards.")
    parser.add_argument("--db", default="data/redteam.db", help="Path to SQLite database.")
    parser.add_argument("--benign", default="data/benign_prompts.jsonl", help="Path to benign prompts JSONL.")
    parser.add_argument("--out-dir", default="data", help="Directory where models (.pkl) will be saved.")
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or path.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedder '%s'...", args.embedder)
    embedder = SentenceTransformer(args.embedder)

    conn = db.get_connection(args.db)
    db.init_db(conn)

    # Pull un-guarded attempts from DB to avoid biased training on filtered data
    df = pd.read_sql_query(
        "SELECT prompt, response, judge_violated FROM attempts WHERE guard_active = 0",
        conn,
    )
    logger.info("Found %d unguarded attempt records in %s.", len(df), args.db)

    benign_texts = load_benign_texts(args.benign)
    logger.info("Loaded %d benign texts from %s.", len(benign_texts), args.benign)

    if len(df) == 0 and len(benign_texts) == 0:
        logger.error("No training data available in database or benign set.")
        return

    # --- Train Input Guard ---
    logger.info("=== Training INPUT Guard ===")
    pos_prompts = df[df["judge_violated"] == 1]["prompt"].tolist() if not df.empty else []
    neg_prompts = df[df["judge_violated"] == 0]["prompt"].tolist() if not df.empty else []
    all_neg_prompts = neg_prompts + benign_texts

    logger.info(
        "Input Guard dataset: %d positive (adversarial) + %d negative (benign/non-violating)",
        len(pos_prompts),
        len(all_neg_prompts),
    )

    if pos_prompts and all_neg_prompts:
        texts_in = pos_prompts + all_neg_prompts
        labels_in = [1] * len(pos_prompts) + [0] * len(all_neg_prompts)
        clf_input = train_classifier(texts_in, labels_in, embedder)
        compute_fpr_on_benign(embedder, clf_input, benign_texts)

        out_path_in = out_dir / "input_guard.pkl"
        with open(out_path_in, "wb") as f:
            pickle.dump(clf_input, f)
        logger.info("Saved input guard classifier to %s", out_path_in)
    else:
        logger.warning("Insufficient positive/negative prompt balance to train input guard.")

    # --- Train Output Guard ---
    logger.info("=== Training OUTPUT Guard ===")
    pos_responses = df[df["judge_violated"] == 1]["response"].tolist() if not df.empty else []
    neg_responses = df[df["judge_violated"] == 0]["response"].tolist() if not df.empty else []
    all_neg_responses = neg_responses + benign_texts

    logger.info(
        "Output Guard dataset: %d positive (violating response) + %d negative (safe/benign)",
        len(pos_responses),
        len(all_neg_responses),
    )

    if pos_responses and all_neg_responses:
        texts_out = pos_responses + all_neg_responses
        labels_out = [1] * len(pos_responses) + [0] * len(all_neg_responses)
        clf_output = train_classifier(texts_out, labels_out, embedder)
        compute_fpr_on_benign(embedder, clf_output, benign_texts)

        out_path_out = out_dir / "output_guard.pkl"
        with open(out_path_out, "wb") as f:
            pickle.dump(clf_output, f)
        logger.info("Saved output guard classifier to %s", out_path_out)
    else:
        logger.warning("Insufficient positive/negative response balance to train output guard.")

    logger.info("Guard training pipeline finished.")


if __name__ == "__main__":
    main()
