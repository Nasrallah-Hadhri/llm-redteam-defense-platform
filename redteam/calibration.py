"""Judge calibration module — stratified sampling, CSV export/import, and agreement metrics.

Provides inter-rater reliability analysis (percent agreement, Cohen's kappa,
and confusion matrix) comparing Judge LLM verdicts against human expert ground truth.
"""
from __future__ import annotations

import csv
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Severity threshold for automatic redaction in exported calibration datasets
REDACTION_SEVERITY_THRESHOLD = 4


def export_calibration_sample(
    conn: sqlite3.Connection,
    out_csv_path: str | Path,
    sample_size: int = 50,
    campaign_id: str | None = None,
    random_state: int = 42,
) -> tuple[Path, int]:
    """Export a stratified random sample of attempts to CSV for human calibration.

    Responses for high severity (>= 4) violations are strictly redacted,
    preserving analytical summaries while preventing raw exploit dissemination.

    Args:
        conn: SQLite connection to audit database.
        out_csv_path: Destination path for exported CSV.
        sample_size: Number of attempts to sample (typically 50-100).
        campaign_id: Optional campaign filter.
        random_state: Seed for reproducible stratified sampling.

    Returns:
        Tuple of (path_to_exported_csv, count_of_exported_rows).
    """
    where_clause = "WHERE campaign_id = ?" if campaign_id else ""
    params = (campaign_id,) if campaign_id else ()

    query = f"""
        SELECT id, campaign_id, target_model, strategy, goal,
               attempt_number, prompt, response, judge_violated,
               judge_category, judge_severity, judge_reasoning,
               judge_model, judge_model_digest
        FROM attempts
        {where_clause}
        ORDER BY id ASC
    """
    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        raise ValueError("No attempt records found in database to export for calibration.")

    # Stratified sampling across (judge_violated, judge_severity)
    strata_col = df["judge_violated"].astype(str) + "_" + df["judge_severity"].fillna(1).astype(str)
    df["_stratum"] = strata_col

    if len(df) <= sample_size:
        sample_df = df.copy()
    else:
        # Sample proportionally per stratum, ensuring at least 1 per stratum if possible
        sample_df = (
            df.groupby("_stratum", group_keys=False)
            .apply(
                lambda x: x.sample(
                    max(1, int(round(len(x) / len(df) * sample_size))),
                    random_state=random_state,
                )
            )
        )
        if len(sample_df) > sample_size:
            sample_df = sample_df.sample(sample_size, random_state=random_state)
        elif len(sample_df) < sample_size:
            remaining = df[~df["id"].isin(sample_df["id"])]
            needed = sample_size - len(sample_df)
            if len(remaining) > 0:
                fill_sample = remaining.sample(min(needed, len(remaining)), random_state=random_state)
                sample_df = pd.concat([sample_df, fill_sample])

    sample_set_id = f"calib_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out_rows = []

    for _, r in sample_df.iterrows():
        is_violated = bool(r["judge_violated"])
        severity = int(r["judge_severity"]) if pd.notnull(r["judge_severity"]) else 1

        # Redact high severity (>=4) violating responses
        if is_violated and severity >= REDACTION_SEVERITY_THRESHOLD:
            reasoning_summary = str(r["judge_reasoning"]).strip() if pd.notnull(r["judge_reasoning"]) else "Policy violation elicited"
            safe_response = f"[REDACTED — Severity {severity}/5 research finding. Analytical summary: {reasoning_summary}]"
        else:
            safe_response = str(r["response"])

        row_dict = {
            "sample_set_id": sample_set_id,
            "attempt_id": int(r["id"]),
            "campaign_id": str(r["campaign_id"]),
            "target_model": str(r["target_model"]),
            "strategy": str(r["strategy"]),
            "goal": str(r["goal"]),
            "attempt_number": int(r["attempt_number"]),
            "prompt": str(r["prompt"]),
            "response": safe_response,
            "judge_violated": 1 if is_violated else 0,
            "judge_severity": severity,
            "judge_reasoning": str(r["judge_reasoning"]) if pd.notnull(r["judge_reasoning"]) else "",
            "judge_model": str(r["judge_model"]) if pd.notnull(r["judge_model"]) else "unknown",
            "judge_model_digest": str(r["judge_model_digest"]) if pd.notnull(r["judge_model_digest"]) else "unknown",
            # Blank columns for human evaluator
            "human_violated": "",
            "human_severity": "",
            "human_notes": "",
        }
        out_rows.append(row_dict)

        # Record in calibration_samples table if table exists
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO calibration_samples
                (sample_id, attempt_id, campaign_id, target_model, strategy, goal,
                 prompt, response, judge_violated, judge_severity, judge_reasoning,
                 judge_model, judge_model_digest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{sample_set_id}_{r['id']}",
                    int(r["id"]),
                    str(r["campaign_id"]),
                    str(r["target_model"]),
                    str(r["strategy"]),
                    str(r["goal"]),
                    str(r["prompt"]),
                    safe_response,
                    is_violated,
                    severity,
                    str(r["judge_reasoning"]),
                    str(r["judge_model"]),
                    str(r["judge_model_digest"]),
                ),
            )
        except sqlite3.OperationalError:
            pass

    conn.commit()

    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_df = pd.DataFrame(out_rows)
    export_df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Exported %d calibration samples to %s", len(export_df), out_path)

    return out_path, len(export_df)


def parse_boolean_label(val: Any) -> bool | None:
    """Parse various representations of truthy/falsy values into a boolean."""
    if pd.isnull(val):
        return None
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "violated"):
        return True
    if s in ("0", "false", "no", "n", "safe"):
        return False
    return None


def import_calibration_labels(
    csv_file: str | Path | pd.DataFrame,
    conn: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    """Load calibration CSV and extract rows with completed human ratings.

    Args:
        csv_file: Path to CSV or loaded DataFrame.
        conn: Optional SQLite connection to persist completed ratings.

    Returns:
        DataFrame containing valid labeled rows with parsed boolean labels.
    """
    if isinstance(csv_file, pd.DataFrame):
        df = csv_file.copy()
    else:
        df = pd.read_csv(csv_file, encoding="utf-8")

    required_cols = {"attempt_id", "judge_violated", "human_violated"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Calibration CSV missing required columns: {missing}")

    df["human_violated_parsed"] = df["human_violated"].apply(parse_boolean_label)
    df["judge_violated_parsed"] = df["judge_violated"].apply(parse_boolean_label)

    # Filter only rows where human filled in the label
    labeled = df[df["human_violated_parsed"].notnull() & df["judge_violated_parsed"].notnull()].copy()

    if labeled.empty:
        raise ValueError("No completed human labels found in the provided CSV file.")

    if conn is not None:
        for _, row in labeled.iterrows():
            attempt_id = int(row["attempt_id"])
            h_viol = bool(row["human_violated_parsed"])
            h_sev = int(row["human_severity"]) if pd.notnull(row.get("human_severity")) and str(row.get("human_severity")).strip().isdigit() else None
            h_notes = str(row.get("human_notes", "") or "")

            try:
                conn.execute(
                    """
                    UPDATE calibration_samples
                    SET human_violated = ?, human_severity = ?, human_notes = ?, labeled_at = CURRENT_TIMESTAMP
                    WHERE attempt_id = ?
                    """,
                    (h_viol, h_sev, h_notes, attempt_id),
                )
            except sqlite3.OperationalError:
                pass
        conn.commit()

    return labeled


def compute_calibration_metrics(labeled_df: pd.DataFrame) -> dict[str, Any]:
    """Compute Cohen's kappa, percent agreement, and confusion matrix.

    Args:
        labeled_df: DataFrame with judge_violated_parsed and human_violated_parsed.

    Returns:
        Dictionary of statistical agreement metrics and confusion matrix counts.
    """
    if "human_violated_parsed" not in labeled_df.columns or "judge_violated_parsed" not in labeled_df.columns:
        labeled_df = labeled_df.copy()
        labeled_df["human_violated_parsed"] = labeled_df["human_violated"].apply(parse_boolean_label)
        labeled_df["judge_violated_parsed"] = labeled_df["judge_violated"].apply(parse_boolean_label)

    valid = labeled_df[labeled_df["human_violated_parsed"].notnull() & labeled_df["judge_violated_parsed"].notnull()]
    n = len(valid)
    if n == 0:
        raise ValueError("Cannot compute calibration metrics on empty dataset.")

    y_judge = valid["judge_violated_parsed"].astype(bool)
    y_human = valid["human_violated_parsed"].astype(bool)

    # Confusion matrix components (Judge vs Human Ground Truth)
    # TP: Judge=True, Human=True
    tp = int(((y_judge == True) & (y_human == True)).sum())
    # FP: Judge=True, Human=False
    fp = int(((y_judge == True) & (y_human == False)).sum())
    # FN: Judge=False, Human=True
    fn = int(((y_judge == False) & (y_human == True)).sum())
    # TN: Judge=False, Human=False
    tn = int(((y_judge == False) & (y_human == False)).sum())

    observed_agreement = (tp + tn) / n

    # Expected chance agreement:
    # P(judge=1) * P(human=1) + P(judge=0) * P(human=0)
    p_judge_yes = (tp + fp) / n
    p_judge_no = (fn + tn) / n
    p_human_yes = (tp + fn) / n
    p_human_no = (fp + tn) / n

    expected_agreement = (p_judge_yes * p_human_yes) + (p_judge_no * p_human_no)

    if expected_agreement >= 1.0:
        kappa = 1.0 if observed_agreement >= 1.0 else 0.0
    else:
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

    # Qualitative classification according to Landis & Koch (1977)
    if kappa < 0.0:
        interpretation = "Poor (Less than chance agreement)"
    elif kappa <= 0.20:
        interpretation = "Slight agreement"
    elif kappa <= 0.40:
        interpretation = "Fair agreement"
    elif kappa <= 0.60:
        interpretation = "Moderate agreement"
    elif kappa <= 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"

    # Severity analysis if human_severity is present
    severity_metrics = {}
    if "human_severity" in valid.columns and "judge_severity" in valid.columns:
        sev_df = valid[valid["human_severity"].notnull() & valid["judge_severity"].notnull()].copy()
        sev_df = sev_df[
            sev_df["human_severity"].astype(str).str.strip().str.isdigit()
            & sev_df["judge_severity"].astype(str).str.strip().str.isdigit()
        ]
        if len(sev_df) > 0:
            j_sev = sev_df["judge_severity"].astype(int)
            h_sev = sev_df["human_severity"].astype(int)
            mae = float((j_sev - h_sev).abs().mean())
            exact_match = float((j_sev == h_sev).mean() * 100.0)
            severity_metrics = {
                "severity_count": len(sev_df),
                "severity_mae": round(mae, 2),
                "severity_exact_match_pct": round(exact_match, 1),
            }

    return {
        "sample_size": n,
        "percent_agreement": round(observed_agreement * 100.0, 2),
        "cohens_kappa": round(kappa, 4),
        "interpretation": interpretation,
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        },
        "marginal_distributions": {
            "judge_positive_rate_pct": round(p_judge_yes * 100.0, 1),
            "human_positive_rate_pct": round(p_human_yes * 100.0, 1),
        },
        **severity_metrics,
    }
