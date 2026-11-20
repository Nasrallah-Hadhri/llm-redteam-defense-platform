"""Database migration utility with automatic pre-migration backup.

Ensures zero row loss and provides safe schema evolution for the SQLite audit database.
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def backup_database(db_path: str | Path) -> Path:
    """Create a timestamped backup of the database file before running migrations.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Path to the created backup file.
    """
    source = Path(db_path)
    if not source.exists():
        raise FileNotFoundError(f"Database file not found for backup: {source}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = source.parent / f"redteam_backup_{timestamp}.db"
    shutil.copy2(source, backup_file)

    logger.info("Created pre-migration database backup: %s", backup_file)
    return backup_file


def apply_migrations(db_path: str | Path = "data/redteam.db") -> dict[str, Any]:
    """Apply non-destructive schema migrations to the database.

    Guarantees:
    1. A timestamped backup is generated before any schema modification.
    2. Total row counts of existing tables are verified before and after migration.
    3. Existing records are preserved completely.

    Returns:
        Dictionary summarizing migration results and verification status.
    """
    path = Path(db_path)
    if not path.exists():
        # Nothing to migrate if database hasn't been initialized yet
        return {
            "migrated": False,
            "reason": "Database file does not exist yet.",
            "pre_count": 0,
            "post_count": 0,
            "backup_path": None,
        }

    # Step 1: Pre-migration verification and backup
    conn_pre = sqlite3.connect(str(path))
    cur_pre = conn_pre.cursor()

    cur_pre.execute("SELECT COUNT(*) FROM attempts")
    pre_attempts_count = cur_pre.fetchone()[0]

    cur_pre.execute("SELECT COUNT(*) FROM campaigns")
    pre_campaigns_count = cur_pre.fetchone()[0]
    conn_pre.close()

    backup_path = backup_database(path)

    # Step 2: Apply non-destructive migrations
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    # Inspect attempts table
    cur.execute("PRAGMA table_info(attempts)")
    existing_columns = {row[1] for row in cur.fetchall()}

    added_columns: list[str] = []

    if "judge_model" not in existing_columns:
        cur.execute("ALTER TABLE attempts ADD COLUMN judge_model TEXT")
        added_columns.append("judge_model")

    if "judge_model_digest" not in existing_columns:
        cur.execute("ALTER TABLE attempts ADD COLUMN judge_model_digest TEXT")
        added_columns.append("judge_model_digest")

    # Step 3: Create calibration_samples table if not present
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_samples (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id         TEXT UNIQUE NOT NULL,
            attempt_id        INTEGER NOT NULL,
            campaign_id       TEXT NOT NULL,
            target_model      TEXT NOT NULL,
            strategy          TEXT NOT NULL,
            goal              TEXT NOT NULL,
            prompt            TEXT NOT NULL,
            response          TEXT NOT NULL,
            judge_violated    BOOLEAN NOT NULL,
            judge_severity    INTEGER,
            judge_reasoning   TEXT,
            judge_model       TEXT,
            judge_model_digest TEXT,
            human_violated    BOOLEAN,
            human_severity    INTEGER,
            human_notes       TEXT,
            exported_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
            labeled_at        DATETIME
        );
        """
    )

    conn.commit()

    # Step 4: Post-migration verification
    cur.execute("SELECT COUNT(*) FROM attempts")
    post_attempts_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM campaigns")
    post_campaigns_count = cur.fetchone()[0]
    conn.close()

    if pre_attempts_count != post_attempts_count:
        raise RuntimeError(
            f"Migration row count mismatch on attempts: pre={pre_attempts_count}, post={post_attempts_count}!"
        )

    if pre_campaigns_count != post_campaigns_count:
        raise RuntimeError(
            f"Migration row count mismatch on campaigns: pre={pre_campaigns_count}, post={post_campaigns_count}!"
        )

    logger.info(
        "Migration successful. Attempts count verified: %d -> %d. Columns added: %s",
        pre_attempts_count,
        post_attempts_count,
        added_columns,
    )

    return {
        "migrated": True,
        "backup_path": str(backup_path),
        "pre_attempts_count": pre_attempts_count,
        "post_attempts_count": post_attempts_count,
        "pre_campaigns_count": pre_campaigns_count,
        "post_campaigns_count": post_campaigns_count,
        "columns_added": added_columns,
    }
