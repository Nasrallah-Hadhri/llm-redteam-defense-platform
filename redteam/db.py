"""SQLite persistence layer — schema, insertion helpers, and query helpers.

All campaign data is written to a single SQLite file (data/redteam.db by
default). The schema is append-only; no rows are ever deleted by the
platform so the full audit trail is preserved.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id        TEXT    NOT NULL,
    target_model       TEXT    NOT NULL,
    strategy           TEXT    NOT NULL,
    goal               TEXT    NOT NULL,
    attempt_number     INTEGER NOT NULL,
    prompt             TEXT    NOT NULL,
    response           TEXT    NOT NULL,
    judge_violated     BOOLEAN NOT NULL,
    judge_category     TEXT,
    judge_severity     INTEGER,
    judge_reasoning    TEXT,
    guard_active       BOOLEAN NOT NULL DEFAULT 0,
    guard_blocked      BOOLEAN NOT NULL DEFAULT 0,
    judge_model        TEXT,
    judge_model_digest TEXT,
    timestamp          DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_attempts_model_strategy
    ON attempts (target_model, strategy, guard_active);

CREATE INDEX IF NOT EXISTS idx_attempts_campaign
    ON attempts (campaign_id);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id   TEXT PRIMARY KEY,
    started_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    finished_at   DATETIME,
    config_json   TEXT
);

CREATE TABLE IF NOT EXISTS calibration_samples (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id          TEXT UNIQUE NOT NULL,
    attempt_id         INTEGER NOT NULL,
    campaign_id        TEXT NOT NULL,
    target_model       TEXT NOT NULL,
    strategy           TEXT NOT NULL,
    goal               TEXT NOT NULL,
    prompt             TEXT NOT NULL,
    response           TEXT NOT NULL,
    judge_violated     BOOLEAN NOT NULL,
    judge_severity     INTEGER,
    judge_reasoning    TEXT,
    judge_model        TEXT,
    judge_model_digest TEXT,
    human_violated     BOOLEAN,
    human_severity     INTEGER,
    human_notes        TEXT,
    exported_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    labeled_at         DATETIME
);
"""

# Columns allowed in GROUP BY to prevent SQL injection from config values.
_ALLOWED_GROUP_COLS = frozenset(
    {
        "target_model",
        "strategy",
        "goal",
        "guard_active",
        "judge_category",
        "campaign_id",
        "judge_model",
    }
)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_connection(db_path: str | Path = "data/redteam.db") -> sqlite3.Connection:
    """Return a sqlite3 connection with WAL mode enabled.

    Creates the parent directory if it does not exist. Uses row_factory
    so rows can be accessed as both index and key.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not already exist."""
    conn.executescript(_SCHEMA)
    conn.commit()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def log_attempt(conn: sqlite3.Connection, **kwargs: Any) -> int:
    """Insert one attempt row and return its auto-assigned row id.

    Keyword arguments must match the attempts table columns (excluding
    ``id`` and ``timestamp``, which are set automatically).
    """
    cols = [
        "campaign_id",
        "target_model",
        "strategy",
        "goal",
        "attempt_number",
        "prompt",
        "response",
        "judge_violated",
        "judge_category",
        "judge_severity",
        "judge_reasoning",
        "guard_active",
        "guard_blocked",
        "judge_model",
        "judge_model_digest",
    ]
    defaults = {
        "guard_active": False,
        "guard_blocked": False,
        "judge_violated": False,
        "judge_category": "none",
        "judge_severity": 1,
        "judge_reasoning": "",
        "judge_model": None,
        "judge_model_digest": None,
    }
    values = [kwargs.get(c, defaults.get(c)) for c in cols]
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    cur = conn.execute(
        f"INSERT INTO attempts ({col_names}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def log_campaign_start(
    conn: sqlite3.Connection, campaign_id: str, config_json: str = ""
) -> None:
    """Record a campaign start in the campaigns table."""
    conn.execute(
        "INSERT OR IGNORE INTO campaigns (campaign_id, config_json) VALUES (?, ?)",
        (campaign_id, config_json),
    )
    conn.commit()


def log_campaign_end(conn: sqlite3.Connection, campaign_id: str) -> None:
    """Stamp the finished_at timestamp for a campaign."""
    conn.execute(
        "UPDATE campaigns SET finished_at = CURRENT_TIMESTAMP WHERE campaign_id = ?",
        (campaign_id,),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def get_campaign_summary(conn: sqlite3.Connection, campaign_id: str) -> dict:
    """Return high-level counts for a campaign as a plain dict."""
    row = conn.execute(
        """
        SELECT
            COUNT(*)                          AS total_attempts,
            SUM(judge_violated)               AS violations,
            SUM(guard_blocked)                AS guard_blocked_total,
            COUNT(DISTINCT target_model)      AS num_targets,
            COUNT(DISTINCT strategy)          AS num_strategies
        FROM attempts
        WHERE campaign_id = ?
        """,
        (campaign_id,),
    ).fetchone()
    return dict(row) if row else {}


def get_success_rate_by(
    conn: sqlite3.Connection,
    group_cols: list[str],
    campaign_id: str | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with success_rate_pct grouped by the given columns.

    Args:
        conn: Active SQLite connection.
        group_cols: Column names to group by. Must be a subset of
            _ALLOWED_GROUP_COLS to prevent SQL injection.
        campaign_id: If provided, filters results to this campaign only.

    Returns:
        DataFrame with columns: *group_cols*, total_attempts, violations,
        guard_blocked_count, success_rate_pct.

    Raises:
        ValueError: If any column name is not in the allowlist.
    """
    bad = set(group_cols) - _ALLOWED_GROUP_COLS
    if bad:
        raise ValueError(f"Disallowed group columns: {bad}")

    cols_sql = ", ".join(group_cols)
    where_clause = "WHERE campaign_id = ?" if campaign_id else ""
    params: tuple = (campaign_id,) if campaign_id else ()

    query = f"""
        SELECT
            {cols_sql},
            COUNT(*)                                                    AS total_attempts,
            SUM(CASE WHEN judge_violated AND NOT guard_blocked
                     THEN 1 ELSE 0 END)                                AS violations,
            SUM(guard_blocked)                                          AS guard_blocked_count,
            ROUND(
                100.0
                * SUM(CASE WHEN judge_violated AND NOT guard_blocked
                           THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
            1)                                                          AS success_rate_pct
        FROM attempts
        {where_clause}
        GROUP BY {cols_sql}
        ORDER BY {cols_sql}
    """
    return pd.read_sql_query(query, conn, params=params)


def get_recent_attempts(
    conn: sqlite3.Connection,
    n: int = 100,
    campaign_id: str | None = None,
) -> pd.DataFrame:
    """Return the most recent *n* attempt rows as a DataFrame."""
    where = "WHERE campaign_id = ?" if campaign_id else ""
    params: tuple = (campaign_id,) if campaign_id else ()
    return pd.read_sql_query(
        f"""
        SELECT * FROM attempts
        {where}
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(*params, n),
    )
