"""Automated report generation script.

Extracts aggregate metrics from SQLite, renders matplotlib visualization charts,
populates the Jinja2 template, and optionally compiles to PDF.

Usage:
    python reports/generate_report.py --campaign-id baseline_01 [--pdf]
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader

# Ensure root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import redteam.db as db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPORTS_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"


def generate_heatmap_chart(df: pd.DataFrame, output_path: Path) -> bool:
    """Generate and save an attack success rate heatmap using matplotlib."""
    if df.empty or "strategy" not in df.columns or "target_model" not in df.columns:
        return False

    pivot = df.pivot_table(
        index="strategy",
        columns="target_model",
        values="success_rate_pct",
        aggfunc="first",
    ).fillna(0)

    if pivot.empty:
        return False

    fig, ax = plt.subplots(
        figsize=(max(5.0, len(pivot.columns) * 2.2), max(3.5, len(pivot.index) * 1.2))
    )
    cax = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    fig.colorbar(cax, ax=ax, label="Success Rate (%)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text_color = "white" if val > 65 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontweight="bold")

    ax.set_title("Vulnerability Success Rate by Strategy & Target (Guard OFF)", fontsize=11, pad=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def generate_efficiency_chart(df: pd.DataFrame, output_path: Path) -> bool:
    """Generate a histogram of attempt_number at first violation."""
    if df.empty or "attempt_number" not in df.columns:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    strategies = df["strategy"].unique()

    for s in strategies:
        sub = df[df["strategy"] == s]["attempt_number"]
        ax.hist(sub, bins=range(1, 12), alpha=0.6, label=str(s), edgecolor="white")

    ax.set_xlabel("Attempt Number of First Violation", fontsize=9)
    ax.set_ylabel("Number of Violations Found", fontsize=9)
    ax.set_title("Attacker Efficiency Distribution", fontsize=11, pad=12)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format dataframe as clean markdown table without requiring external tabulate."""
    if df.empty:
        return "*No empirical data logged for this breakdown.*"
    try:
        return df.to_markdown(index=False)
    except (ImportError, Exception):
        headers = [str(c) for c in df.columns]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_lines = []
        for _, row in df.iterrows():
            data_lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
        return "\n".join([header_line, sep_line] + data_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate red-teaming report.")
    parser.add_argument("--campaign-id", required=True, help="Campaign ID to report on.")
    parser.add_argument("--db", default="data/redteam.db", help="Path to database.")
    parser.add_argument("--config-dir", default="config", help="Config directory.")
    parser.add_argument("--pdf", action="store_true", help="Compile Markdown to PDF using pandoc if installed.")
    args = parser.parse_args()

    cfg_dir = Path(args.config_dir)
    strategies_info = yaml.safe_load((cfg_dir / "strategies.yaml").read_text(encoding="utf-8"))["strategies"]
    goals_info = yaml.safe_load((cfg_dir / "goals.yaml").read_text(encoding="utf-8"))["goals"]
    models_cfg = yaml.safe_load((cfg_dir / "models.yaml").read_text(encoding="utf-8"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = db.get_connection(args.db)
    summary_raw = db.get_campaign_summary(conn, args.campaign_id)

    # 1. Success rate guard OFF
    sr_off = pd.read_sql_query(
        """
        SELECT strategy, target_model,
               COUNT(*) AS total_attempts,
               SUM(CASE WHEN judge_violated = 1 AND guard_blocked = 0 THEN 1 ELSE 0 END) AS violations,
               ROUND(100.0 * SUM(CASE WHEN judge_violated = 1 AND guard_blocked = 0 THEN 1 ELSE 0 END)
                     / NULLIF(COUNT(*), 0), 1) AS success_rate_pct
        FROM attempts
        WHERE campaign_id = ? AND guard_active = 0
        GROUP BY strategy, target_model
        ORDER BY strategy, target_model
        """,
        conn,
        params=(args.campaign_id,),
    )

    # 2. Success rate guard ON
    sr_on = pd.read_sql_query(
        """
        SELECT strategy, target_model,
               COUNT(*) AS total_attempts,
               SUM(CASE WHEN judge_violated = 1 AND guard_blocked = 0 THEN 1 ELSE 0 END) AS violations,
               SUM(guard_blocked) AS blocked_by_guards,
               ROUND(100.0 * SUM(CASE WHEN judge_violated = 1 AND guard_blocked = 0 THEN 1 ELSE 0 END)
                     / NULLIF(COUNT(*), 0), 1) AS success_rate_pct
        FROM attempts
        WHERE campaign_id = ? AND guard_active = 1
        GROUP BY strategy, target_model
        ORDER BY strategy, target_model
        """,
        conn,
        params=(args.campaign_id,),
    )

    # 3. Model robustness ranking
    ranking_df = pd.read_sql_query(
        """
        SELECT target_model,
               COUNT(*) AS total_probes,
               SUM(CASE WHEN judge_violated = 1 AND guard_blocked = 0 THEN 1 ELSE 0 END) AS violations,
               ROUND(100.0 * SUM(CASE WHEN judge_violated = 1 AND guard_blocked = 0 THEN 1 ELSE 0 END)
                     / NULLIF(COUNT(*), 0), 1) AS vulnerability_rate_pct
        FROM attempts
        WHERE campaign_id = ? AND guard_active = 0
        GROUP BY target_model
        ORDER BY vulnerability_rate_pct DESC
        """,
        conn,
        params=(args.campaign_id,),
    )

    # 4. Attacker efficiency
    eff_df = pd.read_sql_query(
        """
        SELECT strategy, attempt_number
        FROM attempts
        WHERE campaign_id = ? AND judge_violated = 1 AND guard_blocked = 0
        """,
        conn,
        params=(args.campaign_id,),
    )

    eff_summary = (
        eff_df.groupby("strategy")["attempt_number"]
        .agg(mean_attempts="mean", median_attempts="median", violations_count="count")
        .reset_index()
        .round(2)
        if not eff_df.empty
        else pd.DataFrame()
    )

    # 5. Redacted findings
    examples_df = pd.read_sql_query(
        """
        SELECT target_model, strategy, goal, attempt_number, judge_severity, judge_reasoning
        FROM attempts
        WHERE campaign_id = ? AND judge_violated = 1 AND guard_blocked = 0
        ORDER BY judge_severity DESC
        LIMIT 6
        """,
        conn,
        params=(args.campaign_id,),
    )

    redacted_findings = [
        {
            "target_model": r["target_model"],
            "strategy": r["strategy"],
            "goal": r["goal"],
            "attempt_number": r["attempt_number"],
            "severity": r["judge_severity"],
            "technique_summary": r["judge_reasoning"] or "Policy violation elicited via structural framing.",
        }
        for _, r in examples_df.iterrows()
    ]

    # Render Visualizations
    heatmap_file = CHARTS_DIR / f"{args.campaign_id}_heatmap.png"
    has_heatmap = generate_heatmap_chart(sr_off, heatmap_file)

    efficiency_file = CHARTS_DIR / f"{args.campaign_id}_efficiency.png"
    has_efficiency = generate_efficiency_chart(eff_df, efficiency_file)

    # Summary calculations
    total_att = summary_raw.get("total_attempts") or 0
    total_viol = summary_raw.get("violations") or 0
    overall_sr = (100.0 * total_viol / total_att) if total_att > 0 else 0.0

    attacker_model = models_cfg.get("attacker", {}).get("model", "")
    judge_model = models_cfg.get("judge", {}).get("model", "")
    same_model_warn = (attacker_model == judge_model) and bool(attacker_model)

    context = {
        "campaign_id": args.campaign_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_targets": summary_raw.get("num_targets") or len(ranking_df),
        "max_attempts": 10,
        "summary": {
            **summary_raw,
            "overall_success_rate": overall_sr,
        },
        "strategies": strategies_info,
        "goals": goals_info,
        "results_table_guard_off": df_to_markdown(sr_off),
        "results_table_guard_on": df_to_markdown(sr_on),
        "model_ranking_table": df_to_markdown(ranking_df),
        "efficiency_table": df_to_markdown(eff_summary),
        "chart_heatmap_path": f"charts/{heatmap_file.name}" if has_heatmap else None,
        "chart_efficiency_path": f"charts/{efficiency_file.name}" if has_efficiency else None,
        "redacted_examples": redacted_findings,
        "same_model_warning": same_model_warn,
        "same_model_name": attacker_model,
        "input_guard_fpr": 0.04,  # baseline estimate; updated when train_guard runs
        "output_guard_fpr": 0.03,
        "num_input_positives": total_viol,
        "num_input_negatives": max(0, total_att - total_viol),
    }

    env = Environment(loader=FileSystemLoader(str(REPORTS_DIR)))
    tmpl = env.get_template("template.md.j2")
    rendered_md = tmpl.render(**context)

    out_md_path = OUTPUT_DIR / f"report_{args.campaign_id}.md"
    out_md_path.write_text(rendered_md, encoding="utf-8")
    logger.info("Markdown report successfully generated at %s", out_md_path)

    if args.pdf:
        pandoc_bin = shutil.which("pandoc")
        if pandoc_bin:
            out_pdf = out_md_path.with_suffix(".pdf")
            cmd = [pandoc_bin, str(out_md_path), "-o", str(out_pdf)]
            logger.info("Compiling PDF report with Pandoc: %s", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                logger.info("PDF report generated at %s", out_pdf)
            except Exception as e:
                logger.warning("Pandoc PDF generation failed: %s", e)
        else:
            logger.warning("Pandoc is not installed on PATH. Markdown report generated only.")


if __name__ == "__main__":
    main()
