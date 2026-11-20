"""CLI entrypoint to re-run campaigns with trained defense guards active.

Usage:
    python scripts/run_defense_eval.py --campaign-id eval_guard_01 --source-campaign baseline_01
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

import redteam.db as db
from redteam.campaign import Campaign
from redteam.defense.input_guard import InputGuard
from redteam.defense.output_guard import OutputGuard
from redteam.judge import Judge
from redteam.llm_client import OllamaClient
from redteam.targets.ollama_adapter import OllamaAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run red-teaming campaign with defense guards active.")
    parser.add_argument("--campaign-id", required=True, help="New campaign ID for the evaluation pass.")
    parser.add_argument(
        "--source-campaign",
        required=False,
        default=None,
        help="Optional source campaign ID for reference logging.",
    )
    parser.add_argument(
        "--targets",
        default="all",
        help="Comma-separated target models or 'all'.",
    )
    parser.add_argument(
        "--strategies",
        default="all",
        help="Comma-separated strategy IDs or 'all'.",
    )
    parser.add_argument(
        "--goals",
        default="all",
        help="Comma-separated goal IDs or 'all'.",
    )
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--db", default="data/redteam.db")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--input-guard", default="data/input_guard.pkl")
    parser.add_argument("--output-guard", default="data/output_guard.pkl")
    args = parser.parse_args()

    cfg_dir = Path(args.config_dir)
    models_cfg = load_yaml(cfg_dir / "models.yaml")
    strategies_data = load_yaml(cfg_dir / "strategies.yaml")["strategies"]
    goals_data = load_yaml(cfg_dir / "goals.yaml")["goals"]

    # Filter targets
    target_cfgs = models_cfg.get("targets", [])
    if args.targets != "all":
        req = {t.strip() for t in args.targets.split(",")}
        target_cfgs = [t for t in target_cfgs if t["model"] in req]

    targets = [
        OllamaAdapter(
            model_name=t["model"],
            base_url=t.get("base_url", "http://localhost:11434"),
            timeout=t.get("timeout", 120),
        )
        for t in target_cfgs
    ]

    # Filter strategies & goals
    if args.strategies != "all":
        req_s = {s.strip() for s in args.strategies.split(",")}
        strategies_data = [s for s in strategies_data if s["id"] in req_s]
    if args.goals != "all":
        req_g = {g.strip() for g in args.goals.split(",")}
        goals_data = [g for g in goals_data if g["id"] in req_g]

    # Load Guards
    input_guard = None
    output_guard = None

    input_guard_path = Path(args.input_guard)
    output_guard_path = Path(args.output_guard)

    if input_guard_path.exists():
        logger.info("Loading input guard from %s...", input_guard_path)
        input_guard = InputGuard.load(input_guard_path)
    else:
        logger.warning("Input guard model file not found at %s. Running without input guard.", input_guard_path)

    if output_guard_path.exists():
        logger.info("Loading output guard from %s...", output_guard_path)
        output_guard = OutputGuard.load(output_guard_path)
    else:
        logger.warning("Output guard model file not found at %s. Running without output guard.", output_guard_path)

    if not input_guard and not output_guard:
        logger.error("Neither input nor output guard was loaded. Train guards first via train_guard.py.")
        sys.exit(1)

    attacker_cfg = models_cfg["attacker"]
    judge_cfg = models_cfg["judge"]

    attacker_client = OllamaClient(
        model=attacker_cfg["model"],
        base_url=attacker_cfg.get("base_url", "http://localhost:11434"),
        timeout=attacker_cfg.get("timeout", 120),
    )
    judge_client = OllamaClient(
        model=judge_cfg["model"],
        base_url=judge_cfg.get("base_url", "http://localhost:11434"),
        timeout=judge_cfg.get("timeout", 120),
    )
    judge = Judge(judge_client)

    conn = db.get_connection(args.db)
    db.init_db(conn)

    warnings.filterwarnings("always")
    campaign = Campaign(
        campaign_id=args.campaign_id,
        attacker_client=attacker_client,
        judge=judge,
        targets=targets,
        strategies=strategies_data,
        goals=goals_data,
        conn=conn,
        max_attempts=args.max_attempts,
        input_guard=input_guard,
        output_guard=output_guard,
        attacker_model_name=attacker_cfg.get("model"),
        judge_model_name=judge_cfg.get("model"),
    )

    logger.info("Starting defense evaluation campaign '%s' with guards active...", args.campaign_id)
    campaign.run()
    logger.info("Defense evaluation completed. Results saved in %s", args.db)


if __name__ == "__main__":
    main()
