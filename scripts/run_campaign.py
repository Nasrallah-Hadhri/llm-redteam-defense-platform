"""CLI entrypoint to launch an automated red-teaming campaign.

Usage:
    python scripts/run_campaign.py --campaign-id baseline_01
    python scripts/run_campaign.py --campaign-id exp_01 --targets llama3:8b,qwen2.5:7b --max-attempts 5
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Ensure root package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

import redteam.db as db
from redteam.campaign import Campaign
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
    parser = argparse.ArgumentParser(description="Run an automated LLM red-teaming campaign.")
    parser.add_argument("--campaign-id", required=True, help="Unique identifier for this campaign run.")
    parser.add_argument(
        "--targets",
        default="all",
        help="Comma-separated model names (e.g. llama3:8b,mistral:7b) or 'all'.",
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
    parser.add_argument("--max-attempts", type=int, default=10, help="Max PAIR attempts per attack chain.")
    parser.add_argument("--db", default="data/redteam.db", help="Path to SQLite database.")
    parser.add_argument("--config-dir", default="config", help="Path to configuration directory.")
    args = parser.parse_args()

    cfg_dir = Path(args.config_dir)
    models_cfg = load_yaml(cfg_dir / "models.yaml")
    strategies_data = load_yaml(cfg_dir / "strategies.yaml")["strategies"]
    goals_data = load_yaml(cfg_dir / "goals.yaml")["goals"]

    # Filter targets
    target_cfgs = models_cfg.get("targets", [])
    if args.targets != "all":
        requested_targets = {t.strip() for t in args.targets.split(",")}
        target_cfgs = [t for t in target_cfgs if t["model"] in requested_targets]
        if not target_cfgs:
            logger.error("No matching targets found for: %s", args.targets)
            sys.exit(1)

    targets = [
        OllamaAdapter(
            model_name=t["model"],
            base_url=t.get("base_url", "http://localhost:11434"),
            timeout=t.get("timeout", 120),
        )
        for t in target_cfgs
    ]

    # Filter strategies
    if args.strategies != "all":
        requested_strats = {s.strip() for s in args.strategies.split(",")}
        strategies_data = [s for s in strategies_data if s["id"] in requested_strats]
        if not strategies_data:
            logger.error("No matching strategies found for: %s", args.strategies)
            sys.exit(1)

    # Filter goals
    if args.goals != "all":
        requested_goals = {g.strip() for g in args.goals.split(",")}
        goals_data = [g for g in goals_data if g["id"] in requested_goals]
        if not goals_data:
            logger.error("No matching goals found for: %s", args.goals)
            sys.exit(1)

    # Client instantiation
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
        attacker_model_name=attacker_cfg.get("model"),
        judge_model_name=judge_cfg.get("model"),
    )

    logger.info("Starting campaign '%s' with %d targets, %d strategies, %d goals.",
                args.campaign_id, len(targets), len(strategies_data), len(goals_data))
    campaign.run()
    logger.info("Campaign completed successfully. Results saved to %s", args.db)


if __name__ == "__main__":
    main()
