"""
SSDP · Main Pipeline Orchestrator
====================================
Usage:
  python pipeline.py --stage all          # run stages 1 → 2 → 4
  python pipeline.py --stage generate     # stage 1 only
  python pipeline.py --stage synthesize   # stage 2 only
  python pipeline.py --stage review       # stage 3 (launches web UI)
  python pipeline.py --stage export       # stage 4 only
  python pipeline.py --config my.yaml     # custom config
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# ── Logging setup ────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger("ssdp")


def load_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Stage runners ─────────────────────────────────────────────

def run_generate(cfg: dict, log: logging.Logger) -> None:
    from stage1_prompt_generator import run
    run(cfg, log)


def run_synthesize(cfg: dict, log: logging.Logger) -> None:
    from stage2_tts_synthesizer import run
    run(cfg, log)


def run_review(cfg: dict, log: logging.Logger) -> None:
    from stage3_review_server import run
    run(cfg, log)


def run_export(cfg: dict, log: logging.Logger) -> None:
    from stage4_exporter import run
    run(cfg, log)


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SSDP · Synthetic Speech Data Pipeline")
    parser.add_argument("--stage",  default="all",
                        choices=["all", "generate", "synthesize", "review", "export"],
                        help="Pipeline stage to run")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(Path(cfg["paths"]["logs_dir"]))

    log.info("=" * 60)
    log.info(f"  SSDP v{cfg['pipeline']['version']}  |  stage={args.stage}")
    log.info("=" * 60)

    stage = args.stage
    if stage in ("all", "generate"):
        run_generate(cfg, log)
    if stage in ("all", "synthesize"):
        run_synthesize(cfg, log)
    if stage == "review":
        run_review(cfg, log)
    if stage in ("all", "export"):
        log.info("[Main] Skipping export in 'all' mode — run --stage review first, then --stage export")
        if stage == "export":
            run_export(cfg, log)

    log.info("Pipeline complete ✓")


if __name__ == "__main__":
    main()
