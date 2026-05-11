"""
SSDP - Main Pipeline Runner
==============================
Orchestrates all four stages of the Synthetic Speech Data Pipeline.

Usage:
    python run_pipeline.py [--stage {1,2,3,4,all}] [--config config.yaml]

Examples:
    python run_pipeline.py                        # run all stages
    python run_pipeline.py --stage 1              # generate prompts only
    python run_pipeline.py --stage 2              # synthesize audio (resumable)
    python run_pipeline.py --stage 3              # launch review UI
    python run_pipeline.py --stage 4              # export approved dataset
    python run_pipeline.py --stage 1 --stage 2   # chain stages
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────

def _ensure_utf8_stdout():
    """Force UTF-8 on Windows consoles that default to cp1252."""
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    elif sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler — full debug
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    ))

    # Console handler — info+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                      datefmt="%H:%M:%S"))

    root.addHandler(fh)
    root.addHandler(ch)

    return logging.getLogger("ssdp")




# ─────────────────────────────────────────────────────────────
#  Config loader
# ─────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Make all paths relative to config file location
    base = path.parent
    for key in cfg.get("paths", {}):
        cfg["paths"][key] = str(base / cfg["paths"][key])
    return cfg


# ─────────────────────────────────────────────────────────────
#  Stage runners
# ─────────────────────────────────────────────────────────────

def run_stage1(cfg: dict, log: logging.Logger):
    log.info("=" * 60)
    log.info("  STAGE 1 - Prompt Generation")
    log.info("=" * 60)
    import stage1_prompt_generator as s1
    return s1.run(cfg, log)


def run_stage2(cfg: dict, log: logging.Logger):
    log.info("=" * 60)
    log.info("  STAGE 2 - TTS Synthesis  (async, resumable)")
    log.info("=" * 60)
    import stage2_tts_synthesizer as s2
    return s2.run(cfg, log)


def run_stage3(cfg: dict, log: logging.Logger):
    log.info("=" * 60)
    log.info("  STAGE 3 - Review UI")
    log.info("=" * 60)
    import stage3_review_server as s3
    return s3.run(cfg, log)


def run_stage4(cfg: dict, log: logging.Logger):
    log.info("=" * 60)
    log.info("  STAGE 4 - Dataset Export")
    log.info("=" * 60)
    import stage4_exporter as s4
    return s4.run(cfg, log)


STAGE_MAP = {
    "1": run_stage1,
    "2": run_stage2,
    "3": run_stage3,
    "4": run_stage4,
}


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="SSDP - Synthetic Speech Data Pipeline (Egyptian Arabic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage", "-s",
        action="append",
        choices=["1", "2", "3", "4", "all"],
        default=None,
        help="Which stage(s) to run. Defaults to 'all'.",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()

    # Default: all stages
    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["1", "2", "3", "4"]
    stages = sorted(set(stages))  # deduplicate, keep order

    # Load config
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent / cfg_path

    cfg = load_config(cfg_path)

    # Setup logging
    log = setup_logging(Path(cfg["paths"]["logs_dir"]))
    log.info(f"SSDP Pipeline  v{cfg['pipeline'].get('version','?')}  starting …")
    log.info(f"Config: {cfg_path}")
    log.info(f"Stages to run: {stages}")

    for stage in stages:
        runner = STAGE_MAP.get(stage)
        if runner:
            try:
                runner(cfg, log)
            except KeyboardInterrupt:
                log.info(f"Stage {stage} interrupted by user.")
                if stage == "3":
                    # Review server was stopped intentionally; continue to export if requested
                    if "4" in stages:
                        log.info("Proceeding to Stage 4 …")
                    continue
                else:
                    sys.exit(0)
            except Exception as exc:
                log.exception(f"Stage {stage} failed: {exc}")
                sys.exit(1)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
