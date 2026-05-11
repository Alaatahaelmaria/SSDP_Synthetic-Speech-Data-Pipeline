"""
SSDP · Stage 4 · Exporter
===========================
Exports accepted samples to training-ready formats:
  - HuggingFace datasets (metadata.jsonl + audio/)
  - LJSpeech  (wavs/ + metadata.csv)
  - CSV summary

Applies train/val/test split, audio resampling, and normalization.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def _copy_audio(src: Path, dst: Path, target_sr: int) -> bool:
    """Copy audio to export dir. WAV conversion attempted if pydub+ffmpeg available."""
    if not src.exists():
        return False
    if PYDUB_AVAILABLE and dst.suffix == ".wav":
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = AudioSegment.from_file(str(src))
                audio = audio.set_frame_rate(target_sr).set_channels(1)
                audio.export(str(dst), format="wav")
                return True
        except Exception:
            pass
    # Fallback: copy as-is (MP3 works fine for review and HF datasets)
    dst_actual = dst.with_suffix(src.suffix)
    shutil.copy2(str(src), str(dst_actual))
    return True


def _split(records: list, cfg: dict) -> dict[str, list]:
    split_cfg = cfg["export"]["split"]
    rng = random.Random(cfg.get("pipeline", {}).get("seed", 42))
    data = records[:]
    rng.shuffle(data)
    n = len(data)
    n_test  = max(1, int(n * split_cfg.get("test", 0.10)))
    n_val   = max(1, int(n * split_cfg.get("validation", 0.10)))
    return {
        "test":       data[:n_test],
        "validation": data[n_test: n_test + n_val],
        "train":      data[n_test + n_val:],
    }


def _export_huggingface(records: list, export_dir: Path, cfg: dict) -> None:
    splits = _split(records, cfg)
    target_sr = cfg["export"].get("sample_rate", 22050)
    fields    = cfg["export"].get("metadata_fields", [])

    for split_name, split_records in splits.items():
        split_dir = export_dir / "huggingface" / split_name
        audio_dir = split_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        with (split_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for rec in split_records:
                src = Path(rec.get("audio_path", ""))
                if not src.exists():
                    continue
                dst = audio_dir / src.name
                _copy_audio(src, dst, target_sr)
                dst_actual = dst.with_suffix(src.suffix)
                row = {k: rec.get(k) for k in fields if k in rec}
                row["audio_path"] = f"audio/{dst_actual.name}"
                row["split"]      = split_name
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(f"[Stage 4] HF {split_name}: {len(split_records)} samples → {split_dir}")


def _export_ljspeech(records: list, export_dir: Path, cfg: dict) -> None:
    target_sr = cfg["export"].get("sample_rate", 22050)
    lj_dir    = export_dir / "ljspeech"
    wavs_dir  = lj_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    with (lj_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for rec in records:
            src = Path(rec.get("audio_path", ""))
            if not src.exists():
                continue
            dst = wavs_dir / src.name
            _copy_audio(src, dst, target_sr)
            writer.writerow([rec["id"], rec["text"], rec.get("text_normalized", rec["text"])])

    logger.info(f"[Stage 4] LJSpeech: {len(records)} samples → {lj_dir}")


def _export_csv(records: list, export_dir: Path, cfg: dict) -> None:
    out = export_dir / "dataset.csv"
    fields = cfg["export"].get("metadata_fields", list(records[0].keys()) if records else [])
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"[Stage 4] CSV: {len(records)} rows → {out}")


def run(cfg: dict, logger_: logging.Logger | None = None) -> Path:
    log = logger_ or logger
    manifests_dir = Path(cfg["paths"]["manifests_dir"])
    export_dir    = Path(cfg["paths"]["export_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    results_file = manifests_dir / "synthesis_results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError("Run stages 1 & 2 first.")

    all_records = [
        json.loads(line)
        for line in results_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Merge review decisions from reviews.json
    reviews_path = manifests_dir / "reviews.json"
    if reviews_path.exists():
        reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
        for rec in all_records:
            if rec["id"] in reviews:
                rec.update(reviews[rec["id"]])

    include_rejected = cfg["export"].get("include_rejected", False)
    include_pending  = cfg["export"].get("include_pending", False)

    def keep(r: dict) -> bool:
        s = r.get("review_status", "pending")
        if s == "approved":  return True
        if s == "pending":   return include_pending
        if s == "rejected":  return include_rejected
        return False

    records = [r for r in all_records if keep(r)]
    log.info(f"[Stage 4] Exporting {len(records)} / {len(all_records)} records …")

    fmt = cfg["export"].get("format", "huggingface")
    if fmt in ("huggingface", "all"):
        _export_huggingface(records, export_dir, cfg)
    if fmt in ("ljspeech", "all"):
        _export_ljspeech(records, export_dir, cfg)
    if fmt in ("csv", "all"):
        _export_csv(records, export_dir, cfg)

    manifest = {
        "stage": "export", "format": fmt,
        "total_exported": len(records),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_file = manifests_dir / "export_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"[Stage 4] Done → {export_dir}")
    return export_dir
