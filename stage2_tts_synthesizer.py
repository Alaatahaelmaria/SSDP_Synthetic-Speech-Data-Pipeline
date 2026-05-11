"""
SSDP · Stage 2 · TTS Synthesizer
===================================
Reads prompts from stage 1 output and synthesizes audio using
Microsoft Edge TTS (edge-tts) which provides:
  - ar-EG-SalmaNeural  (Egyptian Arabic female)
  - ar-EG-ShakirNeural (Egyptian Arabic male)

Features:
  - Async batch processing (configurable concurrency)
  - Per-sample retry with exponential back-off
  - Checkpoint / resume: skips already-synthesized samples
  - Quality gate: duration checks + SNR estimation
  - Manifest updated incrementally (observable progress)

Output:
  data/audio/<id>.wav
  data/manifests/synthesis_manifest.json
  data/manifests/synthesis_checkpoint.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Optional imports  (graceful degradation)
# ─────────────────────────────────────────────────────────────

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge-tts not installed. Run: pip install edge-tts")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")


# ─────────────────────────────────────────────────────────────
#  Audio utilities
# ─────────────────────────────────────────────────────────────

def _get_mp3_duration(mp3_path: Path) -> float:
    """
    Estimate MP3 duration in seconds.
    Uses pydub if available; otherwise estimates from file size
    at a typical 24kbps bitrate (edge-tts default).
    """
    if PYDUB_AVAILABLE:
        try:
            audio = AudioSegment.from_mp3(str(mp3_path))
            return len(audio) / 1000.0
        except Exception:
            pass
    # Fallback: rough estimate from file size (24 kbps = 3000 bytes/sec)
    try:
        size = mp3_path.stat().st_size
        return round(size / 3000.0, 3)
    except Exception:
        return 0.0


def _estimate_snr(wav_path: Path) -> float | None:
    """
    Rough SNR estimation: compare RMS of signal vs. estimated noise floor
    (first 100ms assumed to be silence/lead-in for edge-tts outputs).
    Returns None if numpy/soundfile not available.
    """
    if not (NUMPY_AVAILABLE and SOUNDFILE_AVAILABLE):
        return None
    try:
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        noise_samples = int(0.1 * sr)
        if len(data) <= noise_samples:
            return None
        noise_rms  = np.sqrt(np.mean(data[:noise_samples] ** 2)) + 1e-10
        signal_rms = np.sqrt(np.mean(data[noise_samples:] ** 2)) + 1e-10
        snr_db = 20 * math.log10(signal_rms / noise_rms)
        return round(snr_db, 2)
    except Exception:
        return None


def _mp3_to_wav(mp3_path: Path, wav_path: Path, target_sr: int = 22050) -> bool:
    """Convert MP3 to WAV using pydub (requires ffmpeg)."""
    if not PYDUB_AVAILABLE:
        return False
    try:
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        audio.export(str(wav_path), format="wav")
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ─────────────────────────────────────────────────────────────

def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if checkpoint_path.exists():
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return {"synthesized": {}, "failed": {}}


def _save_checkpoint(checkpoint_path: Path, state: dict) -> None:
    checkpoint_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────────
#  Core synthesizer
# ─────────────────────────────────────────────────────────────

class TTSSynthesizer:
    def __init__(self, cfg: dict):
        self.cfg          = cfg
        tts_cfg           = cfg["tts"]
        self.engine       = tts_cfg.get("engine", "edge-tts")
        self.voice_pool   = tts_cfg.get("voice_pool", ["ar-EG-SalmaNeural"])
        self.rate         = tts_cfg.get("rate", "+0%")
        self.volume       = tts_cfg.get("volume", "+0%")
        self.pitch        = tts_cfg.get("pitch", "+0Hz")
        self.batch_size   = tts_cfg.get("batch_size", 10)
        self.max_retries  = tts_cfg.get("max_retries", 3)
        self.retry_delay  = tts_cfg.get("retry_delay_sec", 2)
        self.timeout      = tts_cfg.get("timeout_sec", 30)
        self.chk_every    = tts_cfg.get("checkpoint_every", 25)

        quality_cfg        = cfg.get("quality", {})
        self.min_dur       = quality_cfg.get("min_duration_sec", 0.5)
        self.max_dur       = quality_cfg.get("max_duration_sec", 15.0)
        self.snr_threshold = quality_cfg.get("snr_threshold_db", 20.0)
        self.check_snr     = quality_cfg.get("enable_snr_check", True)
        self.target_sr     = cfg["export"].get("sample_rate", 22050)

        self.audio_dir     = Path(cfg["paths"]["audio_dir"])
        self.manifests_dir = Path(cfg["paths"]["manifests_dir"])
        self.cache_dir     = Path(cfg["paths"].get("cache_dir", ".cache"))

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.manifests_dir / "synthesis_checkpoint.json"
        self.state           = _load_checkpoint(self.checkpoint_path)

    def _pick_voice(self, idx: int) -> str:
        """Round-robin across voice pool for speaker diversity."""
        return self.voice_pool[idx % len(self.voice_pool)]

    async def _synthesize_one(self, record: dict, voice: str) -> dict:
        """
        Attempt synthesis for a single record.
        Returns an enriched record with audio_path, duration, quality info.
        """
        sample_id  = record["id"]
        text       = record["text"]
        mp3_tmp    = self.cache_dir / f"{sample_id}.mp3"
        wav_path   = self.audio_dir / f"{sample_id}.wav"

        for attempt in range(1, self.max_retries + 1):
            try:
                if not EDGE_TTS_AVAILABLE:
                    raise RuntimeError("edge-tts not installed")

                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice,
                    rate=self.rate,
                    volume=self.volume,
                    pitch=self.pitch,
                )
                await asyncio.wait_for(
                    communicate.save(str(mp3_tmp)),
                    timeout=self.timeout
                )

                # Try WAV conversion; if no ffmpeg, keep as MP3
                final_path = self.audio_dir / f"{sample_id}.mp3"
                wav_path   = self.audio_dir / f"{sample_id}.wav"

                if _mp3_to_wav(mp3_tmp, wav_path, target_sr=self.target_sr):
                    final_path = wav_path
                else:
                    import shutil
                    shutil.move(str(mp3_tmp), str(final_path))

                # Clean up temp mp3 if still exists
                if mp3_tmp.exists():
                    mp3_tmp.unlink()

                # Quality checks
                duration = (_get_wav_duration(final_path)
                            if final_path.suffix == ".wav"
                            else _get_mp3_duration(final_path))
                snr      = (_estimate_snr(final_path)
                            if (self.check_snr and final_path.suffix == ".wav")
                            else None)

                quality_flags: list[str] = []
                if duration < self.min_dur:
                    quality_flags.append("too_short")
                if duration > self.max_dur:
                    quality_flags.append("too_long")
                if snr is not None and snr < self.snr_threshold:
                    quality_flags.append("low_snr")

                quality_score = self._compute_quality_score(duration, snr, quality_flags)

                enriched = {
                    **record,
                    "audio_path":       str(final_path),
                    "voice":            voice,
                    "duration_sec":     round(duration, 3),
                    "snr_db":           snr,
                    "quality_flags":    quality_flags,
                    "quality_score":    quality_score,
                    "review_status":    "pending",
                    "synthesis_engine": self.engine,
                    "synthesized_at":   datetime.now(timezone.utc).isoformat(),
                    "attempts":         attempt,
                }
                return enriched

            except asyncio.TimeoutError:
                logger.warning(f"[{sample_id}] Timeout on attempt {attempt}")
            except Exception as e:
                logger.warning(f"[{sample_id}] Attempt {attempt} failed: {e}")

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * attempt)

        # All retries exhausted
        return {**record, "audio_path": None, "review_status": "failed",
                "quality_score": 0.0, "quality_flags": ["synthesis_failed"],
                "attempts": self.max_retries}

    def _compute_quality_score(
        self,
        duration: float,
        snr: float | None,
        flags: list[str]
    ) -> float:
        """
        Heuristic quality score [0.0 – 1.0].
        Penalizes duration outliers and poor SNR.
        """
        score = 1.0
        if duration < self.min_dur or duration > self.max_dur:
            score -= 0.4
        if snr is not None:
            if snr < self.snr_threshold:
                score -= 0.3
            elif snr > 35:
                score += 0.05
        score -= 0.2 * len(flags)
        return round(max(0.0, min(1.0, score)), 3)

    async def _process_batch(
        self,
        batch: list[tuple[int, dict]],
        results: list[dict]
    ) -> None:
        tasks = []
        for global_idx, record in batch:
            voice = self._pick_voice(global_idx)
            tasks.append(self._synthesize_one(record, voice))
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    async def synthesize_all(self, records: list[dict]) -> list[dict]:
        results: list[dict] = []
        pending = [
            (i, r) for i, r in enumerate(records)
            if r["id"] not in self.state["synthesized"]
               and r["id"] not in self.state.get("failed", {})
        ]

        skipped = len(records) - len(pending)
        if skipped:
            logger.info(f"[Stage 2] Resuming: {skipped} already synthesized, {len(pending)} remaining.")

        # Restore already-done results
        for rec in records:
            if rec["id"] in self.state["synthesized"]:
                results.append(self.state["synthesized"][rec["id"]])

        # Process in batches
        for batch_start in range(0, len(pending), self.batch_size):
            batch = pending[batch_start: batch_start + self.batch_size]
            batch_results: list[dict] = []
            await self._process_batch(batch, batch_results)

            for res in batch_results:
                results.append(res)
                sid = res["id"]
                if res.get("audio_path"):
                    self.state["synthesized"][sid] = res
                    logger.info(
                        f"[Stage 2] ✓ {sid} | {res.get('duration_sec',0):.2f}s | "
                        f"score={res.get('quality_score',0):.2f} | voice={res.get('voice','')}"
                    )
                else:
                    self.state.setdefault("failed", {})[sid] = res
                    logger.warning(f"[Stage 2] ✗ {sid} FAILED")

            # Checkpoint
            if (batch_start // self.batch_size + 1) % max(1, self.chk_every // self.batch_size) == 0:
                _save_checkpoint(self.checkpoint_path, self.state)
                logger.info(f"[Stage 2] Checkpoint saved ({len(self.state['synthesized'])} done)")

        # Final checkpoint
        _save_checkpoint(self.checkpoint_path, self.state)
        return results


# ─────────────────────────────────────────────────────────────
#  Entry-point helper
# ─────────────────────────────────────────────────────────────

def run(cfg: dict, logger_: logging.Logger | None = None) -> Path:
    log = logger_ or logger

    prompts_file  = Path(cfg["paths"]["prompts_dir"]) / "prompts.jsonl"
    manifests_dir = Path(cfg["paths"]["manifests_dir"])
    manifests_dir.mkdir(parents=True, exist_ok=True)

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}. Run stage 1 first.")

    records = [json.loads(line) for line in prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    log.info(f"[Stage 2] Loaded {len(records)} prompts. Starting synthesis …")

    synthesizer = TTSSynthesizer(cfg)
    results     = asyncio.run(synthesizer.synthesize_all(records))

    # Write synthesis JSONL
    out_file = manifests_dir / "synthesis_results.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write manifest
    done    = [r for r in results if r.get("audio_path")]
    failed  = [r for r in results if not r.get("audio_path")]
    manifest = {
        "stage":        "synthesis",
        "total":        len(results),
        "succeeded":    len(done),
        "failed":       len(failed),
        "avg_duration": round(sum(r.get("duration_sec", 0) for r in done) / max(len(done), 1), 3),
        "voices_used":  list({r.get("voice") for r in done if r.get("voice")}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_file":  str(out_file),
    }
    manifest_file = manifests_dir / "synthesis_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(f"[Stage 2] Done. {len(done)} succeeded, {len(failed)} failed.")
    log.info(f"[Stage 2] Manifest → {manifest_file}")
    return out_file
