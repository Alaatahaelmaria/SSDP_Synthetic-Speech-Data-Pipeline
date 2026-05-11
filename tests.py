"""Tests for critical pipeline logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


# ── Stage 1 tests ────────────────────────────────────────────

from stage1_prompt_generator import (
    PromptGenerator,
    _compute_word_count,
    _detect_code_switching,
    _detect_numbers,
    _normalize_arabic,
    _maybe_inject_filler,
    PROMPT_BANK,
)


def _minimal_cfg(n: int = 20) -> dict:
    return {
        "pipeline": {"seed": 0},
        "prompts": {
            "total_count": n,
            "domains": [{"name": d, "weight": 1 / len(PROMPT_BANK)} for d in PROMPT_BANK],
            "styles": ["casual", "formal", "code_switched", "fast_speech"],
            "length_distribution": {"short": 0.3, "medium": 0.5, "long": 0.2},
        },
    }


def test_word_count_basic():
    assert _compute_word_count("عايز أكل") == 2


def test_word_count_strips_diacritics():
    assert _compute_word_count("عَايِزٌ أَكَلَ") == 2


def test_detect_code_switching_true():
    assert _detect_code_switching("ابعتلي الـ location") is True


def test_detect_code_switching_false():
    assert _detect_code_switching("عايز أكل") is False


def test_detect_numbers_arabic():
    assert _detect_numbers("رقم ١٥") is True


def test_detect_numbers_western():
    assert _detect_numbers("رقم 15") is True


def test_detect_numbers_none():
    assert _detect_numbers("لا أرقام") is False


def test_normalize_arabic_removes_tatweel():
    assert "ـ" not in _normalize_arabic("جميـل")


def test_normalize_arabic_alef():
    assert "إ" not in _normalize_arabic("إحنا")
    assert "أ" not in _normalize_arabic("أنا")


def test_generate_count():
    gen = PromptGenerator(_minimal_cfg(30))
    records = gen.generate(30)
    assert len(records) <= 30


def test_generate_unique_ids():
    gen = PromptGenerator(_minimal_cfg(30))
    records = gen.generate(30)
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"


def test_generate_unique_texts():
    gen = PromptGenerator(_minimal_cfg(30))
    records = gen.generate(30)
    texts = [r["text"] for r in records]
    assert len(texts) == len(set(texts)), "Duplicate texts found"


def test_generate_metadata_fields():
    gen = PromptGenerator(_minimal_cfg(5))
    records = gen.generate(5)
    required = {"id", "text", "text_normalized", "domain", "style",
                "dialect", "word_count", "contains_code_switching",
                "contains_numbers", "length_bucket"}
    for r in records:
        assert required.issubset(r.keys()), f"Missing fields: {required - r.keys()}"


def test_dialect_always_egyptian():
    gen = PromptGenerator(_minimal_cfg(10))
    records = gen.generate(10)
    assert all(r["dialect"] == "egyptian_arabic" for r in records)


def test_code_switching_flag_accuracy():
    gen = PromptGenerator(_minimal_cfg(50))
    records = gen.generate(50)
    for r in records:
        # If contains latin chars → must be flagged
        import re
        has_latin = bool(re.search(r'[A-Za-z]', r["text"]))
        if has_latin:
            assert r["contains_code_switching"] is True, f"Not flagged: {r['text']}"


def test_filler_injection_doesnt_break_text():
    import random
    random.seed(0)
    for _ in range(20):
        result = _maybe_inject_filler("عايز أكل", probability=1.0)
        assert "عايز أكل" in result


# ── Stage 4 tests ────────────────────────────────────────────

from stage4_exporter import _split


def _make_records(n: int) -> list[dict]:
    return [{"id": str(i), "text": f"نص {i}"} for i in range(n)]


def test_split_proportions():
    records = _make_records(100)
    cfg = {"pipeline": {"seed": 42}, "export": {"split": {"train": 0.8, "validation": 0.1, "test": 0.1}}}
    splits = _split(records, cfg)
    assert len(splits["test"]) >= 1
    assert len(splits["validation"]) >= 1
    total = sum(len(v) for v in splits.values())
    assert total == 100


def test_split_no_overlap():
    records = _make_records(50)
    cfg = {"pipeline": {"seed": 0}, "export": {"split": {"train": 0.8, "validation": 0.1, "test": 0.1}}}
    splits = _split(records, cfg)
    all_ids = (
        [r["id"] for r in splits["train"]] +
        [r["id"] for r in splits["validation"]] +
        [r["id"] for r in splits["test"]]
    )
    assert len(all_ids) == len(set(all_ids)), "Overlap between splits"


def test_split_deterministic():
    records = _make_records(40)
    cfg = {"pipeline": {"seed": 99}, "export": {"split": {"train": 0.8, "validation": 0.1, "test": 0.1}}}
    s1 = _split(records, cfg)
    s2 = _split(records, cfg)
    assert [r["id"] for r in s1["train"]] == [r["id"] for r in s2["train"]]
