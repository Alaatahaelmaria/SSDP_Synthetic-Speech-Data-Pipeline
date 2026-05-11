"""
SSDP · Stage 1 · Prompt Generator
===================================
Generates diverse Egyptian Arabic text prompts covering:
  - Multiple domains (daily life, tech, food, transport …)
  - Multiple styles (casual, formal, code-switched, fast-speech)
  - Dialect-aware challenges (spelling variants, code-switching,
    contractions, filler words, numbers, brand names …)

Output: data/prompts/prompts.jsonl  +  data/manifests/prompts_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Raw prompt bank  – keyed by (domain, style)
#  Each tuple: (text, metadata_flags)
# ─────────────────────────────────────────────────────────────

PROMPT_BANK: dict[str, list[tuple[str, dict[str, Any]]]] = {

    # ── Daily Conversation ────────────────────────────────────
    "daily_conversation": [
        # casual
        ("إزيك؟ عامل إيه؟", {"style": "casual", "length": "short"}),
        ("والله تعبان شوية النهارده، مش عارف إيه اللي جري.", {"style": "casual", "length": "medium"}),
        ("اللي فات مات، خلينا نتكلم في الحاضر.", {"style": "casual", "length": "medium"}),
        ("انت فين؟ أنا استنيتك من زمان!", {"style": "casual", "length": "medium"}),
        ("معنديش وقت دلوقتي، كلمني بعدين.", {"style": "fast_speech", "length": "medium"}),
        ("قلتلك مش هيجي، وهو فعلاً ماجاش.", {"style": "fast_speech", "length": "medium"}),
        ("يعني مش عارف أعمل إيه، الدنيا بايظة.", {"style": "casual", "length": "medium"}),
        ("طب ماشي، نشوف بكره هيبقى إيه.", {"style": "casual", "length": "medium"}),
        ("بص، أنا حاسس إن في حاجة غلط.", {"style": "casual", "length": "medium"}),
        ("ايوه ايوه، فاهم اللي بتقوله.", {"style": "casual", "length": "short"}),
        ("مش لازم تزعل، الموضوع بسيط.", {"style": "casual", "length": "short"}),
        ("إنت عارف إنك غلطان صح؟", {"style": "casual", "length": "short"}),
        ("عشان كده قلتلك من الأول.", {"style": "fast_speech", "length": "short"}),
        ("أنا قلتلك وقلتلك وانت مسمعتش.", {"style": "fast_speech", "length": "medium"}),
        ("خلاص بقى، كفاية كلام فاضي.", {"style": "casual", "length": "short"}),
        ("الجو جميل النهارده، نطلع نتمشى؟", {"style": "casual", "length": "medium"}),
        ("ربنا يسهّل، إن شاء الله بكره يبقى أحسن.", {"style": "formal", "length": "medium"}),
        ("أنا آسف على اللي حصل، مكنتش قاصدك.", {"style": "formal", "length": "medium"}),
        ("شكراً جدا على مساعدتك.", {"style": "formal", "length": "short"}),
        ("من فضلك ممكن تساعدني في حاجة؟", {"style": "formal", "length": "short"}),

        # Fast speech / contractions
        ("انا قولتلك الكلام ده من زمان بس انت مسمعتش.", {"style": "fast_speech", "length": "long",
                                                       "contains_contractions": True}),
        ("معنديش فلوس دلوقتي، خلسها لبكره.", {"style": "fast_speech", "length": "medium",
                                                 "contains_contractions": True}),
    ],

    # ── Technology ────────────────────────────────────────────
    "technology": [
        ("ابعتلي الـ location على واتساب.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ WiFi فصل، اعملي restart للراوتر.", {"style": "code_switched", "contains_code_switching": True}),
        ("اعمل update للتطبيق من الـ store.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ battery خلصت، محتاج charger بسرعة.", {"style": "code_switched", "contains_code_switching": True}),
        ("كنسل الـ subscription قبل ما يتجدد.", {"style": "code_switched", "contains_code_switching": True}),
        ("شير معايا الـ screen عشان أشوف المشكلة.", {"style": "code_switched", "contains_code_switching": True}),
        ("نزّل الـ app ده، هيفيدك جداً.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ internet بطيء جداً النهارده.", {"style": "code_switched", "contains_code_switching": True}),
        ("عندك password الـ hotspot؟", {"style": "code_switched", "contains_code_switching": True}),
        ("حمل الـ file على الـ cloud عشان ميتضيعش.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ laptop بتاعي علق، محتاج أعمله restart.", {"style": "code_switched", "contains_code_switching": True}),
        ("بعتلك الـ link على الإيميل.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ call بتقطع، نتكلم على zoom.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ OTP جالك على التليفون.", {"style": "code_switched", "contains_code_switching": True,
                                         "contains_abbreviations": True}),
        ("مش شغال الـ GPS عندي دلوقتي.", {"style": "code_switched", "contains_code_switching": True}),
        ("حمّل الـ PDF وابعته على الجروب.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ RAM بتاع الجهاز مش كفاية للـ game ده.", {"style": "code_switched", "contains_code_switching": True}),
        ("عملت backup للتليفون قبل ما أغيره.", {"style": "code_switched", "contains_code_switching": True}),
        ("اشتريت subscription في Netflix بشهر.", {"style": "code_switched", "contains_code_switching": True}),
        ("إيه رأيك في الـ iPhone الجديد؟", {"style": "code_switched", "contains_code_switching": True}),
    ],

    # ── Food & Restaurants ────────────────────────────────────
    "food_and_restaurants": [
        ("عايز طبق كشري بدون خل.", {"style": "casual", "length": "short"}),
        ("الأكل في المطعم ده جامد جداً، ننزل عليه تاني.", {"style": "casual", "length": "medium"}),
        ("اطلب بيتزا، أنا جعان.", {"style": "casual", "length": "short"}),
        ("فيه إيه اليوم للأكل؟", {"style": "casual", "length": "short"}),
        ("عايزة شاي بلبن وعيش تست.", {"style": "casual", "length": "short"}),
        ("الطعمية هنا أحسن من أي حتة تانية.", {"style": "casual", "length": "medium"}),
        ("طلبنا delivery من Uber Eats.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ menu كله باللغة العربية.", {"style": "code_switched", "contains_code_switching": True}),
        ("هات اتنين كوباية قهوة سادة.", {"style": "casual", "length": "short"}),
        ("الأكل البيتي أحسن من أكل المطاعم دايماً.", {"style": "casual", "length": "medium"}),
        ("حجزت طاولة الساعة تمانية بالليل.", {"style": "formal", "length": "medium"}),
        ("محتاج أعرف مكونات الأكل ده، عندي حساسية.", {"style": "formal", "length": "medium"}),
        ("الحساب كام يا فندم؟", {"style": "formal", "length": "short"}),
        ("ممكن الفاتورة لو سمحت؟", {"style": "formal", "length": "short"}),
        ("اتأخرت أوي الأكل، الزبائن زهقوا.", {"style": "casual", "length": "medium"}),
    ],

    # ── Transportation ────────────────────────────────────────
    "transportation": [
        ("فين أقرب محطة مترو من هنا؟", {"style": "casual", "length": "short"}),
        ("اطلب أوبر، أنا مش عارف أوصل.", {"style": "code_switched", "contains_code_switching": True}),
        ("الزحمة في الشيراتون وقفت.", {"style": "casual", "length": "short"}),
        ("المواصلات بقت غالية جداً.", {"style": "casual", "length": "short"}),
        ("امشي على الأوتوستراد لحد الدائري.", {"style": "casual", "length": "medium"}),
        ("الميكروباص مش وصّال لحتة قريبة.", {"style": "casual", "length": "medium"}),
        ("هوصل محطة مصر الساعة كام؟", {"style": "casual", "length": "short"}),
        ("فيه موقف عربيات هنا؟", {"style": "casual", "length": "short"}),
        ("الـ GPS قالي خد الطريق التاني.", {"style": "code_switched", "contains_code_switching": True}),
        ("التذاكر في المطار غالية النهارده.", {"style": "formal", "length": "medium"}),
        ("الأتوبيس رقم ١٥ بيوصل المنصورة؟", {"style": "casual", "length": "medium",
                                               "contains_numbers": True}),
        ("هنوصل في ساعتين لو ما كانش فيه زحمة.", {"style": "casual", "length": "medium"}),
        ("خد التاكسي، أسرع من الأوبر.", {"style": "casual", "length": "short"}),
    ],

    # ── Shopping ──────────────────────────────────────────────
    "shopping": [
        ("بايعهولك بكام الكيلو ده؟", {"style": "casual", "length": "short"}),
        ("فيه خصم على المنتجات دي؟", {"style": "casual", "length": "short"}),
        ("اشتريت جاكت جديد من Zara.", {"style": "code_switched", "contains_code_switching": True}),
        ("المول مفتوح لحد الساعة كام؟", {"style": "casual", "length": "short"}),
        ("عايزة سايز أكبر من ده.", {"style": "casual", "length": "short"}),
        ("ممكن أجرب ده قبل ما أشتريه؟", {"style": "casual", "length": "medium"}),
        ("السعر غالي أوي، ممكن تخفض؟", {"style": "casual", "length": "medium"}),
        ("في offer على المنتجات الجديدة؟", {"style": "code_switched", "contains_code_switching": True}),
        ("هطلب أونلاين وهيتبعتلي لحد البيت.", {"style": "code_switched", "contains_code_switching": True}),
        ("الكوبون ما اشتغلش، عندك غيره؟", {"style": "code_switched", "contains_code_switching": True}),
        ("الإيصال بالعربي ولا الإنجليزي؟", {"style": "formal", "length": "short"}),
        ("هرجعه لو مش عاجبني.", {"style": "casual", "length": "short"}),
    ],

    # ── Health ────────────────────────────────────────────────
    "health": [
        ("عايز موعد عند الدكتور بكره.", {"style": "formal", "length": "short"}),
        ("بقالي يومين مش تمام، عندي سخونية.", {"style": "casual", "length": "medium"}),
        ("الدكتور قالي اشرب دوا كل ست ساعات.", {"style": "casual", "length": "medium"}),
        ("العيادة قريبة من البيت شوية.", {"style": "casual", "length": "short"}),
        ("الـ pharmacy مفتوح دلوقتي؟", {"style": "code_switched", "contains_code_switching": True}),
        ("محتاج تحليل دم وأشعة.", {"style": "formal", "length": "short"}),
        ("ضغطي ارتفع، محتاج أاخد الدوا.", {"style": "casual", "length": "medium"}),
        ("الدكتور قفل عيادته، هروح المستشفى.", {"style": "casual", "length": "medium"}),
        ("الأكل الصحي بيفرق أوي في الصحة.", {"style": "formal", "length": "medium"}),
        ("لازم تاخد الـ vaccine قبل السفر.", {"style": "code_switched", "contains_code_switching": True}),
    ],

    # ── Education ─────────────────────────────────────────────
    "education": [
        ("الامتحان بكره، لازم أذاكر النهارده.", {"style": "casual", "length": "medium"}),
        ("الأستاذ شرح الدرس كويس جداً.", {"style": "casual", "length": "medium"}),
        ("المدرسة بدأت الساعة سبعة الصبح.", {"style": "casual", "length": "medium"}),
        ("هجيب course أونلاين في البرمجة.", {"style": "code_switched", "contains_code_switching": True}),
        ("المكتبة فيها كتب كتير في الموضوع ده.", {"style": "formal", "length": "medium"}),
        ("نتيجة الفصل الدراسي طلعت النهارده.", {"style": "formal", "length": "medium"}),
        ("هتقدم في أنهي كلية؟", {"style": "casual", "length": "short"}),
        ("النظام الجديد في التعليم بيفرق كتير.", {"style": "formal", "length": "medium"}),
    ],

    # ── Social Media ──────────────────────────────────────────
    "social_media": [
        ("شوف الـ post ده على Instagram.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ TikTok ده went viral.", {"style": "code_switched", "contains_code_switching": True}),
        ("اعمل like وsubscribe على القناة.", {"style": "code_switched", "contains_code_switching": True}),
        ("بعتلك الـ story على واتساب.", {"style": "code_switched", "contains_code_switching": True}),
        ("شير الـ reel ده للجروب.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ hashtag ده بينتشر أوي.", {"style": "code_switched", "contains_code_switching": True}),
        ("عمل live على الفيسبوك امبارح.", {"style": "code_switched", "contains_code_switching": True}),
        ("الـ followers زادوا بشكل كبير.", {"style": "code_switched", "contains_code_switching": True}),
    ],
}

# ─────────────────────────────────────────────────────────────
#  Spelling variants  (simulate orthographic inconsistency)
# ─────────────────────────────────────────────────────────────

SPELLING_VARIANTS: dict[str, list[str]] = {
    "عايز":  ["عايز", "عاوز", "عايزة", "عاوزة"],
    "ماشي":  ["ماشي", "ماشى", "ماشيي"],
    "إزيك":  ["إزيك", "ازيك", "ازيك؟", "عامل إيه"],
    "إحنا":  ["إحنا", "احنا", "احنا"],
    "علشان": ["علشان", "عشان", "عشان"],
    "دلوقتي": ["دلوقتي", "دلوقت", "دلوقتى"],
    "إيه":   ["إيه", "ايه", "ايه"],
    "فيه":   ["فيه", "فيها", "فيه"],
}

# ─────────────────────────────────────────────────────────────
#  Filler word injection
# ─────────────────────────────────────────────────────────────

FILLER_WORDS = ["يعني", "بص", "طب", "ايوه", "ماشي", "اممم", "زي ما قلت"]


def _sample_filler() -> str:
    return random.choice(FILLER_WORDS)


def _maybe_inject_filler(text: str, probability: float = 0.15) -> str:
    """Randomly prepend a filler word to simulate spontaneous speech."""
    if random.random() < probability:
        return f"{_sample_filler()}، {text}"
    return text


def _apply_spelling_variant(text: str) -> str:
    """Replace canonical forms with random spelling variants."""
    for canonical, variants in SPELLING_VARIANTS.items():
        if canonical in text:
            text = text.replace(canonical, random.choice(variants), 1)
    return text


def _compute_word_count(text: str) -> int:
    # Strip Arabic diacritics and count space-separated tokens
    clean = re.sub(r'[\u064B-\u065F]', '', text)
    return len(clean.split())


def _detect_code_switching(text: str) -> bool:
    """Return True if the text contains Latin characters."""
    return bool(re.search(r'[A-Za-z]', text))


def _detect_numbers(text: str) -> bool:
    return bool(re.search(r'[\d٠-٩]', text))


def _hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _normalize_arabic(text: str) -> str:
    """
    Light normalization:
    - Remove tatweel (kashida)
    - Normalize alef variants → bare alef
    - Remove extra punctuation
    """
    # Remove tatweel
    text = text.replace('\u0640', '')
    # Normalize alef variants
    text = re.sub(r'[إأآا]', 'ا', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────
#  Core generation logic
# ─────────────────────────────────────────────────────────────

class PromptGenerator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = random.Random(cfg.get("pipeline", {}).get("seed", 42))

    def _weighted_domain_sample(self) -> str:
        domains = self.cfg["prompts"]["domains"]
        names   = [d["name"] for d in domains]
        weights = [d["weight"] for d in domains]
        return self.rng.choices(names, weights=weights, k=1)[0]

    def _build_prompt_record(self, text: str, base_meta: dict, domain: str) -> dict:
        word_count = _compute_word_count(text)
        record: dict[str, Any] = {
            "id":                    _hash_id(text),
            "text":                  text,
            "text_normalized":       _normalize_arabic(text),
            "domain":                domain,
            "style":                 base_meta.get("style", "casual"),
            "dialect":               "egyptian_arabic",
            "word_count":            word_count,
            "contains_code_switching": _detect_code_switching(text) or
                                       base_meta.get("contains_code_switching", False),
            "contains_numbers":      _detect_numbers(text) or
                                     base_meta.get("contains_numbers", False),
            "contains_contractions": base_meta.get("contains_contractions", False),
            "contains_abbreviations": base_meta.get("contains_abbreviations", False),
            "length_bucket":         ("short" if word_count < 5
                                      else "long" if word_count > 15
                                      else "medium"),
            "created_at":            datetime.now(timezone.utc).isoformat(),
        }
        return record

    def generate(self, n: int) -> list[dict]:
        """
        Generate `n` prompt records sampling from the full bank
        with domain weighting, spelling variation, and filler injection.
        """
        all_records: list[dict] = []
        seen_texts:  set[str]   = set()

        # First pass: include all base prompts to ensure coverage
        for domain, prompts in PROMPT_BANK.items():
            for text, meta in prompts:
                if text in seen_texts:
                    continue
                record = self._build_prompt_record(text, meta, domain)
                all_records.append(record)
                seen_texts.add(text)

        # Second pass: generate variants until we reach n
        attempts = 0
        max_attempts = n * 10
        while len(all_records) < n and attempts < max_attempts:
            attempts += 1
            domain  = self._weighted_domain_sample()
            pool    = PROMPT_BANK.get(domain, [])
            if not pool:
                continue
            base_text, base_meta = self.rng.choice(pool)

            # Apply transformations
            variant = base_text
            if self.rng.random() < 0.3:
                variant = _apply_spelling_variant(variant)
            if self.rng.random() < 0.15:
                variant = _maybe_inject_filler(variant)

            if variant in seen_texts:
                continue

            record = self._build_prompt_record(variant, base_meta, domain)
            all_records.append(record)
            seen_texts.add(variant)

        # Shuffle and cap
        self.rng.shuffle(all_records)
        return all_records[:n]


# ─────────────────────────────────────────────────────────────
#  Entry-point helper
# ─────────────────────────────────────────────────────────────

def run(cfg: dict, logger_: logging.Logger | None = None) -> Path:
    log = logger_ or logger
    prompts_dir   = Path(cfg["paths"]["prompts_dir"])
    manifests_dir = Path(cfg["paths"]["manifests_dir"])
    prompts_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    n = cfg["prompts"]["total_count"]
    log.info(f"[Stage 1] Generating {n} prompts …")

    gen     = PromptGenerator(cfg)
    records = gen.generate(n)

    # Write JSONL
    out_file = prompts_dir / "prompts.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write manifest
    manifest = {
        "stage":       "prompt_generation",
        "total":       len(records),
        "domains":     {d: sum(1 for r in records if r["domain"] == d)
                        for d in PROMPT_BANK},
        "styles":      {s: sum(1 for r in records if r["style"] == s)
                        for s in ["casual", "formal", "code_switched", "fast_speech"]},
        "code_switched": sum(1 for r in records if r["contains_code_switching"]),
        "with_numbers":  sum(1 for r in records if r["contains_numbers"]),
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "output_file":   str(out_file),
    }
    manifest_file = manifests_dir / "prompts_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(f"[Stage 1] Done. {len(records)} prompts -> {out_file}")
    log.info(f"[Stage 1] Manifest -> {manifest_file}")
    return out_file
