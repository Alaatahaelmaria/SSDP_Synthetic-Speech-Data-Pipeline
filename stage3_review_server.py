"""
SSDP · Stage 3 · Review Server
================================
Serves a browser-based UI to review (text, audio) pairs.
Reviewers can: approve, reject, or flag samples with specific issues.

Run:  python stage3_review_server.py  (or via main pipeline runner)

Endpoints:
  GET  /            → Review UI (HTML)
  GET  /api/samples → List samples (filterable by status)
  GET  /api/sample/{id} → Single sample
  GET  /audio/{id}  → Serve audio file
  POST /api/review/{id} → Submit review decision
  GET  /api/stats   → Pipeline statistics
  GET  /api/export-ready → Count of approved samples
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Data layer
# ─────────────────────────────────────────────────────────────

class SampleStore:
    """In-memory store backed by synthesis_results.jsonl with persistence."""

    def __init__(self, results_path: Path, reviews_path: Path):
        self.results_path = results_path
        self.reviews_path = reviews_path
        self.samples: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.results_path.exists():
            for line in self.results_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    self.samples[rec["id"]] = rec

        # Apply saved reviews
        if self.reviews_path.exists():
            reviews = json.loads(self.reviews_path.read_text(encoding="utf-8"))
            for sid, review in reviews.items():
                if sid in self.samples:
                    self.samples[sid].update(review)

        logger.info(f"[Stage 3] Loaded {len(self.samples)} samples.")

    def _save_reviews(self):
        reviews = {
            sid: {k: v for k, v in rec.items()
                  if k in ("review_status", "review_flags", "review_note", "reviewed_at")}
            for sid, rec in self.samples.items()
            if rec.get("review_status") in ("approved", "rejected")
        }
        self.reviews_path.write_text(
            json.dumps(reviews, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def list_samples(self, status: str | None = None, page: int = 1, per_page: int = 20) -> dict:
        items = list(self.samples.values())
        if status:
            items = [s for s in items if s.get("review_status") == status]
        total = len(items)
        start = (page - 1) * per_page
        page_items = items[start: start + per_page]
        return {"total": total, "page": page, "per_page": per_page, "items": page_items}

    def get_sample(self, sid: str) -> dict | None:
        return self.samples.get(sid)

    def submit_review(self, sid: str, decision: str, flags: list[str], note: str) -> bool:
        if sid not in self.samples:
            return False
        self.samples[sid]["review_status"] = decision
        self.samples[sid]["review_flags"]  = flags
        self.samples[sid]["review_note"]   = note
        self.samples[sid]["reviewed_at"]   = datetime.now(timezone.utc).isoformat()
        self._save_reviews()
        return True

    def stats(self) -> dict:
        all_s = list(self.samples.values())
        return {
            "total":    len(all_s),
            "pending":  sum(1 for s in all_s if s.get("review_status") == "pending"),
            "approved": sum(1 for s in all_s if s.get("review_status") == "approved"),
            "rejected": sum(1 for s in all_s if s.get("review_status") == "rejected"),
            "failed":   sum(1 for s in all_s if s.get("review_status") == "failed"),
            "avg_quality": round(
                sum(s.get("quality_score", 0) for s in all_s) / max(len(all_s), 1), 3
            ),
        }


# ─────────────────────────────────────────────────────────────
#  HTML UI (self-contained, no external dependencies)
# ─────────────────────────────────────────────────────────────

HTML_UI = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SSDP Review — Egyptian Arabic</title>
  <style>
    :root {
      --bg: #0f1117; --surface: #1a1d27; --card: #22263a;
      --accent: #6c63ff; --accent2: #ff6b6b; --ok: #43d9ad;
      --warn: #ffd166; --text: #e2e8f0; --muted: #8892a4;
      --border: #2e3347; --radius: 12px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', Tahoma, sans-serif; min-height: 100vh; }

    header { background: var(--surface); border-bottom: 1px solid var(--border);
             padding: 16px 32px; display: flex; align-items: center; gap: 16px; }
    header h1 { font-size: 1.3rem; color: var(--accent); font-weight: 700; }
    .badge { background: var(--accent); color: #fff; border-radius: 20px;
             padding: 2px 10px; font-size: .75rem; font-weight: 600; }

    .stats-bar { display: flex; gap: 16px; padding: 16px 32px; flex-wrap: wrap; }
    .stat { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 12px 20px; display: flex; flex-direction: column; align-items: center; min-width: 110px; }
    .stat-val { font-size: 1.6rem; font-weight: 700; }
    .stat-lbl { font-size: .72rem; color: var(--muted); margin-top: 2px; }
    .c-total  { color: var(--text); }
    .c-pending{ color: var(--warn); }
    .c-ok     { color: var(--ok); }
    .c-bad    { color: var(--accent2); }

    .controls { display: flex; gap: 12px; padding: 0 32px 16px; flex-wrap: wrap; align-items: center; }
    select, button { border-radius: 8px; border: 1px solid var(--border); padding: 8px 16px;
                     font-size: .9rem; cursor: pointer; }
    select { background: var(--surface); color: var(--text); }
    button { background: var(--accent); color: #fff; border-color: var(--accent);
             font-weight: 600; transition: opacity .15s; }
    button:hover { opacity: .85; }
    button.sec { background: var(--surface); color: var(--text); border-color: var(--border); }
    button.danger { background: var(--accent2); border-color: var(--accent2); }
    button.ok    { background: var(--ok); color: #000; border-color: var(--ok); }

    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 20px; padding: 0 32px 40px; }

    .card { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 20px; display: flex; flex-direction: column; gap: 12px;
            transition: box-shadow .2s; position: relative; }
    .card:hover { box-shadow: 0 4px 24px #0008; }
    .card.approved { border-color: var(--ok); }
    .card.rejected { border-color: var(--accent2); }

    .card-id  { font-size: .7rem; color: var(--muted); font-family: monospace; }
    .card-text{ font-size: 1rem; line-height: 1.65; direction: rtl; }
    .card-meta{ display: flex; gap: 8px; flex-wrap: wrap; }
    .tag { font-size: .68rem; padding: 2px 8px; border-radius: 20px;
           background: var(--surface); border: 1px solid var(--border); color: var(--muted); }
    .tag.cs  { border-color: #6c63ff44; color: var(--accent); }
    .tag.num { border-color: #ffd16644; color: var(--warn); }

    .score-bar { height: 4px; border-radius: 4px; background: var(--border); overflow: hidden; }
    .score-fill{ height: 100%; border-radius: 4px; background: linear-gradient(90deg,var(--accent2),var(--ok)); transition: width .4s; }

    audio { width: 100%; border-radius: 8px; }

    .flags { display: flex; flex-wrap: wrap; gap: 6px; }
    .flag-btn { font-size: .72rem; padding: 3px 10px; border-radius: 20px; cursor: pointer;
                background: var(--surface); border: 1px solid var(--border); color: var(--muted);
                transition: all .15s; user-select: none; }
    .flag-btn.active { background: #ff6b6b22; border-color: var(--accent2); color: var(--accent2); }

    .card-actions { display: flex; gap: 8px; margin-top: 4px; }
    .card-actions button { flex: 1; padding: 8px; font-size: .82rem; }
    .note-input { width: 100%; background: var(--surface); border: 1px solid var(--border);
                  border-radius: 8px; color: var(--text); padding: 8px; font-size: .82rem;
                  direction: rtl; resize: none; }
    .status-badge { position: absolute; top: 12px; left: 12px; font-size: .68rem;
                    padding: 2px 8px; border-radius: 20px; font-weight: 600; }
    .status-badge.approved { background: #43d9ad22; color: var(--ok); border: 1px solid var(--ok); }
    .status-badge.rejected { background: #ff6b6b22; color: var(--accent2); border: 1px solid var(--accent2); }
    .status-badge.pending  { background: #ffd16622; color: var(--warn); border: 1px solid var(--warn); }
    .status-badge.failed   { background: #8892a422; color: var(--muted); border: 1px solid var(--muted); }

    .pagination { display: flex; gap: 8px; padding: 0 32px 32px; align-items: center; }
    .empty { color: var(--muted); text-align: center; padding: 60px 0; grid-column: 1/-1; font-size: 1.1rem; }

    @media (max-width: 600px) {
      header, .stats-bar, .controls, .grid, .pagination { padding-left: 16px; padding-right: 16px; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<header>
  <h1>🎙 SSDP Review Dashboard</h1>
  <span class="badge">Egyptian Arabic</span>
  <span style="margin-right:auto;"></span>
  <span style="font-size:.8rem;color:var(--muted);">Synthetic Speech Data Pipeline</span>
</header>

<div class="stats-bar" id="statsBar">
  <div class="stat"><span class="stat-val c-total" id="st-total">—</span><span class="stat-lbl">Total</span></div>
  <div class="stat"><span class="stat-val c-pending" id="st-pending">—</span><span class="stat-lbl">Pending</span></div>
  <div class="stat"><span class="stat-val c-ok" id="st-approved">—</span><span class="stat-lbl">Approved</span></div>
  <div class="stat"><span class="stat-val c-bad" id="st-rejected">—</span><span class="stat-lbl">Rejected</span></div>
  <div class="stat"><span class="stat-val" id="st-quality" style="color:var(--accent)">—</span><span class="stat-lbl">Avg Quality</span></div>
</div>

<div class="controls">
  <label style="color:var(--muted);font-size:.85rem;">Filter:</label>
  <select id="filterStatus" onchange="loadPage(1)">
    <option value="">All</option>
    <option value="pending">Pending</option>
    <option value="approved">Approved</option>
    <option value="rejected">Rejected</option>
    <option value="failed">Failed</option>
  </select>
  <label style="color:var(--muted);font-size:.85rem;">Per page:</label>
  <select id="perPage" onchange="loadPage(1)">
    <option value="12">12</option>
    <option value="24" selected>24</option>
    <option value="48">48</option>
  </select>
  <button class="sec" onclick="loadPage(currentPage)">↻ Refresh</button>
</div>

<div class="grid" id="grid"><div class="empty">Loading…</div></div>
<div class="pagination" id="pagination"></div>

<script>
const FLAG_OPTIONS = [
  "wrong_pronunciation", "english_word_issue", "dialect_mismatch",
  "unnatural_speech", "background_noise", "clipping", "too_fast", "too_slow"
];
const FLAG_LABELS = {
  wrong_pronunciation: "❌ نطق غلط",
  english_word_issue:  "🔤 مشكلة إنجليزي",
  dialect_mismatch:    "🗣 لهجة غير مصرية",
  unnatural_speech:    "🤖 كلام غير طبيعي",
  background_noise:    "📢 ضوضاء",
  clipping:            "✂️ قطع",
  too_fast:            "⏩ سريع جداً",
  too_slow:            "⏪ بطيء جداً"
};

let currentPage = 1;
let cardState = {};   // sid -> { flags: Set, note: "" }

async function loadStats() {
  const r = await fetch('/api/stats');
  const s = await r.json();
  document.getElementById('st-total').textContent    = s.total;
  document.getElementById('st-pending').textContent  = s.pending;
  document.getElementById('st-approved').textContent = s.approved;
  document.getElementById('st-rejected').textContent = s.rejected;
  document.getElementById('st-quality').textContent  = (s.avg_quality * 100).toFixed(0) + '%';
}

async function loadPage(page) {
  currentPage = page;
  const status  = document.getElementById('filterStatus').value;
  const perPage = document.getElementById('perPage').value;
  const url = `/api/samples?page=${page}&per_page=${perPage}${status ? '&status='+status : ''}`;
  const r = await fetch(url);
  const data = await r.json();
  renderGrid(data.items);
  renderPagination(data.total, data.page, data.per_page);
  loadStats();
}

function renderGrid(items) {
  const grid = document.getElementById('grid');
  if (!items.length) { grid.innerHTML = '<div class="empty">لا توجد عينات</div>'; return; }
  grid.innerHTML = items.map(s => cardHTML(s)).join('');
}

function cardHTML(s) {
  const score = (s.quality_score || 0);
  const scorePct = Math.round(score * 100);
  if (!cardState[s.id]) cardState[s.id] = { flags: new Set(), note: '' };
  const existingFlags = s.review_flags || [];
  existingFlags.forEach(f => cardState[s.id].flags.add(f));

  const flagsHTML = FLAG_OPTIONS.map(f => {
    const active = cardState[s.id].flags.has(f) ? 'active' : '';
    return `<span class="flag-btn ${active}" onclick="toggleFlag('${s.id}','${f}',this)">${FLAG_LABELS[f]||f}</span>`;
  }).join('');

  const tags = [
    s.domain ? `<span class="tag">${s.domain}</span>` : '',
    s.style  ? `<span class="tag">${s.style}</span>` : '',
    s.contains_code_switching ? `<span class="tag cs">code-switch</span>` : '',
    s.contains_numbers ? `<span class="tag num">أرقام</span>` : '',
    s.word_count ? `<span class="tag">${s.word_count} كلمة</span>` : '',
    s.duration_sec ? `<span class="tag">${s.duration_sec}s</span>` : '',
    s.voice ? `<span class="tag">${s.voice.split('-').pop()}</span>` : '',
  ].filter(Boolean).join('');

  const audioSrc = s.audio_path ? `/audio/${s.id}` : '';
  const audioHTML = audioSrc
    ? `<audio controls preload="none"><source src="${audioSrc}"></audio>`
    : `<div style="color:var(--accent2);font-size:.8rem;">⚠️ No audio (synthesis failed)</div>`;

  const existingNote = s.review_note || '';
  const statusCls = s.review_status || 'pending';

  return `
<div class="card ${statusCls}" id="card-${s.id}">
  <span class="status-badge ${statusCls}">${statusCls}</span>
  <div class="card-id">#${s.id}</div>
  <div class="card-text">${s.text}</div>
  <div class="card-meta">${tags}</div>
  <div>
    <div style="display:flex;justify-content:space-between;font-size:.72rem;color:var(--muted);margin-bottom:4px;">
      <span>Quality</span><span>${scorePct}%</span>
    </div>
    <div class="score-bar"><div class="score-fill" style="width:${scorePct}%"></div></div>
  </div>
  ${audioHTML}
  <div>
    <div style="font-size:.72rem;color:var(--muted);margin-bottom:6px;">Flag issues:</div>
    <div class="flags">${flagsHTML}</div>
  </div>
  <textarea class="note-input" rows="2" placeholder="ملاحظة اختيارية…"
    onchange="cardState['${s.id}'].note=this.value">${existingNote}</textarea>
  <div class="card-actions">
    <button class="ok"    onclick="submitReview('${s.id}','approved')">✓ قبول</button>
    <button class="danger" onclick="submitReview('${s.id}','rejected')">✗ رفض</button>
  </div>
</div>`;
}

function toggleFlag(sid, flag, el) {
  if (!cardState[sid]) cardState[sid] = { flags: new Set(), note: '' };
  if (cardState[sid].flags.has(flag)) {
    cardState[sid].flags.delete(flag);
    el.classList.remove('active');
  } else {
    cardState[sid].flags.add(flag);
    el.classList.add('active');
  }
}

async function submitReview(sid, decision) {
  const state = cardState[sid] || { flags: new Set(), note: '' };
  const body = {
    decision, flags: [...state.flags], note: state.note || ''
  };
  const r = await fetch(`/api/review/${sid}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (r.ok) {
    const card = document.getElementById(`card-${sid}`);
    if (card) {
      card.className = `card ${decision}`;
      const badge = card.querySelector('.status-badge');
      if (badge) { badge.className = `status-badge ${decision}`; badge.textContent = decision; }
    }
    loadStats();
  }
}

function renderPagination(total, page, perPage) {
  const totalPages = Math.ceil(total / perPage);
  const pag = document.getElementById('pagination');
  let html = `<span style="color:var(--muted);font-size:.85rem;">${total} sample(s)</span>`;
  if (page > 1)        html += `<button class="sec" onclick="loadPage(${page-1})">← Prev</button>`;
  html += `<span style="color:var(--muted);font-size:.85rem;">Page ${page}/${totalPages||1}</span>`;
  if (page < totalPages) html += `<button class="sec" onclick="loadPage(${page+1})">Next →</button>`;
  pag.innerHTML = html;
}

loadPage(1);
setInterval(() => loadStats(), 10000);
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────
#  HTTP Request Handler
# ─────────────────────────────────────────────────────────────

class ReviewHandler(BaseHTTPRequestHandler):
    store: SampleStore = None   # injected before server start
    cfg:   dict        = None

    def log_message(self, fmt, *args):
        logger.debug(fmt % args)

    def _send(self, code: int, body: bytes, content_type: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: Any):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(code, body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send(200, HTML_UI.encode("utf-8"), "text/html; charset=utf-8")

        elif path == "/api/stats":
            self._json(200, self.store.stats())

        elif path == "/api/samples":
            status   = qs.get("status",   [None])[0]
            page     = int(qs.get("page",     ["1"])[0])
            per_page = int(qs.get("per_page", ["24"])[0])
            self._json(200, self.store.list_samples(status, page, per_page))

        elif path.startswith("/api/sample/"):
            sid = path.split("/")[-1]
            sample = self.store.get_sample(sid)
            if sample:
                self._json(200, sample)
            else:
                self._json(404, {"error": "not found"})

        elif path.startswith("/audio/"):
            sid = path.split("/")[-1]
            sample = self.store.get_sample(sid)
            if not sample or not sample.get("audio_path"):
                self._json(404, {"error": "audio not found"})
                return
            audio_path = Path(sample["audio_path"])
            if not audio_path.exists():
                self._json(404, {"error": "audio file missing"})
                return
            mime = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
            data = audio_path.read_bytes()
            self._send(200, data, mime)

        elif path == "/api/export-ready":
            stats = self.store.stats()
            self._json(200, {"approved": stats["approved"]})

        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path.startswith("/api/review/"):
            sid    = path.split("/")[-1]
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length).decode("utf-8"))
            ok = self.store.submit_review(
                sid,
                decision=body.get("decision", "pending"),
                flags=body.get("flags", []),
                note=body.get("note", ""),
            )
            if ok:
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": "sample not found"})
        else:
            self._json(404, {"error": "not found"})


# ─────────────────────────────────────────────────────────────
#  Entry-point helper
# ─────────────────────────────────────────────────────────────

def run(cfg: dict, logger_: logging.Logger | None = None):
    log = logger_ or logger

    manifests_dir = Path(cfg["paths"]["manifests_dir"])
    results_path  = manifests_dir / "synthesis_results.jsonl"
    reviews_path  = manifests_dir / "reviews.json"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    store = SampleStore(results_path, reviews_path)

    host = cfg["review"].get("server_host", "0.0.0.0")
    port = cfg["review"].get("server_port", 8000)

    ReviewHandler.store = store
    ReviewHandler.cfg   = cfg

    server = HTTPServer((host, port), ReviewHandler)
    url    = f"http://localhost:{port}"
    log.info(f"[Stage 3] Review server running → {url}")
    log.info("[Stage 3] Press Ctrl+C to stop.")

    if cfg["review"].get("auto_open_browser", True):
        import threading, webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("[Stage 3] Server stopped.")
    finally:
        server.server_close()
        # Write final reviews manifest
        stats = store.stats()
        manifest = {"stage": "review", **stats, "completed_at": datetime.now(timezone.utc).isoformat()}
        (manifests_dir / "review_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info(f"[Stage 3] Review manifest saved. Approved: {stats['approved']}")
