#!/usr/bin/env python3
"""
sortling: tidy screenshots, downloads and documents with decision models.

One file, standard library only. Download it and run it:

    python3 sortling.py                     # interactive: pick a folder, a profile, a model
    python3 sortling.py --root ~/Downloads  # scripted dry run with the default backend
    python3 sortling.py --undo <manifest.json>

Modes:
  fast      System One decision models (llama.cpp /v1/systemone, TypeSafe Jev).
            One quick pass per file with real probabilities.
  thorough  Any OpenAI-compatible chat endpoint. Explains each file; slower;
            confidence is self-reported by the model.

Config: $XDG_CONFIG_HOME/sortling/config.json (see --init-config). API keys are never
stored there, only the names of environment variables that hold them.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import ipaddress
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__version__ = "0.2.0"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

# system folders are underscore-prefixed so they sort first
REVIEW_FOLDER = "_review"
DELETE_FOLDER = "_delete"
DUPLICATES_FOLDER = "_duplicates"
SYSTEM_FOLDERS: Tuple[str, ...] = (REVIEW_FOLDER, DELETE_FOLDER, DUPLICATES_FOLDER)
# folderal's numbered layout; still treated as already-sorted so old trees aren't re-processed
LEGACY_TARGET_FOLDERS: Tuple[str, ...] = (
    "10_Personal_Keep",
    "20_Finance_Keep",
    "30_Work_Archive",
    "40_Learning_Core",
    "41_Learning_Archive",
    "90_Review",
    "98_Delete",
    "99_Duplicates",
)

# built-in "default" profile: docs/<category>, screenshots/<topic>, photos/
DOC_FOLDERS: Dict[str, str] = {
    "personal": "docs/personal",
    "medical": "docs/medical",
    "legal": "docs/legal",
    "finance": "docs/finance",
    "work": "docs/work",
    "travel": "docs/travel",
    "learning": "docs/learning",
}
LEARNING_ARCHIVE_FOLDER = "docs/learning/archive"
PHOTOS_FOLDER = "photos"
MEMES_FOLDER = "memes"
DEFAULT_PROFILE_ROOTS: Tuple[str, ...] = ("docs", "screenshots", PHOTOS_FOLDER, MEMES_FOLDER)

# camera photos: photos/<category>. Kept coarse on purpose; photo apps do faces and places better
PHOTO_CATEGORIES: Dict[str, str] = {
    "people": "people, selfies, family, friends, group photos",
    "places": "landscapes, cities, buildings, travel scenery",
    "food": "meals, drinks, cooking, restaurants",
    "pets": "pets and other animals",
    "events": "celebrations, weddings, parties, concerts, festivals, ceremonies",
    "other": "objects, everyday scenes or anything else",
}

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "personal": "identity documents, personal records, resumes, certificates, personal letters or chats",
    "medical": "health records, lab reports, prescriptions, insurance claims for care",
    "finance": "bank or card statements, invoices, receipts, bills, taxes, payments, payslips",
    "work": "employer documents, job-related technical material, work chats, dashboards or tickets",
    "learning": "courses, lecture notes, tutorials, papers, cheat sheets, code or reference material",
    "travel": "bookings, boarding passes, tickets, itineraries, maps, visas in use for a trip",
    "legal": "contracts, leases, property papers, legal notices, agreements",
    "spam": "marketing, promotions, ads, junk, memes or throwaway content with no lasting value",
    "unknown": "cannot tell what this is",
}

# screenshots that aren't records: topic -> description; folder is screenshots/<topic>
SCREENSHOT_TOPICS: Dict[str, str] = {
    "code": "code, terminal, error messages, logs, developer tools",
    "chats": "messaging or chat conversations, emails, comments",
    "reading": "articles, documentation, posts or text worth reading later",
    "apps": "app or website UI, settings, dashboards, design inspiration",
    "social": "social media posts and feeds, videos from feeds",
    "shopping": "products, prices, wishlists, carts (not completed purchases)",
    "maps": "maps, directions, places, addresses",
    "media": "music, movies, TV, games, video stills",
    "other": "none of the above",
}

IMAGE_KINDS: Dict[str, str] = {
    "screenshot": "a capture of a screen: app, website, chat, code, map, social feed, game",
    "photo": "a real camera photo of people, places, food, pets, objects or scenery",
    "meme": (
        "a meme, joke image, forwarded graphic, good morning or festival wishes card, "
        "motivational or religious quote image, or other made-to-share picture"
    ),
    "document": (
        "a scan or photo of a paper document, or a screenshot of a record such as a receipt, "
        "order confirmation, booking, ticket, bill, ID or statement"
    ),
}

# concrete yes/no questions answer more reliably than an abstract keep/archive/delete choice
JUNK_QUESTION = (
    "Is this junk with no lasting value, such as an ad, spam, or a blurry, "
    "accidental, blank or black photo or screenshot?"
)
CORE_REFERENCE_QUESTION = "Is this core reference material worth coming back to often, rather than old notes to archive?"
NONE_OPTION = "none"

CATEGORY_ALIASES: Dict[str, str] = {
    "personal_records": "personal",
    "identity": "personal",
    "health": "medical",
    "financial": "finance",
    "banking": "finance",
    "career": "work",
    "employment": "work",
    "education": "learning",
    "study": "learning",
    "property": "legal",
    "junk": "spam",
    "marketing": "spam",
    "unclear": "unknown",
    "misc": "unknown",
}

PDF_EXTS: Tuple[str, ...] = (".pdf",)
# png/jpeg/gif/bmp are decoded natively by llama.cpp; the rest are converted to PNG first
IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff")
NATIVE_IMAGE_MIME: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


# ---------------------------------------------------------------------------
# Terminal flair
# ---------------------------------------------------------------------------

BANNER_ART = r"""
                   _    _  _
 ___   ___   _ __ | |_ | |(_) _ __    __ _
/ __| / _ \ | '__|| __|| || || '_ \  / _` |
\__ \| (_) || |   | |_ | || || | | || (_| |
|___/ \___/ |_|    \__||_||_||_| |_| \__, |
                                     |___/"""


def unicode_ok(stream=None) -> bool:
    enc = (getattr(stream or sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


def icon(name: str) -> str:
    fancy = {"fast": "⚡", "thorough": "🐢", "hybrid": "⚡🐢", "hosted": "☁", "local": "⌂", "ok": "✓", "warn": "!"}
    plain = {"fast": ">>", "thorough": "~~", "hybrid": ">~", "hosted": "[cloud]", "local": "[local]", "ok": "ok", "warn": "!"}
    return (fancy if unicode_ok() else plain).get(name, "")


def color(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


def print_banner() -> None:
    print(color(BANNER_ART, "36"))
    print(color(f"  tidy screenshots, downloads and documents with decision models  v{__version__}", "2"))
    print()


class ProgressBar:
    def __init__(self, total: int, label: str, enabled: bool = True, width: int = 30, stream=None) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.enabled = enabled and total > 0
        self.width = max(10, int(width))
        self.stream = stream or sys.stderr
        self.dynamic = bool(
            getattr(self.stream, "isatty", lambda: False)()
            and os.environ.get("TERM", "").lower() not in {"", "dumb"}
        )
        self._last_non_dynamic_done = -1
        if self.enabled:
            self.update(0, "")

    def _format(self, done: int, detail: str = "") -> str:
        ratio = min(max(done / self.total, 0.0), 1.0)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        msg = f"{self.label} [{bar}] {done}/{self.total} {ratio * 100:5.1f}%"
        if detail:
            msg += f" | {detail[:52]}"
        return msg

    def update(self, done: int, detail: str = "") -> None:
        if not self.enabled:
            return
        done = min(max(int(done), 0), self.total)
        msg = self._format(done, detail)
        if self.dynamic:
            self.stream.write("\r\033[K" + msg)
            self.stream.flush()
            return
        step = max(1, self.total // 20)
        if done in {0, self.total} or done - self._last_non_dynamic_done >= step:
            self.stream.write(msg + "\n")
            self.stream.flush()
            self._last_non_dynamic_done = done

    def finish(self, detail: str = "done") -> None:
        if not self.enabled:
            return
        self.update(self.total, detail)
        if self.dynamic:
            self.stream.write("\n")
            self.stream.flush()


def stage_note(message: str, enabled: bool) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# The face: a Doom-guy-style status head that reacts to each decision
# ---------------------------------------------------------------------------

# packed by tools/encode_faces.py from assets/faces/head-*.png; regenerate there, don't edit by hand
# --- FACES BEGIN
FACES_DATA = (
    "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"
    "751DzyqfMcMpNMsPRIZ//qmi4L///X+kacan"
)
# --- FACES END

_FACES: Optional[dict] = None


def load_faces() -> dict:
    """{"w", "h", "frames": {name: [[rgb or None] * w] * h}} decoded from FACES_DATA."""
    global _FACES
    if _FACES is None:
        import zlib

        blob = json.loads(zlib.decompress(base64.b64decode("".join(FACES_DATA))))
        palette = [None] + [tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) for c in blob["palette"]]
        w, h = blob["w"], blob["h"]
        frames = {}
        for name, data in blob["frames"].items():
            px = base64.b64decode(data)
            frames[name] = [[palette[px[y * w + x]] for x in range(w)] for y in range(h)]
        _FACES = {"w": w, "h": h, "frames": frames}
    return _FACES


def _xterm256(rgb: Tuple[int, int, int]) -> int:
    r, g, b = (int(round(c / 255 * 5)) for c in rgb)
    return 16 + 36 * r + 6 * g + b


def render_face(name: str, mirror: bool, truecolor: bool) -> List[str]:
    """One string per terminal row: two pixel rows per row with half-block characters."""
    faces = load_faces()
    rows = faces["frames"][name]
    if mirror:
        rows = [list(reversed(r)) for r in rows]

    def fg(c):
        return f"\033[38;2;{c[0]};{c[1]};{c[2]}m" if truecolor else f"\033[38;5;{_xterm256(c)}m"

    def bg(c):
        return f"\033[48;2;{c[0]};{c[1]};{c[2]}m" if truecolor else f"\033[48;5;{_xterm256(c)}m"

    out = []
    for y in range(0, faces["h"], 2):
        top, bot = rows[y], rows[y + 1] if y + 1 < faces["h"] else [None] * faces["w"]
        line = []
        for t, b in zip(top, bot):
            if t is None and b is None:
                line.append("\033[0m ")
            elif b is None:
                line.append("\033[0m" + fg(t) + "▀")
            elif t is None:
                line.append("\033[0m" + fg(b) + "▄")
            else:
                line.append(fg(t) + bg(b) + "▀")
        out.append("".join(line) + "\033[0m")
    return out


def face_mode(stream=None) -> str:
    """'truecolor', '256' or '' (no face). SORTLING_FACE=0 turns it off, =force renders anywhere."""
    stream = stream or sys.stderr
    env = os.environ.get("SORTLING_FACE", "").lower()
    if env in ("0", "off", "no") or os.environ.get("NO_COLOR"):
        return ""
    forced = env == "force"
    if not forced:
        if not getattr(stream, "isatty", lambda: False)() or os.environ.get("TERM", "dumb") in ("", "dumb"):
            return ""
        if not unicode_ok(stream):
            return ""
        size = shutil.get_terminal_size((80, 24))
        if size.columns < 72 or size.lines < 18:
            return ""
    colorterm = os.environ.get("COLORTERM", "").lower()
    return "truecolor" if forced or colorterm in ("truecolor", "24bit") else "256"


# what the face does for each outcome, and the caption beside it
MOODS: Dict[str, Tuple[str, str]] = {
    "working": ("smile", "sorting..."),
    "confident": ("happy", "easy one"),
    "unsure": ("confused", "hmm, not sure -> _review"),
    "junk": ("surprised", "yikes, junk -> _delete"),
    "duplicate": ("nerd", "seen this one before"),
    "meme": ("cool", "a meme, keeping it in memes/"),
    "pet": ("love", "aww"),
    "record": ("searching", "a record, filing it with the docs"),
    "error": ("sad", "the model didn't answer, guessing from the filename"),
    "fallback": ("dizzy", "no model, guessing from the filename"),
    "loading": ("sleepy", "waking the model..."),
    "done": ("starstruck", "all sorted!"),
}


def mood_for(rec: "DocRecord") -> str:
    if "model_error" in rec.reason:
        return "error"
    if rec.classifier.startswith("fallback"):
        return "fallback"
    if rec.duplicate_of:
        return "duplicate"
    if rec.target_folder == REVIEW_FOLDER or rec.target_folder.endswith("/other"):
        return "unsure"
    if rec.target_folder == DELETE_FOLDER:
        return "junk"
    if rec.target_folder == MEMES_FOLDER:
        return "meme"
    if rec.target_folder == f"{PHOTOS_FOLDER}/pets":
        return "pet"
    if rec.kind == "image" and rec.target_folder.startswith("docs"):
        return "record"  # a screenshot or photo of a receipt, ticket or form
    return "confident" if rec.confidence >= 0.9 else "working"


class FaceBar(ProgressBar):
    """ProgressBar with the face beside it. A background thread keeps the face alive between
    updates: it glances left and right and blinks, like Doom guy waiting for something to happen."""

    REACT_SECONDS = 1.1
    TICK_SECONDS = 0.35

    def __init__(self, total: int, label: str, mode: str, stream=None, width: int = 30) -> None:
        import threading

        self.mode = mode
        self.lock = threading.Lock()
        self.mood = "working"
        self.mood_until = 0.0
        self.tick = 0
        self.done = 0
        self.detail = ""
        self.counts: Dict[str, int] = {}
        self.rows = (load_faces()["h"] + 1) // 2
        self.drawn = False
        self._stop = threading.Event()
        super().__init__(total, label, enabled=True, width=width, stream=stream)
        self.dynamic = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def _frame(self) -> Tuple[str, bool]:
        mood = self.mood if time.time() < self.mood_until or self.mood in ("loading", "done") else "working"
        name = MOODS[mood][0]
        if mood != "working":
            return name, False
        # idle: look left, look right, blink, and now and then peer at the file through a magnifier
        beat = self.tick % 16
        if beat == 7:
            return "content", False
        if beat in (11, 12, 13):
            return "searching", beat == 12
        return "smile", beat in (3, 4, 5)

    def _draw(self) -> None:
        name, mirror = self._frame()
        face = render_face(name, mirror, self.mode == "truecolor")
        mood = self.mood if time.time() < self.mood_until or self.mood in ("loading", "done") else "working"
        side = [""] * self.rows
        side[2] = self._format(self.done, "")
        side[4] = self.detail[:60]
        side[6] = color(MOODS[mood][1], "2") if mood != "working" else ""
        if self.counts:
            tally = "  ".join(f"{k} {v}" for k, v in sorted(self.counts.items(), key=lambda kv: -kv[1])[:4])
            side[8] = color(tally[:60], "2")
        out = []
        if self.drawn:
            out.append(f"\033[{self.rows}F")
        for row, text in zip(face, side):
            out.append(f"\033[2K  {row}   {text}\n")
        self.stream.write("".join(out))
        self.stream.flush()
        self.drawn = True

    def _animate(self) -> None:
        while not self._stop.wait(self.TICK_SECONDS):
            with self.lock:
                self.tick += 1
                self._draw()

    def update(self, done: int, detail: str = "") -> None:
        with self.lock:
            self.done = min(max(int(done), 0), self.total)
            if detail:
                self.detail = detail
            self._draw()

    def react(self, mood: str, folder: str = "") -> None:
        with self.lock:
            if folder:
                top = folder.split("/", 1)[0]
                self.counts[top] = self.counts.get(top, 0) + 1
            if mood != "working" or time.time() >= self.mood_until:
                self.mood, self.mood_until = mood, time.time() + self.REACT_SECONDS
            self._draw()

    def finish(self, detail: str = "done") -> None:
        self._stop.set()
        self.thread.join(timeout=1)
        with self.lock:
            self.done, self.detail = self.total, detail
            self.mood, self.mood_until = "done", time.time() + 60
            self._draw()


def make_bar(total: int, label: str, enabled: bool, width: int, face: bool = True) -> ProgressBar:
    mode = face_mode() if enabled and face and FACES_DATA else ""
    return FaceBar(total, label, mode, width=width) if mode else ProgressBar(total, label, enabled=enabled, width=width)


class FaceWait:
    """The sleepy face while something slow happens (a router loading a model)."""

    def __init__(self, label: str, enabled: bool) -> None:
        mode = face_mode() if enabled and FACES_DATA else ""
        self.bar: Optional[FaceBar] = FaceBar(1, label, mode) if mode else None
        if self.bar:
            self.bar.react("loading")

    def close(self, label: str) -> None:
        if self.bar:
            self.bar._stop.set()
            self.bar.thread.join(timeout=1)
            with self.bar.lock:
                self.bar.done, self.bar.detail, self.bar.mood = 1, label, "confident"
                self.bar.mood_until = time.time() + 60
                self.bar._draw()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "default_backend": "local-d3",
    "backends": {
        "local-d3": {
            "type": "systemone",
            "label": "vLLM-SR d3-flash via llama-swap",
            "url": "http://localhost:8080/upstream/vllm-d3-flash/v1/systemone",
            "api_key_env": "LLAMA_SWAP_API_KEY",
        },
        "local-clef": {
            "type": "systemone",
            "label": "Cloudflare Clef-Flash via llama-swap (faster on images)",
            "url": "http://localhost:8080/upstream/cloudflare-clef-flash/v1/systemone",
            "api_key_env": "LLAMA_SWAP_API_KEY",
        },
        "jev": {
            "type": "systemone",
            "label": "TypeSafe Jev (hosted, text only)",
            "url": "https://api.typesafe.ai/v1/systemone",
            "model": "jev-latest",
            "api_key_env": "TYPESAFE_API_KEY",
            "vision": False,
        },
        "local-llm": {
            "type": "openai",
            "label": "any chat model on llama-swap / Ollama / LM Studio",
            "url": "http://localhost:8080/v1",
            "model": "set-me",
            "api_key_env": "LLAMA_SWAP_API_KEY",
        },
        "openai": {
            "type": "openai",
            "label": "OpenAI or any hosted OpenAI-compatible API",
            "url": "https://api.openai.com/v1",
            "model": "set-me",
            "api_key_env": "OPENAI_API_KEY",
        },
        "hybrid-local": {
            "type": "hybrid",
            "label": "d3 decides; a local chat LLM re-checks only the uncertain files",
            "first": "local-d3",
            "second": "local-llm",
        },
    },
    "profiles": {},
}


def config_path(override: str = "") -> Path:
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "sortling" / "config.json"


def load_config(path: Path) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path.exists():
        user = json.loads(path.read_text(encoding="utf-8"))
        cfg["default_backend"] = user.get("default_backend", cfg["default_backend"])
        # user entries replace built-in ones of the same name
        cfg["backends"].update(user.get("backends", {}))
        cfg["profiles"].update(user.get("profiles", {}))
    return cfg


def save_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_user_config(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"default_backend": DEFAULT_CONFIG["default_backend"], "backends": {}, "profiles": {}}


def is_hosted_url(url: str) -> bool:
    """True unless the host is loopback, private, link-local or a bare LAN name."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if not host or host == "localhost" or host.endswith((".local", ".lan", ".home", ".internal")):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local)
    except ValueError:
        return "." in host


# ---------------------------------------------------------------------------
# Backends: both answer System One questions ({id: {type, instructions, criteria}})
# ---------------------------------------------------------------------------


class Backend:
    mode = "fast"

    def __init__(self, name: str, cfg: dict) -> None:
        self.name = name
        self.cfg = cfg
        self.url = str(cfg.get("url", ""))
        self.model = str(cfg.get("model", ""))
        self.label = str(cfg.get("label", ""))
        self.key_env = str(cfg.get("api_key_env", ""))
        self.api_key = os.environ.get(self.key_env, "") if self.key_env else ""
        self.hosted = bool(cfg["hosted"]) if "hosted" in cfg else is_hosted_url(self.url)
        self.vision = bool(cfg.get("vision", True))

    def headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def post(self, url: str, body: dict, timeout_s: int) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=self.headers())
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def decide(self, state: str, images: Sequence[str], questions: dict, timeout_s: int) -> Tuple[dict, str]:
        """Return (answers in System One shape, optional free-text reason)."""
        raise NotImplementedError

    def precheck(self, timeout_s: int) -> None:
        self.decide("precheck", [], {"ok": {"type": "noul", "instructions": "Is this a test?"}}, timeout_s)

    def describe(self) -> str:
        where = f"{icon('hosted')} hosted" if self.hosted else f"{icon('local')} local"
        eyes = "" if self.vision else ", text only"
        return f"{icon(self.mode)} {self.mode:<8} {self.name:<12} {where}{eyes}  {self.label}"


class SystemOneBackend(Backend):
    mode = "fast"

    def decide(self, state: str, images: Sequence[str], questions: dict, timeout_s: int) -> Tuple[dict, str]:
        body: dict = {"state": state, "questions": questions}
        if self.model:
            body["model"] = self.model
        if images and self.vision:
            body["images"] = list(images)
        data = self.post(self.url, body, timeout_s)
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise ValueError(f"decision response has no answers: {str(data)[:200]}")
        return answers, ""


class OpenAIBackend(Backend):
    """Emulates System One answers with a chat model and a JSON reply."""

    mode = "thorough"

    def _prompt(self, state: str, questions: dict) -> str:
        lines = [state, "", "Questions:"]
        for qid, q in questions.items():
            if q["type"] == "choice":
                lines.append(f'- "{qid}" (choose exactly one option): {q["instructions"]}')
                for opt, desc in q["criteria"].items():
                    lines.append(f"    {opt}: {desc}" if desc else f"    {opt}")
            else:
                lines.append(f'- "{qid}" (yes or no): {q["instructions"]}')
        return "\n".join(lines)

    def decide(self, state: str, images: Sequence[str], questions: dict, timeout_s: int) -> Tuple[dict, str]:
        system = (
            "You sort personal files into folders. You are shown one file as text and/or an image. "
            "Answer every question. Reply with ONLY a JSON object:\n"
            '{"answers": {"<question id>": {"choice": "<one option name>", "confidence": <0..1>} '
            'or {"yes": true|false, "confidence": <0..1>}}, "reason": "<one short sentence about the file>"}'
        )
        content: list = [{"type": "text", "text": self._prompt(state, questions)}]
        if images and self.vision:
            content += [{"type": "image_url", "image_url": {"url": u}} for u in images]
        body: dict = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 2048,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
        }
        url = self.url.rstrip("/") + "/chat/completions"
        try:
            data = self.post(url, body, timeout_s)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            body.pop("response_format")  # some servers reject json mode
            data = self.post(url, body, timeout_s)
        text = str(data["choices"][0]["message"].get("content") or "")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # reasoning models
        parsed = extract_json_object(text)
        raw = parsed.get("answers", parsed)
        answers: dict = {}
        for qid, q in questions.items():
            a = raw.get(qid) if isinstance(raw.get(qid), dict) else {}
            conf = to_float_0_1(a.get("confidence", 0.5))
            if q["type"] == "choice":
                opts = list(q["criteria"])
                pick = str(a.get("choice", "")).strip().lower()
                pick = next((o for o in opts if o.lower() == pick), "")
                if not pick:
                    conf, pick = 0.0, opts[-1]  # unparseable: lowest confidence, routes to review
                rest = (1.0 - conf) / max(1, len(opts) - 1)
                answers[qid] = {"type": "choice", "choice": pick, "probabilities": {o: conf if o == pick else rest for o in opts}}
            else:
                yes = a.get("yes")
                yes = yes if isinstance(yes, bool) else str(yes).lower() in {"true", "yes", "1"}
                answers[qid] = {"type": "noul", "noul": conf if yes else 1.0 - conf}
        return answers, str(parsed.get("reason", "")).strip()[:160]

    def precheck(self, timeout_s: int) -> None:
        if not self.model or self.model == "set-me":
            raise ValueError(f'set "model" for backend "{self.name}" in the config')
        super().precheck(timeout_s)


def make_backend(name: str, cfg: dict) -> Backend:
    kind = cfg.get("type", "systemone")
    if kind == "systemone":
        return SystemOneBackend(name, cfg)
    if kind == "openai":
        return OpenAIBackend(name, cfg)
    if kind == "hybrid":
        raise ValueError(f'backend "{name}" is a hybrid; resolve it with make_backends')
    raise ValueError(f'backend "{name}": unknown type "{kind}" (use systemone, openai or hybrid)')


def make_backends(name: str, cfg: dict, url: str = "", second: str = "") -> Tuple[Backend, Optional[Backend]]:
    """Resolve a backend name into (first pass, optional second pass)."""
    backends = cfg["backends"]
    if name not in backends:
        raise ValueError(f'unknown backend "{name}" (configured: {", ".join(backends)})')
    bcfg = dict(backends[name])
    if bcfg.get("type") == "hybrid":
        first_name, second_name = str(bcfg.get("first", "")), second or str(bcfg.get("second", ""))
        for part in (first_name, second_name):
            if part not in backends or backends[part].get("type") == "hybrid":
                raise ValueError(f'hybrid "{name}": "{part}" must be a configured non-hybrid backend')
        bcfg, name = dict(backends[first_name]), first_name
    elif second:
        second_name = second
        if second_name not in backends or backends[second_name].get("type") == "hybrid":
            raise ValueError(f'--second-pass "{second_name}" must be a configured non-hybrid backend')
    else:
        second_name = ""
    if url:
        bcfg["url"] = url
        bcfg.pop("hosted", None)
    first_backend = make_backend(name, bcfg)
    return first_backend, (make_backend(second_name, backends[second_name]) if second_name else None)


def describe_entry(name: str, cfg: dict) -> str:
    bcfg = cfg["backends"][name]
    if bcfg.get("type") == "hybrid":
        return f"{icon('hybrid')} hybrid   {name:<12} {bcfg.get('first')} first, {bcfg.get('second')} re-checks uncertain files"
    return make_backend(name, bcfg).describe()


# ---------------------------------------------------------------------------
# Profiles: which questions to ask and how answers become folders
# ---------------------------------------------------------------------------


@dataclass
class Thresholds:
    min_confidence: float = 0.6
    screenshot_min_confidence: float = 0.4
    delete_min_prob: float = 0.85


@dataclass
class Profile:
    name: str
    description: str = ""
    # "default" and "photos" are built in; "custom" uses folders (folder -> description)
    layout: str = "custom"
    folders: Dict[str, str] = field(default_factory=dict)
    source: str = ""  # where it came from: built-in, config, file path

    @property
    def builtin(self) -> bool:
        return self.layout != "custom"

    def roots(self) -> Tuple[str, ...]:
        """Top-level folders this profile writes, besides the system folders."""
        if self.layout == "default":
            return DEFAULT_PROFILE_ROOTS
        if self.layout == "photos":
            return (PHOTOS_FOLDER, MEMES_FOLDER, "screenshots", "docs")
        return tuple(self.folders)


BUILTIN_PROFILES: Dict[str, Profile] = {
    "default": Profile(
        name="default",
        layout="default",
        source="built-in",
        description="docs/<category>, screenshots/<topic>, photos/<category>, memes/",
    ),
    "photos": Profile(
        name="photos",
        layout="photos",
        source="built-in",
        description="camera rolls and WhatsApp dumps: photos/<category>, memes/, screenshots/, docs/",
    ),
}
DEFAULT_PROFILE = BUILTIN_PROFILES["default"]
LOCAL_PROFILE_FILE = ".sortling.json"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "other"


def profile_from_dict(raw: dict, name: str, source: str, cfg: dict) -> Profile:
    """A profile object: {"profile": "<built-in or saved name>"} or {"folders": {name: description}}."""
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: profile must be a JSON object")
    if "profile" in raw and "folders" not in raw:
        return load_profile(str(raw["profile"]), cfg)
    folders = {slugify(k): str(v or "") for k, v in (raw.get("folders") or {}).items()}
    if len(folders) < 2:
        raise ValueError(f"{source}: a custom profile needs at least 2 folders")
    clash = sorted(set(folders) & {*SYSTEM_FOLDERS, NONE_OPTION})
    if clash:
        raise ValueError(f"{source}: reserved folder names: {', '.join(clash)}")
    return Profile(
        name=str(raw.get("name") or name),
        description=str(raw.get("description", "")),
        layout="custom",
        folders=folders,
        source=source,
    )


def load_profile(name: str, cfg: dict) -> Profile:
    if name in ("", "default") or name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name or "default"]
    raw = cfg.get("profiles", {}).get(name)
    if not raw:
        saved = ", ".join(cfg.get("profiles", {})) or "none"
        raise ValueError(f'unknown profile "{name}" (built-in: {", ".join(BUILTIN_PROFILES)}; saved: {saved})')
    return profile_from_dict(raw, name, "config", cfg)


def load_profile_file(path: str, cfg: dict) -> Profile:
    """--profile-file: a JSON profile from a file, or from stdin with '-'."""
    text = sys.stdin.read() if path == "-" else Path(path).expanduser().read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}")
    stem = "stdin" if path == "-" else slugify(Path(path).stem.lstrip("."))
    return profile_from_dict(raw, stem, "stdin" if path == "-" else str(Path(path).expanduser()), cfg)


def local_profile(root: Path, cfg: dict) -> Optional[Profile]:
    """A .sortling.json in the folder being sorted: that folder always sorts its own way."""
    path = root / LOCAL_PROFILE_FILE
    return load_profile_file(str(path), cfg) if path.is_file() else None


def build_questions(profile: Profile, kind: str, has_image: bool) -> dict:
    junk = {"type": "noul", "instructions": JUNK_QUESTION}
    if profile.layout == "custom":
        criteria = dict(profile.folders)
        criteria[NONE_OPTION] = "none of these folders fit"
        return {
            "folder": {"type": "choice", "instructions": "Which folder does this file belong in?", "criteria": criteria},
            "junk": junk,
        }
    image_kind = {"type": "choice", "instructions": "What kind of image is this?", "criteria": IMAGE_KINDS}
    photo = {
        "type": "choice",
        "instructions": "If this is a camera photo, what is it mainly of?",
        "criteria": PHOTO_CATEGORIES,
    }
    if profile.layout == "photos":
        # the lean profile: 3 questions per image, documents are not sub-sorted
        if kind == "image" and has_image:
            return {"kind": image_kind, "junk": junk, "photo": photo}
        return {"junk": junk}
    category = {
        "type": "choice",
        "instructions": "If this is a document or record, which category of personal file is it?",
        "criteria": CATEGORY_DESCRIPTIONS,
    }
    if kind == "image" and has_image:
        # images first split into documents, screenshots, photos and memes, each with its own tree
        return {
            "kind": image_kind,
            "junk": junk,
            "category": category,
            "topic": {
                "type": "choice",
                "instructions": "If this is a screenshot, what is it mainly about?",
                "criteria": SCREENSHOT_TOPICS,
            },
            "photo": photo,
        }
    return {"category": category, "junk": junk, "core": {"type": "noul", "instructions": CORE_REFERENCE_QUESTION}}


def top_choice(answers: dict, qid: str) -> Tuple[str, float]:
    ans = answers[qid]
    choice = str(ans.get("choice", ""))
    return choice, to_float_0_1(ans.get("probabilities", {}).get(choice, 0.0))


def normalize_category(value: str) -> str:
    v = CATEGORY_ALIASES.get((value or "").strip().lower(), (value or "").strip().lower())
    return v if v in CATEGORY_DESCRIPTIONS else "unknown"


@dataclass
class Routed:
    category: str
    decision: str
    folder: str
    confidence: float
    reason: str
    # a hybrid run sends these to the second, thorough pass
    uncertain: bool = False


def route(profile: Profile, answers: dict, th: Thresholds) -> Routed:
    junk_p = to_float_0_1(answers["junk"].get("noul"))
    # junk the model half-believes: not deleted, but worth a second look
    junkish = 0.5 <= junk_p < th.delete_min_prob

    if profile.layout == "custom":
        folder, p = top_choice(answers, "folder")
        reason = f"decision:{folder}={p:.2f},junk={junk_p:.2f}"
        if junk_p >= th.delete_min_prob:
            return Routed("junk", "delete_candidate", DELETE_FOLDER, junk_p, reason)
        if folder not in profile.folders or p < th.min_confidence:
            return Routed("unknown", "review", REVIEW_FOLDER, p, reason + " | low-confidence->review", True)
        return Routed(folder, "keep", folder, p, reason, junkish)

    if "kind" in answers:
        image_kind, kind_p = top_choice(answers, "kind")
        reason = f"decision:{image_kind}={kind_p:.2f},junk={junk_p:.2f}"
        if junk_p >= th.delete_min_prob:
            return Routed(f"{image_kind}:junk", "delete_candidate", DELETE_FOLDER, junk_p, reason)
        if kind_p < th.min_confidence:
            # can't tell whether it is a record; a person should look
            return Routed("unknown", "review", REVIEW_FOLDER, kind_p, reason + " | kind-unsure->review", True)
        if image_kind == "meme":
            return Routed("meme", "keep", MEMES_FOLDER, kind_p, reason, junkish)
        if image_kind == "photo":
            sub, sub_p = top_choice(answers, "photo")
            reason += f",photo:{sub}={sub_p:.2f}"
            if sub not in PHOTO_CATEGORIES or sub_p < th.screenshot_min_confidence:
                return Routed("photo:other", "keep", f"{PHOTOS_FOLDER}/other", kind_p, reason + " | ->other", True)
            return Routed(f"photo:{sub}", "keep", f"{PHOTOS_FOLDER}/{sub}", min(kind_p, sub_p), reason, junkish)
        if image_kind == "screenshot":
            if profile.layout == "photos":
                return Routed("screenshot", "keep", "screenshots", kind_p, reason, junkish)
            topic, topic_p = top_choice(answers, "topic")
            reason += f",topic:{topic}={topic_p:.2f}"
            # misfiled screenshots cost little: unsure ones go to screenshots/other, not the review inbox
            if topic not in SCREENSHOT_TOPICS or topic_p < th.screenshot_min_confidence:
                reason += f" | topic<{th.screenshot_min_confidence:.2f}->other"
                return Routed("screenshot:other", "keep", "screenshots/other", kind_p, reason, True)
            return Routed(f"screenshot:{topic}", "keep", f"screenshots/{topic}", min(kind_p, topic_p), reason, junkish)
        # image_kind == "document": a scanned or screenshotted record, filed with the docs below
        if profile.layout == "photos":
            return Routed("document", "keep", "docs", kind_p, reason, junkish)
    elif profile.layout == "photos":
        # PDFs in the photos profile are not sub-sorted
        reason = f"decision:junk={junk_p:.2f}"
        if junk_p >= th.delete_min_prob:
            return Routed("document:junk", "delete_candidate", DELETE_FOLDER, junk_p, reason)
        return Routed("document", "keep", "docs", 1.0 - junk_p, reason, junkish)

    # documents: the folder comes from the category; yes/no answers pick delete or learning core vs archive
    cat_choice, cat_p = top_choice(answers, "category")
    category = normalize_category(cat_choice)
    core_p = to_float_0_1(answers["core"].get("noul")) if "core" in answers else 1.0
    doc_reason = f"{category}={cat_p:.2f}" + (f",core={core_p:.2f}" if "core" in answers else "")
    reason = (reason + "," + doc_reason) if "kind" in answers else f"decision:{doc_reason},junk={junk_p:.2f}"
    if junk_p >= th.delete_min_prob:
        return Routed(category, "delete_candidate", DELETE_FOLDER, junk_p, reason)
    if category == "unknown" or cat_p < th.min_confidence:
        return Routed(category, "review", REVIEW_FOLDER, cat_p, reason + " | low-confidence->review", True)
    if category == "spam":
        return Routed(category, "review", REVIEW_FOLDER, min(cat_p, 1.0 - junk_p), reason + f" | junk<{th.delete_min_prob:.2f}->review", True)
    if category == "learning" and core_p < 0.5:
        return Routed(category, "archive", LEARNING_ARCHIVE_FOLDER, cat_p, reason, junkish)
    return Routed(category, "keep", DOC_FOLDERS[category], cat_p, reason, junkish)


def fallback_classify(profile: Profile, filename: str, text_excerpt: str) -> Routed:
    """Keyword rules for when no model is reachable. Custom profiles just go to review."""
    return Routed(*keyword_rules(profile, filename, text_excerpt))


def keyword_rules(profile: Profile, filename: str, text_excerpt: str) -> Tuple[str, str, str, float, str]:
    if not profile.builtin:
        return ("unknown", "review", REVIEW_FOLDER, 0.0, "fallback:no-model")
    hay = f"{filename} {text_excerpt}".lower()

    def has(*tokens: str) -> bool:
        return any(tok in hay for tok in tokens)

    if has("congrats on winning", "contest", "promo", "unsubscribe", "marketing"):
        return ("spam", "delete_candidate", DELETE_FOLDER, 0.65, "keyword:spam")
    if has("study permit", "passport", "oci", "visa", "pr card"):
        return ("personal", "keep", DOC_FOLDERS["personal"], 0.78, "keyword:identity")
    if has("invoice", "statement", "receipt", "tangerine", "cibc", "tax", "w8", "payment"):
        return ("finance", "keep", DOC_FOLDERS["finance"], 0.73, "keyword:finance")
    if has("resume", "employment", "offer letter", "reference check", "certificate"):
        return ("personal", "keep", DOC_FOLDERS["personal"], 0.74, "keyword:personal")
    if has("blood report", "prescription", "health", "medical", "lab"):
        return ("medical", "keep", DOC_FOLDERS["medical"], 0.75, "keyword:medical")
    if has("oracle", "supply chain", "ebs", "oaf", "doc id", "plsql"):
        return ("work", "archive", DOC_FOLDERS["work"], 0.72, "keyword:work")
    if has("lecture", "cheat sheet", "numpy", "pandas", "machine learning", "deep learning"):
        return ("learning", "archive", LEARNING_ARCHIVE_FOLDER, 0.71, "keyword:learning")
    if has("screenshot", "screen shot", "screen_shot"):
        return ("screenshot:other", "keep", "screenshots/other", 0.5, "keyword:screenshot")
    if has("flight", "booking", "ticket", "map", "transit"):
        return ("travel", "review", REVIEW_FOLDER, 0.61, "keyword:travel")
    return ("unknown", "review", REVIEW_FOLDER, 0.35, "fallback:uncertain")


# ---------------------------------------------------------------------------
# Files: discovery, duplicates, text and image extraction
# ---------------------------------------------------------------------------


@dataclass
class DocRecord:
    file: str
    kind: str
    source_relpath: str
    sha1: str
    duplicate_of: str
    extractor: str
    text_chars: int
    used_vision: bool
    classifier: str
    standardized_category: str
    keep_decision: str
    target_folder: str
    confidence: float
    reason: str
    action: str
    result: str = ""
    error: str = ""
    second_pass: str = ""


def file_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        return "pdf"
    if ext in IMAGE_EXTS:
        return "image"
    return ""


def is_in_sorted_folder(root: Path, file_path: Path, sorted_roots: Sequence[str]) -> bool:
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        return False
    return len(rel.parts) > 1 and rel.parts[0] in sorted_roots


def list_source_files(root: Path, recursive: bool, kinds: Sequence[str], sorted_roots: Sequence[str]) -> List[Path]:
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        p
        for p in candidates
        if p.is_file() and file_kind(p) in kinds and not is_in_sorted_folder(root, p, sorted_roots)
    )


def count_supported(root: Path) -> Dict[str, int]:
    counts = {"pdf": 0, "image": 0}
    try:
        for p in root.iterdir():
            k = file_kind(p) if p.is_file() else ""
            if k:
                counts[k] += 1
    except OSError:
        pass
    return counts


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_penalty(path: Path, source_set: set, root: Path) -> Tuple[int, int, int, int, str]:
    rel = str(path.relative_to(root))
    n = path.name.lower()
    in_source = 1 if path in source_set else 0
    deleteish = (DELETE_FOLDER, DUPLICATES_FOLDER, "98_Delete", "99_Duplicates")
    in_deleteish = 1 if rel.split("/", 1)[0] in deleteish else 0
    copyish = 1 if (" copy" in n or re.search(r"\(\d+\)\.[a-z0-9]+$", n) is not None) else 0
    numericish = 1 if re.match(r"^\d{7,}", n) else 0
    return (in_source, in_deleteish, copyish + numericish, len(n), n)


def build_duplicate_lookup(
    root: Path, source_files: Sequence[Path], kinds: Sequence[str], show_progress: bool, width: int
) -> Dict[Path, Path]:
    """Exact duplicates by content across the whole tree. The kept copy prefers already-sorted, non-'copy' names."""
    source_set = set(source_files)
    scan_files = sorted(p for p in root.rglob("*") if p.is_file() and file_kind(p) in kinds)
    hash_groups: Dict[str, List[Path]] = {}
    progress = ProgressBar(total=len(scan_files), label="Hash scan", enabled=show_progress, width=width)
    for idx, p in enumerate(scan_files, start=1):
        hash_groups.setdefault(sha1_file(p), []).append(p)
        progress.update(idx, p.name)
    progress.finish("hash complete")
    duplicate_of: Dict[Path, Path] = {}
    for group in hash_groups.values():
        if len(group) < 2:
            continue
        primary = sorted(group, key=lambda p: file_penalty(p, source_set, root))[0]
        for p in group:
            if p != primary and p in source_set:
                duplicate_of[p] = primary
    return duplicate_of


def extract_ascii_pdf_strings(path: Path, max_chars: int) -> Tuple[str, str]:
    try:
        raw = path.read_bytes()[: 6 * 1024 * 1024]
    except Exception as exc:
        return "", f"ascii-scan:{exc}"
    out: List[str] = []
    total = 0
    for m in re.findall(rb"[ -~]{6,}", raw):
        s = m.decode("ascii", errors="ignore").strip()
        if not s:
            continue
        out.append(s)
        total += len(s) + 1
        if total >= max_chars:
            break
    joined = "\n".join(out).strip()
    if not joined:
        return "", "ascii-scan:no-text"
    return joined[:max_chars], "ascii-scan"


def extract_text_from_pdf(path: Path, max_pages: int, max_chars: int) -> Tuple[str, str]:
    errors: List[str] = []
    for module, reader_name in (("pypdf", "PdfReader"), ("PyPDF2", "PdfReader")):
        try:
            reader_cls = getattr(__import__(module), reader_name)
            reader = reader_cls(str(path))
            chunks = []
            for i in range(min(len(reader.pages), max_pages)):
                text = reader.pages[i].extract_text() or ""
                if text.strip():
                    chunks.append(text.strip())
                if len("\n\n".join(chunks)) >= max_chars:
                    break
            joined = "\n\n".join(chunks)
            if joined.strip():
                return joined[:max_chars], module
            errors.append(f"{module}:no-text")
        except ImportError:
            errors.append(f"{module}:missing")
        except Exception as exc:
            errors.append(f"{module}:{exc}")

    if shutil.which("pdftotext"):
        try:
            cmd = ["pdftotext", "-f", "1", "-l", str(max_pages), "-layout", str(path), "-"]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout[:max_chars], "pdftotext"
            errors.append(f"pdftotext:rc{proc.returncode}")
        except Exception as exc:
            errors.append(f"pdftotext:{exc}")
    else:
        errors.append("pdftotext:missing")

    text, source = extract_ascii_pdf_strings(path, max_chars=max_chars)
    if text.strip():
        return text[:max_chars], source
    errors.append(source)
    return "", ";".join(errors)


def render_first_page_png_base64(path: Path, max_dim: int = 1100) -> Tuple[Optional[str], str]:
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(path))
        if doc.page_count < 1:
            doc.close()
            return None, "fitz:no-pages"
        page = doc.load_page(0)
        scale = min(max_dim / max(page.rect.width, 1.0), max_dim / max(page.rect.height, 1.0))
        scale = max(0.7, min(scale, 2.0))
        raw = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")
        doc.close()
        return base64.b64encode(raw).decode("ascii"), "fitz"
    except ImportError:
        pass
    except Exception as exc:
        return None, f"fitz:{exc}"

    if shutil.which("pdftoppm"):
        try:
            # no output root: pdftoppm writes to stdout (a "-" root writes a file named "-.png")
            cmd = ["pdftoppm", "-png", "-f", "1", "-l", "1", "-singlefile", "-scale-to", str(max_dim), str(path)]
            proc = subprocess.run(cmd, check=False, capture_output=True)
            if proc.returncode == 0 and proc.stdout:
                return base64.b64encode(proc.stdout).decode("ascii"), "pdftoppm"
            return None, f"pdftoppm:rc{proc.returncode}"
        except Exception as exc:
            return None, f"pdftoppm:{exc}"

    if shutil.which("qlmanage"):
        try:
            with tempfile.TemporaryDirectory(prefix="sortling_preview_") as tmp:
                cmd = ["qlmanage", "-t", "-s", str(max_dim), "-o", tmp, str(path)]
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if proc.returncode != 0:
                    return None, f"qlmanage:rc{proc.returncode}"
                previews = sorted(Path(tmp).glob("*.png"))
                if not previews:
                    return None, "qlmanage:no-preview"
                return base64.b64encode(previews[0].read_bytes()).decode("ascii"), "qlmanage"
        except Exception as exc:
            return None, f"qlmanage:{exc}"

    return None, "fitz:missing;pdftoppm:missing;qlmanage:missing"


def load_image_base64(path: Path) -> Tuple[Optional[str], str, str]:
    """Return (base64, mime, source). Formats llama.cpp can't decode are converted to PNG."""
    mime = NATIVE_IMAGE_MIME.get(path.suffix.lower())
    if mime:
        try:
            return base64.b64encode(path.read_bytes()).decode("ascii"), mime, "file"
        except Exception as exc:
            return None, "", f"read:{exc}"
    # sips ships with macOS; ImageMagick elsewhere
    if shutil.which("sips") and platform.system() == "Darwin":
        with tempfile.TemporaryDirectory(prefix="sortling_img_") as tmp:
            out = Path(tmp) / "img.png"
            proc = subprocess.run(["sips", "-s", "format", "png", str(path), "--out", str(out)], check=False, capture_output=True)
            if proc.returncode == 0 and out.exists():
                return base64.b64encode(out.read_bytes()).decode("ascii"), "image/png", "sips"
    tool = shutil.which("magick") or shutil.which("convert")
    if not tool:
        return None, "", "convert:missing"
    try:
        # [0] takes the first frame of animated or multi-page files
        proc = subprocess.run([tool, f"{path}[0]", "png:-"], check=False, capture_output=True)
        if proc.returncode == 0 and proc.stdout:
            return base64.b64encode(proc.stdout).decode("ascii"), "image/png", "magick"
        return None, "", f"magick:rc{proc.returncode}"
    except Exception as exc:
        return None, "", f"magick:{exc}"


def build_state(filename: str, kind: str, text_excerpt: str, has_image: bool) -> str:
    if kind == "pdf":
        what = "PDF document" + (", first page shown above" if has_image else "")
    else:
        what = "image file (screenshot, photo or scan)" + (", shown above" if has_image else ", not shown")
    lines = [f"Filename: {filename}", f"File type: {what}"]
    if text_excerpt:
        lines += ["Extracted text (may be noisy):", text_excerpt]
    elif kind == "pdf":
        lines.append("Extracted text: none")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def to_float_0_1(value: object) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f < 0.0 else 1.0 if f > 1.0 else f


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError("Unbalanced JSON object in model output")


def is_timeout_like_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError) or (reason and "timed out" in str(reason).lower()):
            return True
    txt = str(exc).lower()
    return "timed out" in txt or "timeout" in txt


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Planning, applying, undo
# ---------------------------------------------------------------------------


@dataclass
class RunOptions:
    root: Path
    kinds: Tuple[str, ...]
    recursive: bool
    profile: Profile
    backend: Optional[Backend]
    thresholds: Thresholds
    # hybrid: uncertain first-pass answers get a second look from this (usually thorough) backend
    second: Optional[Backend] = None
    vision_mode: str = "auto"
    vision_min_text_chars: int = 400
    max_pages: int = 3
    max_chars: int = 7000
    timeout_s: int = 120
    retry_timeouts: int = 1
    retry_backoff_sec: float = 1.2
    max_consecutive_timeouts: int = 3
    delete_exact_duplicates: bool = False
    hard_delete: bool = False
    progress: bool = True
    progress_width: int = 32
    face: bool = True
    status: str = ""
    second_status: str = ""


def sorted_roots(profile: Profile) -> Tuple[str, ...]:
    return (*profile.roots(), *SYSTEM_FOLDERS, *LEGACY_TARGET_FOLDERS)


def _check(backend: Backend, timeout_s: int) -> str:
    """Empty string when the backend answers, otherwise why it can't."""
    try:
        # first call may wait for a router to load the model
        backend.precheck(timeout_s=max(timeout_s, 60))
        return ""
    except Exception as exc:
        detail = f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else str(exc)[:80]
        return f"{type(exc).__name__}:{detail}"


def precheck_backend(opts: RunOptions) -> None:
    """Disable backends that can't answer, recording why. A missing first pass falls back to keywords."""
    if opts.backend is None:
        opts.status = "fallback:disabled"
    else:
        wait = FaceWait(f"waking {opts.backend.name}...", opts.progress and opts.face)
        why = _check(opts.backend, opts.timeout_s)
        wait.close(f"{opts.backend.name} {'is awake' if not why else 'did not answer'}")
        opts.status = f"fallback:{opts.backend.name}-unavailable:{why}" if why else f"{opts.backend.mode}:{opts.backend.name}"
        if why:
            opts.backend = None
    if opts.second is not None:
        why = _check(opts.second, opts.timeout_s) if opts.backend is not None else "first pass unavailable"
        opts.second_status = f"off:{opts.second.name}-unavailable:{why}" if why else f"{opts.second.mode}:{opts.second.name}"
        if why:
            opts.second = None


@dataclass
class FileInput:
    path: Path
    kind: str
    text: str
    extractor: str
    image_b64: Optional[str]
    image_mime: str
    vision_source: str


def prepare_input(path: Path, opts: RunOptions) -> FileInput:
    kind = file_kind(path)
    image_b64: Optional[str] = None
    image_mime = "image/png"
    vision_source = ""
    if kind == "pdf":
        text, extractor = extract_text_from_pdf(path, opts.max_pages, opts.max_chars)
        # ascii-scan output on scanned PDFs is mostly PDF syntax, not document text
        weak_text = extractor.startswith("ascii-scan")
        use_vision = opts.vision_mode == "always" or (
            opts.vision_mode == "auto" and (weak_text or len(text) < opts.vision_min_text_chars)
        )
        if use_vision:
            image_b64, vision_source = render_first_page_png_base64(path)
            if image_b64 is None:
                extractor = f"{extractor};{vision_source}"
        if weak_text and image_b64 is not None:
            text = ""
    else:
        # images are classified from the pixels; there is no OCR step
        text, extractor = "", "image"
        if opts.vision_mode != "off":
            image_b64, image_mime, vision_source = load_image_base64(path)
            if image_b64 is None:
                extractor = f"image;{vision_source}"
    return FileInput(path, kind, text, extractor, image_b64, image_mime, vision_source)


def classify(backend: Backend, fi: FileInput, opts: RunOptions) -> Tuple[Routed, Optional[Exception]]:
    """Ask one backend about one file, with timeout retries. On failure, keyword rules decide."""
    sent_image = fi.image_b64 if backend.vision else None
    questions = build_questions(opts.profile, fi.kind, sent_image is not None)
    state = build_state(fi.path.name, fi.kind, fi.text, sent_image is not None)
    images = [f"data:{fi.image_mime};base64,{sent_image}"] if sent_image else []
    err: Optional[Exception] = None
    for attempt in range(opts.retry_timeouts + 1):
        try:
            answers, why = backend.decide(state, images, questions, opts.timeout_s)
            routed = route(opts.profile, answers, opts.thresholds)
            if why:
                routed.reason = f"{routed.reason} | {why}"
            return routed, None
        except Exception as exc:
            err = exc
            if is_timeout_like_error(exc) and attempt < opts.retry_timeouts:
                time.sleep(max(0.0, opts.retry_backoff_sec) * (attempt + 1))
                continue
            break
    routed = fallback_classify(opts.profile, fi.path.name, fi.text)
    routed.reason = f"{routed.reason} | model_error:{err}"
    return routed, err


def plan(opts: RunOptions, on_file: Optional[Callable[[DocRecord], None]] = None) -> List[DocRecord]:
    root = opts.root
    source_files = list_source_files(root, opts.recursive, opts.kinds, sorted_roots(opts.profile))
    if not source_files:
        return []

    stage_note("Stage 1/3: scanning duplicates", opts.progress)
    duplicate_of = build_duplicate_lookup(root, source_files, opts.kinds, opts.progress, opts.progress_width)

    stage_note("Stage 2/3: extracting + classifying files", opts.progress)
    bar = make_bar(len(source_files), "Classify", opts.progress, opts.progress_width, opts.face)
    consecutive_timeouts = 0
    records: List[DocRecord] = []
    uncertain: List[Tuple[DocRecord, FileInput]] = []
    for idx, path in enumerate(source_files, start=1):
        fi = prepare_input(path, opts)
        backend = opts.backend
        if backend is not None and fi.kind == "image" and not backend.vision:
            backend = None  # a text-only model would only see the filename
            classifier = f"fallback:{opts.backend.name}-text-only" if opts.backend else "fallback"
        else:
            classifier = opts.status if backend is None else f"{backend.mode}:{backend.name}"

        is_uncertain = False
        if backend is not None:
            routed, err = classify(backend, fi, opts)
            is_uncertain = err is None and routed.uncertain
            if err is not None:
                classifier = f"fallback:{type(err).__name__}"
                if is_timeout_like_error(err):
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= max(1, opts.max_consecutive_timeouts):
                        opts.backend = None
                        opts.status = "fallback:timeouts"
            else:
                consecutive_timeouts = 0
        else:
            routed = fallback_classify(opts.profile, path.name, fi.text)

        rec = make_record(root, fi, routed, classifier, duplicate_of)
        assign_action(rec, opts.delete_exact_duplicates, opts.hard_delete)
        records.append(rec)
        if is_uncertain and not rec.duplicate_of:
            uncertain.append((rec, fi))
        bar.update(idx, f"{path.name} -> {rec.target_folder}")
        if isinstance(bar, FaceBar):
            bar.react(mood_for(rec), rec.target_folder)
        if on_file:
            on_file(rec)
    bar.finish("classification complete")

    if opts.second is not None and uncertain:
        second_pass(opts, uncertain)
    return records


def second_pass(opts: RunOptions, uncertain: Sequence[Tuple[DocRecord, FileInput]]) -> None:
    """Hybrid: re-ask the uncertain files with the second backend; its answer wins."""
    second = opts.second
    assert second is not None
    stage_note(f"Stage 2b/3: second look at {len(uncertain)} uncertain files ({second.mode}:{second.name})", opts.progress)
    bar = make_bar(len(uncertain), "Re-check", opts.progress, opts.progress_width, opts.face)
    for idx, (rec, fi) in enumerate(uncertain, start=1):
        if fi.kind == "image" and not second.vision:
            rec.second_pass = f"skipped:{second.name}-text-only"
        else:
            routed, err = classify(second, fi, opts)
            if err is not None:
                rec.second_pass = f"failed:{type(err).__name__}"
            else:
                first = rec.target_folder
                rec.second_pass = f"{second.mode}:{second.name}:{first}->{routed.folder}"
                rec.classifier = f"hybrid:{rec.classifier}>{second.mode}:{second.name}"
                rec.standardized_category = routed.category
                rec.keep_decision = routed.decision
                rec.target_folder = routed.folder
                rec.confidence = round(routed.confidence, 4)
                rec.reason = (f"{routed.reason} | first:{first}" + (f" | vision:{fi.vision_source}" if fi.image_b64 else ""))[:260]
                assign_action(rec, opts.delete_exact_duplicates, opts.hard_delete)
        bar.update(idx, f"{rec.file} -> {rec.target_folder}")
        if isinstance(bar, FaceBar):
            bar.react(mood_for(rec), rec.target_folder)
    bar.finish("second pass complete")


def make_record(root: Path, fi: FileInput, routed: Routed, classifier: str, duplicate_of: Dict[Path, Path]) -> DocRecord:
    reason = routed.reason
    if fi.image_b64 is not None and fi.vision_source:
        reason = f"{reason} | vision:{fi.vision_source}"
    return DocRecord(
        file=fi.path.name,
        kind=fi.kind,
        source_relpath=str(fi.path.relative_to(root)),
        sha1=sha1_file(fi.path),
        duplicate_of=str(duplicate_of[fi.path].relative_to(root)) if fi.path in duplicate_of else "",
        extractor=fi.extractor,
        text_chars=len(fi.text),
        used_vision=fi.image_b64 is not None,
        classifier=classifier,
        standardized_category=routed.category,
        keep_decision=routed.decision,
        target_folder=routed.folder,
        confidence=round(routed.confidence, 4),
        reason=reason[:260],
        action="",
    )


def assign_action(record: DocRecord, delete_exact_duplicates: bool, hard_delete: bool) -> None:
    if record.duplicate_of:
        if delete_exact_duplicates:
            record.action = "DELETE_EXACT_DUPLICATE"
        else:
            record.target_folder = DUPLICATES_FOLDER
            record.action = "MOVE_TO_DUPLICATES"
        return
    if record.keep_decision == "delete_candidate" and hard_delete:
        record.action = "DELETE_CANDIDATE"
        return
    record.action = "MOVE"


def unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    i = 2
    while True:
        candidate = dest.with_name(f"{dest.stem} ({i}){dest.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def apply_record(root: Path, rec: DocRecord) -> None:
    src = root / rec.source_relpath
    if not src.exists():
        rec.error = "source-missing"
        rec.result = "ERROR"
        return
    if rec.action in {"DELETE_EXACT_DUPLICATE", "DELETE_CANDIDATE"}:
        src.unlink()
        rec.result = "DELETED"
        return
    dest_dir = root / rec.target_folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_destination(dest_dir / src.name)
    shutil.move(str(src), str(dest))
    rec.result = f"MOVED:{dest.relative_to(root)}"


def apply_all(opts: RunOptions, records: Sequence[DocRecord]) -> None:
    stage_note("Stage 3/3: applying actions", opts.progress)
    bar = make_bar(len(records), "Apply", opts.progress, opts.progress_width, opts.face)
    for idx, rec in enumerate(records, start=1):
        try:
            apply_record(opts.root, rec)
        except Exception as exc:
            rec.error = str(exc)
            rec.result = "ERROR"
        bar.update(idx, f"{rec.action} {rec.file}")
    bar.finish("apply complete")


def manifest_payload(opts: RunOptions, records: Sequence[DocRecord], dry_run: bool) -> dict:
    return {
        "generated_at": datetime.now().isoformat(),
        "sortling_version": __version__,
        "mode": "dry_run" if dry_run else "apply",
        "root": str(opts.root),
        "profile": opts.profile.name,
        "profile_source": opts.profile.source,
        "profile_roots": list(opts.profile.roots()),
        "classifier": opts.status,
        "second_pass": opts.second_status,
        "summary": {
            "files": len(records),
            "by_folder": {f: sum(1 for r in records if r.target_folder == f) for f in sorted({r.target_folder for r in records})},
        },
        "records": [asdict(r) for r in records],
    }


def write_logs(output_dir: Path, opts: RunOptions, records: Sequence[DocRecord], dry_run: bool) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    csv_path = output_dir / f"sortling_manifest_{stamp}.csv"
    json_path = output_dir / f"sortling_manifest_{stamp}.json"
    fieldnames = [f.name for f in DocRecord.__dataclass_fields__.values()]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))
    payload = manifest_payload(opts, records, dry_run)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def write_review_note(root: Path, records: Sequence[DocRecord], apply: bool) -> Path:
    review_dir = root / REVIEW_FOLDER
    review_dir.mkdir(parents=True, exist_ok=True)
    note = review_dir / "REVIEW_LIST.txt"
    if apply:
        files = sorted(p.name for p in review_dir.iterdir() if p.is_file() and file_kind(p))
        lines = ["Manual review list (current snapshot)", f"Total files in {REVIEW_FOLDER}: {len(files)}", ""]
        lines.extend(f"- {name}" for name in files)
    else:
        candidates = sorted(rec.file for rec in records if rec.target_folder == REVIEW_FOLDER)
        lines = ["Manual review list (dry-run projection)", f"Would route to {REVIEW_FOLDER}: {len(candidates)}", ""]
        lines.extend(f"- {name}" for name in candidates)
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note


def undo(manifest: Path) -> int:
    """Move files from an applied manifest back where they came from. Deleted files can't return."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("mode") != "apply" or "root" not in data:
        print(f"error: {manifest} is not an applied sortling manifest", file=sys.stderr)
        return 2
    root = Path(data["root"])
    restored = skipped = gone = 0
    for rec in data.get("records", []):
        result = str(rec.get("result", ""))
        if result == "DELETED":
            gone += 1
            continue
        if not result.startswith("MOVED:"):
            continue
        dest, src = root / result[len("MOVED:"):], root / rec["source_relpath"]
        if not dest.exists() or src.exists():
            print(f"skip: {rec['source_relpath']} ({'moved again or removed' if not dest.exists() else 'original path is taken'})")
            skipped += 1
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dest), str(src))
        restored += 1
    # remove folders the run created, if they are now empty
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            if d != root and not any(d.iterdir()) and d.relative_to(root).parts[0] in (*SYSTEM_FOLDERS, *data.get("profile_roots", DEFAULT_PROFILE_ROOTS)):
                d.rmdir()
        except OSError:
            pass
    print(f"UNDO: restored {restored}, skipped {skipped}, deleted-not-restorable {gone}")
    return 0


def summarize(records: Sequence[DocRecord]) -> str:
    by_action: Dict[str, int] = {}
    by_folder: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for rec in records:
        by_action[rec.action] = by_action.get(rec.action, 0) + 1
        by_folder[rec.target_folder] = by_folder.get(rec.target_folder, 0) + 1
        by_source[rec.classifier] = by_source.get(rec.classifier, 0) + 1
    parts = ["Action counts:", *(f"- {k}: {v}" for k, v in sorted(by_action.items()))]
    parts += ["Target folder counts:", *(f"- {k}: {v}" for k, v in sorted(by_folder.items()))]
    parts += ["Classifier source:", *(f"- {k}: {v}" for k, v in sorted(by_source.items()))]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        raise KeyboardInterrupt
    return value or default


def ask_choice(title: str, options: Sequence[Tuple[str, object]], default: int = 1) -> object:
    print(color(title, "1"))
    for i, (label, _) in enumerate(options, start=1):
        print(f"  {i}) {label}")
    while True:
        raw = ask("  choose", str(default))
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            print()
            return options[int(raw) - 1][1]
        print("  enter a number from the list")


def ask_yes(prompt: str, default: bool = False) -> bool:
    raw = ask(f"{prompt} ({'Y/n' if default else 'y/N'})", "").lower()
    return default if not raw else raw.startswith("y")


def screenshot_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        try:
            proc = subprocess.run(
                ["defaults", "read", "com.apple.screencapture", "location"], capture_output=True, text=True, check=False
            )
            loc = proc.stdout.strip()
            if proc.returncode == 0 and loc:
                return Path(loc).expanduser()
        except OSError:
            pass
        return Path.home() / "Desktop"  # macOS default
    candidates: List[Path] = []
    try:
        proc = subprocess.run(["xdg-user-dir", "PICTURES"], capture_output=True, text=True, check=False)
        pictures = Path(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip() else None
        # xdg-user-dir prints $HOME when no Pictures dir is configured
        if pictures and pictures != Path.home():
            candidates.append(pictures / "Screenshots")
    except OSError:
        pass
    candidates += [Path.home() / "Pictures" / "Screenshots", Path.home() / "Screenshots"]
    return next((c for c in candidates if c.is_dir()), candidates[0])


def describe_counts(path: Path) -> str:
    if not path.is_dir():
        return "not found"
    c = count_supported(path)
    return f"{c['image']} images, {c['pdf']} PDFs"


def create_profile_wizard(cfg_file: Path) -> Profile:
    print(color("New profile", "1"))
    print("  Name each folder and describe what goes in it in a few words.")
    print("  The description is what the model reads, so be specific. Blank name to finish.\n")
    name = slugify(ask("  profile name", "my-folders"))
    folders: Dict[str, str] = {}
    while True:
        folder = ask(f"  folder {len(folders) + 1} name (blank to finish)")
        if not folder:
            if len(folders) >= 2:
                break
            print("  add at least 2 folders")
            continue
        desc = ask(f"    what goes in '{slugify(folder)}'")
        folders[slugify(folder)] = desc
    profile = Profile(name=name, description="custom", folders=folders)
    print(f"\n  uncertain files go to {REVIEW_FOLDER}/, junk to {DELETE_FOLDER}/")
    if ask_yes(f"  save '{name}' to {cfg_file}", True):
        user = read_user_config(cfg_file)
        user.setdefault("profiles", {})[name] = {"description": "custom", "folders": folders}
        save_config(cfg_file, user)
        print(f"  {icon('ok')} saved")
    print()
    return profile


def interactive(args: argparse.Namespace, cfg: dict, cfg_file: Path) -> int:
    if not args.no_banner:
        print_banner()

    # 1. what to clean up
    shots = screenshot_dir()
    downloads = Path.home() / "Downloads"
    desktop = Path.home() / "Desktop"
    targets: List[Tuple[str, object]] = [
        (f"Screenshots   {shots}  ({describe_counts(shots)})", (shots, ("image",))),
        (f"Downloads     {downloads}  ({describe_counts(downloads)})", (downloads, ("pdf", "image"))),
    ]
    if desktop != shots:
        targets.append((f"Desktop       {desktop}  ({describe_counts(desktop)})", (desktop, ("pdf", "image"))))
    targets.append(("Another folder...", None))
    picked = ask_choice("What do you want to tidy?", targets)
    if picked is None:
        root = Path(ask("  folder path")).expanduser()
        kinds: Tuple[str, ...] = ("pdf", "image")
        print()
    else:
        root, kinds = picked  # type: ignore[misc]
    root = root.resolve()
    if not root.is_dir():
        print(f"error: not a folder: {root}", file=sys.stderr)
        return 2

    # 2. how to sort
    profiles: List[Tuple[str, object]] = []
    try:
        here = local_profile(root, cfg)
    except (OSError, ValueError) as exc:
        print(f"  {icon('warn')} ignoring {root / LOCAL_PROFILE_FILE}: {exc}\n")
        here = None
    if here is not None:
        profiles.append((f"{LOCAL_PROFILE_FILE:<9} this folder's own profile: {', '.join(here.roots())}", here))
    for bname, builtin in BUILTIN_PROFILES.items():
        profiles.append((f"{bname:<9} {builtin.description}", builtin))
    for pname in cfg.get("profiles", {}):
        try:
            saved = load_profile(pname, cfg)
            profiles.append((f"{pname:<9} {', '.join(saved.roots())}", saved))
        except ValueError as exc:
            print(f"  {icon('warn')} skipping saved profile {pname}: {exc}")
    profiles.append(("create your own...", None))
    picked_profile = ask_choice("How should files be sorted?", profiles)
    profile = create_profile_wizard(cfg_file) if picked_profile is None else picked_profile
    assert isinstance(profile, Profile)

    # 3. which model
    names = list(cfg["backends"])
    print(color("Which model?", "1"))
    print(f"  {icon('fast')} fast      decision model: one quick pass per file, real probabilities. Best default.")
    print(f"  {icon('thorough')} thorough  a general chat LLM: explains each file, slower, confidence is self-reported,")
    print("               so the review/delete thresholds are less trustworthy.")
    print(f"  {icon('hybrid')} hybrid    fast on everything, thorough only on the files fast is unsure about.")
    default_idx = next((i for i, n in enumerate(names, start=1) if n == cfg.get("default_backend")), 1)
    picked_name = str(ask_choice("", [(describe_entry(n, cfg), n) for n in names], default_idx))
    try:
        backend, second = make_backends(picked_name, cfg)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for b in (backend, second):
        if b is None:
            continue
        if b.key_env and not b.api_key:
            print(f"  {icon('warn')} ${b.key_env} is not set; {b.name} requests may be rejected.")
        if b.hosted:
            host = urllib.parse.urlparse(b.url).hostname
            what = "file text and images" if b.vision else "file text and names"
            scope = "uncertain files' " if b is second else ""
            print(f"  {icon('warn')} {b.name} is hosted: {scope}{what} will be uploaded to {host}.")
            if not ask_yes("  continue", False):
                return 1
    if not backend.vision and "image" in kinds:
        print(f"  {icon('warn')} {backend.name} is text-only: images will be sorted by filename keywords only.")
    print()

    opts = RunOptions(
        root=root,
        kinds=kinds,
        recursive=False,
        profile=profile,
        backend=backend,
        second=second,
        thresholds=Thresholds(args.min_confidence, args.screenshot_min_confidence, args.delete_min_prob),
        timeout_s=args.timeout_s,
        progress=True,
        progress_width=args.progress_width,
        face=not args.no_face,
    )
    precheck_backend(opts)
    if opts.backend is None:
        print(f"  {icon('warn')} model unavailable ({opts.status}); using filename keywords instead.")
        if not ask_yes("  continue anyway", False):
            return 1
    if second is not None and opts.second is None:
        print(f"  {icon('warn')} second pass unavailable ({opts.second_status}); fast results only.")

    # 4. dry run, show, confirm
    records = plan(opts)
    if not records:
        print(f"Nothing to sort in {root}.")
        return 0
    print()
    by_folder: Dict[str, int] = {}
    for rec in records:
        by_folder[rec.target_folder] = by_folder.get(rec.target_folder, 0) + 1
    print(color(f"Plan for {len(records)} files in {root}:", "1"))
    for folder, n in sorted(by_folder.items(), key=lambda kv: (kv[0].startswith("_"), kv[0])):
        print(f"  {n:>4}  -> {folder}/")
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else root / "_sortling"
    _, json_path = write_logs(out_dir, opts, records, dry_run=True)
    print(f"\n  details: {json_path}")
    if not ask_yes("\nMove the files now", False):
        print("No files moved.")
        return 0
    apply_all(opts, records)
    _, json_path = write_logs(out_dir, opts, records, dry_run=False)
    errors = sum(1 for r in records if r.result == "ERROR")
    print(f"\n{icon('ok')} moved {len(records) - errors} files" + (f", {errors} errors" if errors else ""))
    print(f"  undo with: python3 {Path(sys.argv[0]).name} --undo '{json_path}'")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sortling",
        description="Tidy screenshots, downloads and documents with decision models. Run with no arguments for interactive mode.",
    )
    p.add_argument("--root", help="Folder to sort (scripted mode). Without it on a terminal, sortling asks.")
    p.add_argument("-i", "--interactive", action="store_true", help="Force the interactive wizard")
    p.add_argument("--apply", action="store_true", help="Apply moves/deletes. Default is dry-run")
    p.add_argument("--undo", metavar="MANIFEST", help="Move files from an applied manifest back")
    p.add_argument("--types", default="pdf,image", help="Comma-separated file kinds: pdf, image (default: pdf,image)")
    p.add_argument("--recursive-source", action="store_true", help="Also sort files in subfolders (not already-sorted ones)")
    p.add_argument("--profile", default="", help="Sorting profile: default, photos, or one saved in the config")
    p.add_argument("--profile-file", default="", metavar="JSON",
                   help='Profile from a JSON file ("-" for stdin): {"folders": {"name": "what goes in it", ...}}')
    p.add_argument("--backend", default="", help="Backend name from the config (default: config default_backend)")
    p.add_argument("--second-pass", default="", metavar="BACKEND",
                   help="Hybrid: re-check uncertain files with this backend (usually a thorough one)")
    p.add_argument("--json", action="store_true", help="Print the plan as JSON on stdout (for agents and scripts)")
    p.add_argument("--url", default="", help="Override the backend URL")
    p.add_argument("--config", default="", help="Config file (default: $XDG_CONFIG_HOME/sortling/config.json)")
    p.add_argument("--init-config", action="store_true", help="Write a starter config with all backends and exit")
    p.add_argument("--list-backends", action="store_true", help="Show configured backends and exit")
    p.add_argument("--disable-llm", action="store_true", help="Skip all models and use filename keywords")
    p.add_argument("--min-confidence", type=float, default=0.6, help="Below this, documents go to _review")
    p.add_argument("--screenshot-min-confidence", type=float, default=0.4, help="Below this topic probability, screenshots go to screenshots/other")
    p.add_argument("--delete-min-prob", type=float, default=0.85, help="Minimum junk probability for _delete")
    p.add_argument("--delete-exact-duplicates", action="store_true", help="Delete exact duplicates instead of moving them to _duplicates")
    p.add_argument("--hard-delete-delete-candidates", action="store_true", help="Hard-delete junk instead of moving it to _delete")
    p.add_argument("--timeout-s", type=int, default=120, help="Model request timeout (seconds)")
    p.add_argument("--max-pages", type=int, default=3, help="Max PDF pages for text extraction")
    p.add_argument("--max-chars", type=int, default=7000, help="Max text chars sent to the model")
    p.add_argument("--vision-mode", choices=("off", "auto", "always"), default="auto",
                   help="Send images: PDF first-page snapshots (auto: when text is missing or weak) and image files (off: never)")
    p.add_argument("--vision-min-text-chars", type=int, default=400, help="In auto mode, snapshot PDFs with less text than this")
    p.add_argument("--output-dir", default="", help="Manifest directory (default: <root>/_sortling)")
    p.add_argument("--write-review-note", action="store_true", help="Update _review/REVIEW_LIST.txt")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars and stage logs")
    p.add_argument("--no-banner", action="store_true", help="Skip the banner in interactive mode")
    p.add_argument("--no-face", action="store_true", help="Plain progress bar instead of the animated face (also SORTLING_FACE=0)")
    p.add_argument("--progress-width", type=int, default=32, help="Progress bar width")
    p.add_argument("--max-consecutive-timeouts", type=int, default=3, help="Stop using the model after this many timeouts in a row")
    p.add_argument("--retry-timeouts", type=int, default=1, help="Per-file retries on timeouts")
    p.add_argument("--retry-backoff-sec", type=float, default=1.2, help="Base backoff before a retry")
    p.add_argument("--version", action="version", version=f"sortling {__version__}")
    return p.parse_args(argv)


def resolve_backends(args: argparse.Namespace, cfg: dict) -> Tuple[Backend, Optional[Backend]]:
    name = args.backend or cfg.get("default_backend", "")
    return make_backends(name, cfg, url=args.url, second=args.second_pass)


def resolve_profile(args: argparse.Namespace, cfg: dict, root: Path) -> Profile:
    """--profile-file, then --profile, then the folder's .sortling.json, then default."""
    if args.profile_file:
        return load_profile_file(args.profile_file, cfg)
    if args.profile:
        return load_profile(args.profile, cfg)
    return local_profile(root, cfg) or DEFAULT_PROFILE


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg_file = config_path(args.config)

    if args.init_config:
        if cfg_file.exists():
            print(f"config already exists: {cfg_file}")
            return 1
        save_config(cfg_file, DEFAULT_CONFIG)
        print(f"wrote {cfg_file}")
        return 0
    if args.undo:
        return undo(Path(args.undo).expanduser())

    try:
        cfg = load_config(cfg_file)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read config {cfg_file}: {exc}", file=sys.stderr)
        return 2

    if args.list_backends:
        for n in cfg["backends"]:
            mark = "*" if n == cfg.get("default_backend") else " "
            print(f"{mark} {describe_entry(n, cfg)}")
        return 0

    if args.interactive or (not args.root and not args.json and sys.stdin.isatty() and sys.stdout.isatty()):
        try:
            return interactive(args, cfg, cfg_file)
        except KeyboardInterrupt:
            print("\ncancelled, no files moved.")
            return 130

    root = Path(args.root or ".").expanduser().resolve()
    if not root.is_dir():
        print(f"error: invalid root directory: {root}", file=sys.stderr)
        return 2
    kinds = tuple(k.strip() for k in args.types.split(",") if k.strip())
    if not kinds or any(k not in ("pdf", "image") for k in kinds):
        print(f"error: --types must be a comma-separated subset of pdf,image: {args.types}", file=sys.stderr)
        return 2
    try:
        profile = resolve_profile(args, cfg, root)
        backend, second = (None, None) if args.disable_llm else resolve_backends(args, cfg)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    opts = RunOptions(
        root=root,
        kinds=kinds,
        recursive=args.recursive_source,
        profile=profile,
        backend=backend,
        second=second,
        thresholds=Thresholds(args.min_confidence, args.screenshot_min_confidence, args.delete_min_prob),
        vision_mode=args.vision_mode,
        vision_min_text_chars=args.vision_min_text_chars,
        max_pages=args.max_pages,
        max_chars=args.max_chars,
        timeout_s=args.timeout_s,
        retry_timeouts=max(0, args.retry_timeouts),
        retry_backoff_sec=args.retry_backoff_sec,
        max_consecutive_timeouts=args.max_consecutive_timeouts,
        delete_exact_duplicates=args.delete_exact_duplicates,
        hard_delete=args.hard_delete_delete_candidates,
        progress=not args.no_progress,
        progress_width=args.progress_width,
        face=not args.no_face,
    )
    precheck_backend(opts)
    records = plan(opts)
    if not records:
        if args.json:
            print(json.dumps(manifest_payload(opts, [], dry_run=not args.apply)))
        else:
            print(f"No source files ({','.join(kinds)}) found to process.")
        return 0
    if args.apply:
        apply_all(opts, records)
    else:
        stage_note("Stage 3/3: dry-run (no file changes)", opts.progress)
        for rec in records:
            rec.result = "DRY_RUN"

    out_dir = Path(args.output_dir).expanduser() if args.output_dir else root / "_sortling"
    csv_path, json_path = write_logs(out_dir.resolve(), opts, records, dry_run=not args.apply)
    review_note = write_review_note(root, records, args.apply) if args.write_review_note else None

    if args.json:
        # machine-readable plan for agents and scripts: the manifest plus where it was written
        payload = manifest_payload(opts, records, dry_run=not args.apply)
        payload["manifest_json"], payload["manifest_csv"] = str(json_path), str(csv_path)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"MODE: {'APPLY' if args.apply else 'DRY_RUN'}")
    print(f"ROOT: {root}")
    print(f"SOURCE_FILES: {len(records)} ({','.join(kinds)})")
    print(f"PROFILE: {profile.name} ({profile.source})")
    print(f"BACKEND: {backend.describe() if backend else 'none'}")
    if second is not None or opts.second_status:
        print(f"SECOND_PASS: {opts.second_status}")
    print(f"PRECHECK: {opts.status}")
    print(summarize(records))
    print(f"MANIFEST_CSV: {csv_path}")
    print(f"MANIFEST_JSON: {json_path}")
    if review_note:
        print(f"REVIEW_NOTE: {review_note}")
    if args.apply:
        print(f"UNDO: python3 {Path(sys.argv[0]).name} --undo '{json_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
