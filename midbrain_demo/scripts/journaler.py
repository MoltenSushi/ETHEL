#!/usr/bin/env python
# journaler.py — ETHEL midbrain demo: JSONL → SQLite journal DB
#
# This script is a standalone demo version of ETHEL's Stage 3 journaler.
# It is NOT the full ETHEL pipeline. It demonstrates:
#   - discovery of detector JSONL event shards under BASE/events/
#   - ingestion into a simplified SQLite journal database
#   - basic deduplication and clip inference (when segments exist)
#   - caption extraction for speech / speech.suppressed events
#   - optional tools: verify DB, peek JSONL, JSONL→clip checks, tail mode
#
# DIRECTORY CONTRACT (DEMO)
#   BASE/
#     scripts/    - this script (journaler.py)
#     events/     - detector event shards: events/YYYY-MM-DD/events_YYYYMMDD_HH.jsonl
#     db/         - journal database: ethel_journal_demo.db
#     record/     - optional media segment root (usually empty in demo)
#     logs/       - journaler log file: journaler_demo.log
#
# CONFIG FILES
#
#   (optional) indexing_config.toml
#     [indexing]
#       pre_ms = 2000              # milliseconds before event ts to include
#       post_ms = 3000             # milliseconds after event ts to include
#
#
# CLI USAGE (DEMO)
#   python journaler.py
#       # one-shot ingest: all JSONL under BASE/events → BASE/db/ethel_journal_demo.db
#
#   python journaler.py --dry-run
#       # show ingest plan (no DB writes)
#
#   python journaler.py --verify-db
#       # print DB structure + table counts and exit
#
#   python journaler.py --peek-jsonl 20
#       # show first 20 JSONL events + inferred clip windows and exit
#
#   python journaler.py --verify-jsonl
#       # sample-check JSONL → clip path inference and exit
#
#   python journaler.py --tail
#       # tail JSONL shards and continuously ingest new events into the DB
#
# FLAGS
#   --base PATH           : override BASE (default: parent of scripts/)
#   --events-root PATH    : override events/ root (default: BASE/events)
#   --events-glob-jsonl G : glob pattern under events_root (default: '*/*.jsonl')
#   --db-path PATH        : override DB path (default: BASE/db/ethel_journal_demo.db)
#   --clips-dir PATH      : override record/ root for clip inference (default: BASE/record)
#   --camera-stub NAME    : camera stub for segment guessing (default: cam_sim)
#   --container-ext EXT   : container extension for guessed clips (default: mkv)
#   --indexing-config TOML: optional TOML file with [indexing] pre_ms/post_ms
#   --limit N             : maximum JSONL events to process in one run
#   --dedupe-seconds S    : deduplicate events within S seconds around t0_ms
#   --dry-run             : do not write to DB, just print ingest plan
#   --verify-db           : check DB structure/counts and exit
#   --peek-jsonl N        : show N JSONL events + inferred windows and exit
#   --verify-jsonl        : sample JSONL→clip mapping and exit
#   --tail                : tail mode (loop, polling events_root)
#   --interval S          : tail polling interval in seconds (default: 2.0)
#   --log-file PATH       : override log file (default: BASE/logs/journaler_demo.log)
#   --quiet               : disable stdout logging (log file only)
#
# OUTPUT (DEMO)
#   - Creates or updates the SQLite journal DB:
#       db/ethel_journal_demo.db
#   - Populates tables:
#       events   : one row per detector event (motion, track, summary, speech, etc.)
#       clips    : one row per known media segment (only if segments exist)
#       captions : one row per speech / speech.suppressed event with text and timing
#   - All DB timestamps are stored in UTC ISO format; the detector’s local-time JSONL
#     is normalized during ingest so Stage 4 summarizer can reason about days/hours.


from __future__ import annotations
import argparse, sys, itertools, json, hashlib, time, re, sqlite3, logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Iterator, Optional, Tuple, Dict, Any, List

# ----------------------------
# Paths / base
# ----------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent

# ----------------------------
# Logging
# ----------------------------
def setup_logging(log_file: Path | None, verbose: bool = True) -> logging.Logger:
    lg = logging.getLogger("ethel.stage3.demo")
    lg.setLevel(logging.INFO)
    for h in list(lg.handlers):
        lg.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        lg.addHandler(ch)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    return lg

# ----------------------------
# TOML (optional)
# ----------------------------
try:
    import tomllib  # 3.11+
except Exception:     # pragma: no cover
    import tomli as tomllib  # type: ignore

def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)

# ----------------------------
# Hash helpers
# ----------------------------
def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def sha1_short(s: str, n: int = 10) -> str:
    return sha1_hex(s)[:n]

# ----------------------------
# Time helpers
# ----------------------------
def parse_any_iso(ts: str | None) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        s = ts.strip()
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        if len(s) >= 5 and (s[-5] in "+-") and s[-3] != ":":
            s = s[:-2] + ":" + s[-2:]
        return datetime.fromisoformat(s)
    except Exception:
        return None

def to_utc_string(dt: Optional[datetime], fallback: Optional[datetime] = None) -> str:
    if dt is None:
        dt = fallback or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ----------------------------
# Segment bounds (HH-mm)
# ----------------------------
SEG_RX = re.compile(r".*_(\d{2})-(\d{2})\.(?:mkv|mp4|avi|mov)$", re.IGNORECASE)

def quarter_floor(minute: int) -> int:
    return (minute // 15) * 15

def segment_bounds_from_record_name(p: Path) -> Optional[Tuple[datetime, datetime]]:
    """
    record/YYYY-MM-DD/cam_stub_HH-mm.{mkv|mp4|...}
    → returns (segment_start_utc, segment_end_utc)

    For demo, we mostly expect this to fail (no record files), its fine.
    """
    try:
        date_dir = p.parent.name  # YYYY-MM-DD
        m = SEG_RX.match(p.name)
        if not m:
            hour = int(p.name.split("_")[-1].split("-")[0])
            start = datetime.strptime(f"{date_dir} {hour:02d}", "%Y-%m-%d %H")
            end = start + timedelta(hours=1)
            return start.replace(tzinfo=timezone.utc), end.replace(tzinfo=timezone.utc)
        hour = int(m.group(1))
        minute = int(m.group(2))
        minute = quarter_floor(minute)
        start = datetime.strptime(f"{date_dir} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M")
        end = start + timedelta(minutes=15)
        return start.replace(tzinfo=timezone.utc), end.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def guess_segment_from_ts(record_root: Path,
                          camera_stub: str,
                          ts_utc: datetime,
                          container: str = "mkv") -> Path:
    """
    Construct expected 15-min filename under record/YYYY-MM-DD/ using event timestamp.
    In demo, this usually won't exist; that's fine too (clip_id stays empty).
    """
    ts_local = ts_utc  # keep UTC tree for demo
    day = ts_local.strftime("%Y-%m-%d")
    HH = ts_local.strftime("%H")
    MM = f"{quarter_floor(ts_local.minute):02d}"
    rel = f"{day}/{camera_stub}_{HH}-{MM}.{container}"
    return (record_root / rel).resolve()

# ----------------------------
# Payload helpers
# ----------------------------
def infer_kind_from_payload(obj: Dict[str, Any]) -> str:
    if isinstance(obj.get("type"), str):
        return obj["type"]
    if "motion_score" in obj:
        return "motion"
    if "objects" in obj:
        return "objects"
    return "event"

# ----------------------------
# Media path resolution
# ----------------------------
def resolve_media_path(rel: str, jsonl_file: Path, clips_dir: Path) -> Path:
    rel_norm = rel.replace("\\", "/")
    if rel_norm.startswith(("media/", "record/")):
        if rel_norm.startswith("media/"):
            return (jsonl_file.parent / rel_norm).resolve()
        else:
            return (clips_dir.parent / rel_norm).resolve()
    p = Path(rel)
    return p if p.is_absolute() else (jsonl_file.parent / rel).resolve()

def resolve_primary_clip(obj: Dict[str, Any], jsonl_file: Path, clips_dir: Path) -> Optional[Path]:
    clip_field = obj.get("clip")
    if isinstance(clip_field, str) and clip_field.strip():
        return resolve_media_path(clip_field.strip(), jsonl_file, clips_dir)
    return None

# ----------------------------
# SQLite helpers (WAL + indexes)
# ----------------------------
def setup_db(con: sqlite3.Connection) -> None:
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("PRAGMA cache_size=-8000;")  # ~8MB page cache

def ensure_schema(cur: sqlite3.Cursor) -> None:
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY, value TEXT
    );
    CREATE TABLE IF NOT EXISTS clips (
        clip_id   TEXT PRIMARY KEY,
        path      TEXT NOT NULL,
        start_utc TEXT,
        end_utc   TEXT
    );
    CREATE TABLE IF NOT EXISTS events (
        event_id    TEXT PRIMARY KEY,
        source      TEXT,
        kind        TEXT,
        t0_ms       INTEGER,
        t1_ms       INTEGER,
        clip_id     TEXT,
        novelty     TEXT,
        confidence  REAL,
        payload     TEXT,
        created_utc TEXT
    );
    CREATE TABLE IF NOT EXISTS captions (
        caption_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id    TEXT,
        text        TEXT,
        model       TEXT,
        tokens      INTEGER,
        wav         TEXT,
        start_utc   TEXT,
        end_utc     TEXT,
        sr          INTEGER,
        lang        TEXT,
        created_utc TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_utc);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_kind_created ON events(kind, created_utc);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_clips_path ON clips(path);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_captions_start ON captions(start_utc);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_captions_end ON captions(end_utc);")

def ensure_clips_row(cur: sqlite3.Cursor, clip_path: Path) -> Tuple[str, Dict[str, str]]:
    bounds = segment_bounds_from_record_name(clip_path)
    start = end = ""
    if bounds:
        start = bounds[0].strftime("%Y-%m-%dT%H:%M:%SZ")
        end = bounds[1].strftime("%Y-%m-%dT%H:%M:%SZ")
    clip_id = sha1_hex(f"{clip_path}|{start}|{end}")
    row = cur.execute("SELECT clip_id FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
    if not row:
        cur.execute(
            "INSERT INTO clips(clip_id, path, start_utc, end_utc) VALUES(?,?,?,?)",
            (clip_id, str(clip_path), start, end),
        )
    return clip_id, {"start_utc": start, "end_utc": end}

def compute_event_id(source: str, kind: str, clip_id: str, t0_ms: int, payload_str: str) -> str:
    basis = f"{source}|{kind}|{clip_id}|{t0_ms}|{sha1_short(payload_str, 16)}"
    return sha1_hex(basis)

# ----------------------------
# JSONL iter
# ----------------------------
def iter_jsonl(file_path: Path) -> Iterator[Tuple[Dict[str, Any], int]]:
    with file_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                yield json.loads(s), idx
            except Exception:
                yield {"__parse_error__": s[:200]}, idx

# ----------------------------
# Offsets (no real clips in demo)
# ----------------------------
def compute_offsets_ms_for_jsonl(obj: Dict[str, Any],
                                 jsonl_file: Path,
                                 clips_dir: Path,
                                 indexing: Dict[str, Any],
                                 clip_path: Optional[Path]) -> Tuple[int, int]:
    """
    In demo, there are no MKV segments; clip_path is usually None.
    Will still generate a consistent t0_ms/t1_ms using obj.ts and an hour anchor.
    """
    pre_ms = int(indexing.get("pre_ms", 2000))
    post_ms = int(indexing.get("post_ms", 3000))

    # Explicit offsets win if present.
    for a, b in (("t0_ms", "t1_ms"), ("start_ms", "end_ms")):
        if a in obj and b in obj:
            try:
                return int(obj[a]), int(obj[b])
            except Exception:
                pass

    dt = parse_any_iso(obj.get("ts")) if isinstance(obj.get("ts"), str) else None
    seg_bounds = segment_bounds_from_record_name(clip_path) if clip_path else None

    if dt is not None:
        dt_utc = dt.astimezone(timezone.utc)
        if seg_bounds:
            seg_start, seg_end = seg_bounds
            delta_ms = int((dt_utc - seg_start).total_seconds() * 1000)
            if delta_ms < 0 or dt_utc > seg_end:
                delta_ms = int((dt_utc - seg_start).total_seconds() * 1000)
            return max(0, delta_ms - pre_ms), delta_ms + post_ms
        else:
            hour_anchor = dt_utc.replace(minute=0, second=0, microsecond=0)
            delta_ms = int((dt_utc - hour_anchor).total_seconds() * 1000)
            return max(0, delta_ms - pre_ms), delta_ms + post_ms

    # Fallback: small window if we somehow lost ts
    return (0, pre_ms + post_ms)

# ----------------------------
# Captions insertion helper
# ----------------------------
def maybe_insert_caption(cur: sqlite3.Cursor,
                         obj: Dict[str, Any],
                         kind: str,
                         event_id: str,
                         created_utc: str) -> None:
    """
    If this event is a stage2 speech/speech.suppressed event, insert a row into captions.

    Expected payload from Stage 2 speech events:
      {
        "type": "speech" | "speech.suppressed",
        "wav": "...",
        "start_utc": "...",
        "end_utc": "...",
        "sr": 16000,
        "lang": "en",
        "model": "medium.en",
        "text": "....",              # speech only; suppressed may omit
        "created_utc": "..."
      }
    """
    if kind not in ("speech", "speech.suppressed"):
        return

    payload = obj
    wav = payload.get("wav", "")
    start_utc = payload.get("start_utc")
    end_utc = payload.get("end_utc")
    sr = payload.get("sr")
    lang = payload.get("lang")
    model = payload.get("model")
    text = payload.get("text", "")
    created = payload.get("created_utc") or created_utc

    # Normalize start/end; fall back to created_utc if missing
    dt_start = parse_any_iso(start_utc) if isinstance(start_utc, str) else None
    dt_end = parse_any_iso(end_utc) if isinstance(end_utc, str) else None

    start_utc_str = to_utc_string(dt_start, fallback=parse_any_iso(created) or None)
    end_utc_str = to_utc_string(dt_end, fallback=parse_any_iso(created) or None)

    tokens = None  # Demo summarizer doesn't currently use this

    cur.execute(
        """
        INSERT INTO captions(
            event_id, text, model, tokens, wav, start_utc, end_utc, sr, lang, created_utc
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            event_id,
            text,
            model,
            tokens,
            wav,
            start_utc_str,
            end_utc_str,
            sr,
            lang,
            created,
        ),
    )

# ----------------------------
# Ingestion (JSONL only)
# ----------------------------
def ingest_jsonl(db_path: Path,
                 jsonl_files: List[Path],
                 clips_dir: Path,
                 indexing: Dict[str, Any],
                 limit: Optional[int],
                 commit: bool,
                 dedupe_seconds: Optional[float],
                 camera_stub: str,
                 container_ext: str,
                 logger: logging.Logger) -> List[str]:

    lines: List[str] = []
    con = sqlite3.connect(str(db_path))
    setup_db(con)
    cur = con.cursor()
    ensure_schema(cur)

    inserted_events = 0
    skipped_dupes = 0
    processed = 0

    ev_rows: List[Tuple] = []

    def flush():
        nonlocal ev_rows, inserted_events
        if ev_rows:
            cur.executemany(
                """INSERT OR IGNORE INTO events(
                       event_id, source, kind, t0_ms, t1_ms, clip_id,
                       novelty, confidence, payload, created_utc
                   ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                ev_rows,
            )
            inserted_events += cur.rowcount or 0
            ev_rows = []
        con.commit()

    try:
        for jf in jsonl_files:
            for obj, line_no in iter_jsonl(jf):
                if limit and processed >= limit:
                    break
                processed += 1

                if "__parse_error__" in obj:
                    logger.warning(f"[SKIP] JSON parse error {jf}:{line_no}")
                    continue

                kind = infer_kind_from_payload(obj)
                source = str(obj.get("source", "stage2"))

                # For demo, clips don't really exist; clip_id will be empty.
                clip_path = resolve_primary_clip(obj, jf, clips_dir)
                if not clip_path:
                    ts = parse_any_iso(obj.get("ts"))
                    if ts:
                        guess = guess_segment_from_ts(
                            clips_dir, camera_stub, ts.astimezone(timezone.utc), container_ext
                        )
                        if guess.exists():
                            clip_path = guess

                clip_id = ""
                if clip_path is not None and clip_path.exists():
                    clip_id, _ = ensure_clips_row(cur, clip_path)

                t0_ms, t1_ms = compute_offsets_ms_for_jsonl(
                    obj, jf, clips_dir, indexing, clip_path if (clip_path and clip_path.exists()) else None
                )

                if clip_path is not None and clip_path.exists():
                    bounds = segment_bounds_from_record_name(clip_path)
                    if bounds:
                        seg_start_ms = int(bounds[0].timestamp() * 1000)
                        t0_ms += seg_start_ms
                        t1_ms += seg_start_ms

                payload_str = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                created_dt = parse_any_iso(obj.get("ts")) if isinstance(obj.get("ts"), str) else None
                created_utc = to_utc_string(created_dt)

                # Dedup within a time window around t0 for same kind+clip_id
                is_dupe = False
                if dedupe_seconds is not None and dedupe_seconds >= 0 and clip_id:
                    window_ms = int(dedupe_seconds * 1000)
                    lo = max(0, t0_ms - window_ms)
                    hi = t0_ms + window_ms
                    payload_sig = sha1_short(payload_str, 16)
                    existing = cur.execute(
                        "SELECT event_id, payload FROM events "
                        "WHERE clip_id=? AND kind=? AND t0_ms BETWEEN ? AND ?",
                        (clip_id, kind, lo, hi),
                    ).fetchall()
                    for _, pay in existing:
                        try:
                            if sha1_short(pay, 16) == payload_sig:
                                is_dupe = True
                                break
                        except Exception:
                            pass
                if is_dupe:
                    skipped_dupes += 1
                    continue

                event_id = compute_event_id(source, kind, clip_id or "-", t0_ms, payload_str)

                if commit:
                    ev_rows.append(
                        (
                            event_id,
                            source,
                            kind,
                            t0_ms,
                            t1_ms,
                            clip_id,
                            obj.get("novelty"),
                            obj.get("confidence"),
                            payload_str,
                            created_utc,
                        )
                    )
                    # Insert caption immediately for speech events
                    maybe_insert_caption(cur, obj, kind, event_id, created_utc)

                    if len(ev_rows) >= 512:
                        flush()
                else:
                    lines.append(
                        f"[DRY ] {jf.name}:{line_no} → kind={kind} t0={t0_ms}ms "
                        f"clip={'-' if not clip_id else clip_id[:10] + '…'}"
                    )

            if limit and processed >= limit:
                break

        if commit:
            flush()
    finally:
        con.close()

    lines.append(
        f"\nIngest(JSONL) summary: events+={inserted_events}, skipped_dupes={skipped_dupes}, "
        f"mode={'COMMIT' if commit else 'DRY-RUN'}"
    )
    return lines

# ----------------------------
# Verify / Peek / DB check
# ----------------------------
def summarize_paths(paths: List[Path], limit: int = 10) -> str:
    out = [f"- {p}" for p in itertools.islice(paths, 0, limit)]
    if len(paths) > limit:
        out.append(f"... and {len(paths) - limit} more")
    return "\n".join(out) if out else "(none found)"

def verify_event_to_clip_jsonl(jsonl_files: List[Path],
                               clips_dir: Path,
                               sample: int = 10) -> List[str]:
    lines: List[str] = []
    count = 0
    for jf in jsonl_files:
        for obj, line_no in iter_jsonl(jf):
            if "__parse_error__" in obj:
                continue
            clip_p = resolve_primary_clip(obj, jf, clips_dir)
            if not clip_p:
                lines.append(f"[MISS] {jf.name}:{line_no}  clip=(missing)")
            else:
                tag = "OK" if clip_p.exists() else "MISS"
                lines.append(f"[{tag:4}] {jf.name}:{line_no}  →  {clip_p}")
            count += 1
            if count >= sample:
                break
        if count >= sample:
            break
    lines.append(f"\nVerify(JSONL) summary: checked={count}")
    return lines

def peek_jsonl(jsonl_files: List[Path],
               clips_dir: Path,
               indexing: Dict[str, Any],
               n: int) -> List[str]:
    out: List[str] = []
    shown = 0
    for jf in jsonl_files:
        for obj, line_no in iter_jsonl(jf):
            if "__parse_error__" in obj:
                out.append(f"[ERR ] {jf.name}:{line_no} parse failed (skipped)")
                continue
            kind = infer_kind_from_payload(obj)
            keys = list(obj.keys())
            clip_p = resolve_primary_clip(obj, jf, clips_dir)
            pre_ms = int(indexing.get("pre_ms", 2000))
            post_ms = int(indexing.get("post_ms", 3000))
            out.append(
                f"[PEEK] {jf.name}:{line_no} kind={kind} keys={keys[:8]}{' …' if len(keys) > 8 else ''}"
            )
            out.append(
                f"       clip={clip_p if clip_p else '(none)'} window pre={pre_ms} post={post_ms} ms"
            )
            shown += 1
            if shown >= n:
                return out
    if not out:
        out.append("(no JSONL events to peek)")
    return out

def verify_db(db_path: Path) -> List[str]:
    if not db_path.exists():
        return [f"[MISS] DB not found: {db_path}"]
    lines: List[str] = []
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        tables = [
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            ).fetchall()
        ]
        lines.append(f"[OK] Opened DB: {db_path}")
        lines.append("Tables: " + (", ".join(tables) if tables else "(none)"))
        if "events" in tables:
            try:
                n = cur.execute("SELECT COUNT(*) FROM events;").fetchone()[0]
                lines.append(f"count(events)={n}")
            except sqlite3.Error:
                lines.append("[WARN] count(events) failed")
        if "clips" in tables:
            try:
                n = cur.execute("SELECT COUNT(*) FROM clips;").fetchone()[0]
                lines.append(f"count(clips)={n}")
            except sqlite3.Error:
                lines.append("[WARN] count(clips) failed")
        if "captions" in tables:
            try:
                n = cur.execute("SELECT COUNT(*) FROM captions;").fetchone()[0]
                lines.append(f"count(captions)={n}")
            except sqlite3.Error:
                lines.append("[WARN] count(captions) failed")
        try:
            integ = cur.execute("PRAGMA integrity_check;").fetchone()[0]
            lines.append(f"integrity_check={integ}")
        except sqlite3.Error:
            lines.append("[WARN] integrity_check failed")
    finally:
        con.close()
    return lines

# ----------------------------
# Discovery
# ----------------------------
def discover_jsonl(events_root: Path, events_glob_jsonl: str) -> List[Path]:
    return sorted(events_root.glob(events_glob_jsonl))

# ----------------------------
# Tail JSONL (optional)
# ----------------------------
def tail_loop(db_path: Path,
              events_root: Path,
              events_glob_jsonl: str,
              clips_dir: Path,
              indexing: Dict[str, Any],
              commit: bool,
              dedupe_seconds: Optional[float],
              interval: float,
              camera_stub: str,
              container_ext: str,
              logger: logging.Logger) -> None:

    seen_keys: set[str] = set()
    logger.info("[TAIL] starting... Ctrl+C to stop")
    try:
        while True:
            jsonl_files = discover_jsonl(events_root, events_glob_jsonl)
            if not jsonl_files:
                time.sleep(max(0.2, interval))
                continue

            con = sqlite3.connect(str(db_path))
            setup_db(con)
            cur = con.cursor()
            ensure_schema(cur)
            inserted = 0

            for jf in jsonl_files:
                for obj, line_no in iter_jsonl(jf):
                    if "__parse_error__" in obj:
                        continue
                    payload_str = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                    key = f"{jf}|{line_no}|{sha1_short(payload_str, 8)}"
                    if key in seen_keys:
                        continue

                    kind = infer_kind_from_payload(obj)
                    source = str(obj.get("source", "stage2"))

                    clip_path = resolve_primary_clip(obj, jf, clips_dir)
                    if not clip_path:
                        ts = parse_any_iso(obj.get("ts"))
                        if ts:
                            guess = guess_segment_from_ts(
                                clips_dir,
                                camera_stub,
                                ts.astimezone(timezone.utc),
                                container_ext,
                            )
                            if guess.exists():
                                clip_path = guess

                    clip_id = ""
                    if clip_path is not None and clip_path.exists():
                        clip_id, _ = ensure_clips_row(cur, clip_path)

                    t0_ms, t1_ms = compute_offsets_ms_for_jsonl(
                        obj, jf, clips_dir, indexing, clip_path if (clip_path and clip_path.exists()) else None
                    )
                    if clip_path is not None and clip_path.exists():
                        bounds = segment_bounds_from_record_name(clip_path)
                        if bounds:
                            seg_start_ms = int(bounds[0].timestamp() * 1000)
                            t0_ms += seg_start_ms
                            t1_ms += seg_start_ms

                    created_dt = parse_any_iso(obj.get("ts")) if isinstance(obj.get("ts"), str) else None
                    created_utc = to_utc_string(created_dt)
                    event_id = compute_event_id(source, kind, clip_id or "-", t0_ms, payload_str)

                    if commit:
                        cur.execute(
                            """INSERT OR IGNORE INTO events(
                                   event_id, source, kind, t0_ms, t1_ms, clip_id,
                                   novelty, confidence, payload, created_utc
                               ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                            (
                                event_id,
                                source,
                                kind,
                                t0_ms,
                                t1_ms,
                                clip_id,
                                obj.get("novelty"),
                                obj.get("confidence"),
                                payload_str,
                                created_utc,
                            ),
                        )
                        if cur.rowcount:
                            inserted += 1
                            # Insert caption for speech events in tail mode as well
                            maybe_insert_caption(cur, obj, kind, event_id, created_utc)

                    seen_keys.add(key)

            if commit:
                con.commit()
            con.close()
            if inserted:
                logger.info(f"[TAIL] inserted {inserted} new event(s)")
            time.sleep(max(0.2, interval))
    except KeyboardInterrupt:
        logger.info("[TAIL] interrupted by user; stopping.")

# ----------------------------
# CLI / main
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ETHEL Stage 3 demo – journaler (Stage2 JSONL → SQLite)"
    )

    ap.add_argument(
        "--base",
        type=str,
        help="Base ETHEL demo dir (default: parent of scripts/, e.g. C:\\AI\\demo)",
    )
    ap.add_argument(
        "--events-root",
        type=str,
        help="Root events dir (default: BASE/events)",
    )
    ap.add_argument(
        "--events-glob-jsonl",
        type=str,
        default="*/*.jsonl",
        help="Glob pattern under events_root (default: '*/*.jsonl')",
    )
    ap.add_argument(
        "--db-path",
        type=str,
        help="SQLite DB path (default: BASE/db/ethel_journal_demo.db)",
    )
    ap.add_argument(
        "--clips-dir",
        type=str,
        help="Record/clips root (default: BASE/record)",
    )
    ap.add_argument(
        "--camera-stub",
        type=str,
        default="cam_sim",
        help="Camera stub name for guessing segments (default: cam_sim)",
    )
    ap.add_argument(
        "--container-ext",
        type=str,
        default="mkv",
        help="Container extension for guessed clips (default: mkv)",
    )
    ap.add_argument(
        "--indexing-config",
        type=str,
        help="TOML with [indexing] pre_ms/post_ms (optional)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        help="Max JSONL events to process (per run)",
    )
    ap.add_argument(
        "--dedupe-seconds",
        type=float,
        default=None,
        help="Dedup window in seconds around t0_ms for same kind+clip_id (optional)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (no DB writes, just print ingest plan)",
    )
    ap.add_argument(
        "--verify-db",
        action="store_true",
        help="Check DB structure and counts, then exit",
    )
    ap.add_argument(
        "--peek-jsonl",
        type=int,
        metavar="N",
        help="Show first N JSONL events + inferred clip windows, then exit",
    )
    ap.add_argument(
        "--verify-jsonl",
        action="store_true",
        help="Verify JSONL → clip paths mapping (sample set), then exit",
    )
    ap.add_argument(
        "--tail",
        action="store_true",
        help="Tail events JSONL into DB (looping)",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Tail polling interval in seconds (default: 2.0)",
    )
    ap.add_argument(
        "--log-file",
        type=str,
        help="Optional log file (default: BASE/logs/journaler_demo.log)",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Disable stdout logging (errors only to log file)",
    )

    return ap.parse_args()

def main() -> None:
    args = parse_args()

    base = Path(args.base).resolve() if args.base else SCRIPTS_DIR.parent
    events_root = Path(args.events_root).resolve() if args.events_root else (base / "events")
    db_path = Path(args.db_path).resolve() if args.db_path else (base / "db" / "ethel_journal_demo.db")
    clips_dir = Path(args.clips_dir).resolve() if args.clips_dir else (base / "record")
    log_file = Path(args.log_file).resolve() if args.log_file else (base / "logs" / "journaler_demo.log")

    logger = setup_logging(log_file, verbose=not args.quiet)

    # Indexing defaults (demo-friendly)
    indexing: Dict[str, Any] = {"pre_ms": 2000, "post_ms": 3000}
    if args.indexing_config:
        cfg_path = Path(args.indexing_config)
        if cfg_path.exists():
            try:
                cfg = load_config(cfg_path)
                if isinstance(cfg.get("indexing"), dict):
                    indexing.update(cfg["indexing"])
            except Exception as e:
                logger.warning(f"Failed to load indexing config {cfg_path}: {e!r}")

    events_glob_jsonl = args.events_glob_jsonl or "*/*.jsonl"
    jsonl_files = discover_jsonl(events_root, events_glob_jsonl)

    # Mode: verify DB only
    if args.verify_db:
        lines = verify_db(db_path)
        for ln in lines:
            print(ln)
        return

    # Mode: peek JSONL
    if args.peek_jsonl:
        lines = peek_jsonl(jsonl_files, clips_dir, indexing, args.peek_jsonl)
        for ln in lines:
            print(ln)
        return

    # Mode: verify JSONL → clips (mostly no-op in demo, but kept)
    if args.verify_jsonl:
        lines = verify_event_to_clip_jsonl(jsonl_files, clips_dir, sample=20)
        for ln in lines:
            print(ln)
        return

    # Mode: tail
    if args.tail:
        if not jsonl_files:
            print(f"[TAIL] No JSONL files under {events_root} matching {events_glob_jsonl}")
        tail_loop(
            db_path=db_path,
            events_root=events_root,
            events_glob_jsonl=events_glob_jsonl,
            clips_dir=clips_dir,
            indexing=indexing,
            commit=not args.dry_run,
            dedupe_seconds=args.dedupe_seconds,
            interval=args.interval,
            camera_stub=args.camera_stub,
            container_ext=args.container_ext,
            logger=logger,
        )
        return

    # Default: one-shot ingest
    if not jsonl_files:
        print(f"[INGEST] No JSONL files under {events_root} matching {events_glob_jsonl}")
        return

    lines = ingest_jsonl(
        db_path=db_path,
        jsonl_files=jsonl_files,
        clips_dir=clips_dir,
        indexing=indexing,
        limit=args.limit,
        commit=not args.dry_run,
        dedupe_seconds=args.dedupe_seconds,
        camera_stub=args.camera_stub,
        container_ext=args.container_ext,
        logger=logger,
    )
    for ln in lines:
        print(ln)

if __name__ == "__main__":
    main()
