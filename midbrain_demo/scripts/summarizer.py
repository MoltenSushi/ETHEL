#!/usr/bin/env python
# summarizer.py — ETHEL midbrain demo: SQLite → hourly/daily summaries
#
# This script is a standalone demo version of ETHEL's Stage 4 summarizer.
# It is NOT the full ETHEL pipeline. It demonstrates:
#   - reading the demo journal DB produced by journaler.py
#   - rolling events into per-hour metric buckets (motion + speech)
#   - deriving a per-day summary with basic lexical stats
#   - writing a compact hourly JSONL and a daily JSON file
#   - verifying that summarized event_ids still exist in the DB
#
# DIRECTORY CONTRACT (DEMO)
#   BASE/
#     scripts/     - this script (summarizer.py)
#     db/          - journal database: ethel_journal_demo.db
#     summaries/   - summarizer outputs (per-hour JSONL + per-day JSON)
#     logs/        - summarizer log file: stage4.log
#
# CONFIG FILES
#
#   (none required for the demo)
#   - summarizer reads only from the SQLite DB (events table) and doesn't
#     use separate config files; the time window is selected via CLI flags
#     or inferred from the latest events in the DB.
#
#
# CLI USAGE (DEMO)
#   python summarizer.py
#       # auto-detect the latest local day with events and summarize that window
#
#   python summarizer.py --date 2025-11-20
#       # summarize a specific local date (00:00–23:59 local, converted to UTC)
#
#   python summarizer.py --start 2025-11-20T00:00:00Z --end 2025-11-20T23:59:59Z
#       # summarize an explicit UTC window instead of a calendar day
#
# FLAGS
#   --date YYYY-MM-DD     : summarize one local calendar day extrapolated from UTC
#   --start ISO           : summary window start (ISO string, usually UTC)
#   --end ISO             : summary window end (ISO string, usually UTC)
#
# OUTPUT (DEMO)
#   - Writes per-hour rollups to:
#       summaries/summ_hourly.jsonl
#         * one JSON object per hour, with counts, motion/speech metrics,
#           lexical top-words, and an excitement_score
#   - Writes one per-day summary file to:
#       summaries/YYYY-MM-DD.json
#         * includes day-level totals, per-hour sections, lexical stats,
#           and a simple verification block for event_ids
#   - Connects to the journal DB read-only (db/ethel_journal_demo.db) and
#     never mutates the DB; all work is done via SELECTs and new summary files.


import argparse, collections, json, os, re, sqlite3, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta


# ============================================================
#   Paths / Base  
# ============================================================

SCRIPTS_DIR = Path(__file__).resolve().parent          # midbrain_demo/scripts
BASE = SCRIPTS_DIR.parent                              # midbrain_demo/

DB_DIR = BASE / "db"
SUMM_DIR = BASE / "summaries"
LOG_DIR = BASE / "logs"

DB_PATH = DB_DIR / "ethel_journal_demo.db"
LOG_PATH = LOG_DIR / "stage4.log"
HOURLY_JSONL_PATH = SUMM_DIR / "summ_hourly.jsonl"


# ============================================================
#   Constants/Configs
# ============================================================

LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "of", "for",
    "to", "from", "by", "with", "is", "am", "are", "was", "were", "be", "been",
    "it", "this", "that", "these", "those", "you", "i", "we", "they", "he",
    "she", "them", "him", "her", "as", "so", "not", "no", "yes", "uh", "um",
}


# ============================================================
#   Helpers
# ============================================================

def iso(dt: datetime) -> str:
    """UTC ISO string with trailing Z."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        elif len(s) >= 5 and (s[-5] in "+-") and s[-3] != ":":
            s = s[:-2] + ":" + s[-2:]
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def day_bounds_utc_for(date_str: str):
    d0 = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    start = d0.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1) - timedelta(seconds=1)
    return start, end


def log(msg: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)


def connect_ro(db_path: Path) -> sqlite3.Connection:
    """
    Open DB read-only using a proper SQLite file: URI.

    On Windows, db_path is like C:\\midbrain_demo\\db\\file.db.
    We must convert backslashes to forward slashes for URI mode.
    """
    uri_path = db_path.as_posix()  # e.g. C:/midbrain_demo/db/ethel_journal_demo.db
    uri = f"file:{uri_path}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def table_has_column(cur, table: str, col: str) -> bool:
    try:
        cur.execute(f"PRAGMA table_info({table})")
        return col in [r[1] for r in cur.fetchall()]
    except sqlite3.Error:
        return False


def table_exists(cur, table: str) -> bool:
    try:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (table,),
        )
        return cur.fetchone() is not None
    except sqlite3.Error:
        return False


def detect_latest_day(cur):
    # Find latest local-day window from events.created_utc. 
    try:
        if not table_has_column(cur, "events", "created_utc"):
            return None

        cur.execute(
            "SELECT MIN(created_utc), MAX(created_utc) FROM events WHERE created_utc IS NOT NULL"
        )
        row = cur.fetchone()
        if not row or not row[1]:
            return None

        max_dt_utc = parse_iso(row[1])
        if not max_dt_utc:
            return None

        max_dt_local = max_dt_utc.astimezone(LOCAL_TZ)
        day_start_local = max_dt_local.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end_local = day_start_local + timedelta(days=1) - timedelta(seconds=1)

        return (
            day_start_local.astimezone(timezone.utc),
            day_end_local.astimezone(timezone.utc),
        )
    except sqlite3.Error:
        return None


def hour_bucket(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def word_tokens(text: str):
    for tok in re.split(r"[^\w']+", text):
        t = tok.strip()
        if t:
            yield t


def update_lexical_counters(bucket: dict, text: str):
    if "_word_counts" not in bucket:
        bucket["_word_counts"] = collections.Counter()
    if "_proper_counts" not in bucket:
        bucket["_proper_counts"] = collections.Counter()

    wc = bucket["_word_counts"]
    pc = bucket["_proper_counts"]

    for tok in word_tokens(text):
        lower = tok.lower()
        if lower not in STOPWORDS:
            wc[lower] += 1

            # naive proper name heuristic
            if tok[0].isupper() and not tok.isupper() and tok != "I":
                pc[tok] += 1


# ============================================================
#   Rollup/Hourly
# ============================================================

def hourly_rollup(cur, start_utc: datetime, end_utc: datetime):
    if not table_exists(cur, "events"):
        return {}

    needed_cols = (
        "event_id", "source", "kind", "t0_ms", "t1_ms",
        "clip_id", "created_utc", "payload"
    )
    for col in needed_cols:
        if not table_has_column(cur, "events", col):
            log(f"(U) Warning: column events.{col} missing; partial metrics.")

    start_s = iso(start_utc)
    end_s = iso(end_utc)

    try:
        cur.execute(
            """
            SELECT event_id, source, kind, t0_ms, t1_ms, clip_id,
                   created_utc, payload
            FROM events
            WHERE created_utc BETWEEN ? AND ?
            ORDER BY created_utc ASC
            """,
            (start_s, end_s),
        )
        ev_rows = cur.fetchall()
    except sqlite3.Error:
        ev_rows = []

    per_hour = {}

    # Pre-seed hours
    h = hour_bucket(start_utc)
    while h <= hour_bucket(end_utc):
        k = iso(h)
        per_hour[k] = {
            "hour_start_utc": iso(h),
            "hour_end_utc": iso(h + timedelta(hours=1) - timedelta(seconds=1)),
            "total_events": 0,
            "counts_by_kind": {},
            "counts_by_source": {},
            "unique_sources": 0,
            "unique_clips": 0,
            "sum_duration_ms": 0,
            "first_event_created_utc": None,
            "last_event_created_utc": None,
            "event_ids": [],
            "speech_events": 0,
            "speech_suppressed_events": 0,
            "speech_seconds": 0.0,
            "speech_words": 0,
            "speech_chars": 0,
            "motion_summary_events": 0,
            "motion_peak_score_sum": 0.0,
            "motion_peak_score_max": 0.0,
            "excitement_score": 0.0,
            "excitement_basis": "none",
            "top_words": [],
            "top_proper_names": [],
        }
        h += timedelta(hours=1)

    # Accumulate
    for event_id, source, kind, t0_ms, t1_ms, clip_id, created_utc, payload_str in ev_rows:
        dt = parse_iso(created_utc)
        if not dt:
            continue

        hb = iso(hour_bucket(dt))
        bucket = per_hour.get(hb)
        if not bucket:
            # dynamic expansion if needed
            bucket = {
                "hour_start_utc": hb,
                "hour_end_utc": iso(hour_bucket(dt) + timedelta(hours=1) - timedelta(seconds=1)),
                "total_events": 0,
                "counts_by_kind": {},
                "counts_by_source": {},
                "unique_sources": 0,
                "unique_clips": 0,
                "sum_duration_ms": 0,
                "first_event_created_utc": None,
                "last_event_created_utc": None,
                "event_ids": [],
                "speech_events": 0,
                "speech_suppressed_events": 0,
                "speech_seconds": 0.0,
                "speech_words": 0,
                "speech_chars": 0,
                "motion_summary_events": 0,
                "motion_peak_score_sum": 0.0,
                "motion_peak_score_max": 0.0,
                "excitement_score": 0.0,
                "excitement_basis": "none",
                "top_words": [],
                "top_proper_names": [],
                "_word_counts": collections.Counter(),
                "_proper_counts": collections.Counter(),
            }
            per_hour[hb] = bucket

        bucket["total_events"] += 1
        if kind:
            bucket["counts_by_kind"][kind] = bucket["counts_by_kind"].get(kind, 0) + 1
        if source:
            bucket["counts_by_source"][source] = bucket["counts_by_source"].get(source, 0) + 1

        if bucket["first_event_created_utc"] is None or created_utc < bucket["first_event_created_utc"]:
            bucket["first_event_created_utc"] = created_utc
        if bucket["last_event_created_utc"] is None or created_utc > bucket["last_event_created_utc"]:
            bucket["last_event_created_utc"] = created_utc

        # uniques
        bucket.setdefault("_sources", set()).add(source)
        if clip_id:
            bucket.setdefault("_clips", set()).add(clip_id)

        # duration
        if t0_ms is not None and t1_ms is not None:
            try:
                if int(t1_ms) >= int(t0_ms):
                    bucket["sum_duration_ms"] += (int(t1_ms) - int(t0_ms))
            except Exception:
                pass

        if event_id:
            bucket["event_ids"].append(str(event_id))

        # payload
        try:
            payload = json.loads(payload_str) if isinstance(payload_str, str) else {}
        except Exception:
            payload = {}

        p_type = payload.get("type")

        # speech
        if kind in ("speech", "speech.suppressed"):
            is_speech = (kind == "speech")
            if is_speech:
                bucket["speech_events"] += 1
            else:
                bucket["speech_suppressed_events"] += 1

            s0 = parse_iso(payload.get("start_utc"))
            s1 = parse_iso(payload.get("end_utc"))
            dur = 0.0
            if s0 and s1 and s1 >= s0:
                dur = (s1 - s0).total_seconds()
            elif t0_ms and t1_ms:
                try:
                    dur = max(0.0, (int(t1_ms) - int(t0_ms)) / 1000.0)
                except Exception:
                    pass

            bucket["speech_seconds"] += float(dur)

            if is_speech:
                text = (payload.get("text") or "").strip()
                if text:
                    bucket["speech_chars"] += len(text)
                    bucket["speech_words"] += len([w for w in text.split() if w.strip()])
                    update_lexical_counters(bucket, text)

        # motion.summary
        if kind == "motion.summary" or p_type == "motion.summary":
            bucket["motion_summary_events"] += 1
            try:
                score = float(payload.get("motion_score", 0.0))
            except Exception:
                score = 0.0
            bucket["motion_peak_score_sum"] += score
            bucket["motion_peak_score_max"] = max(bucket["motion_peak_score_max"], score)

    # finalize buckets
    for b in per_hour.values():
        b["unique_sources"] = len(b.get("_sources", []))
        b["unique_clips"] = len(b.get("_clips", []))
        b.pop("_sources", None)
        b.pop("_clips", None)

        # avg motion
        if b["motion_summary_events"] > 0:
            b["avg_motion_score"] = (
                b["motion_peak_score_sum"] / b["motion_summary_events"]
            )
        else:
            b["avg_motion_score"] = 0.0

        motion_component = b["avg_motion_score"] * 100.0
        speech_component = b["speech_seconds"]
        excit = motion_component + speech_component

        b["excitement_score"] = float(round(excit, 3))

        if excit <= 0:
            b["excitement_basis"] = "none"
        else:
            if motion_component > 2 * speech_component:
                b["excitement_basis"] = "motion"
            elif speech_component > 2 * motion_component:
                b["excitement_basis"] = "speech"
            else:
                b["excitement_basis"] = "mixed"

        wc = b.get("_word_counts", collections.Counter())
        pc = b.get("_proper_counts", collections.Counter())

        b["top_words"] = [{"word": w, "count": int(c)} for w, c in wc.most_common(3)]
        b["top_proper_names"] = [{"name": w, "count": int(c)} for w, c in pc.most_common(3)]

        b["_word_counts"] = wc
        b["_proper_counts"] = pc

    return per_hour


def write_hourly_jsonl(per_hour: dict):
    SUMM_DIR.mkdir(parents=True, exist_ok=True)
    tmp = str(HOURLY_JSONL_PATH) + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        for k in sorted(per_hour.keys()):
            b = dict(per_hour[k])
            b.pop("_word_counts", None)
            b.pop("_proper_counts", None)
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    os.replace(tmp, HOURLY_JSONL_PATH)
    return HOURLY_JSONL_PATH


# ============================================================
#   Summary/Daily
# ============================================================

def build_daily_summary(per_hour, start_utc, end_utc):
    total_events = 0
    counts_by_kind = collections.Counter()
    counts_by_source = collections.Counter()
    sum_duration_ms = 0
    all_ids = []

    speech_events = 0
    speech_suppressed = 0
    speech_seconds = 0
    speech_words = 0
    speech_chars = 0

    motion_summary_events = 0
    motion_peak_max = 0

    day_wc = collections.Counter()
    day_pc = collections.Counter()

    max_ex = None
    min_ex = None

    for hk in sorted(per_hour.keys()):
        h = per_hour[hk]
        total_events += h["total_events"]

        counts_by_kind.update(h["counts_by_kind"])
        counts_by_source.update(h["counts_by_source"])

        sum_duration_ms += h.get("sum_duration_ms", 0)
        all_ids.extend(h.get("event_ids", []))

        speech_events += h.get("speech_events", 0)
        speech_suppressed += h.get("speech_suppressed_events", 0)
        speech_seconds += h.get("speech_seconds", 0.0)
        speech_words += h.get("speech_words", 0)
        speech_chars += h.get("speech_chars", 0)

        motion_summary_events += h.get("motion_summary_events", 0)
        motion_peak_max = max(motion_peak_max, h.get("motion_peak_score_max", 0.0))

        ex = h.get("excitement_score", 0.0)
        if max_ex is None or ex > max_ex["score"]:
            max_ex = {
                "hour_start_utc": h["hour_start_utc"],
                "hour_end_utc": h["hour_end_utc"],
                "score": ex,
                "basis": h.get("excitement_basis", "none"),
            }
        if min_ex is None or ex < min_ex["score"]:
            min_ex = {
                "hour_start_utc": h["hour_start_utc"],
                "hour_end_utc": h["hour_end_utc"],
                "score": ex,
                "basis": h.get("excitement_basis", "none"),
            }

        wc = h.get("_word_counts", collections.Counter())
        pc = h.get("_proper_counts", collections.Counter())
        if wc:
            day_wc.update(wc)
        if pc:
            day_pc.update(pc)

    # lexical top
    w = day_wc.most_common(1)
    most_common_word = w[0][0] if w else None
    most_common_word_count = w[0][1] if w else 0

    p = day_pc.most_common(1)
    most_common_name = p[0][0] if p else None
    most_common_name_count = p[0][1] if p else 0

    summary = {
        "date_utc": start_utc.strftime("%Y-%m-%d"),
        "window_start_utc": iso(start_utc),
        "window_end_utc": iso(end_utc),

        "total_events": int(total_events),
        "counts_by_kind": dict(counts_by_kind),
        "counts_by_source": dict(counts_by_source),
        "sum_duration_ms": int(sum_duration_ms),

        "hours": [],

        "speech": {
            "speech_events": int(speech_events),
            "speech_suppressed_events": int(speech_suppressed),
            "speech_seconds": float(round(speech_seconds, 3)),
            "speech_words": int(speech_words),
            "speech_chars": int(speech_chars),
        },

        "motion": {
            "motion_summary_events": int(motion_summary_events),
            "max_motion_peak_score": float(round(motion_peak_max, 3)),
        },

        "excitement_daily": {
            "max_hour": max_ex,
            "min_hour": min_ex,
        },

        "lexical": {
            "most_common_word": most_common_word,
            "most_common_word_count": int(most_common_word_count),
            "most_common_proper_name": most_common_name,
            "most_common_proper_name_count": int(most_common_name_count),
        },

        "verification": {
            "checked_event_ids": 0,
            "missing_event_ids": 0,
            "missing_ids_sample": [],
        },
    }

    for hk in sorted(per_hour.keys()):
        h = dict(per_hour[hk])
        h.pop("_word_counts", None)
        h.pop("_proper_counts", None)
        summary["hours"].append(h)

    summary["_event_ids"] = all_ids
    return summary


# ============================================================
#   Write the daily file
# ============================================================

def write_daily_file(summary):
    SUMM_DIR.mkdir(parents=True, exist_ok=True)
    date = summary["date_utc"]
    out = SUMM_DIR / f"{date}.json"
    tmp = str(out) + ".tmp"

    to_write = dict(summary)
    to_write.pop("_event_ids", None)

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_write, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out)
    return out


# ============================================================
#   Verify the Event IDS
# ============================================================

def verify_ids(cur, summary, start_utc, end_utc):
    ids = [str(x) for x in summary.get("_event_ids", []) if x]

    if not ids:
        # fallback: count DB events in window
        try:
            start_s = iso(start_utc)
            end_s = iso(end_utc)
            cur.execute(
                "SELECT COUNT(1) FROM events WHERE created_utc BETWEEN ? AND ?",
                (start_s, end_s),
            )
            c = cur.fetchone()[0] or 0
        except sqlite3.Error:
            c = 0

        summary["verification"] = {
            "checked_event_ids": 0,
            "missing_event_ids": 0,
            "missing_ids_sample": [],
            "db_count_in_window": int(c),
        }
        return summary

    missing = 0
    sample = []
    checked = 0

    try:
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS _stage4_ids(id TEXT PRIMARY KEY)")
        cur.execute("DELETE FROM _stage4_ids")

        CH = 1000
        for i in range(0, len(ids), CH):
            chunk = [(ids[i + j],) for j in range(0, min(CH, len(ids) - i))]
            cur.executemany("INSERT OR IGNORE INTO _stage4_ids(id) VALUES (?)", chunk)

        cur.execute("SELECT COUNT(*) FROM _stage4_ids")
        checked = cur.fetchone()[0] or 0

        cur.execute(
            """
            SELECT s.id
            FROM _stage4_ids s
            LEFT JOIN events e ON e.event_id = s.id
            WHERE e.event_id IS NULL
            LIMIT 20
            """
        )
        sample = [r[0] for r in cur.fetchall()]

        cur.execute(
            """
            SELECT COUNT(*)
            FROM _stage4_ids s
            LEFT JOIN events e ON e.event_id = s.id
            WHERE e.event_id IS NULL
            """
        )
        missing = cur.fetchone()[0] or 0

    except sqlite3.Error:
        sample_local = []
        checked = len(ids)
        CH = 500
        for i in range(0, len(ids), CH):
            chunk = ids[i:i + CH]
            try:
                placeholders = ",".join(["?"] * len(chunk))
                cur.execute(f"SELECT event_id FROM events WHERE event_id IN ({placeholders})", chunk)
                present = {str(r[0]) for r in cur.fetchall()}
            except sqlite3.Error:
                present = set()
            for eid in chunk:
                if eid not in present:
                    if len(sample_local) < 20:
                        sample_local.append(eid)
                    missing += 1
        sample = sample_local

    summary["verification"] = {
        "checked_event_ids": int(checked),
        "missing_event_ids": int(missing),
        "missing_ids_sample": sample,
    }
    return summary


# ============================================================
#   MAIN + CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Stage 4 demo summarizer")
    ap.add_argument("--date", type=str)
    ap.add_argument("--start", type=str)
    ap.add_argument("--end", type=str)
    return ap.parse_args()


def main():
    SUMM_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    args = parse_args()

    # Determine window
    if args.date:
        start_utc, end_utc = day_bounds_utc_for(args.date)
    else:
        start_utc = parse_iso(args.start) if args.start else None
        end_utc = parse_iso(args.end) if args.end else None

        if not start_utc or not end_utc:
            try:
                with connect_ro(DB_PATH) as cx:
                    cur = cx.cursor()
                    detected = detect_latest_day(cur)
                    if not detected:
                        log("No events found in DB; nothing to summarize.")
                        return 0
                    start_utc, end_utc = detected
            except sqlite3.Error as e:
                log(f"DB open error: {e}")
                return 1

    log(f"Stage 4 starting. Window: {iso(start_utc)} .. {iso(end_utc)}")

    try:
        cx = connect_ro(DB_PATH)
        cur = cx.cursor()
        log("Connected to DB (read-only). (v.)")
    except sqlite3.Error as e:
        log(f"ERROR: Could not connect read-only to DB: {e}")
        return 1

    try:
        per_hour = hourly_rollup(cur, start_utc, end_utc)
        hp = write_hourly_jsonl(per_hour)
        log(f"4.1 Hourly roll-up complete. Hours: {len(per_hour)}. Output: {hp} (v.)")

        daily = build_daily_summary(per_hour, start_utc, end_utc)
        log("4.2 Daily summary constructed. (v.)")

        daily = verify_ids(cur, daily, start_utc, end_utc)
        log(
            f"4.4 Verification: checked={daily['verification']['checked_event_ids']}, "
            f"missing={daily['verification']['missing_event_ids']} (v.)"
        )

        outp = write_daily_file(daily)
        log(f"4.3 Daily file written: {outp} (v.)")

        log("Stage 4 complete. (v.)")
        return 0

    except Exception as e:
        log(f"ERROR during Stage 4: {e}")
        return 2

    finally:
        try:
            cx.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
