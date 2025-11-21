#!/usr/bin/env python
# detector.py — ETHEL midbrain demo: video → JSONL event stream
#
# This script is a standalone demo version of ETHEL's midbrain detector.
# It is NOT the full ETHEL pipeline. It demonstrates:
#   - motion detection
#   - YOLO object detection (optional)
#   - novelty detection (pHash)
#   - track enter/exit
#   - burst detection + motion summary
# and writes event JSONL files that the demo journaler can ingest.
#
# DIRECTORY CONTRACT (DEMO)
#   BASE/
#     ingest/    - input media + demo JSONL files (before_demo.jsonl, after_demo.jsonl,
#                  transcript_demo.jsonl or transcript_demo.txt, shimmer_test.mp4)
#     events/    - JSONL event shards written by this script (per-day subfolders)
#     configs/   - simple key=value config files used at startup
#     logs/      - detector log file: stage2_demo.log
#     scripts/   - this script (detector.py) and other demo scripts
#
# CONFIG FILES (all simple key=value)
#
# ingest/source_config.txt
#   source=shimmer_test.mp4        # or camera index (0) or RTSP/HTTP URL
#
# configs/motion_config.txt
#   motion_threshold=0.35          # minimum motion fraction
#   min_motion_frames=1            # consecutive frames required above threshold
#   min_motion_area_frac=0.0       # fraction of frame that must be in motion
#
# configs/detector_config.txt
#   model=yolov8n.pt               # YOLO model file
#   cadence_fps=10                 # detector cadence (frames per second)
#   conf_threshold=0.2             # YOLO confidence threshold
#   nms_iou=0.50                   # NMS IoU threshold
#   max_detections=20              # max YOLO detections per frame
#   labels=person,dog,cat          # allowed YOLO classes
#
# configs/novelty_config.txt
#   phash_distance_max=8           # max Hamming distance for pHash novelty filter
#   novelty_window_s=30            # window for image-hash novelty memory
#
# configs/event_config.txt
#   event_jsonl_roll=hour          # 'hour' or 'file' to determine date shards or a single event file
#   side_margin_frac=0.15          # fraction of width for side-detection bands
#   micro_event_cooldown_s=2       # cooldown between enter/exit for same track
#   min_track_frames=2             # frames required before declaring track entry
#   min_track_area_px=1500         # minimum bbox area to count as a valid track
#   presence_hold_s=3              # how long presence holds scene as "active"
#   merge_grace_s=4                # seconds allowed before burst is considered ended
#   clip_pre_s=3                   # seconds included before burst
#   clip_post_s=3                  # seconds included after burst
#   min_event_sec=1.0              # minimum duration to emit summary events
#
#
# CLI USAGE (DEMO)
#   python detector.py                # use source_config.txt or fallback to ingest/shimmer_test.mp4
#   python detector.py --source 0     # use local camera index 0
#   python detector.py --source rtsp://...    # use RTSP/HTTP stream
#
# FLAGS
#   --source   : override video source (file, index, or URL)
#   --duration : stop after N seconds (test runs)
#   --status   : oneline | lines | none  (console status style defualts to oneline)
#   --tv       : TV test mode (more sensitive motion, faster YOLO cadence)
#   --yolo-off : disable YOLO even if ultralytics is installed
#
# OUTPUT (DEMO)
#   - Writes events to:
#       events/YYYY-MM-DD/events_YYYYMMDD_HH.jsonl
#   - On exit, also:
#       * Retimes/copies before_demo.jsonl into the previous hour shard.
#       * Retimes/copies after_demo.jsonl into the next hour shard.
#       * Retimes a transcript demo file into transcript_YYYYMMDD_HH_demo.jsonl (or .txt)
#         aligned to the BEFORE shard’s first event.
#   - All event timestamps use system *local* time with numeric offset; original Whisper UTC times stay in the payload but are ignored for ordering.

import os, sys, time, json, math, random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
from PIL import Image
import imagehash

# ============================================================
# Paths: keep everything rooted at BASE (ethel/demo parent)
# ============================================================
SCRIPTS_DIR = Path(__file__).resolve().parent
if (SCRIPTS_DIR.parent / "configs").exists():
    BASE = SCRIPTS_DIR.parent
elif (SCRIPTS_DIR.parent.parent / "configs").exists():
    BASE = SCRIPTS_DIR.parent.parent
else:
    BASE = SCRIPTS_DIR.parent

DIRS = {
    "ingest":  BASE / "ingest",
    "events":  BASE / "events",
    "logs":    BASE / "logs",
    "configs": BASE / "configs",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = DIRS["logs"] / "stage2_demo.log"

# ============================================================
# Timezone: use system local timezone only
# ============================================================
LOCAL_TZ = datetime.now().astimezone().tzinfo


def now_local() -> datetime:
    return datetime.now().astimezone(LOCAL_TZ)


def log(line: str, *, also_stdout: bool = True) -> None:
    ts = now_local().strftime("%Y-%m-%dT%H:%M:%S%z")
    msg = f"[{ts}] {line}"
    if also_stdout:
        print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write(msg + "\n")


# ============================================================
# Config (simple key=value files)
# ============================================================
def load_kv(path: Path, default: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if default:
        out.update(default)
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


source_cfg = load_kv(DIRS["ingest"] / "source_config.txt")
motion_cfg = load_kv(DIRS["configs"] / "motion_config.txt",
                     {"motion_threshold": "0.35"})
det_cfg = load_kv(DIRS["configs"] / "detector_config.txt", {
    "model": "yolov8n.pt",
    "cadence_fps": "30",
    "conf_threshold": "0.2",
    "nms_iou": "0.50",
    "max_detections": "20",
    "labels": "person,dog,cat,bird",
})
novelty_cfg = load_kv(DIRS["configs"] / "novelty_config.txt", {
    "phash_distance_max": "8",
    "novelty_window_s": "30",
})
event_cfg = load_kv(DIRS["configs"] / "event_config.txt", {
    "event_jsonl_roll": "hour",
    "side_margin_frac": "0.15",
    "micro_event_cooldown_s": "8",
    "min_track_frames": "2",
    "min_track_area_px": "1500",
    "presence_hold_s": "2.5",
    "merge_grace_s": "4",
    "clip_pre_s": "5",
    "clip_post_s": "6",
    "min_event_sec": "1.0",
})

# Names for demo ingest files
BEFORE_DEMO_NAME = "before_demo.jsonl"
AFTER_DEMO_NAME = "after_demo.jsonl"
TRANSCRIPT_DEMO_CANDIDATES = ["transcript_demo.jsonl", "transcript_demo.txt"]


# ============================================================
# CLI
# ============================================================
import argparse


def parse_args():
    ap = argparse.ArgumentParser(description="ETHEL midbrain demo – detector + tracker → JSONL events")
    ap.add_argument(
        "--source",
        help="Video file, camera index, or RTSP/HTTP URL (demo default: ingest/shimmer_test.mp4)"
    )
    ap.add_argument("--duration", type=int, help="Stop after N seconds (for tests)")
    ap.add_argument("--status", choices=["oneline", "lines", "none"], default="oneline",
                    help="Console status style")
    ap.add_argument("--tv", action="store_true", help="TV test mode: more sensitive motion + faster YOLO cadence")
    ap.add_argument("--yolo-off", action="store_true", help="Disable YOLO even if model is available")
    return ap.parse_args()


ARGS = parse_args()


# ============================================================
# Helpers
# ============================================================
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ts_iso(dt: datetime) -> str:
    # canonical local timestamp with numeric offset
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    else:
        dt = dt.astimezone(LOCAL_TZ)
    return dt.strftime("%Y-%m-%dT%H:%M:%S%z")


def label_for_motion(score: float) -> str:
    if score < 0.10:
        return "none"
    if score < 0.25:
        return "minor"
    if score < 0.50:
        return "clear"
    return "heavy"


def short_id_from_ts(now_s: float) -> str:
    return f"{int((now_s * 1000) % 1e9):08x}{random.randint(0, 0xFFF):03x}"[-9:]


# ============================================================
# Events: JSONL only (no media output for the demo)
# ============================================================
def day_dir_for(dt_local: datetime) -> Path:
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=LOCAL_TZ)
    dd = DIRS["events"] / dt_local.strftime("%Y-%m-%d")
    dd.mkdir(parents=True, exist_ok=True)
    return dd


def events_jsonl_path_for(dt_local: datetime) -> Path:
    roll = event_cfg.get("event_jsonl_roll", "hour").lower()
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=LOCAL_TZ)
    if roll == "hour":
        return day_dir_for(dt_local) / f"events_{dt_local.strftime('%Y%m%d_%H')}.jsonl"
    return day_dir_for(dt_local) / "events.jsonl"


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_event_line(ts_local: datetime, payload: Dict[str, Any]) -> None:
    append_jsonl(events_jsonl_path_for(ts_local), payload)


def _parse_iso_dt_or_none(s: str) -> Optional[datetime]:
    """
    Parse an ISO timestamp string into a datetime, handling both offset forms
    and trailing 'Z' (Zulu / UTC) as used in transcript start_utc/end_utc.
    """
    try:
        s = s.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _retime_jsonl_file(src: Path, dst: Path, target_hour_local: datetime) -> bool:
    """
    Copy src → dst, shifting ISO timestamps so the content appears in target_hour_local.
    Only structured fields (ts/start_ts/end_ts) are adjusted; free-form strings are left as-is.
    Returns True on success, False on failure.
    """
    try:
        if not src.exists():
            log(f"Demo helper: source JSONL not found: {src}")
            return False

        lines = src.read_text(encoding="utf-8").splitlines()
        events: List[Dict[str, Any]] = []
        first_ts_dt: Optional[datetime] = None

        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                # If a line isn't valid JSON, keep it as-is in passthrough mode
                log(f"Demo helper: non-JSON line in {src.name}; copying file without retime.")
                ensure_parent(dst)
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                return True
            events.append(obj)
            if first_ts_dt is None and isinstance(obj, dict) and "ts" in obj:
                cand = _parse_iso_dt_or_none(obj["ts"])
                if cand is not None:
                    first_ts_dt = cand

        if not events or first_ts_dt is None:
            # Nothing usable to retime; just copy
            ensure_parent(dst)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            log(f"Demo helper: {src.name} had no usable ts; copied without retime.")
            return True

        # Normalize first_ts to local before computing delta
        if first_ts_dt.tzinfo is None:
            first_local = first_ts_dt.replace(tzinfo=LOCAL_TZ)
        else:
            first_local = first_ts_dt.astimezone(LOCAL_TZ)

        if target_hour_local.tzinfo is None:
            target_hour_local = target_hour_local.replace(tzinfo=LOCAL_TZ)
        else:
            target_hour_local = target_hour_local.astimezone(LOCAL_TZ)

        old_hour_start = first_local.replace(minute=0, second=0, microsecond=0)
        new_hour_start = target_hour_local.replace(minute=0, second=0, microsecond=0)
        delta = new_hour_start - old_hour_start

        out_lines: List[str] = []
        for obj in events:
            if isinstance(obj, dict):
                for key in ("ts", "start_ts", "end_ts"):
                    if key in obj and isinstance(obj[key], str):
                        dt = _parse_iso_dt_or_none(obj[key])
                        if dt is not None:
                            if dt.tzinfo is None:
                                dt_local = dt.replace(tzinfo=LOCAL_TZ)
                            else:
                                dt_local = dt.astimezone(LOCAL_TZ)
                            shifted_local = dt_local + delta
                            obj[key] = ts_iso(shifted_local)
            out_lines.append(json.dumps(obj, ensure_ascii=False))

        ensure_parent(dst)
        dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        return True
    except Exception as e:
        log(f"Demo helper: error retiming {src.name} → {dst.name}: {e}")
        return False


def _find_transcript_demo() -> Optional[Path]:
    for name in TRANSCRIPT_DEMO_CANDIDATES:
        p = DIRS["ingest"] / name
        if p.exists():
            return p
    return None


def _maybe_retime_transcript_demo(
    run_start_local: datetime,
    before_events_path: Optional[Path],
) -> Tuple[bool, str]:
    """
    Best-effort retime for a demo transcript file from ingest → events/day.

    Behavior:

    - Finds transcript JSONL in ingest.
    - Reads ALL JSONL lines (no early break).
    - Computes a shift so that the *first* transcript time aligns with the
      *first ts* found in the BEFORE-DEMO events shard (after its own retime and if available).
    - Alignment and all canonical timestamps are based on system local time.
    - Adds a `loc_ts` field per line representing the local canonical timestamp.
    - Never touches created_utc in transcript file.
    """
    src = _find_transcript_demo()
    if src is None:
        return False, "no transcript demo file found in ingest"

    # Target in the events day dir, keep the original suffix
    if run_start_local.tzinfo is None:
        run_start_local = run_start_local.replace(tzinfo=LOCAL_TZ)
    else:
        run_start_local = run_start_local.astimezone(LOCAL_TZ)

    day_dir = day_dir_for(run_start_local)
    suffix = src.suffix or ".txt"
    dst = day_dir / f"transcript_{run_start_local.strftime('%Y%m%d_%H')}_demo{suffix}"

    try:
        lines = src.read_text(encoding="utf-8").splitlines()
        objs: List[Dict[str, Any]] = []
        first_ts_dt: Optional[datetime] = None

        # --- Load ALL JSONL lines; capture first usable time but do not early-break
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # Not JSONL; fall back, just plain copy
                ensure_parent(dst)
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                return True, (
                    f"copied transcript demo {src.name} → {dst.name} "
                    f"(format not JSONL; timestamps not adjusted)"
                )

            objs.append(obj)
            if first_ts_dt is None and isinstance(obj, dict):
                # Prefer the "true" utterance clock if present
                for key in ("start_utc", "start_ts", "ts", "end_ts", "end_utc"):
                    if key in obj and isinstance(obj[key], str):
                        cand = _parse_iso_dt_or_none(obj[key])
                        if cand is not None:
                            first_ts_dt = cand
                            break

        if not objs or first_ts_dt is None:
            # Nothing usable to retime; just copy
            ensure_parent(dst)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            return True, (
                f"copied transcript demo {src.name} → {dst.name} "
                f"(no usable timestamps found)"
            )

        # --- Determine anchor: first ts in BEFORE-DEMO events shard, if available
        anchor_dt_local: Optional[datetime] = None
        if before_events_path is not None and before_events_path.exists():
            try:
                for raw in before_events_path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(ev, dict) and isinstance(ev.get("ts"), str):
                        cand = _parse_iso_dt_or_none(ev["ts"])
                        if cand is not None:
                            # Anchor on the first event ts, in local time
                            if cand.tzinfo is None:
                                anchor_dt_local = cand.replace(tzinfo=LOCAL_TZ)
                            else:
                                anchor_dt_local = cand.astimezone(LOCAL_TZ)
                            break
            except Exception as e:
                log(f"Demo helper: failed reading before-demo events for transcript anchor: {e}")

        # Fallback: anchor to previous hour if no before-demo ts is available
        if anchor_dt_local is None:
            anchor_hour = run_start_local.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            anchor_dt_local = anchor_hour

        # Normalize first transcript timestamp to local for delta computation
        if first_ts_dt.tzinfo is None:
            first_local = first_ts_dt.replace(tzinfo=LOCAL_TZ)
        else:
            first_local = first_ts_dt.astimezone(LOCAL_TZ)

        delta = anchor_dt_local - first_local

        out_lines: List[str] = []
        for obj in objs:
            if isinstance(obj, dict):
                shifted_local_by_key: Dict[str, datetime] = {}

                for key in ("ts", "start_ts", "end_ts", "start_utc", "end_utc"):
                    val = obj.get(key)
                    if not isinstance(val, str):
                        continue
                    dt = _parse_iso_dt_or_none(val)
                    if dt is None:
                        continue

                    # Treat any *_utc as actual UTC on input, convert to local for alignment
                    if key.endswith("_utc"):
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                        dt_local = dt.astimezone(LOCAL_TZ)
                    else:
                        if dt.tzinfo is None:
                            dt_local = dt.replace(tzinfo=LOCAL_TZ)
                        else:
                            dt_local = dt.astimezone(LOCAL_TZ)

                    shifted_local = dt_local + delta
                    shifted_local_by_key[key] = shifted_local

                # Write shifted times back to object
                for key, shifted_local in shifted_local_by_key.items():
                    if key in ("start_utc", "end_utc"):
                        # keep *_utc as proper UTC, but derived from local-aligned time
                        shifted_utc = shifted_local.astimezone(timezone.utc)
                        obj[key] = shifted_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
                    else:
                        obj[key] = ts_iso(shifted_local)

                # Add canonical local timestamp field
                if shifted_local_by_key:
                    canonical_local: Optional[datetime] = None
                    for pref in ("start_ts", "ts", "start_utc", "end_ts", "end_utc"):
                        if pref in shifted_local_by_key:
                            canonical_local = shifted_local_by_key[pref]
                            break
                    if canonical_local is not None:
                        obj["loc_ts"] = ts_iso(canonical_local)

                # created_utc is left untouched on purpose

            out_lines.append(json.dumps(obj, ensure_ascii=False))

        ensure_parent(dst)
        dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

        if before_events_path is not None and before_events_path.exists():
            detail_anchor = f"first ts in {before_events_path.name}"
        else:
            detail_anchor = "previous-hour anchor (no before-demo ts found)"

        return True, (
            f"retimed transcript demo {src.name} → {dst.name}, "
            f"anchored first utterance to {detail_anchor}"
        )
    except Exception as e:
        return False, f"error processing transcript demo {src.name}: {e}"


def _handle_demo_files(run_start_local: datetime) -> None:
    """
    After capture, make sure journal has something real to use:
    - Find the primary events JSONL file written from the MP4/camera.
    - Copy + retime before_demo.jsonl and after_demo.jsonl from ingest so they
      appear as the previous/next hour around the primary.
    - Try to align a transcript demo file into the same 3-hour window, with
      transcript aligned to the BEFORE shard's first ts.
    Logs a final summary line about what happened.
    """
    roll = event_cfg.get("event_jsonl_roll", "hour").lower()
    if roll != "hour":
        log("Demo helper: event_jsonl_roll != 'hour'; skipping demo companion setup.")
        return

    if run_start_local.tzinfo is None:
        run_start_local = run_start_local.replace(tzinfo=LOCAL_TZ)
    else:
        run_start_local = run_start_local.astimezone(LOCAL_TZ)

    primary_path = events_jsonl_path_for(run_start_local)
    if not primary_path.exists():
        log(f"Demo helper: no primary events JSONL found at {primary_path}; demo companions not created.")
        return

    anchor_hour = run_start_local.replace(minute=0, second=0, microsecond=0)
    prev_hour = anchor_hour - timedelta(hours=1)
    next_hour = anchor_hour + timedelta(hours=1)

    before_src = DIRS["ingest"] / BEFORE_DEMO_NAME
    after_src = DIRS["ingest"] / AFTER_DEMO_NAME

    before_dst = events_jsonl_path_for(prev_hour)
    after_dst = events_jsonl_path_for(next_hour)

    summary_bits = []

    # ----- BEFORE DEMO -----
    if before_src.exists():
        ok_before = _retime_jsonl_file(before_src, before_dst, prev_hour)
        if ok_before:
            msg = (f"Took demo file from {before_src} → {before_dst.name}, "
                   f"lined up filenames + timestamps for hour {prev_hour.strftime('%Y-%m-%d %H:00')} – SUCCESS")
        else:
            msg = (f"Tried demo file from {before_src} → {before_dst.name}, "
                   f"attempted to line up filenames + timestamps – DID NOT WORK")
        log(msg)
        summary_bits.append(msg)
    else:
        msg = f"Expected before-demo file {before_src} not found – DID NOT WORK"
        log(msg)
        summary_bits.append(msg)

    # ----- AFTER DEMO -----
    if after_src.exists():
        ok_after = _retime_jsonl_file(after_src, after_dst, next_hour)
        if ok_after:
            msg = (f"Took demo file from {after_src} → {after_dst.name}, "
                   f"lined up filenames + timestamps for hour {next_hour.strftime('%Y-%m-%d %H:00')} – SUCCESS")
        else:
            msg = (f"Tried demo file from {after_src} → {after_dst.name}, "
                   f"attempted to line up filenames + timestamps – DID NOT WORK")
        log(msg)
        summary_bits.append(msg)
    else:
        msg = f"Expected after-demo file {after_src} not found – DID NOT WORK"
        log(msg)
        summary_bits.append(msg)

    # ----- TRANSCRIPT DEMO -----
    before_events_for_anchor = before_dst if before_dst.exists() else None
    ok_tx, tx_detail = _maybe_retime_transcript_demo(run_start_local, before_events_for_anchor)
    if ok_tx:
        msg = f"Transcript demo: {tx_detail} – SUCCESS"
    else:
        msg = f"Transcript demo: {tx_detail} – DID NOT WORK"
    log(msg)
    summary_bits.append(msg)

    # Final one-line summary for the user
    log("Demo helper summary: " + " | ".join(summary_bits))


# ============================================================
# Ring buffer – "what just happened around this?"
# ============================================================
BUF_SECONDS = 12.0
BUF_LIMIT_MB = 256.0
BUF_LIMIT_BYTES = int(BUF_LIMIT_MB * 1024 * 1024)


def bytes_estimate(frame: np.ndarray) -> int:
    h, w = frame.shape[:2]
    chans = frame.shape[2] if len(frame.shape) == 3 else 1
    return h * w * chans


class RingBuffer:
    def __init__(self, seconds: float, bytes_cap: int):
        self.seconds = seconds
        self.bytes_cap = bytes_cap
        self.deq: deque = deque()
        self.total_bytes = 0

    def push(self, ts_utc: datetime, ts_local: datetime, frame: np.ndarray, phash_val) -> None:
        b = bytes_estimate(frame)
        self.deq.append((ts_utc, ts_local, frame, phash_val, b))
        self.total_bytes += b
        self._prune(ts_utc)

    def _prune(self, now_utc: datetime) -> None:
        cutoff = now_utc.timestamp() - self.seconds
        while self.deq and self.deq[0][0].timestamp() < cutoff:
            *_, b = self.deq.popleft()
            self.total_bytes -= b
        while self.deq and self.total_bytes > self.bytes_cap:
            *_, b = self.deq.popleft()
            self.total_bytes -= b

    def nearest(self, target_s: float):
        if not self.deq:
            return None
        idx = min(range(len(self.deq)),
                  key=lambda i: abs(self.deq[i][0].timestamp() - target_s))
        return self.deq[idx]


ring = RingBuffer(BUF_SECONDS, BUF_LIMIT_BYTES)


# ============================================================
# Motion (MOG2 + diff)
# ============================================================
tv_mode = bool(ARGS.tv)
MOTION_THRESH = float(motion_cfg.get("motion_threshold", "0.35"))
if tv_mode:
    MOTION_THRESH = min(0.08, MOTION_THRESH)

MIN_MOTION_FRAMES = int(motion_cfg.get("min_motion_frames", "1"))
MIN_MOTION_AREA_FRAC = float(motion_cfg.get("min_motion_area_frac", "0.0"))
consec_over_gate = 0

bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
_prev_gray = None


def motion_fraction_for(frame: np.ndarray) -> float:
    global _prev_gray
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg = bg.apply(frame)
    _, m1 = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)

    if _prev_gray is None:
        _prev_gray = g
        return np.count_nonzero(m1) / m1.size

    diff = cv2.absdiff(g, _prev_gray)
    _prev_gray = g
    _, m2 = cv2.threshold(diff, 25 if tv_mode else 35, 255, cv2.THRESH_BINARY)

    m = cv2.bitwise_or(m1, m2)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    return float(np.count_nonzero(m)) / m.size


# ============================================================
# Novelty (pHash window)
# ============================================================
PHASH_MAX_D = int(novelty_cfg.get("phash_distance_max", "8"))
NOVELTY_WINDOW_S = float(novelty_cfg.get("novelty_window_s", "30"))
recent_hashes: deque = deque()  # (ts_s, phash)


def phash_for(frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    return imagehash.phash(im)


def hamming(a, b) -> int:
    return (a - b)


def novelty_check(ph):
    now = time.time()
    while recent_hashes and (now - recent_hashes[0][0]) > NOVELTY_WINDOW_S:
        recent_hashes.popleft()
    if not recent_hashes:
        recent_hashes.append((now, ph))
        return "new", None
    dmin = min(hamming(ph, h) for _, h in recent_hashes)
    recent_hashes.append((now, ph))
    if dmin <= PHASH_MAX_D:
        return "duplicate", int(dmin)
    return "new", int(dmin)


# ============================================================
# YOLO (optional)
# ============================================================
YOLO_AVAILABLE = False
yolo_model = None
yolo_labels = [x.strip() for x in det_cfg.get("labels", "").split(",") if x.strip()]
yolo_disabled = bool(ARGS.yolo_off)

yolo_cadence_fps = float(det_cfg.get("cadence_fps", "2"))
if tv_mode:
    yolo_cadence_fps = max(4.0, yolo_cadence_fps)
yolo_conf = float(det_cfg.get("conf_threshold", "0.35"))
yolo_iou = float(det_cfg.get("nms_iou", "0.50"))
yolo_max_det = int(det_cfg.get("max_detections", "20"))

if not yolo_disabled:
    try:
        from ultralytics import YOLO
        model_name = det_cfg.get("model", "yolov8n.pt")
        yolo_model = YOLO(model_name)
        YOLO_AVAILABLE = True
        log(f"YOLO loaded ({model_name})")
    except Exception as e:
        log(f"YOLO not available ({e}); continuing without detector.")
        YOLO_AVAILABLE = False


def yolo_detect(frame_bgr: np.ndarray):
    if not YOLO_AVAILABLE:
        return [], []
    inp = frame_bgr
    if tv_mode and max(inp.shape[0], inp.shape[1]) > 720:
        scale = 720 / max(inp.shape[0], inp.shape[1])
        inp = cv2.resize(inp, (int(inp.shape[1] * scale), int(inp.shape[0] * scale)))

    res = yolo_model.predict(
        source=inp[:, :, ::-1],
        verbose=False,
        conf=yolo_conf,
        iou=yolo_iou,
        max_det=yolo_max_det,
    )

    dets = []
    objs = set()
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_idx = int(b.cls.item())
            name = r.names.get(cls_idx, str(cls_idx))
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]

            # --- apply labels whitelist from detector_config.txt ---
            if yolo_labels and name not in yolo_labels:
                continue

            dets.append((name, conf, (x1, y1, x2, y2)))
            objs.add(name)
    return dets, sorted(objs)


# ============================================================
# Simple tracker (IOU + side tagging)
# ============================================================
SIDE_MARGIN = float(event_cfg.get("side_margin_frac", "0.15"))
COOLDOWN_S = float(event_cfg.get("micro_event_cooldown_s", "8"))
MIN_TRACK_FRAMES = int(event_cfg.get("min_track_frames", "6"))
MIN_TRACK_AREA = int(event_cfg.get("min_track_area_px", "1500"))


@dataclass
class Track:
    tid: int
    cls: str
    conf: float
    bbox: Tuple[float, float, float, float]
    last_ts: float
    first_ts: float
    frames: int = 0
    last_side: Optional[str] = None
    first_side: Optional[str] = None
    # "center" here means "not near left/right margin", not "dead center".


class SimpleTracker:
    def __init__(self, iou_thr: float = 0.3, ttl_s: float = 2.0):
        self.iou_thr = iou_thr
        self.ttl_s = ttl_s
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    @staticmethod
    def center(b) -> Tuple[float, float]:
        x1, y1, x2, y2 = b
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def side_for_bbox(self, bb, img_w: int) -> str:
        cx, _ = self.center(bb)
        if cx < img_w * SIDE_MARGIN:
            return "left"
        if cx > img_w * (1.0 - SIDE_MARGIN):
            return "right"
        return "center"

    def update(self, dets: List[Tuple[str, float, Tuple[float, float, float, float]]],
               now_s: float,
               img_w: int) -> Dict[int, Track]:
        # drop expired tracks
        for tid in list(self.tracks.keys()):
            if (now_s - self.tracks[tid].last_ts) > self.ttl_s:
                del self.tracks[tid]

        unmatched = []
        used = set()

        for det in dets:
            cls, conf, bb = det
            best_tid, best_iou = None, 0.0
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                if tr.cls != cls:
                    continue
                iou = self.iou(tr.bbox, bb)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None and best_iou >= self.iou_thr:
                used.add(best_tid)
                tr = self.tracks[best_tid]
                tr.bbox = bb
                tr.conf = conf
                tr.last_ts = now_s
                tr.frames += 1
                side = self.side_for_bbox(bb, img_w)
                tr.last_side = side
                if tr.first_side is None:
                    tr.first_side = side
            else:
                unmatched.append(det)

        for cls, conf, bb in unmatched:
            tid = self.next_id
            self.next_id += 1
            side = self.side_for_bbox(bb, img_w)
            self.tracks[tid] = Track(
                tid=tid, cls=cls, conf=conf, bbox=bb,
                last_ts=now_s, first_ts=now_s,
                frames=1, last_side=side, first_side=side
            )

        return self.tracks


cooldowns: Dict[Tuple[str, int], float] = {}


def under_cooldown(etype: str, tid: int, now_s: float) -> bool:
    key = (etype, tid)
    last = cooldowns.get(key, -1e9)
    if now_s - last < COOLDOWN_S:
        return True
    cooldowns[key] = now_s
    return False


# ============================================================
# Burst parameters
# ============================================================
PRESENCE_HOLD_S = float(event_cfg.get("presence_hold_s", "15"))
DEBOUNCE_S = float(event_cfg.get("merge_grace_s", "4"))
PRE_S = float(event_cfg.get("clip_pre_s", "5"))
POST_S = float(event_cfg.get("clip_post_s", "6"))
MIN_EVENT_SEC = float(event_cfg.get("min_event_sec", "1.0"))


# ============================================================
# Source - demo is file-first, no fancy RTSP tuning this demo
# ============================================================
def resolve_source() -> Tuple[cv2.VideoCapture, str, float]:
    def is_stream(s: str) -> bool:
        s = s.lower()
        return s.startswith(("rtsp://", "rtsps://", "http://", "https://"))

    # 1) Explicit CLI
    if ARGS.source:
        src = ARGS.source.strip().strip('"').strip("'")
        if src.isdigit():
            cap = cv2.VideoCapture(int(src))
            if cap.isOpened():
                return cap, f"camera:{src}", cap.get(cv2.CAP_PROP_FPS) or 15.0
        if is_stream(src) or "://" in src:
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                return cap, src, cap.get(cv2.CAP_PROP_FPS) or 15.0
        p = Path(src)
        if not p.is_absolute():
            p = DIRS["ingest"] / p
        cap = cv2.VideoCapture(str(p))
        if cap.isOpened():
            return cap, str(p), cap.get(cv2.CAP_PROP_FPS) or 15.0
        log(f"Failed to open --source={src}")

    # 2) source_config.txt
    cfg_src = (source_cfg.get("source") or "").strip().strip('"').strip("'")
    if cfg_src:
        if cfg_src.isdigit():
            cap = cv2.VideoCapture(int(cfg_src))
            if cap.isOpened():
                return cap, f"camera:{cfg_src}", cap.get(cv2.CAP_PROP_FPS) or 15.0
        if is_stream(cfg_src) or "://" in cfg_src:
            cap = cv2.VideoCapture(cfg_src)
            if cap.isOpened():
                return cap, cfg_src, cap.get(cv2.CAP_PROP_FPS) or 15.0
        p = Path(cfg_src)
        if not p.is_absolute():
            p = DIRS["ingest"] / p
        cap = cv2.VideoCapture(str(p))
        if cap.isOpened():
            return cap, str(p), cap.get(cv2.CAP_PROP_FPS) or 15.0

    # 3) Demo default
    demo = DIRS["ingest"] / "shimmer_test.mp4"
    cap = cv2.VideoCapture(str(demo))
    if cap.isOpened():
        return cap, str(demo), cap.get(cv2.CAP_PROP_FPS) or 15.0

    log("ERROR: could not open any source (CLI, source_config, or ingest/shimmer_test.mp4).")
    sys.exit(1)


# ============================================================
# Console status
# ============================================================
class StatusLine:
    def __init__(self, mode: str = "oneline"):
        self.mode = mode
        try:
            import shutil as _sh
            self.width = _sh.get_terminal_size((100, 20)).columns
        except Exception:
            self.width = 100
        self.last_print = 0.0

    def print(self, s: str) -> None:
        if self.mode == "none":
            return
        if self.mode == "lines":
            print(s, flush=True)
            return
        now = time.time()
        if now - self.last_print < 0.12:
            return
        self.last_print = now
        msg = s
        if len(msg) > self.width - 1:
            msg = msg[: self.width - 4] + "..."
        pad = " " * max(0, self.width - len(msg) - 1)
        sys.stdout.write("\r" + msg + pad)
        sys.stdout.flush()

    def done(self) -> None:
        if self.mode == "oneline":
            sys.stdout.write("\n")
            sys.stdout.flush()


STATUS = StatusLine(ARGS.status)


# ============================================================
# Helper: compute exit_side + exit_reason from sides
# ============================================================
def infer_exit_side_and_reason(first_side: Optional[str],
                               last_side: Optional[str]) -> Tuple[str, str]:
    f = first_side or last_side or "center"
    l = last_side or f

    if l in ("left", "right"):
        if f != l:
            return l, "trajectory_side_crossing"
        return l, "edge_exit"

    # last_side is "center" or missing → basically lost it there
    return "center", "lost_in_center"


# ============================================================
# Main loop – demo version
# ============================================================
def main():
    global _prev_gray, consec_over_gate

    cap, src_desc, src_fps = resolve_source()
    log(f"Opened source: {src_desc}")

    ok, frame = cap.read()
    if not ok:
        log("Failed to read first frame; exiting.")
        return

    h0, w0 = frame.shape[:2]
    log(f"First frame size: {w0}x{h0}")

    # Seed ring + novelty
    ts_local = now_local()
    ts_utc = ts_local.astimezone(timezone.utc)
    run_start_local = ts_local  # anchor for demo companions
    ph0 = phash_for(frame)
    ring.push(ts_utc, ts_local, frame, ph0)
    recent_hashes.clear()
    recent_hashes.append((ts_utc.timestamp(), ph0))
    _prev_gray = None

    # Tracker + burst state
    tracker = SimpleTracker(iou_thr=0.3, ttl_s=2.0)
    last_presence_time = -1e9

    burst_open = False
    burst_start_s: float = 0.0
    burst_last_seen_s: float = 0.0
    burst_peak_motion: float = 0.0
    burst_first_ts_utc: Optional[datetime] = None
    burst_last_ts_utc: Optional[datetime] = None
    burst_first_objects: List[str] = []
    burst_seen_objects: List[str] = []

    frame_idx = 0
    t0 = time.time()

    # YOLO cadence
    fps = src_fps or 15.0
    cadence = max(0.5, yolo_cadence_fps)
    yolo_interval = max(1, int(round(fps / cadence)))

    log(f"Starting capture loop (fps≈{fps:.1f}, yolo_interval={yolo_interval})")

    while True:
        if ARGS.duration and (time.time() - t0) >= ARGS.duration:
            log("Duration reached; stopping capture.")
            break

        ok, frame = cap.read()
        if not ok:
            log("End of source or read error; stopping.")
            break

        ts_local = now_local()
        ts_utc = ts_local.astimezone(timezone.utc)
        now_s = ts_utc.timestamp()

        h, w = frame.shape[:2]

        # motion
        mot_frac = motion_fraction_for(frame)
        mot_lbl = label_for_motion(mot_frac)

        # novelty
        ph = phash_for(frame)
        nov_type, nov_extra = novelty_check(ph)

        # motion gate counter
        mot_effective = mot_frac if mot_frac >= MIN_MOTION_AREA_FRAC else 0.0
        if mot_effective >= MOTION_THRESH:
            consec_over_gate += 1
        else:
            consec_over_gate = 0

        # YOLO cadence
        dets, objects = [], []
        if YOLO_AVAILABLE and (frame_idx % yolo_interval) == 0:
            dets, objects = yolo_detect(frame)
        objects = list(objects)

        # tracking
        tracks = tracker.update(dets, now_s, w)

        # presence vs idle
        presence = bool(objects) or bool(tracks)
        if presence:
            last_presence_time = now_s

        # ring
        ring.push(ts_utc, ts_local, frame, ph)

        # status line
        if nov_type == "duplicate" and nov_extra is not None:
            nov_str = f"dup(d={nov_extra})"
        elif nov_type == "new" and nov_extra is not None:
            nov_str = f"new(d={nov_extra})"
        else:
            nov_str = nov_type
        STATUS.print(
            f"[{ts_utc.strftime('%H:%M:%S')}] mot={mot_lbl}({mot_frac:.2f}) "
            f"gate={MOTION_THRESH:.2f} nov={nov_str} yolo={len(objects)} trk={len(tracks)}"
        )

        # ---------------- Track micro-events (enter/exit) + object.summary ----------------
        static_prev_ids = getattr(main, "_prev_ids", set())
        static_last_known = getattr(main, "_last_known", {})
        obj_spans = getattr(main, "_obj_spans", {})

        # track.enter.* when a track has enough frames + area
        for tid, tr in tracks.items():
            x1, y1, x2, y2 = tr.bbox
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if tr.frames == MIN_TRACK_FRAMES and area >= MIN_TRACK_AREA:
                etype = f"track.enter.{tr.first_side or 'center'}"
                if not under_cooldown(etype, tid, now_s):
                    bbox_pxls = [
                        round(x1, 1), round(y1, 1),
                        round(x2, 1), round(y2, 1)
                    ]
                    msg = f"{tr.cls}[tid={tid}] entered via {tr.first_side or 'center'} side"
                    payload = {
                        "type": etype,
                        "ts": ts_iso(ts_local),
                        "source": "cam_sim",
                        "tid": tid,
                        "cls": tr.cls,
                        "confidence": round(tr.conf, 3),
                        "bbox_pxls": bbox_pxls,
                        "msg": msg,
                    }
                    write_event_line(ts_local, payload)
                    # start per-object span for object.summary
                    if tid not in obj_spans:
                        obj_spans[tid] = {
                            "cls": tr.cls,
                            "start_ts_iso": ts_iso(ts_local),
                            "start_epoch": now_s,
                            "first_side": tr.first_side or "center",
                        }

        cur_ids = set(tracks.keys())
        gone_ids = static_prev_ids - cur_ids

        # track.exit when a track disappears; emit object.summary here too
        for gid in gone_ids:
            tr = static_last_known.get(gid)
            if tr:
                x1, y1, x2, y2 = tr["bbox"]
                first_side = tr.get("first_side")
                last_side = tr.get("last_side")
                exit_side, exit_reason = infer_exit_side_and_reason(first_side, last_side)

                etype = "track.exit"
                if not under_cooldown(etype, gid, now_s):
                    bbox_pxls = [
                        round(x1, 1), round(y1, 1),
                        round(x2, 1), round(y2, 1)
                    ]

                    if exit_side == "center" and exit_reason == "lost_in_center":
                        msg = (
                            f"{tr['cls']}[tid={gid}] lost near center; "
                            f"exit_side=center, exit_reason=lost_in_center "
                            f"(not literally, the detector lost it)"
                        )
                    else:
                        msg = (
                            f"{tr['cls']}[tid={gid}] exited toward {exit_side} side "
                            f"(exit_reason={exit_reason})"
                        )

                    payload = {
                        "type": etype,
                        "ts": ts_iso(ts_local),
                        "source": "cam_sim",
                        "tid": gid,
                        "cls": tr["cls"],
                        "bbox_pxls": bbox_pxls,
                        "last_side": last_side,
                        "exit_side": exit_side,
                        "exit_reason": exit_reason,
                        "msg": msg,
                    }
                    write_event_line(ts_local, payload)

                    # close per-object span and emit object.summary
                    span = obj_spans.pop(gid, None)
                    if span:
                        start_iso = span["start_ts_iso"]
                        end_iso = ts_iso(ts_local)
                        duration_s = round(now_s - span["start_epoch"], 3)
                        first_side_span = span.get("first_side") or first_side or exit_side

                        # use exit_side for "where it ended"
                        if first_side_span and exit_side and first_side_span != exit_side and exit_side != "center":
                            side_phrase = f" from {first_side_span} to {exit_side}"
                        elif exit_side and exit_side != "center":
                            side_phrase = f" near {exit_side} side"
                        elif first_side_span:
                            side_phrase = f" near {first_side_span} side (lost_in_center)"
                        else:
                            side_phrase = ""

                        msg_summary = (
                            f"{span['cls']}[tid={gid}] seen{side_phrase} "
                            f"for {duration_s:.1f}s ({start_iso} → {end_iso})"
                        )

                        if exit_side == "center" and exit_reason == "lost_in_center":
                            msg_summary += (
                                " – track lost near center; not always exited"
                            )

                        obj_summary = {
                            "type": "object.summary",
                            "ts": ts_iso(ts_local),
                            "source": "cam_sim",
                            "tid": gid,
                            "cls": span["cls"],
                            "start_ts": start_iso,
                            "end_ts": end_iso,
                            "duration_s": duration_s,
                            "first_side": first_side_span,
                            "last_side": last_side,
                            "exit_side": exit_side,
                            "exit_reason": exit_reason,
                            "msg": msg_summary,
                        }
                        write_event_line(ts_local, obj_summary)

        # refresh last_known state for the next frame
        new_last_known = {}
        for tid, tr in tracks.items():
            new_last_known[tid] = {
                "cls": tr.cls,
                "bbox": tr.bbox,
                "last_side": tr.last_side,
                "first_side": tr.first_side,
            }

        main._prev_ids = cur_ids
        main._last_known = new_last_known
        main._obj_spans = obj_spans

        # ---------------- Burst logic (scene-level: something appears, hangs around, vanishes) ----------------
        if (consec_over_gate >= MIN_MOTION_FRAMES) or presence:
            # burst is "live"
            if not burst_open:
                # open new burst
                burst_open = True
                burst_start_s = now_s - PRE_S
                burst_last_seen_s = now_s
                burst_peak_motion = mot_frac
                burst_first_ts_utc = ts_utc
                burst_last_ts_utc = ts_utc
                burst_first_objects = sorted(set(objects))
                burst_seen_objects = sorted(set(objects))
                label = "/".join(burst_first_objects) if burst_first_objects else "-"
                log(f"BURST start: objs={burst_first_objects or ['-']} at {burst_first_ts_utc.isoformat()}")
                msg_start = f"burst started when {label} seen; motion_score={mot_frac:.2f}"
                start_ev = {
                    "type": "burst.start",
                    "ts": ts_iso(ts_local),
                    "source": "cam_sim",
                    "objects": burst_first_objects,
                    "motion_score": round(mot_frac, 3),
                    "msg": msg_start,
                }
                write_event_line(ts_local, start_ev)
            else:
                # update existing burst
                burst_last_seen_s = now_s
                burst_last_ts_utc = ts_utc
                burst_peak_motion = max(burst_peak_motion, mot_frac)
                if objects:
                    # union of everything we've seen during this burst
                    burst_seen_objects = sorted(set(burst_seen_objects).union(objects))
        else:
            # no motion/presence right now; maybe close burst
            if burst_open:
                no_motion_long_enough = (now_s - burst_last_seen_s) >= DEBOUNCE_S
                presence_still = (now_s - last_presence_time) < PRESENCE_HOLD_S
                if no_motion_long_enough and not presence_still:
                    end_s = (burst_last_seen_s or now_s) + POST_S
                    duration = max(0.0, end_s - (burst_start_s or now_s))

                    if duration >= MIN_EVENT_SEC:
                        # wording: "cat was seen at X, last seen at Y, gone"
                        start_iso = burst_first_ts_utc.isoformat() if burst_first_ts_utc else "?"
                        end_iso = burst_last_ts_utc.isoformat() if burst_last_ts_utc else "?"
                        label = "/".join(burst_seen_objects) if burst_seen_objects else "-"
                        STATUS.print(
                            f"burst: {label} start={start_iso} last={end_iso} dur={duration:.1f}s"
                        )

                        objects_for_event = burst_seen_objects or burst_first_objects or []
                        msg_motion = (
                            f"{label if label != '-' else 'motion'} from {start_iso} "
                            f"to {end_iso} (~{duration:.1f}s), peak motion_score={burst_peak_motion:.2f}"
                        )
                        motion_ev = {
                            "type": "motion.summary",
                            "event_id": f"ev_{short_id_from_ts(now_s)}",
                            "ts": ts_iso(ts_local),
                            "source": "cam_sim",
                            "motion_score": round(burst_peak_motion, 3),
                            "novelty": None,  # demo keeps novelty internal; jouirnal ignores it anyway
                            "objects": objects_for_event,
                            "frames": {},
                            "clip": "",
                            "notes": f"burst start={start_iso} last={end_iso} dur={duration:.1f}s",
                            "msg": msg_motion,
                        }
                        write_event_line(ts_local, motion_ev)

                        msg_end = (
                            f"burst ended for {label if label != '-' else 'motion'}; "
                            f"duration {duration:.1f}s ({start_iso} → {end_iso})"
                        )
                        # also emit a burst.end event for anyone reading raw JSON later
                        end_ev = {
                            "type": "burst.end",
                            "ts": ts_iso(ts_local),
                            "source": "cam_sim",
                            "objects": objects_for_event,
                            "start_ts": start_iso,
                            "end_ts": end_iso,
                            "duration_s": round(duration, 3),
                            "msg": msg_end,
                        }
                        write_event_line(ts_local, end_ev)

                    # reset burst state
                    burst_open = False
                    burst_peak_motion = 0.0
                    burst_start_s = 0.0
                    burst_last_seen_s = 0.0
                    burst_first_ts_utc = None
                    burst_last_ts_utc = None
                    burst_first_objects = []
                    burst_seen_objects = []

        frame_idx += 1

    # ---- Flush any open burst on exit (EOF / ctrl-c mid-burst) ----
    if burst_open:
        now_s = time.time()
        end_s = (burst_last_seen_s or now_s) + POST_S
        duration = max(0.0, end_s - (burst_start_s or now_s))
        if duration >= MIN_EVENT_SEC:
            ts_local = now_local()
            ts_utc = ts_local.astimezone(timezone.utc)
            start_iso = burst_first_ts_utc.isoformat() if burst_first_ts_utc else "?"
            end_iso = burst_last_ts_utc.isoformat() if burst_last_ts_utc else "?"
            label = "/".join(burst_seen_objects) if burst_seen_objects else "-"
            STATUS.print(
                f"burst[flush]: {label} start={start_iso} last={end_iso} dur={duration:.1f}s"
            )

            objects_for_event = burst_seen_objects or burst_first_objects or []
            msg_motion = (
                f"{label if label != '-' else 'motion'} from {start_iso} "
                f"to {end_iso} (~{duration:.1f}s), peak motion_score={burst_peak_motion:.2f} (flush)"
            )
            motion_ev = {
                "type": "motion.summary",
                "event_id": f"ev_{short_id_from_ts(now_s)}",
                "ts": ts_iso(ts_local),
                "source": "cam_sim",
                "motion_score": round(burst_peak_motion, 3),
                "novelty": None,
                "objects": objects_for_event,
                "frames": {},
                "clip": "",
                "notes": f"flush_on_exit start={start_iso} last={end_iso} dur={duration:.1f}s",
                "msg": msg_motion,
            }
            write_event_line(ts_local, motion_ev)

            msg_end = (
                f"burst ended for {label if label != '-' else 'motion'} on flush; "
                f"duration {duration:.1f}s ({start_iso} → {end_iso})"
            )
            end_ev = {
                "type": "burst.end",
                "ts": ts_iso(ts_local),
                "source": "cam_sim",
                "objects": objects_for_event,
                "start_ts": start_iso,
                "end_ts": end_iso,
                "duration_s": round(duration, 3),
                "msg": msg_end,
            }
            write_event_line(ts_local, end_ev)

        # reset burst state (not strictly necessary at process end)
        burst_open = False

    try:
        cap.release()
    except Exception:
        pass
    STATUS.done()

    # ---- Demo companion shard wiring (before/after/demo + transcript) ----
    try:
        _handle_demo_files(run_start_local)
    except Exception as e:
        log(f"Demo helper: unexpected error while setting up demo companions: {e}")

    log("Capture stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        STATUS.done()
        log("Interrupted by user.")
