# ETHEL Midbrain Demo (Detector → Journaler → Summarizer)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Summary

This repo contains a compact, self-contained demonstration of the ETHEL midbrain (spine) pipeline — the part of ETHEL responsible for perceiving the world, writing down what happened, and summarizing it into something usable for later stages.  
It’s a focused, runnable demo that shows the core loop:  

Video → JSONL events → SQLite journal → Hourly/Daily summaries  

As a demo, it represents a minimal version of the real flow, packaged so anyone can clone it and run it easily.

---

## What This Demo Is (and Isn’t)

### Is:

- A faithful representation of ETHEL’s core perception pipeline behaviour, simplified in structure  
- A clear, runnable example of the system’s backbone  
- A portable piece showing how ETHEL processes stimuli  

### Is not:

- Live Whisper audio. A sample transcript is included for demonstration.  
- The full ETHEL architecture  
- A complete media/clip subsystem  
- A full speech pipeline  
- A production DB schema or detector stack  

This is the “midbrain”: enough to show ingest → perceive → log → summarize, without the broader director/LLM layers.

---

## What This Demo Shows

- Motion detection  
- YOLO object detection (default but optional)  
- Novelty detection via pHash  
- Track enter/exit events  
- Burst detection + motion summaries  
- Conversion of event shards into a structured SQLite journal  
- Hourly and daily summary generation (motion, speech, lexical stats)  

All scripts are simplified and single-purpose.  
The structure mirrors ETHEL’s real midbrain stages, but is intentionally lightweight.

---

## Prerequisites

- Python 3.12  
- Windows 10/11 or Linux  
  (Paths in examples are Windows style, but the scripts are cross-platform.)  

- ffmpeg installed and available in your PATH  
  (Only required if you switch the video source to formats that OpenCV can’t decode natively.)  

- GPU optional  
  YOLO runs on CPU if CUDA isn’t available; performance is fine for the demo.

---

## Repository Layout

```txt
midbrain_demo/
  LICENSE
  README.md
  requirements.txt
  
  configs/
    detector_config.txt
    event_config.txt
    motion_config.txt
    novelty_config.txt
  db/
  events/
  ingest/
    before_demo.jsonl
    after_demo.jsonl
    shimmer_test.mp4
    source_config.txt
    transcript_demo.jsonl
  logs/
  scripts/
    Detect.py
    journaler.py
    summarizer.py
    yolov8n.pt
  summaries/
```

---

## Installation

You can use a virtual environment or install globally.

### Create venv (recommended)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run the Demo

### 1. Stage 2 — Detector (video → events)

Runs as a one-liner:

```bash
python scripts/Detect.py
```

Uses `ingest/source_config.txt` unless you override:

```bash
python scripts/Detect.py --source 0
python scripts/Detect.py --source rtsp://...
```

Outputs JSONL event shards to:

```txt
events/YYYY-MM-DD/events_YYYYMMDD_HH.jsonl
```

It also retimes the demo `before_*`, `after_*`, and transcript files into the correct hour window so the journaler can ingest them cleanly.

---

### 2. Stage 3 — Journaler (events → SQLite)

One-shot ingest:

```bash
python scripts/journaler.py
```

Follow detector live (tail mode):

```bash
python scripts/journaler.py --tail
```

This creates and updates:

- `db/ethel_journal_demo.db` with three tables: events, clips (empty in demo), captions.

---

### 3. Stage 4 — Summarizer (SQLite → summaries)

Run after you have events in the DB:

```bash
python scripts/summarizer.py
```

or summarize a specific UTC day:

```bash
python scripts/summarizer.py --date YYYY-MM-DD
```

Outputs:

- Hourly rollups → `summaries/summ_hourly.jsonl`  
- Daily summary → `summaries/YYYY-MM-DD.json`

---

## Config Files

All configs are human-readable key=value files under `configs/` and `ingest/`.  
Defaults are tuned for running the demo straight out of the box.

- `detector_config.txt` — model, YOLO settings, cadence  
- `motion_config.txt` — motion thresholds  
- `novelty_config.txt` — pHash novelty window and distance  
- `event_config.txt` — track rules, burst rules, shard strategy  
- `ingest/source_config.txt` — video file, camera index, or stream URL  

---

## License

© 2025 Kena Teite. Released under the MIT License.  
See the LICENSE file for full terms.
