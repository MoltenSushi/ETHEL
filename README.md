TL;DR
- ETHEL is a fully local AI system that watches a single environment, writes down what it sees and hears, and builds a long-term memory of that space. The core pipeline is working today (vision → captions → database → summaries → analytics → chat). A per-entity weight system (novelty, comfort, expectation) and proper identity layer (“Sparky vs dog”) are designed but not mathematically specified or implemented yet.

-----

ETHEL:

Emergent Tethered Habitat-aware Engram Lattice

ETHEL is a local, self-contained system built from multiple open-source components working together: vision, audio, language, and memory. It runs on a single Windows machine, stays in one environment, and forms an understanding of that space through repetition, absence, observation of interactions, and change.

This isn’t a single model. ETHEL is a stack of parts, each doing one job, connected by a simple bridge so they act like one system.

-----

What ETHEL Is Trying to Be --
- A continuous observer of one physical space, not a general chatbot.
- Local-first: no cloud calls; all models and data live on the machine.
- Transparent: every detection, caption, summary, and decision is logged in human readable entries -- plain text or SQLite.
- A long-running experiment in environmental continuity and emergent bias/personality from accumulated experience.

-----
-----

Pipeline Overview (Parts)

Part 1 – Inputs (Camera & Mic)
- Takes a single primary video source (RTSP, USB, or file).
- Takes a single audio source for ambient sound and speech.

-----

Part 2 – Eyes (Video Detector / Recorder – detectorv3.py)

Eyes = what happened, frame by frame, and where the evidence is stored

What it does:
- Captures video from the configured source (RTSP/USB/file).
- Runs YOLO-based detection on the stream.
- Maintains a rolling 15-minute MKV buffer:
  - Writes fixed-length 15-minute recording chunks.
  - Keeps a 7-day history of these MKVs for later re-scan or model retraining.
- Generates a continuous stream of still frames (≈3 fps) into a rolling 15-minute stills buffer used by Qwen.
- Computes motion scores and perceptual hashes to:
  - decide when something counts as an “event”
  - reduce duplicate events.
- Creates structured event JSONL files with:
  - timestamps
  - object detections (boxes, classes, confidences)
  - motion metrics
  - links to stills and clips.
- Cleans up old stills and recordings according to retention rules.
- Emits everything into a standard directory tree (record/, events/, rolling media/), so later parts can treat it as a stable data source.

-----

Part 2.5 – Lens (Qwen Vision Server + Adapter – qwen_server.py, vision_qwen.py) 

Lens = turn these stills into concise language.

What it does:
- Wraps Qwen2-VL as a local HTTP server, using an OpenAI-style /v1/chat/completions interface.
- Uses an adapter (vision_qwen.py) to:
  - load Qwen with int8/int4/fp16 quantization
  - downscale images to a fixed max side
  - build proper chat templates for text-only or image+text queries.
- Supports:
  - text-only prompts
  - image+text prompts for vision questions
  - “fake streaming” by chunking single outputs.
- On visual requests, finds the latest still in the rolling media buffer and sends it to Qwen with an instruction prompt.
- Returns short, single-sentence captions for each burst of stills (“a person walks through the room and sits down”).
- Acts as the visual “front end” for both:
  - the Qwen Observer (below)
  - any tools that need direct visual descriptions.

-----

Part 2.8 – Ears (Audio Ingest + Whisper – 2_audio.py)

What it does:
- Uses FFmpeg to stream raw audio from the source.
- Runs VAD (webrtcvad) to detect actual speech vs background noise.
- Chunks speech into WAV files and drops very short/quiet fragments to reduce junk.
- Sends speech chunks to a background Whisper worker for transcription.
- Writes output into:
  - events/YYYY-MM-DD/audio/chunks/… (audio files)
  - events/YYYY-MM-DD/audio/transcripts.jsonl (aligned transcripts).
- Handles FFmpeg restarts with backoff and logs failures separately.
- Keeps everything time-aligned so the Journaler can treat audio like just another event stream.
- Right now, Whisper logs to disk; it isn’t yet wired directly through the bridge to Llama.

-----

Part 2.9 – Lens-Observer (Qwen Observer – qwen_observer.py)

The “visual narrator” that glues Eyes + Lens + DB together.

Lens-Observer = what just happened in the last second or two, in one sentence?

What it does:
- Watches events/YYYY-MM-DD/.../stills/ for new stills coming from Eyes.
- Groups stills into ~1-second bursts (small temporal windows).
- For each burst:
  - sends multiple frames to the Qwen server (Lens)
  - asks for a single concise caption of the burst.
  - De-duplicates captions that are near-identical to avoid spam.
  - Inserts one row per burst into vision_events in the SQLite journal, including:
    - timestamp
    - caption text
    - list of frame paths used.
- Runs continuously with a small polling interval and batching window.

-----

Part 3 – Journaler (SQLite Event Store – journalerv2.py)

Journaler = make all this chaos queryable.

What it does:
- Consumes outputs from:
  - Eyes (events, boxes, clips, MKV refs)
  - Ears (transcripts)
  - Lens-Observer (vision event captions).
- Creates and maintains the journal database with tables like:
  - meta
  - clips
  - events
  - boxes
  - captions
  - responses
  - vision_events (linked to still bursts).
- Uses WAL mode and tuned pragmas for high write throughput.
- Normalizes paths and IDs so every event is queryable by time and type.
- Aligns everything to time segments (e.g., 15-minute chunks) for easier rollups.
- Is idempotent and schema-safe: can be re-run without destroying existing data.
- Acts as the single source of truth for “what ETHEL has seen and heard.”
- Feeds downstream summary/analytics jobs (Parts 4 and 5).

-----

Part 4 – Summarizer (Hourly + Daily Rollups – stage4_summarizer.py)

-- This runs periodically, not as a permanent process.

Summarizer = yesterday on one page

What it does:
- Connects read-only to the journal DB.
- Produces hourly rollups into summaries/summ_hourly.jsonl:
  - counts of events, motion, captions
  - speech totals (seconds, words, characters)
  - object and presence stats.
- Produces daily summaries into summaries/YYYY-MM-DD.json:
  - aggregated event patterns
  - who/what appeared that day
  - activity profile across the day.
- Verifies that summarized IDs actually exist in the DB (sanity checking).
- Acts as an intermediate compression layer between raw events and long-term analytics.
- Gives higher-level parts a “what today looked like” view instead of raw events.
- Can be re-run to regenerate summaries if the schema is extended.
- Serves as the main input to analytics and context memory.

-----

Part 5 – Analytics (Trends & Anomalies – stage5_analytics.py)

-- Also a periodic job.

Analytics = how this week compares to last week.

What it does:
- Reads hourly and daily summary files from Part 4.
- Computes per-day metrics such as:
  - total events
  - motion vs object mix
  - novelty scores
  - confidence patterns
  - uptime/downtime
  - duplicate/stale rates.
- Builds rolling baselines and labels them (e.g., cold / stale / ready).
- Detects anomalies:
  - spikes or drops in activity
  - shifts in object mix
  - missing expected motion
  - unusual novelty or confidence changes.
- Writes outputs to:
  - analytics/day_YYYY-MM-DD.json
  - analytics/trends.json
  - analytics/flags.jsonl (and optional CSV/PNG).
- Acts as the long-term health and environment trend monitor.
- Gives higher-level reasoning a handle on “is today weird compared to last week?”
- Provides the natural place to hook in future weight adjustments.

-----

Part 6 – Cortex (Llama + State Daemon – llama_state_daemon.py)

Cortex = if you ask ETHEL a question, this is "who" answers.

What it does:
- Uses Llama 3.x 8B via Ollama as the main language model.
- Receives chat and reasoning requests from the Bridge (Part 7).
- Periodically:
  - reads recent events, captions, and summaries
  - compresses them into short contextual summaries.
  - Writes these into state_summaries in the DB (or adjacent files) as:
    - short-term context
    - “what ETHEL has been seeing lately.”
- Keeps the context window manageable by summarizing instead of replaying raw logs.
- Treats visual and audio information indirectly, via the written captions and summaries.
- Is entirely local; no external calls.
- Is the main place where “memory” is turned into answerable text when you talk to ETHEL.

-----

Part 7 – Bridge / “Thalamus” (Router – ethel_bridge_server.py)

This is the internal API that makes ETHEL feel like one brain.

Thalamus = which part of the brain should handle this?

What it does:
- Exposes an OpenAI-style /v1/chat/completions endpoint.
- Routes each request to:
  - Llama only,
  - Qwen only, or
  - a chained path (Qwen first for vision, then Llama for reasoning).
- Uses meta tags and lightweight routing logic to:
  - distinguish visual vs non-visual queries
  - decide when to refresh the latest frame
  - suspend vision entirely for text-only questions.
- Accepts control tags like [VISION=REFRESH] in the raw input to force a fresh look.
- Adds timing and logging so behavior is auditable.
- Supports streaming and non-streaming responses.
- Does not yet integrate Whisper directly; audio is still going via transcripts → DB → summaries.
- Is the single entry point that external clients can talk to without knowing about YOLO, Qwen, or Llama.

-----

Part 7.5 – Context Proxy (Front Door – context_proxy.py)

-- A thin layer in front of the Bridge. 

Context Proxy = add what ETHEL remembers to whatever you just asked.

What it does:
- Exposes another /v1/chat/completions endpoint.
- Before forwarding a request to the Bridge, pulls recent short-term summaries from the DB.
- Injects this into the prompt as a system-level “persistent memory” block, so:
  - Llama always sees the latest context
  - client code doesn’t need to manage history.
- Passes through all the normal OpenAI semantics (models, messages, streaming).
- Keeps the external interface simple while ETHEL’s internals stay complex.
- Acts as the layer that gives Llama continuity — it pulls recent hourly/daily summaries and injects them into each request.
- Prevents overload by sending Llama compact summaries instead of raw event logs.

-----

Part 8 – Qwen Chat Adapter (ETHEL ↔ Qwen conversational layer – ethel_chat_qwen.py)

Optional direct chat path to the vision model, mainly for debugging and direct Qwen conversations when needed

What it does:
- Sends chat-style prompts directly to the Qwen server.
- Adds an ETHEL persona and rules for:
  - being brief and factual
  - not leaking file paths or overlays
  - ignoring OSD timestamps.
- Handles tool gating, including a denylist so unsafe tools (e.g., PTZ control - its a long story of camera-specific hell) are blocked by pattern.
- Detects visual vs non-visual asks and sets meta flags accordingly.
- Cleans up control tags so users don’t see them echoed back.

-----
-----

Planned Behavior: Weights & “Personality” (Not Implemented Yet)

A central design goal is to attach simple, per-entity weights that change over time:
- Novelty – how new, interesting, or attention-grabbing an entity or pattern is.
- Comfort – how familiar, common, or “normal” it feels based on repeated exposure.
- Expectation – how strongly ETHEL predicts that entity or pattern “should” be present at a given time, based on history.

These weights are not coded yet. They’re intended to sit on top of the existing logs, summaries, and analytics, and influence what ETHEL pays attention to, re-checks, or asks about.

-----

Example 1 – “Sparky”

ETHEL sees an unfamiliar dog.
- Captures multiple angles, stores stills.
- Asks: “What’s this dog’s name?”
- Creates a named record with notes (e.g., dog_sparky, barks a lot, brown fur, seen first at x time on x day, etc).
- Initializes weights: novelty high, comfort low, expectation low.
  - Over repeated visits:
    - novelty decreases
    - comfort rises
    - expectation forms (“Sparky usually appears at these times”).

Later behavior around Sparky uses these weights: more attention early, more relaxed later.

-----

Example 2 – “Bill the Cat”

Bill shows up once or twice a day, every day, for months:
- novelty drops
- comfort rises
- expectation becomes strong: “Bill should appear today.”

Bill goes missing for a week:
- absence is logged as a break in pattern
- novelty may spike when he returns
- comfort may dip or be temporarily capped due to unpredictability/unreliability.

Years later, Bill dies:
- ETHEL is told he will not return.
- That fact is trained in, but the old expectation (“Bill appears daily”) is not deleted.
- A small, permanent conflict remains: “Bill should appear, but doesn’t.”

That unresolved mismatch then biases how ETHEL treats:
- new cats,
- long absences,
- entities ETHEL recognises as temporal,
- broken patterns in general.

-----

Example 3 – “Too Many Unknown Faces”

ETHEL sees a room full of unfamiliar people at once:
- novelty rises across multiple entities
- comfort drops globally
- expectation becomes unreliable because the scene no longer matches normal.
- She does not immediately ask “who is this?” for every face.
- Instead, she waits until:
  - the room is quieter, and
  - someone with a higher comfort weight is present.
- Then she shows stills and asks for names.

This isn’t because of politeness or direct instruction, it’s the weight system deciding when it “feels” safe or worthwhile to resolve unknowns.

All of this is design, not reality yet. The groundwork (events, captions, summaries, analytics) is in place so these weights can be implemented later without rewriting the whole pipeline.

-----
-----

Limitations / Gaps

Current, known limitations:
- Weight system (novelty/comfort/expectation) is not implemented.
- Identity resolution layer (“Sparky vs generic dog”) is not implemented.
- Whisper → Llama bridge (audio fed through the same routing as vision) is not wired up yet.
- Piper TTS for ETHEL’s spoken replies is not integrated.
- Expectation math (decay, thresholds, conflict handling) is only on paper.
- No formal evaluation metrics yet:
  - no YOLO accuracy/false-positive stats
  - no latency benchmarks
  - no robustness tests on anomaly flags.
- No test suite yet:
  - no unit/integration tests
  - no CI
  - no containerized deployment story.
- All inference is local, so performance depends heavily on the specific GPU/CPU.

-- I feel these are normal early-project gaps and am explicitly acknowledging them here to avoid overclaiming.

-----
-----

Current Status (~6 Weeks In)

Working today:
- Video capture, detection, 15-minute MKV recording, and stills.
- Audio capture + Whisper transcription.
- Qwen vision server + observer turning still bursts into scene captions (logged to DB and used later by the summarizer and context layers).
- SQLite journal logging everything in a normalized schema.
- Hourly and daily summarization into compact JSON.
- Analytics pass that computes trends and flags anomalies.
- Llama-based cortex with a context proxy for chat.
- Bridge routing between Qwen and Llama with visual/non-visual intent.

Designed but not built:
- Identity resolution and per-entity weight system.
- Structured expectation modeling.
- Whisper-aware reasoning through the bridge.
- Long-term “remember when…” style replay jobs.
- Voice output via Piper.

ETHEL is a live, evolving prototype: the pipeline is real and running; the personality and weighting system are the next major layer.
