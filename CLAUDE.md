# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice AI Agent built with [Pipecat](https://github.com/pipecat-ai/pipecat) v0.0.101. Wires up **Deepgram STT → Anthropic Claude LLM → Deepgram TTS** into a conversational voice pipeline over WebRTC (browser). Includes a two-bot arguing mode where two AI personas debate a topic, and a recruiter intake trainer mode where a human recruiter practices intake calls against an AI hiring manager.

## Commands

### Setup
```bash
micromamba create -n dev_conf python=3.12 -c conda-forge
micromamba activate dev_conf
pip install poetry
poetry install
```

### Run
```bash
# Default voice chat — browser at http://localhost:7860/client
python default_runner.py

# Arguing mode — two bots debate, browser at http://localhost:7860
python arguing_runner.py

# Recruiter intake trainer — practice intake calls, browser at http://localhost:7860
# You speak (STT), HM responds as text only (no voice output)
python recruiter_runner.py
```

There are no tests, linting, or build steps in this project.

## Architecture

### Default Mode

Three-file architecture with a shared pipeline core and a transport runner:

- **`bot.py`** — Core pipeline. `run_bot(transport, runner_args)` constructs the full STT → LLM → TTS pipeline with Silero VAD (200ms stop threshold).
- **`default_runner.py`** — WebRTC entry point. Creates a `SmallWebRTC` transport via `create_transport()`, serves browser client at `/client` on port 7860.

Pipeline order in `bot.py`:
```
transport.input() → STT → user_aggregator → LLM → TTS → transport.output() → assistant_aggregator
```

### Arguing Mode

- **`arguing_runner.py`** — Custom FastAPI app (no `pipecat.runner.run.main()`) with a custom `DebateManager` processor. Uses the Pipecat prebuilt WebRTC UI at `/client` for audio. A simple config page at `/` lets you set the topic and bot roles before connecting.

Key endpoints:
- `GET /` — Simple config page for topic, bot names/roles. Saves and redirects to `/client`.
- `GET /api/config` — Returns current debate configuration as JSON
- `POST /api/config` — Receives debate configuration from the UI
- `POST /api/offer` — WebRTC SDP offer
- `PATCH /api/offer` — WebRTC ICE trickle
- `/client` — Pipecat prebuilt WebRTC UI (`SmallWebRTCPrebuiltUI` mount)

Pipeline order in `arguing_runner.py`:
```
transport.input() → STT → LLM → DebateManager → TTS → transport.output()
```

The DebateManager:
- Maintains separate conversation histories for two bots (configurable names/roles)
- Collects LLM output text between `LLMFullResponseStartFrame` and `LLMFullResponseEndFrame`
- On response end, switches to the other bot: swaps voice via `tts.set_voice()`, builds new `LLMContext`, and queues `LLMContextFrame` via `task.queue_frames()`
- Intercepts `TranscriptionFrame` for user interjections and adds them to both bots' histories

Bot-to-bot mode: The user connects via the prebuilt UI at `/client`. The debate auto-starts on connection. Users configure the debate via the config page at `/` before connecting.

### Recruiter Intake Trainer Mode

- **`recruiter_runner.py`** — Custom FastAPI app with a custom split-screen frontend (no Pipecat prebuilt UI). The recruiter (human) speaks via mic (Deepgram STT transcribes), and the AI hiring manager responds as **text only** (no voice output). The HM has a "can't explain properly" personality. Includes a dynamic checklist, live transcript, and LLM-powered question suggestions.

Key files:
- `static/config.html` — Landing page where the recruiter enters the role name
- `static/index.html` — Split-screen session UI with WebRTC, transcript, checklist, suggest button

Key endpoints:
- `GET /` — Config/landing page (role name input)
- `POST /api/config` — Receives role name, generates hidden requirements and HM system prompt
- `GET /session` — Main session split-screen UI
- `POST /start` — WebRTC session creation
- `POST /api/offer` — WebRTC SDP offer
- `PATCH /api/offer` — WebRTC ICE trickle
- `GET /api/transcript/stream` — SSE endpoint for real-time transcript updates
- `POST /api/checklist/generate` — Generates dynamic checklist items for the role
- `POST /api/checklist/evaluate` — Evaluates conversation against checklist, auto-ticks items
- `POST /api/suggest` — Side-channel LLM call for question suggestions

Pipeline order:
```
transport.input() → STT → UserTranscriptProcessor → user_aggregator → LLM → HMTranscriptProcessor → transport.output() → assistant_aggregator
```

Key architectural decisions:
- Two separate `FrameProcessor` subclasses: `UserTranscriptProcessor` (captures STT output, placed before aggregator) and `HMTranscriptProcessor` (captures LLM output, placed after LLM)
- SSE via native `StreamingResponse` (no sse_starlette dependency)
- Three separate LLM contexts: voice pipeline (HM persona), checklist evaluation (side-channel), question suggestions (side-channel)
- Hidden requirements generated per-role at session start, embedded in HM system prompt but never shown to the recruiter
- Custom vanilla JS WebRTC client (no React/build step) with audio-only (3 transceivers to match pipecat server expectations)

## Configuration

Environment variables in `.env`:
```
DEEPGRAM_API_KEY=your-deepgram-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Customization Points

### Default mode (bot.py)
- System prompt (the `messages` list)
- LLM model (`claude-sonnet-4-5-20250929`), temperature, max_tokens
- TTS voice (`aura-2-helena-en`)
- VAD `stop_secs` (currently 0.2s) — controls turn-taking sensitivity

### Arguing mode (arguing_runner.py)
- Topic, bot names, and bot roles — configured via the web UI at `/`
- `BOT_A_VOICE` / `BOT_B_VOICE` — Deepgram Aura 2 voice IDs (hardcoded defaults)
- `MAX_TURNS` — total debate turns before auto-stopping
- LLM temperature (0.9 for more creative arguments)

### Recruiter trainer mode (recruiter_runner.py)
- HM personality — currently "can't explain properly"; add more personalities by modifying `build_hm_system_prompt()`
- No TTS — HM responds as text only, displayed in the transcript panel
- Hidden requirements — generated dynamically per role via `generate_hidden_requirements()`
- Checklist items — generated dynamically per role via `/api/checklist/generate`
- VAD `stop_secs` (0.4s) — slightly longer than default to let the recruiter finish speaking
- LLM temperature (0.85) — slightly creative for natural-sounding vagueness

Pipecat supports swapping in other providers (OpenAI, ElevenLabs, Cartesia, etc.) by changing the service classes.
