# Voice AI Agent — Quick Starter

A minimal Voice AI Agent built with [Pipecat](https://github.com/pipecat-ai/pipecat) that wires up **Deepgram STT + Anthropic Claude LLM + Deepgram TTS** into a conversational voice pipeline over **WebRTC** (browser). Includes a **two-bot arguing mode** where two AI personas debate while you listen.

## What This Agent Does

- **Speech-to-Text**: Deepgram Nova for accurate real-time transcription
- **LLM**: Anthropic Claude Sonnet for fast, conversational responses
- **Text-to-Speech**: Deepgram Aura 2 for natural-sounding voice output
- **WebRTC**: Talk to the agent directly from your browser
- **Voice Activity Detection**: Silero VAD for natural turn-taking (200ms stop threshold)
- **Arguing Mode**: Two AI bots debate a topic while you listen and can interject
- **Metrics**: Built-in pipeline and usage metrics

## Architecture

### Default Mode

```
┌──────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐     ┌──────────┐
│  Audio   │────>│ Deepgram │────>│ Anthropic │────>│ Deepgram │────>│  Audio   │
│  Input   │     │ STT      │     │ Claude    │     │ TTS      │     │  Output  │
└──────────┘     └──────────┘     └───────────┘     └──────────┘     └──────────┘
   WebRTC          Nova           Claude Sonnet      Aura 2            WebRTC
```

### Arguing Mode

```
                 ┌──────────┐     ┌───────────┐     ┌─────────────┐     ┌──────────┐     ┌──────────┐
  Audio In ─────>│ Deepgram │────>│ Anthropic │────>│   Debate    │────>│ Deepgram │────>│  Audio   │
  (user can      │ STT      │     │ Claude    │     │  Manager    │     │ TTS      │     │  Output  │
   interject)    └──────────┘     └───────────┘     └─────────────┘     └──────────┘     └──────────┘
                                                     Turn-taking          Voice
                                                     & context mgmt      switching
```

Two bots (Nova the optimist and Rex the skeptic) take turns arguing. The Debate Manager handles turn-taking, maintains separate conversation histories for each bot, and switches TTS voices between turns.

## Project Structure

```
voicebot-quick-starter/
├── bot.py              # Core pipeline: STT → LLM → TTS (default mode)
├── default_runner.py   # WebRTC runner — browser-based voice chat
├── arguing_runner.py   # Arguing runner — two bots debate over WebRTC
├── pyproject.toml      # Poetry project config & dependencies
├── .env                # Environment variables (not committed)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Quick Start

### Prerequisites

- Python 3.10+
- A [Deepgram](https://deepgram.com) API key (free tier available)
- An [Anthropic](https://console.anthropic.com) API key

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/exotel/voicebot-quick-starter.git
   cd voicebot-quick-starter
   ```

2. **Set up Python environment with micromamba:**

   Install micromamba (one-time setup):

   ```bash
   "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
   ```

   The installer will ask a few questions — accept the defaults:

   | Prompt | What to enter |
   |--------|---------------|
   | Micromamba binary folder? `[~/.local/bin]` | Press **Enter** |
   | Init shell (bash)? `[Y/n]` | Type **Y**, press **Enter** |
   | Configure conda-forge? `[Y/n]` | Type **Y**, press **Enter** |
   | Prefix location? `[~/micromamba]` | Press **Enter** |

   Then reload your shell and create the environment:

   ```bash
   source ~/.bashrc

   micromamba create -n dev_conf python=3.12 -c conda-forge
   micromamba activate dev_conf
   ```

3. **Install dependencies with Poetry:**

   ```bash
   pip install poetry
   poetry install
   ```

4. **Configure environment variables:**

   Create a `.env` file inside the `voicebot-quick-starter/` directory:

   ```bash
   DEEPGRAM_API_KEY=your-deepgram-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

### Running the Agent

#### Default Mode (Voice Chat)

```bash
python default_runner.py
```

Open **http://localhost:7860/client** in your browser and click **Connect**. Speak into your microphone to chat with the agent.

#### Arguing Mode (Two Bots Debate)

```bash
python arguing_runner.py
```

Open **http://localhost:7860/client** in your browser and click **Connect**. Two AI bots will begin debating. You can speak to interject — your comment is added to both bots' conversation context.

## Customizing the Agent

### Default Mode (`bot.py`)

- **System prompt** — Change the agent's personality and behavior
- **LLM model** — Currently using `claude-sonnet-4-5-20250929`
- **TTS voice** — Currently using `aura-2-helena-en`; see [Deepgram voices](https://developers.deepgram.com/docs/tts-models)
- **VAD sensitivity** — Adjust `stop_secs` in `VADParams` for turn-taking timing

### Arguing Mode (`arguing_runner.py`)

- **Debate topic** — Change `DEBATE_TOPIC` at the top of the file
- **Bot personas** — Edit `BOT_A_SYSTEM` and `BOT_B_SYSTEM` prompts
- **Bot voices** — Change `BOT_A_VOICE` and `BOT_B_VOICE` (Deepgram Aura 2 voices)
- **Max turns** — Adjust `MAX_TURNS` to control debate length
- **LLM temperature** — Higher values make arguments more creative/unpredictable

### Swap in Your Preferred STT, LLM, or TTS

Pipecat supports a wide range of providers. You can replace any component in the pipeline by swapping the service class. Refer to the examples in the Pipecat repo for guidance:

| Component | Supported Providers | Examples |
|-----------|-------------------|----------|
| **STT** | Deepgram, AssemblyAI, Whisper, Azure, Google | [STT examples](https://github.com/pipecat-ai/pipecat/tree/main/examples/foundational) |
| **LLM** | OpenAI, Anthropic, Google Gemini, Azure, Groq | [LLM examples](https://github.com/pipecat-ai/pipecat/tree/main/examples/foundational) |
| **TTS** | Cartesia, ElevenLabs, PlayHT, Deepgram, Google, Azure | [TTS examples](https://github.com/pipecat-ai/pipecat/tree/main/examples/foundational) |

For the full list of supported services, see the [Pipecat services documentation](https://docs.pipecat.ai/server/services/overview).

## Deployment

Deploy to any cloud provider (GCP, AWS, etc.) and ensure:

- WebSocket connections are supported by your load balancer
- Environment variables (`DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`) are injected via a secrets manager

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | [Pipecat](https://github.com/pipecat-ai/pipecat) v0.0.101 |
| STT | [Deepgram](https://deepgram.com) Nova |
| LLM | [Anthropic](https://anthropic.com) Claude Sonnet |
| TTS | [Deepgram](https://deepgram.com) Aura 2 |
| VAD | Silero VAD |
| WebRTC | Pipecat SmallWebRTC |

## Troubleshooting

### No Audio Output

- Ensure your browser allows microphone access
- Check that `DEEPGRAM_API_KEY` and `ANTHROPIC_API_KEY` are set correctly in `.env`
- Review the terminal logs for `ErrorFrame` messages

### High Latency

- Adjust `stop_secs` in VAD params — lower values mean faster turn-taking but may cut off speech

### Arguing Mode Issues

- If bots stop responding, check the terminal for error messages — API rate limits can cause interruptions
- If voices sound the same, verify the Deepgram voice IDs are valid Aura 2 models

## License

MIT

## Acknowledgments

- [Pipecat](https://github.com/pipecat-ai/pipecat) — Open-source framework for voice and multimodal AI agents
- [Deepgram](https://deepgram.com) — Speech-to-text and text-to-speech
- [Anthropic](https://anthropic.com) — Claude LLM
