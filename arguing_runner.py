"""Arguing runner — two AI bots debate while the user listens over WebRTC.

Run:  python arguing_runner.py
Then open http://localhost:7860 in your browser.

Configure the debate topic and bot roles in the UI, then click "Start Debate".
Two bots take turns arguing. The user's microphone is muted (bot-to-bot mode).
"""

import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from http import HTTPMethod

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

# --- Validate API keys early ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not ANTHROPIC_API_KEY:
    logger.error("ANTHROPIC_API_KEY not set. Check your .env file.")
    sys.exit(1)
if not DEEPGRAM_API_KEY:
    logger.error("DEEPGRAM_API_KEY not set. Check your .env file.")
    sys.exit(1)

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI
from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# ---------------------------------------------------------------------------
# Default debate config (overridden by UI)
# ---------------------------------------------------------------------------

MAX_TURNS = 20

debate_config = {
    "topic": "Will artificial intelligence ultimately be a net positive or net negative for humanity?",
    "bot_a_name": "Nova",
    "bot_a_role": "a passionate optimist who firmly believes AI will be an enormous net positive for humanity",
    "bot_a_voice": "aura-2-helena-en",
    "bot_b_name": "Rex",
    "bot_b_role": "a sharp skeptic who believes unchecked AI poses serious risks to humanity",
    "bot_b_voice": "aura-2-zeus-en",
}


# ---------------------------------------------------------------------------
# DebateManager — custom processor for turn-taking
# ---------------------------------------------------------------------------


class DebateManager(FrameProcessor):
    """Manages turn-taking between two arguing bots.

    Sits after the LLM in the pipeline. Collects LLM output text, and when
    the response ends, switches to the other bot and triggers a new turn
    by queuing an LLMContextFrame via the pipeline task.
    """

    def __init__(self, tts: DeepgramTTSService, task_ref: list, config: dict, **kwargs):
        super().__init__(**kwargs)
        self._tts = tts
        self._task_ref = task_ref
        self._config = config

        # Per-bot conversation histories
        self._history_a: list[dict] = []
        self._history_b: list[dict] = []

        self._current_bot = "A"
        self._turn_count = 0
        self._debating = False

        self._response_text = ""
        self._collecting = False

    @property
    def _task(self) -> PipelineTask:
        return self._task_ref[0]

    def _system_prompt(self, name: str, role: str) -> str:
        return (
            f"You are {name}, {role} in a live voice debate. "
            "Argue with conviction and directly counter your opponent's points. "
            "Keep each response to 2-3 sentences — this is a fast-paced spoken debate, not an essay. "
            "Sound natural and conversational. Never use bullet points, markdown, or emojis. "
            "Address your opponent directly."
        )

    async def start_debate(self):
        """Kick off the debate with Bot A's opening statement."""
        cfg = self._config
        self._debating = True
        self._current_bot = "A"
        self._turn_count = 0

        sys_a = self._system_prompt(cfg["bot_a_name"], cfg["bot_a_role"])
        sys_b = self._system_prompt(cfg["bot_b_name"], cfg["bot_b_role"])

        opening_user_msg = (
            f"The debate topic is: {cfg['topic']}\n\n"
            "You are going first. Make your opening argument."
        )
        self._history_a = [
            {"role": "system", "content": sys_a},
            {"role": "user", "content": opening_user_msg},
        ]
        self._history_b = [
            {"role": "system", "content": sys_b},
        ]

        self._tts.set_voice(cfg["bot_a_voice"])

        ctx = LLMContext(messages=list(self._history_a))
        await self._task.queue_frames([LLMContextFrame(context=ctx)])
        logger.info(f"Debate started — {cfg['bot_a_name']} goes first")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._collecting = True
            self._response_text = ""
            await self.push_frame(frame, direction)

        elif isinstance(frame, (LLMTextFrame, TextFrame)):
            if self._collecting:
                self._response_text += frame.text
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._collecting = False
            await self.push_frame(frame, direction)

            if self._debating:
                asyncio.create_task(self._next_turn())

        elif isinstance(frame, TranscriptionFrame):
            # In bot-to-bot mode mic is muted, but if user somehow speaks
            # we still handle it gracefully
            if self._debating and frame.text and frame.text.strip():
                logger.info(f"User interjected: {frame.text}")
                audience_msg = f"[Audience member says: {frame.text.strip()}]"
                self._history_a.append({"role": "user", "content": audience_msg})
                self._history_b.append({"role": "user", "content": audience_msg})
        else:
            await self.push_frame(frame, direction)

    async def _next_turn(self):
        """Switch to the other bot and trigger a new LLM call."""
        await asyncio.sleep(0.8)

        self._turn_count += 1
        if self._turn_count >= MAX_TURNS:
            self._debating = False
            logger.info("Debate concluded after max turns")
            return

        response = self._response_text.strip()
        if not response:
            logger.warning("Empty response from bot, ending debate")
            self._debating = False
            return

        cfg = self._config

        if self._current_bot == "A":
            self._history_a.append({"role": "assistant", "content": response})
            self._history_b.append(
                {"role": "user", "content": f"{cfg['bot_a_name']} said: {response}"}
            )
            self._current_bot = "B"
            self._tts.set_voice(cfg["bot_b_voice"])
            ctx = LLMContext(messages=list(self._history_b))
        else:
            self._history_b.append({"role": "assistant", "content": response})
            self._history_a.append(
                {"role": "user", "content": f"{cfg['bot_b_name']} said: {response}"}
            )
            self._current_bot = "A"
            self._tts.set_voice(cfg["bot_a_voice"])
            ctx = LLMContext(messages=list(self._history_a))

        bot_name = cfg["bot_a_name"] if self._current_bot == "A" else cfg["bot_b_name"]
        logger.info(f"Turn {self._turn_count}/{MAX_TURNS} — {bot_name} is up")
        await self._task.queue_frames([LLMContextFrame(context=ctx)])


# ---------------------------------------------------------------------------
# Bot pipeline
# ---------------------------------------------------------------------------


async def run_debate_bot(connection: SmallWebRTCConnection, config: dict):
    """Build and run the debate pipeline for one WebRTC session."""

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)

    llm = AnthropicLLMService(
        api_key=ANTHROPIC_API_KEY,
        model="claude-sonnet-4-5-20250929",
        params=AnthropicLLMService.InputParams(
            temperature=0.9,
            max_tokens=150,
        ),
    )

    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY,
        voice=config["bot_a_voice"],
    )

    task_ref: list[PipelineTask] = []
    debate_manager = DebateManager(tts=tts, task_ref=task_ref, config=config)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            llm,
            debate_manager,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    task_ref.append(task)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — starting debate")
        await debate_manager.start_debate()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

small_webrtc_handler = SmallWebRTCRequestHandler()

# In-memory session store (mimics Pipecat Cloud)
active_sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await small_webrtc_handler.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DebateConfigRequest(BaseModel):
    topic: str
    bot_a_name: str
    bot_a_role: str
    bot_b_name: str
    bot_b_role: str


@app.post("/api/config")
async def set_config(req: DebateConfigRequest):
    """Receive debate configuration from the UI."""
    debate_config["topic"] = req.topic
    debate_config["bot_a_name"] = req.bot_a_name
    debate_config["bot_a_role"] = req.bot_a_role
    debate_config["bot_b_name"] = req.bot_b_name
    debate_config["bot_b_role"] = req.bot_b_role
    logger.info(f"Debate config updated: {debate_config}")
    return {"status": "ok"}


@app.get("/api/config")
async def get_config():
    """Return current debate configuration."""
    return JSONResponse(content=debate_config)


@app.post("/start")
async def rtvi_start(request: Request):
    """Mimic Pipecat Cloud's /start endpoint for the prebuilt UI."""
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    session_id = str(uuid.uuid4())
    active_sessions[session_id] = request_data.get("body", {})

    result: dict = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }

    logger.info(f"Session created: {session_id}")
    return result


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Handle WebRTC SDP offer."""

    async def on_connection(connection: SmallWebRTCConnection):
        config_snapshot = dict(debate_config)
        background_tasks.add_task(run_debate_bot, connection, config_snapshot)

    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    """Handle WebRTC ICE trickle."""
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_session_request(
    session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
):
    """Proxy requests to session-scoped endpoints (mimics Pipecat Cloud)."""
    if session_id not in active_sessions:
        return Response(content="Invalid or not-yet-ready session_id", status_code=404)

    active_session = active_sessions[session_id]

    if path.endswith("api/offer"):
        request_data = await request.json()
        if request.method == HTTPMethod.POST.value:
            webrtc_request = SmallWebRTCRequest(
                sdp=request_data["sdp"],
                type=request_data["type"],
                pc_id=request_data.get("pc_id"),
                restart_pc=request_data.get("restart_pc"),
                request_data=request_data.get("request_data")
                or request_data.get("requestData")
                or active_session,
            )
            return await offer(webrtc_request, background_tasks)
        elif request.method == HTTPMethod.PATCH.value:
            patch_request = SmallWebRTCPatchRequest(
                sdp_mid=request_data.get("sdp_mid") or request_data.get("sdpMid"),
                sdp_mline_index=request_data.get("sdp_mline_index")
                or request_data.get("sdpMLineIndex"),
                candidate=request_data.get("candidate"),
                pc_id=request_data.get("pc_id"),
            )
            return await ice_candidate(patch_request)

    return Response(content="Not found", status_code=404)


# ---------------------------------------------------------------------------
# Config UI — simple form, no WebRTC (that's handled by /client)
# ---------------------------------------------------------------------------

CONFIG_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Debate Config</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: #0b0b11;
    color: #c8c8d4;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    padding: 3rem 1rem;
  }
  .container { max-width: 520px; width: 100%; }
  h1 { color: #fff; font-size: 1.6rem; margin-bottom: 0.3rem; }
  .subtitle { color: #6b6b7b; font-size: 0.85rem; margin-bottom: 2rem; }
  label { display: block; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
          text-transform: uppercase; color: #6b6b7b; margin-bottom: 0.4rem; margin-top: 1.2rem; }
  input, textarea {
    width: 100%; padding: 0.6rem 0.8rem; background: #111119; border: 1px solid #1e1e2a;
    border-radius: 4px; color: #fff; font-family: inherit; font-size: 0.95rem;
    transition: border-color 0.2s;
  }
  input:focus, textarea:focus { outline: none; border-color: #3a9dd4; }
  textarea { resize: vertical; min-height: 60px; line-height: 1.5; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .section-header { color: #d4943a; font-size: 0.7rem; font-weight: 600;
                     letter-spacing: 0.15em; text-transform: uppercase; margin-top: 1.8rem; }
  .section-header.b { color: #3a9dd4; }
  .btn {
    display: block; width: 100%; margin-top: 2rem; padding: 0.85rem;
    background: #fff; color: #0b0b11; border: none; border-radius: 4px;
    font-size: 1rem; font-weight: 700; cursor: pointer;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.9; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .status { text-align: center; margin-top: 0.8rem; font-size: 0.8rem; color: #6b6b7b; }
  .status.error { color: #e83f3f; }
</style>
</head>
<body>
<div class="container">
  <h1>Debate Night</h1>
  <p class="subtitle">Configure the debate, then connect via the player to listen.</p>

  <label for="topic">Topic / Resolution</label>
  <textarea id="topic" rows="2"></textarea>

  <div class="section-header">Corner A</div>
  <div class="row">
    <div><label for="botAName">Name</label><input id="botAName"></div>
    <div><label for="botARole">Stance</label><input id="botARole"></div>
  </div>

  <div class="section-header b">Corner B</div>
  <div class="row">
    <div><label for="botBName">Name</label><input id="botBName"></div>
    <div><label for="botBRole">Stance</label><input id="botBRole"></div>
  </div>

  <button class="btn" onclick="saveConfig()">Save &amp; Open Player</button>
  <div id="status" class="status"></div>
</div>

<script>
// Load current config on page load
fetch("/api/config").then(r => r.json()).then(cfg => {
  document.getElementById("topic").value = cfg.topic || "";
  document.getElementById("botAName").value = cfg.bot_a_name || "";
  document.getElementById("botARole").value = cfg.bot_a_role || "";
  document.getElementById("botBName").value = cfg.bot_b_name || "";
  document.getElementById("botBRole").value = cfg.bot_b_role || "";
});

async function saveConfig() {
  const btn = document.querySelector(".btn");
  const status = document.getElementById("status");
  btn.disabled = true;
  status.textContent = "Saving...";
  status.className = "status";

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        topic: document.getElementById("topic").value,
        bot_a_name: document.getElementById("botAName").value,
        bot_a_role: document.getElementById("botARole").value,
        bot_b_name: document.getElementById("botBName").value,
        bot_b_role: document.getElementById("botBRole").value,
      }),
    });
    if (!res.ok) throw new Error("Failed to save config");
    window.location.href = "/client";
  } catch (e) {
    status.textContent = "Error: " + e.message;
    status.className = "status error";
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


@app.get("/")
async def serve_config_ui():
    """Serve the debate configuration page."""
    return HTMLResponse(content=CONFIG_HTML)


# Mount the Pipecat prebuilt WebRTC UI at /client
app.mount("/client", SmallWebRTCPrebuiltUI)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    print()
    print("AI Debate Arena ready!")
    print("  -> Configure debate: http://localhost:7860")
    print("  -> Connect & listen: http://localhost:7860/client")
    print()

    uvicorn.run(app, host="localhost", port=7860)
