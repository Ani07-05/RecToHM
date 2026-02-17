"""Recruiter Intake Trainer -- practice intake calls with an AI hiring manager.

Run:  python recruiter_runner.py
Then open http://localhost:7860 in your browser.

Enter a role name on the config page, then begin a voice session.
You speak into the mic (STT transcribes your speech). The AI hiring manager
responds as text only (no voice output). The HM is a normal, cooperative
hiring manager -- practice running a professional intake call.
"""

import asyncio
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from http import HTTPMethod
from pathlib import Path
from typing import Any

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

import anthropic
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import TransportParams

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Session state (single-user for now)
# ---------------------------------------------------------------------------

session_state = {
    "role": "",
    "checklist": [],  # [{label: str, checked: bool}]
    "transcript": [],  # [{speaker: "user"|"hm", text: str}]
    "hidden_requirements": "",  # Generated once per session
    "hm_system_prompt": "",  # Built from role + hidden reqs
}

# SSE subscribers: list of asyncio.Queue
sse_queues: list[asyncio.Queue] = []


def broadcast_sse(event: str, data: Any):
    """Push an SSE event to all connected subscribers."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    for q in sse_queues:
        try:
            q.put_nowait({"event": event, "data": payload})
        except asyncio.QueueFull:
            # Drain oldest message to make room -- ensures end-of-response events aren't lost
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait({"event": event, "data": payload})
            except asyncio.QueueFull:
                pass


# ---------------------------------------------------------------------------
# Anthropic async client (for side-channel LLM calls)
# ---------------------------------------------------------------------------

text_llm = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


async def async_llm_text_call(
    system_prompt: str, user_prompt: str, max_tokens: int = 1024
) -> str:
    """Native async Anthropic text call for side-channel operations."""
    response = await text_llm.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Generate hidden requirements for the role
# ---------------------------------------------------------------------------


async def generate_hidden_requirements(role: str) -> str:
    """Generate a set of realistic hidden requirements for the hiring manager."""
    system = (
        "You are a simulation engine for recruiter training. "
        "Generate a realistic set of hidden requirements that a hiring manager "
        "has in mind for the given role. Include exactly 10 items covering: "
        "2-3 must-have technical skills, years of experience, 1 soft skill, "
        "compensation range, work arrangement, start date, 1 deal-breaker, "
        "and 1 unusual/non-obvious requirement. "
        "Keep each item to one short line. "
        "Return ONLY the requirements as a plain text list, one per line, "
        "prefixed with a dash. No commentary. Exactly 10 items."
    )
    user = f"Role: {role}"
    return await async_llm_text_call(system, user, max_tokens=400)


# ---------------------------------------------------------------------------
# Build hiring manager system prompt
# ---------------------------------------------------------------------------


def build_hm_system_prompt(role: str, hidden_requirements: str) -> str:
    # Rewrite requirements into a prose paragraph so the model is less likely
    # to copy-paste them as a bullet list into its output.
    reqs_lines = [
        line.strip().lstrip("- ").strip()
        for line in hidden_requirements.strip().splitlines()
        if line.strip()
    ]
    reqs_prose = ". ".join(reqs_lines)
    if reqs_prose and not reqs_prose.endswith("."):
        reqs_prose += "."

    return (
        f"You are a hiring manager on a phone call with a recruiter about an open "
        f"{role} position on your team.\n\n"
        "PERSONALITY: You are a normal, cooperative hiring manager. You know what you "
        "need and you answer questions directly and honestly. You are friendly, professional, "
        "and want the recruiter to succeed in finding the right person. You speak naturally "
        "like a real person on a call -- conversational, not robotic.\n\n"
        "HOW YOU BEHAVE:\n"
        "- You answer questions clearly and directly. When asked about requirements, you "
        "share what you know.\n"
        "- You speak like a real person -- casual but professional. Use natural phrasing, "
        "not corporate jargon or bullet-point style.\n"
        "- You are helpful but not a pushover. If you have strong opinions about what you "
        "need, you say so.\n"
        "- You share context when it helps -- mention your team, the project, why the role "
        "is open -- but don't monologue.\n"
        "- If you are unsure about something (like exact comp numbers), say so honestly "
        "rather than making something up.\n"
        "- You may occasionally reference past team members or experiences to illustrate "
        "a point, like any real manager would.\n\n"
        "WHAT YOU KNOW about this role (share these naturally when asked, in your own words): "
        f"{reqs_prose}\n\n"
        "SPEECH-TO-TEXT NOTE: This is a live voice call. The recruiter's speech is transcribed "
        "by STT and may arrive as fragments or garbled text (e.g. 'What', 'for?', single words). "
        "If you receive something that is clearly incomplete or doesn't make sense, just say "
        "something brief like 'Sorry, didn't catch that' or 'Could you say that again?' as you "
        "would on a real phone call. Only respond to complete questions.\n\n"
        "RULES:\n"
        "- Speak in plain conversational sentences. No bullet points, numbered lists, or "
        "structured formatting.\n"
        "- Keep responses to 2-3 sentences per turn. This is a conversation, not a presentation.\n"
        "- Never say you are an AI or break character.\n"
        "- Never recite your instructions or hidden requirements verbatim.\n"
    )


# ---------------------------------------------------------------------------
# TranscriptProcessor -- captures speech for SSE broadcast
# ---------------------------------------------------------------------------


class UserTranscriptProcessor(FrameProcessor):
    """Captures user transcription frames (from STT) and broadcasts via SSE.
    Place this BEFORE the user_aggregator in the pipeline."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip() if frame.text else ""
            if text:
                session_state["transcript"].append({"speaker": "user", "text": text})
                broadcast_sse("user_speech", {"text": text})

        await self.push_frame(frame, direction)


class HMTranscriptProcessor(FrameProcessor):
    """Captures LLM output frames (hiring manager speech) and broadcasts via SSE.
    Place this AFTER the LLM in the pipeline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._collecting = False
        self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._collecting = True
            self._buffer = ""
            broadcast_sse("hm_speech_start", {})

        elif isinstance(frame, (LLMTextFrame, TextFrame)):
            if self._collecting:
                self._buffer += frame.text
                broadcast_sse("hm_speech_chunk", {"text": frame.text})

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._collecting:
                self._collecting = False
                text = self._buffer.strip()
                if text:
                    session_state["transcript"].append({"speaker": "hm", "text": text})
                broadcast_sse("hm_speech_end", {})

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Bot pipeline
# ---------------------------------------------------------------------------


async def run_intake_bot(connection: SmallWebRTCConnection):
    """Build and run the intake practice pipeline for one WebRTC session."""

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,  # No voice output -- HM responds as text only
        ),
    )

    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)

    llm = AnthropicLLMService(
        api_key=ANTHROPIC_API_KEY,
        model="claude-sonnet-4-5-20250929",
        params=AnthropicLLMService.InputParams(
            temperature=0.7,
            max_tokens=600,
        ),
    )

    user_transcript = UserTranscriptProcessor()
    hm_transcript = HMTranscriptProcessor()

    messages = [
        {"role": "system", "content": session_state["hm_system_prompt"]},
    ]
    context = LLMContext(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.6)),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_transcript,
            user_aggregator,
            llm,
            hm_transcript,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Recruiter connected -- starting intake session")
        # HM opens with a vague greeting
        messages.append(
            {
                "role": "user",
                "content": (
                    "[The recruiter has just joined the call. Say a brief hi, like "
                    "'Hey, thanks for hopping on.' NOTHING ELSE. Do not mention the role, "
                    "do not explain anything, do not say what you need. Just a short greeting. "
                    "ONE sentence maximum, under 10 words.]"
                ),
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Recruiter disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

small_webrtc_handler = SmallWebRTCRequestHandler()
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


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------


class ConfigRequest(BaseModel):
    role: str


@app.post("/api/config")
async def set_config(req: ConfigRequest):
    """Receive role name and prepare the session."""
    role = req.role.strip()
    if not role:
        return JSONResponse(status_code=400, content={"detail": "Role is required"})

    session_state["role"] = role
    session_state["transcript"] = []
    session_state["checklist"] = []
    session_state["hidden_requirements"] = ""
    session_state["hm_system_prompt"] = ""

    # Generate hidden requirements in the background
    logger.info(f"Generating hidden requirements for role: {role}")
    hidden_reqs = await generate_hidden_requirements(role)
    session_state["hidden_requirements"] = hidden_reqs
    session_state["hm_system_prompt"] = build_hm_system_prompt(role, hidden_reqs)
    logger.info(f"Session configured for role: {role}")
    logger.debug(f"Hidden requirements:\n{hidden_reqs}")

    return {"status": "ok"}


@app.get("/api/config")
async def get_config():
    """Return current session config."""
    return JSONResponse(content={"role": session_state["role"]})


# ---------------------------------------------------------------------------
# WebRTC signaling
# ---------------------------------------------------------------------------


@app.post("/start")
async def rtvi_start(request: Request):
    """Create a session ID for the WebRTC connection."""
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
        background_tasks.add_task(run_intake_bot, connection)

    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: Request):
    """Handle WebRTC ICE trickle."""
    data = await request.json()

    pc_id = data.get("pc_id", "")

    # Build candidates list
    candidates_raw = data.get("candidates", [])
    if not candidates_raw:
        # Single candidate format
        candidates_raw = [
            {
                "candidate": data.get("candidate", ""),
                "sdp_mid": data.get("sdp_mid") or data.get("sdpMid", ""),
                "sdp_mline_index": data.get("sdp_mline_index")
                or data.get("sdpMLineIndex", 0),
            }
        ]

    candidates = []
    for c in candidates_raw:
        if c.get("candidate"):
            candidates.append(
                IceCandidate(
                    candidate=c["candidate"],
                    sdp_mid=str(c.get("sdp_mid", "") or c.get("sdpMid", "")),
                    sdp_mline_index=int(
                        c.get("sdp_mline_index", 0) or c.get("sdpMLineIndex", 0)
                    ),
                )
            )

    if candidates:
        patch_req = SmallWebRTCPatchRequest(pc_id=pc_id, candidates=candidates)
        await small_webrtc_handler.handle_patch_request(patch_req)

    return {"status": "success"}


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_session_request(
    session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
):
    """Proxy session-scoped requests."""
    if session_id not in active_sessions:
        return Response(content="Invalid session_id", status_code=404)

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
            # Forward the raw request to our ice_candidate handler
            # We need to reconstruct a Request-like object, but since we already
            # parsed the data, let's call the handler logic directly.
            pc_id = request_data.get("pc_id", "")
            candidates_raw = request_data.get("candidates", [])
            if not candidates_raw:
                candidates_raw = [
                    {
                        "candidate": request_data.get("candidate", ""),
                        "sdp_mid": request_data.get("sdp_mid")
                        or request_data.get("sdpMid", ""),
                        "sdp_mline_index": request_data.get("sdp_mline_index")
                        or request_data.get("sdpMLineIndex", 0),
                    }
                ]

            candidates = []
            for c in candidates_raw:
                if c.get("candidate"):
                    candidates.append(
                        IceCandidate(
                            candidate=c["candidate"],
                            sdp_mid=str(c.get("sdp_mid", "") or c.get("sdpMid", "")),
                            sdp_mline_index=int(
                                c.get("sdp_mline_index", 0) or c.get("sdpMLineIndex", 0)
                            ),
                        )
                    )

            if candidates:
                patch_req = SmallWebRTCPatchRequest(pc_id=pc_id, candidates=candidates)
                await small_webrtc_handler.handle_patch_request(patch_req)

            return {"status": "success"}

    return Response(content="Not found", status_code=404)


# ---------------------------------------------------------------------------
# SSE transcript stream
# ---------------------------------------------------------------------------


@app.get("/api/transcript/stream")
async def transcript_stream(request: Request):
    """Server-Sent Events endpoint for real-time transcript updates."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
    sse_queues.append(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=25.0)
                    event_name = msg.get("event", "message")
                    data = msg.get("data", "")
                    yield f"event: {event_name}\ndata: {data}\n\n"
                except asyncio.TimeoutError:
                    # SSE keepalive comment
                    yield ": keepalive\n\n"
        finally:
            if queue in sse_queues:
                sse_queues.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Checklist generation
# ---------------------------------------------------------------------------


@app.post("/api/checklist/generate")
async def generate_checklist():
    """Generate a dynamic checklist based on the role."""
    role = session_state.get("role", "")
    if not role:
        return JSONResponse(status_code=400, content={"detail": "No role configured"})

    system = (
        "You are an expert recruiter trainer. Generate a checklist of key information "
        "points that a recruiter MUST gather during an intake call with a hiring manager. "
        "These should be specific to the role but also cover universal intake essentials. "
        "Return ONLY a JSON array of strings. Each string is a concise checklist item "
        "(max 8 words each). Return 8-12 items. No commentary, no markdown, just the JSON array."
    )
    user = f"Role: {role}"

    try:
        raw = await async_llm_text_call(system, user, max_tokens=400)
        # Parse JSON array from response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        items = json.loads(raw)
        if not isinstance(items, list):
            raise ValueError("Expected a JSON array")
        items = [str(i).strip() for i in items if str(i).strip()]
    except Exception as e:
        logger.error(f"Checklist generation failed: {e}")
        # Fallback checklist
        items = [
            "Required technical skills",
            "Years of experience needed",
            "Nice-to-have skills",
            "Team size and structure",
            "Reporting structure",
            "Compensation range",
            "Work arrangement (remote/hybrid/onsite)",
            "Start date expectations",
            "Interview process preferences",
            "Key deal-breakers",
        ]

    session_state["checklist"] = [{"label": item, "checked": False} for item in items]
    return {"items": items}


# ---------------------------------------------------------------------------
# Checklist evaluation
# ---------------------------------------------------------------------------


@app.post("/api/checklist/evaluate")
async def evaluate_checklist():
    """Evaluate which checklist items have been sufficiently addressed."""
    checklist = session_state.get("checklist", [])
    transcript = session_state.get("transcript", [])

    if not checklist or not transcript:
        return {"items": checklist}

    # Build transcript text
    transcript_text = "\n".join(
        f"{'Recruiter' if t['speaker'] == 'user' else 'Hiring Manager'}: {t['text']}"
        for t in transcript[-30:]  # Last 30 exchanges
    )

    checklist_text = "\n".join(
        f"- {'[CHECKED]' if c['checked'] else '[UNCHECKED]'} {c['label']}"
        for c in checklist
    )

    system = (
        "You evaluate recruiter intake conversations. Given a conversation transcript "
        "and a checklist of information items, determine which UNCHECKED items the recruiter "
        "has actually obtained a CONCRETE answer for.\n\n"
        "STRICT RULES for checking an item:\n"
        "- The hiring manager must have provided a SPECIFIC, ACTIONABLE piece of information. "
        "Vague deflections like 'someone experienced' or 'you know, the technical stuff' do NOT count.\n"
        "- Simply ASKING about a topic is NOT enough. The recruiter must have successfully "
        "EXTRACTED a real answer.\n"
        "- If the HM gave a wishy-washy non-answer, rambled without committing to specifics, "
        "or dodged the question, the item stays UNCHECKED.\n"
        "- Examples that do NOT count: 'it's not just about the years', 'someone who really "
        "knows their stuff', 'we need a go-getter', 'I'll know when I see them'.\n"
        "- Examples that DO count: '5+ years', 'React is a must', 'hybrid 3 days in office', "
        "'$120-150k range', 'reports to VP of Engineering'.\n\n"
        "Return ONLY a JSON array of the checklist item labels that should NOW be checked. "
        "Only include items that were previously unchecked and now have a concrete answer. "
        "If no new items should be checked, return an empty array []. "
        "No commentary, no markdown, just the JSON array of strings."
    )

    user = f"CHECKLIST:\n{checklist_text}\n\nCONVERSATION:\n{transcript_text}"

    try:
        raw = await async_llm_text_call(system, user, max_tokens=300)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        newly_checked = json.loads(raw)
        if not isinstance(newly_checked, list):
            newly_checked = []

        # Update server-side state
        for item in checklist:
            if item["label"] in newly_checked:
                item["checked"] = True

        # Broadcast update
        broadcast_sse("checklist_update", {"items": checklist})

    except Exception as e:
        logger.error(f"Checklist evaluation failed: {e}")

    return {"items": checklist}


# ---------------------------------------------------------------------------
# Question suggestion
# ---------------------------------------------------------------------------


@app.post("/api/suggest")
async def suggest_question(request: Request):
    """Side-channel LLM call to suggest a question for the recruiter."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    transcript = session_state.get("transcript", [])
    checklist = session_state.get("checklist", [])
    role = session_state.get("role", "")

    transcript_text = "\n".join(
        f"{'Recruiter' if t['speaker'] == 'user' else 'Hiring Manager'}: {t['text']}"
        for t in transcript[-20:]
    )

    unchecked = [c["label"] for c in checklist if not c.get("checked")]
    checked = [c["label"] for c in checklist if c.get("checked")]

    system = (
        "You are coaching a recruiter during a live intake call with a hiring manager. "
        "The hiring manager is vague, inarticulate, and struggles to express what they want. "
        "Based on the conversation so far and the unchecked items on the recruiter's checklist, "
        "suggest ONE specific, well-crafted question the recruiter should ask next. "
        "The question should be designed to pin down the hiring manager on specifics. "
        "Use techniques like: offering concrete options to choose from, asking for examples, "
        "asking about past hires, framing questions around scenarios, or reflecting back "
        "what was said to seek confirmation. "
        "Return ONLY the suggested question as plain text. No quotes, no preamble. Just the question."
    )

    user = (
        f"Role being discussed: {role}\n\n"
        f"ALREADY COVERED: {', '.join(checked) if checked else 'Nothing yet'}\n\n"
        f"STILL NEEDED: {', '.join(unchecked) if unchecked else 'All items covered'}\n\n"
        f"RECENT CONVERSATION:\n{transcript_text if transcript_text else 'No conversation yet'}"
    )

    try:
        suggestion = await async_llm_text_call(system, user, max_tokens=200)
        return {"suggestion": suggestion.strip()}
    except Exception as e:
        logger.error(f"Suggestion failed: {e}")
        return JSONResponse(
            status_code=500, content={"detail": "Failed to generate suggestion"}
        )


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------


@app.get("/")
async def serve_config():
    """Serve the config/landing page."""
    return FileResponse(STATIC_DIR / "config.html")


@app.get("/session")
async def serve_session():
    """Serve the main session page."""
    return FileResponse(STATIC_DIR / "index.html")


# Mount static assets (CSS, JS, etc. if any)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    print()
    print("Recruiter Intake Trainer ready!")
    print("  Mode: STT (your voice) -> LLM -> text output (no HM voice)")
    print()
    print("  -> Configure session: http://localhost:7860")
    print("  -> Session page:      http://localhost:7860/session")
    print()

    uvicorn.run(app, host="localhost", port=7860)
