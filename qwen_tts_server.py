"""Qwen3-TTS server -- serves TTS synthesis over HTTP.

Run:  python qwen_tts_server.py
Listens on http://localhost:8100

The model is loaded once at startup. Synthesis runs in a thread pool
to avoid blocking the async event loop.

Endpoints:
  POST /synthesize   -- synthesize text to raw PCM audio
  GET  /health       -- check server readiness
  GET  /speakers     -- list available speakers
"""

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from loguru import logger
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
OUTPUT_SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

model = None
model_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Qwen3-TTS model at startup."""
    global model, model_ready
    logger.info(f"Loading {MODEL_NAME} on {DEVICE} with {DTYPE}...")
    start = time.time()

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        dtype=DTYPE,
    )

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")

    speakers = model.get_supported_speakers()
    logger.info(f"Supported speakers: {speakers}")
    model_ready = True

    yield

    model_ready = False
    model = None


app = FastAPI(title="Qwen3-TTS Server", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    speaker: str = "Ryan"
    language: str = "English"
    instruct: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ready" if model_ready else "loading"}


@app.get("/speakers")
async def speakers():
    if not model_ready:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
    return {"speakers": model.get_supported_speakers()}


def _synthesize_sync(text: str, speaker: str, language: str, instruct: Optional[str]):
    """Run synthesis on the GPU. Called from a thread pool."""
    kwargs = {}
    if instruct:
        kwargs["instruct"] = instruct

    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        non_streaming_mode=True,
        max_new_tokens=1024,
        **kwargs,
    )

    waveform = wavs[0]
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm_int16 = (waveform * 32767).astype(np.int16)
    return pcm_int16.tobytes(), sr, len(waveform) / sr


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Synthesize text to raw 16-bit PCM audio.

    Returns raw PCM bytes. Synthesis runs in a thread pool so the
    server stays responsive to health checks and concurrent requests.
    """
    if not model_ready:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})

    text = req.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"detail": "Empty text"})

    logger.debug(f"Synthesizing: [{text[:80]}] speaker={req.speaker}")
    start = time.time()

    try:
        loop = asyncio.get_running_loop()
        pcm_bytes, sr, audio_duration = await loop.run_in_executor(
            None, _synthesize_sync, text, req.speaker, req.language, req.instruct
        )

        elapsed = time.time() - start
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        logger.info(
            f"Synthesized {audio_duration:.2f}s audio in {elapsed:.2f}s "
            f"(RTF={rtf:.2f}) [{text[:50]}]"
        )

        return Response(
            content=pcm_bytes,
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(sr),
                "X-Channels": "1",
                "X-Bit-Depth": "16",
                "X-Audio-Duration": f"{audio_duration:.3f}",
            },
        )

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    print()
    print("Qwen3-TTS Server")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print(f"  URL:    http://localhost:8100")
    print()

    uvicorn.run(app, host="localhost", port=8100)
