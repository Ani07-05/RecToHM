"""Custom pipecat TTS service that calls a Qwen3-TTS HTTP server.

Usage in a pipecat pipeline:
    tts = QwenTTSService(
        base_url="http://localhost:8100",
        voice="Ryan",
        language="English",
    )

The service sends text to the Qwen3-TTS server's /synthesize endpoint
and streams back raw PCM audio frames compatible with pipecat's pipeline.
"""

from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class QwenTTSService(TTSService):
    """HTTP-based pipecat TTS service backed by a Qwen3-TTS server.

    The server must expose POST /synthesize returning raw 16-bit PCM.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8100",
        voice: str = "Ryan",
        language: str = "English",
        instruct: Optional[str] = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._instruct = instruct
        self._session: Optional[aiohttp.ClientSession] = None
        self.set_voice(voice)

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def set_language(self, language: str):
        self._language = language

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"QwenTTS: synthesizing [{text[:80]}...]")

        if not self._session:
            self._session = aiohttp.ClientSession()

        try:
            await self.start_ttfb_metrics()

            payload = {
                "text": text,
                "speaker": self._voice_id,
                "language": self._language,
            }
            if self._instruct:
                payload["instruct"] = self._instruct

            async with self._session.post(
                f"{self._base_url}/synthesize",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"QwenTTS server error: {response.status} {error_text}"
                    )
                    yield ErrorFrame(error=f"TTS server error: {response.status}")
                    return

                # Read sample rate from response headers
                server_sr = int(response.headers.get("X-Sample-Rate", self.sample_rate))

                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()

                await self.stop_ttfb_metrics()

                # Stream PCM data in chunks
                # Each chunk: 0.5s of 16-bit mono audio
                chunk_size = server_sr * 2 * 1  # 1 second of 16-bit mono
                chunk_size = chunk_size // 2  # 0.5s chunks

                buffer = b""
                async for data in response.content.iter_any():
                    buffer += data
                    while len(buffer) >= chunk_size:
                        chunk = buffer[:chunk_size]
                        buffer = buffer[chunk_size:]
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=server_sr,
                            num_channels=1,
                        )

                # Flush remaining buffer (ensure even byte count for int16)
                if buffer:
                    if len(buffer) % 2 != 0:
                        buffer = buffer[:-1]
                    if buffer:
                        yield TTSAudioRawFrame(
                            audio=buffer,
                            sample_rate=server_sr,
                            num_channels=1,
                        )

            yield TTSStoppedFrame()

        except aiohttp.ClientError as e:
            logger.error(f"QwenTTS connection error: {e}")
            yield TTSStoppedFrame()
            yield ErrorFrame(error=f"Cannot reach TTS server at {self._base_url}: {e}")
        except Exception as e:
            logger.error(f"QwenTTS error: {e}")
            yield TTSStoppedFrame()
            yield ErrorFrame(error=f"TTS error: {e}")
