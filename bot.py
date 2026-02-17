"""Common bot logic: Deepgram STT, Anthropic Claude LLM, Deepgram TTS.

This module exposes run_bot() which builds the full pipeline.
Use default_runner.py as the entry point.
"""

import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import BaseTransport


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
):
    """Build and run the voice-bot pipeline.

    Args:
        transport: The transport to use (WebRTC).
        runner_args: Pipecat runner arguments.
    """
    # Deepgram Speech-to-Text
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Anthropic Claude LLM
    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-5-20250929",
        params=AnthropicLLMService.InputParams(
            temperature=0.7,
            max_tokens=200,
        ),
    )

    # Deepgram Text-to-Speech
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-2-helena-en",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly assistant. Keep answers short and conversational. "
                "You are helping developers to build voice agents quickly. "
                "Generate TTS friendly text. No emojis etc. Keep answers brief. "
                "Don't generate long sentences. Please note that you are a voice assistant. "
                "Don't reveal your details on LLM model or company that trained you. "
                "Keep the conversation centered around building voice agents."
            ),
        }
    ]
    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    pipeline_params = PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    task = PipelineTask(pipeline, params=pipeline_params)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        messages.append({"role": "system", "content": "Say hello briefly."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)
