"""Paradee TTS API — an OpenAI-compatible speech endpoint on an 8 MB CPU model.

POST /v1/audio/speech                 -> audio/wav
POST /v1/audio/speech/with-timestamps -> base64 WAV + word timings
GET  /voices, /v1/models, /health, /
"""

import base64
import math
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from . import config, engine, errors
from .errors import OpenAIError

STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the model (and both accents) before /health can answer 200, so the
    # platform healthcheck gates traffic until the service can actually speak.
    engine.load()
    yield


app = FastAPI(
    title="Paradee TTS API",
    version="1.0.0",
    description="OpenAI-compatible text-to-speech on Paradee 1.2 (8M params, CPU, 24 kHz WAV).",
    lifespan=lifespan,
)
errors.install(app)


class SpeechRequest(BaseModel):
    input: str
    voice: str = config.DEFAULT_VOICE
    response_format: str = config.DEFAULT_FORMAT
    speed: float = config.DEFAULT_SPEED
    # OpenAI SDK always sends `model`; it selects nothing here (one model per image).
    model: str | None = None


class TimestampRequest(BaseModel):
    input: str
    voice: str = config.DEFAULT_VOICE
    speed: float = config.DEFAULT_SPEED
    model: str | None = None


def _validate_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise OpenAIError(
            400,
            "input must be a non-empty string of English text.",
            code="empty_input",
            param="input",
        )
    if len(text) > config.MAX_INPUT_CHARS:
        raise OpenAIError(
            400,
            f"input is {len(text)} characters; the maximum is {config.MAX_INPUT_CHARS} "
            f"(about 4 minutes of audio). Split long text and call the endpoint per part.",
            code="input_too_long",
            param="input",
        )
    return text


def _validate_voice(voice: str) -> str:
    voice = (voice or "").strip()
    if voice not in config.VOICE_NAMES:
        raise OpenAIError(
            400,
            f"voice {voice!r} is not available; choose one of {', '.join(config.VOICE_NAMES)}.",
            code="invalid_voice",
            param="voice",
        )
    return voice


def _validate_format(fmt: str):
    fmt = (fmt or config.DEFAULT_FORMAT).strip().lower()
    if fmt not in config.SUPPORTED_FORMATS:
        raise OpenAIError(
            400,
            f"response_format {fmt!r} is not supported: this service returns 16-bit PCM WAV "
            f"at 24 kHz mono only. Pass response_format=\"wav\". No mp3 is produced, and a WAV "
            f"body is never labelled as mp3.",
            code="unsupported_response_format",
            param="response_format",
        )
    return fmt


def _validate_speed(speed) -> float:
    try:
        value = float(speed)
    except (TypeError, ValueError):
        raise OpenAIError(400, "speed must be a number between 0.5 and 2.0.",
                          code="invalid_speed", param="speed")
    if not math.isfinite(value):
        raise OpenAIError(400, "speed must be a finite number between 0.5 and 2.0.",
                          code="invalid_speed", param="speed")
    return max(config.SPEED_MIN, min(config.SPEED_MAX, value))


def _speech_headers(voice: str, speed: float, clamped: bool, seconds: float | None = None):
    headers = {
        "Content-Disposition": 'inline; filename="speech.wav"',
        "X-Paradee-Voice": voice,
        "X-Paradee-Speed": f"{speed:g}",
        "X-Paradee-Clamped": "true" if clamped else "false",
    }
    if seconds is not None:
        headers["X-Paradee-Audio-Seconds"] = f"{seconds:g}"
    return headers


@app.get("/")
def index():
    return {
        "service": "Paradee TTS API",
        "model": config.MODEL_ID,
        "endpoints": [
            "POST /v1/audio/speech",
            "POST /v1/audio/speech/with-timestamps",
            "GET /voices",
            "GET /v1/models",
            "GET /health",
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    loaded = engine.is_loaded()
    body = {
        "status": "ok" if loaded else "unavailable",
        "model_loaded": loaded,
        "model": config.MODEL_ID,
        "rss_mb": engine._rss_mb(),
        "voices": engine.voices(),
        "sample_rate": config.SAMPLE_RATE,
        "g2p_loaded": engine.g2p_ready(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "uptime_s": round(time.time() - STARTED_AT, 1),
    }
    if engine.load_error():
        body["error"] = engine.load_error()
    return JSONResponse(body, status_code=200 if loaded else 503)


@app.get("/voices")
def list_voices():
    return {
        "voices": config.VOICES,
        "default_voice": config.DEFAULT_VOICE,
        "sample_rate": config.SAMPLE_RATE,
        "formats": config.SUPPORTED_FORMATS,
        "max_input_chars": config.MAX_INPUT_CHARS,
        "speed_range": [config.SPEED_MIN, config.SPEED_MAX],
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": config.MODEL_ID, "object": "model", "owned_by": "paradee",
             "created": 0, "permission": []}
        ],
    }


@app.post("/v1/audio/speech")
def create_speech(body: SpeechRequest):
    text = _validate_text(body.input)
    voice = _validate_voice(body.voice)
    _validate_format(body.response_format)
    requested = float(body.speed) if isinstance(body.speed, (int, float)) else 1.0
    speed = _validate_speed(body.speed)
    clamped = speed != requested

    audio = engine.synthesize(text, voice=voice, speed=speed)
    if len(audio) == 0:
        raise OpenAIError(
            400,
            "the input produced no phonemes (it may contain no speakable English text).",
            code="no_speech",
            param="input",
        )
    wav = engine.wav_bytes(audio)
    # A long response must not sit inside a proxy buffer; Content-Length is set.
    return Response(
        content=wav,
        media_type="audio/wav",
        headers=_speech_headers(voice, speed, clamped, engine.audio_seconds(audio)),
    )


@app.post("/v1/audio/speech/with-timestamps")
def create_speech_with_timestamps(body: TimestampRequest):
    text = _validate_text(body.input)
    voice = _validate_voice(body.voice)
    requested = float(body.speed) if isinstance(body.speed, (int, float)) else 1.0
    speed = _validate_speed(body.speed)
    clamped = speed != requested

    audio, words = engine.synthesize_with_timestamps(text, voice=voice, speed=speed)
    if len(audio) == 0:
        raise OpenAIError(
            400,
            "the input produced no phonemes (it may contain no speakable English text).",
            code="no_speech",
            param="input",
        )
    wav = engine.wav_bytes(audio)
    return JSONResponse(
        {
            "audio_b64": base64.b64encode(wav).decode("ascii"),
            "content_type": "audio/wav",
            "format": "wav",
            "sample_rate": config.SAMPLE_RATE,
            "channels": config.CHANNELS,
            "sample_width_bytes": config.SAMPLE_WIDTH,
            "duration_s": engine.audio_seconds(audio),
            "voice": voice,
            "speed": speed,
            "speed_clamped": clamped,
            "words": words,
        },
        headers=_speech_headers(voice, speed, clamped),
    )
