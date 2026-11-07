"""Fixed configuration for the Paradee TTS API.

Everything here is baked in: the template deploys with zero user-set variables.
`PARADEE_MODEL_PATH` / `PARADEE_CONFIG_PATH` exist only so the image (and a local
checkout) can point at the baked weights without touching code.
"""

import os

MODEL_PATH = os.environ.get("PARADEE_MODEL_PATH", "/app/models/paradee_int8.onnx")
CONFIG_PATH = os.environ.get("PARADEE_CONFIG_PATH", "/app/models/config.json")

# Paradee 1.2 has four voices in one file. The order is the model's own voice
# table (paradee.tts.VOICES["1.2"]); do not reorder.
VOICES = [
    {"name": "af_heart", "accent": "American", "lang": "en-us"},
    {"name": "bf_isabella", "accent": "British", "lang": "en-gb"},
    {"name": "am_puck", "accent": "American", "lang": "en-us"},
    {"name": "bm_fable", "accent": "British", "lang": "en-gb"},
]
VOICE_NAMES = [v["name"] for v in VOICES]
DEFAULT_VOICE = "af_heart"

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM

# OpenAI's own speech endpoint caps `input` at 4096 characters; we cap a little
# lower because every chunk is generated serially on one CPU thread.
MAX_INPUT_CHARS = 4000

SPEED_MIN = 0.5
SPEED_MAX = 2.0
DEFAULT_SPEED = 1.0

# Only WAV is produced: float32 samples from the model are quantised to 16-bit
# PCM in-process with the standard library. Nothing is transcoded.
SUPPORTED_FORMATS = ["wav"]
DEFAULT_FORMAT = "wav"

MODEL_ID = "paradee-1.2"
PARADEE_VERSION = "1.2"


def voice_entry(name):
    for v in VOICES:
        if v["name"] == name:
            return v
    return None


def is_british(voice):
    return voice.startswith("b")
