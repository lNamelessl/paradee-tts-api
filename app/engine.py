"""Model lifecycle and inference for Paradee 1.2.

One model instance per process: the ONNX graph is loaded from the weights baked
into the image, and every generation runs under a single lock because the
session is shared. One worker + this lock is the documented concurrency model.
"""

import gc
import io
import os
import threading
import time
import wave

import numpy as np

from . import config

_lock = threading.Lock()

_tts = None
_load_error = None
_g2p_ready = {"american": False, "british": False}
_loaded_at = None


def _rss_mb():
    """Resident set size of this process in MB.

    Linux (the deployed image) reads /proc/self/status; Windows is supported so
    the local pre-flight battery can report the same figure.
    """
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            # K32GetProcessMemoryInfo (Win7+) with explicit argtypes: without
            # them ctypes truncates the 64-bit HANDLE and the call fails.
            query = kernel32.K32GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            query.restype = wintypes.BOOL
            if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return round(counters.WorkingSetSize / (1024 * 1024), 1)
        except Exception:
            return None
    return None


def is_loaded():
    return _tts is not None


def load_error():
    return _load_error


def loaded_at():
    return _loaded_at


def voices():
    return [v["name"] for v in config.VOICES]


def g2p_ready():
    return dict(_g2p_ready)


def load():
    """Load the baked weights and warm both accents. Called once from lifespan."""
    global _tts, _load_error, _loaded_at
    try:
        from paradee import Paradee  # imported late so /health can report import errors

        missing = [p for p in (config.MODEL_PATH, config.CONFIG_PATH) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "missing baked model files: " + ", ".join(missing)
                + " (set PARADEE_MODEL_PATH / PARADEE_CONFIG_PATH)"
            )

        tts = Paradee(
            version=config.PARADEE_VERSION,
            quantized=True,
            model_path=config.MODEL_PATH,
            config_path=config.CONFIG_PATH,
            threads=1,
        )

        # Establish the American G2P (spaCy model + misaki lexicon + eSpeak
        # backend) before /health turns green. The British G2P is left to load
        # lazily on the first British request: misaki keeps a full gold+silver
        # lexicon per language, and on a 512 MB-class instance warming both
        # accents at boot is what pushes a 1500-character request into the
        # allocator plateau. The first British call therefore costs one extra
        # ~2-4 s; /health reports which accents are ready.
        for voice in ("af_heart",):
            tts.phonemize("warm up.", voice)
            _g2p_ready["american"] = True

        # One short synthesis per American voice: proves the voice table and the
        # decoder path on this machine, and removes first-call graph setup from
        # latency.
        for voice in ("af_heart", "am_puck"):
            tts("Warm up.", speed=1.0, voice=voice)

        _tts = tts
        _loaded_at = time.time()
    except Exception as exc:  # surfaced through /health, never kills the worker
        _load_error = f"{type(exc).__name__}: {exc}"


def _require_model():
    if _tts is None:
        raise RuntimeError(f"model is not loaded: {_load_error or 'not initialised'}")
    return _tts


def _mark_g2p(voice):
    """Record that a voice's accent lexicon is now resident (lazy British load)."""
    _g2p_ready["british" if voice.startswith("b") else "american"] = True


def synthesize(text, voice=config.DEFAULT_VOICE, speed=config.DEFAULT_SPEED):
    """text -> float32 mono waveform at 24 kHz."""
    tts = _require_model()
    with _lock:
        audio = tts(text, speed=speed, voice=voice)
    _mark_g2p(voice)
    return audio


def synthesize_with_timestamps(text, voice=config.DEFAULT_VOICE, speed=config.DEFAULT_SPEED):
    """text -> (float32 waveform, [{"word","start","end"}])."""
    tts = _require_model()
    with _lock:
        audio, words = tts.with_timestamps(text, speed=speed, voice=voice)
    _mark_g2p(voice)
    # paradee returns [(text, start, end)] tuples; the API speaks dicts.
    stamps = [{"word": w, "start": round(float(s), 3), "end": round(float(e), 3)} for w, s, e in words]
    return audio, stamps


def wav_bytes(audio):
    """float32 [-1, 1] -> 16-bit PCM WAV (mono, 24 kHz), stdlib only."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(config.CHANNELS)
        handle.setsampwidth(config.SAMPLE_WIDTH)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return buf.getvalue()


def audio_seconds(audio):
    return round(len(audio) / float(config.SAMPLE_RATE), 3)


def release_lexicon_cache():
    """Drop one accent's G2P (used by the memory fallback described in the README)."""
    tts = _tts
    if tts is None:
        return
    with _lock:
        tts._g2p.clear()
        _g2p_ready["american"] = False
        _g2p_ready["british"] = False
    gc.collect()
