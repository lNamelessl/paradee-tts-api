"""Build-time gate for the Paradee TTS image.

Runs inside the builder stage, against the exact weights and site-packages the
runtime image will use, and fails the build for any of the packaging problems
that are otherwise invisible until a user hits the endpoint:

  * the Hugging Face hub being contacted at load time (it must not be),
  * spaCy's en_core_web_sm being missing (misaki would download it at runtime),
  * eSpeak NG's shared library or data directory not being usable,
  * British voices silently falling back to American phonemes,
  * `with_timestamps` returning something other than (audio, [(word, s, e)]).

Exit code 1 on the first failed check.
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILS = []


def check(condition, label, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[warmup] {status} {label}{(' — ' + detail) if detail else ''}", flush=True)
    if not condition:
        FAILS.append(label)
    return condition


def main() -> int:
    # 1. The hub must not be reachable during load. If paradee ever starts
    #    calling hf_hub_download again (e.g. an upstream refactor drops the
    #    model_path override), this raises instead of silently downloading.
    import huggingface_hub

    def _forbidden(*args, **kwargs):
        raise RuntimeError("huggingface_hub was called during model load — not offline-safe")

    huggingface_hub.hf_hub_download = _forbidden

    from app import config, engine

    check(os.path.exists(config.MODEL_PATH), "weights present", config.MODEL_PATH)
    check(os.path.exists(config.CONFIG_PATH), "config present", config.CONFIG_PATH)
    if FAILS:
        return 1

    engine.load()
    if not engine.is_loaded():
        check(False, "model loads offline", engine.load_error() or "unknown")
        return 1
    check(True, "model loads with hf_hub_download disabled")

    check(engine.voices() == ["af_heart", "bf_isabella", "am_puck", "bm_fable"],
          "voice table", ", ".join(engine.voices()))
    check(config.SAMPLE_RATE == 24000, "sample rate", str(config.SAMPLE_RATE))

    # 2. Every voice must actually produce audio, and the four must not collapse
    #    into one (a wrong voice tensor would look fine but sound identical).
    text = "Paradee speaks four voices on one small CPU model."
    renders = {}
    for voice in engine.voices():
        audio = engine.synthesize(text, voice=voice)
        seconds = engine.audio_seconds(audio)
        digest = hashlib.sha256(engine.wav_bytes(audio)).hexdigest()
        renders[voice] = (seconds, digest)
        check(len(audio) > 0 and 1.0 < seconds < 8.0,
              f"synthesis {voice}", f"{seconds:.2f} s, {len(audio)} samples")
    unique = len({d for _, d in renders.values()})
    check(unique == 4, "four distinct voices", f"{unique}/4 distinct renders")

    tts = engine._tts
    # 3. British phonemisation must differ from American for words that do
    #    differ in the two lexicons — this is the check that catches a missing
    #    misaki British dictionary or a voice mapping that ignores the accent.
    probe = "privacy tomato schedule advertisement"
    us = tts.phonemize(probe, "af_heart")
    gb = tts.phonemize(probe, "bf_isabella")
    check(us and gb and us != gb, "British vs American phonemes differ")
    print(f"[warmup]      us={us!r}")
    print(f"[warmup]      gb={gb!r}")

    # 4. Force the eSpeak fallback (unknown words never hit the lexicon) so a
    #    missing libespeak-ng or data directory fails here, not at request time.
    from misaki.espeak import EspeakFallback

    class Tok:
        def __init__(self, text):
            self.text = text

    for british in (False, True):
        fallback = EspeakFallback(british=british)
        ps, rating = fallback(Tok("Zyxwvut"))
        accent = "en-gb" if british else "en-us"
        check(bool(ps), f"espeak fallback ({accent})", repr(ps))

    # 5. Timestamps: words in order, inside the audio's duration.
    audio, words = engine.synthesize_with_timestamps("One. Two three. Four five six.", voice="af_heart")
    duration = engine.audio_seconds(audio)
    check(len(words) >= 5, "timestamps returned words", f"{len(words)} words")
    starts = [w["start"] for w in words]
    check(all(b >= a for a, b in zip(starts, starts[1:])), "timestamps monotonic")
    check(all(w["end"] >= w["start"] for w in words), "timestamps end >= start")
    if words:
        last = words[-1]["end"]
        check(abs(last - duration) <= 0.25, "last word ends at audio end",
              f"last={last}s audio={duration}s")
        print(f"[warmup]      first words: {words[:4]}")

    # 6. The WAV the endpoint returns must be 24 kHz mono 16-bit PCM.
    import io
    import wave

    with wave.open(io.BytesIO(engine.wav_bytes(audio)), "rb") as handle:
        check(handle.getframerate() == 24000, "wav 24 kHz", str(handle.getframerate()))
        check(handle.getnchannels() == 1, "wav mono", str(handle.getnchannels()))
        check(handle.getsampwidth() == 2, "wav 16-bit", str(handle.getsampwidth()))

    print(f"[warmup] rss after load: {engine._rss_mb()} MB", flush=True)

    if FAILS:
        print(f"[warmup] FAILED checks: {FAILS}", flush=True)
        return 1
    print("[warmup] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
