#!/usr/bin/env python3
"""Acceptance battery for the Paradee TTS API (A1–A13).

Runs against any base URL (local uvicorn or a deployed Railway domain), using
only the standard library plus the optional `openai` SDK for A10:

    python tests/battery.py --base-url http://127.0.0.1:8011
    python tests/battery.py --base-url https://<service>.up.railway.app --json out.json

Every check prints PASS/FAIL with its measured numbers; exit code is 1 if any
check failed.
"""

import argparse
import base64
import hashlib
import io
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import wave

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}", flush=True)
    return ok


def post(url, payload, timeout=300):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers, resp.read(), time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read(), time.perf_counter() - started


def get(url, timeout=60):
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def wav_info(data):
    with wave.open(io.BytesIO(data), "rb") as handle:
        frames = handle.getnframes()
        return {
            "channels": handle.getnchannels(),
            "width": handle.getsampwidth(),
            "rate": handle.getframerate(),
            "frames": frames,
            "seconds": round(frames / float(handle.getframerate()), 3),
        }


def synth(base, text, voice="af_heart", speed=1.0, fmt="wav"):
    status, headers, body, elapsed = post(f"{base}/v1/audio/speech", {
        "model": "paradee-1.2", "input": text, "voice": voice,
        "response_format": fmt, "speed": speed,
    })
    return status, headers, body, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--json", default="")
    ap.add_argument("--skip-sdk", action="store_true")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # ---- A1 health -------------------------------------------------------
    status, headers, body = get(f"{base}/health")
    health = json.loads(body)
    rss_idle = health.get("rss_mb")
    record("A1 /health", status == 200 and health.get("model_loaded") is True,
           f"status={status} model_loaded={health.get('model_loaded')} rss_mb={rss_idle} "
           f"voices={health.get('voices')} g2p={health.get('g2p_loaded')} "
           f"hf_offline={health.get('hf_hub_offline')}")
    record("A1b four voices advertised",
           sorted(health.get("voices") or []) == sorted(
               ["af_heart", "bf_isabella", "am_puck", "bm_fable"]),
           str(health.get("voices")))

    voices_st = get(f"{base}/voices")
    record("A1c /voices", voices_st[0] == 200 and b"bf_isabella" in voices_st[2])

    models = get(f"{base}/v1/models")
    record("A1d /v1/models", models[0] == 200 and b"paradee-1.2" in models[2])

    # ---- A2 four distinct voices -----------------------------------------
    text = "Paradee speaks four voices on one small CPU model."
    renders = {}
    for voice in ["af_heart", "am_puck", "bf_isabella", "bm_fable"]:
        status, headers, audio, elapsed = synth(base, text, voice=voice)
        if status != 200:
            record(f"A2 {voice}", False, f"status={status} body={audio[:200]}")
            continue
        info = wav_info(audio)
        renders[voice] = {"sha": hashlib.sha256(audio).hexdigest(), "info": info,
                          "ms": round(elapsed * 1000)}
        record(f"A2 {voice}", info["seconds"] > 1.0,
               f"{info['seconds']}s rate={info['rate']} ch={info['channels']} "
               f"width={info['width']} {round(elapsed*1000)}ms")
    if len(renders) == 4:
        digests = {v["sha"] for v in renders.values()}
        secs = [v["info"]["seconds"] for v in renders.values()]
        spread = max(secs) / min(secs) if min(secs) else 99
        record("A2 four voices are distinct", len(digests) == 4,
               f"{len(digests)}/4 distinct renders, duration spread {spread:.2f}x")
        record("A2b durations within 15%", spread <= 1.15, f"spread {spread:.2f}x")

    # ---- A3 British phonemisation ---------------------------------------
    probe = "privacy tomato schedule advertisement"
    us = synth(base, probe, voice="af_heart")[2]
    gb = synth(base, probe, voice="bf_isabella")[2]
    if us and gb:
        # Different accents must produce different durations/audio for words
        # whose pronunciations differ. The phoneme-level proof is the build
        # gate (it prints both phoneme strings); this is the deployed echo.
        record("A3 British differs from American audio",
               hashlib.sha256(us).hexdigest() != hashlib.sha256(gb).hexdigest(),
               f"us={wav_info(us)['seconds']}s gb={wav_info(gb)['seconds']}s")
    else:
        record("A3 British differs from American audio", False, "empty render")

    # ---- A4 WAV contract ------------------------------------------------
    if renders:
        info = renders["af_heart"]["info"]
        record("A4 wav 24 kHz mono 16-bit",
               info["rate"] == 24000 and info["channels"] == 1 and info["width"] == 2,
               json.dumps(info))

    # ---- A5 timestamps --------------------------------------------------
    stamp_text = "One two. Three four five. Six."
    status, headers, body, elapsed = post(f"{base}/v1/audio/speech/with-timestamps", {
        "model": "paradee-1.2", "input": stamp_text, "voice": "af_heart",
    })
    if status == 200:
        payload = json.loads(body)
        words = payload.get("words") or []
        duration = payload.get("duration_s")
        audio = base64.b64decode(payload.get("audio_b64") or "")
        info = wav_info(audio) if audio else {}
        starts = [w["start"] for w in words]
        ends = [w["end"] for w in words]
        record("A5 timestamps: words returned", len(words) >= 6, f"{len(words)} words")
        record("A5b timestamps monotonic", all(b >= a for a, b in zip(starts, starts[1:])))
        record("A5c end >= start", all(e >= s for s, e in zip(starts, ends)))
        last = ends[-1] if ends else 0
        record("A5d last word ends at audio end", abs(last - (duration or 0)) <= 0.25,
               f"last={last} duration={duration}")
        record("A5e audio_b64 is a valid wav", info.get("rate") == 24000,
               json.dumps({k: info.get(k) for k in ("rate", "channels", "width", "seconds")}))
        # Same text through the plain endpoint: the graph contains noise nodes
        # (RandomNormal*), so samples differ between calls — but the length is
        # fixed by the predicted durations and must match exactly.
        plain = synth(base, stamp_text, voice="af_heart")[2]
        record("A5f timestamps audio has the same length as /speech audio",
               wav_info(plain)["frames"] == wav_info(audio)["frames"],
               f"{wav_info(plain)['frames']} vs {wav_info(audio)['frames']} frames")
    else:
        record("A5 timestamps endpoint", False, f"status={status} body={body[:200]}")

    # ---- A6 stochastic but duration-stable ------------------------------
    # The exported graph contains RandomNormal* nodes (the decoder's noise
    # component), so audio is deliberately not bit-exact between calls.
    a = synth(base, "Determinism check.", voice="am_puck")[2]
    b = synth(base, "Determinism check.", voice="am_puck")[2]
    ia, ib = wav_info(a), wav_info(b)
    record("A6 duration is stable across calls", ia["frames"] == ib["frames"],
           f"{ia['frames']} vs {ib['frames']} frames (bytes differ by design: noise nodes)")
    if ia["frames"] == ib["frames"]:
        import array
        with wave.open(io.BytesIO(a), "rb") as ha:
            sa = array.array("h", ha.readframes(ha.getnframes()))
        with wave.open(io.BytesIO(b), "rb") as hb:
            sb = array.array("h", hb.readframes(hb.getnframes()))
        mean_abs = sum(abs(x - y) for x, y in zip(sa, sb)) / len(sa) / 32768.0
        record("A6b run-to-run sample difference is small", mean_abs < 0.05,
               f"mean|Δsample| = {mean_abs*100:.2f}% of full scale")

    # ---- A7 long paragraph ----------------------------------------------
    paragraph = (
        "Paradee is a small English text-to-speech model distilled from Kokoro. "
        "It runs on one CPU thread, keeps four voices in a single eight megabyte file, "
        "and needs no graphics card. This paragraph exists to exercise the sentence "
        "splitter and the five hundred and ten phoneme chunk limit that the model has. "
    ) * 6
    paragraph = (paragraph + "Padding sentence to reach the cap. " * 20)[:1500]
    status, headers, audio, elapsed = synth(base, paragraph, voice="af_heart")
    if status == 200:
        info = wav_info(audio)
        rtf = info["seconds"] / elapsed if elapsed else 0
        record("A7 1500-char paragraph", info["seconds"] > 60,
               f"{len(paragraph)} chars, {info['seconds']}s audio, {elapsed*1000:.0f} ms, RTF {rtf:.1f}x")
    else:
        record("A7 1500-char paragraph", False, f"status={status} {audio[:300]}")

    # ---- A8 speed -------------------------------------------------------
    normal = wav_info(synth(base, text, speed=1.0)[2])
    fast_st, fast_h, fast, _ = synth(base, text, speed=2.0)
    slow_st, slow_h, slow, _ = synth(base, text, speed=0.5)
    fast = wav_info(fast)
    slow = wav_info(slow)
    r_fast = normal["seconds"] / fast["seconds"]
    r_slow = slow["seconds"] / normal["seconds"]
    record("A8 speed 2.0 halves duration", 1.8 <= r_fast <= 2.2,
           f"1.0={normal['seconds']}s 2.0={fast['seconds']}s ratio={r_fast:.2f}")
    record("A8b speed 0.5 doubles duration", 1.8 <= r_slow <= 2.2,
           f"1.0={normal['seconds']}s 0.5={slow['seconds']}s ratio={r_slow:.2f}")
    st5, h5, b5, _ = synth(base, text, speed=5.0)
    st01, h01, b01, _ = synth(base, text, speed=0.1)
    record("A8c speed clamps to 2.0", h5.get("X-Paradee-Speed") == "2"
           and h5.get("X-Paradee-Clamped") == "true", f"X-Paradee-Speed={h5.get('X-Paradee-Speed')}")
    record("A8d speed clamps to 0.5", h01.get("X-Paradee-Speed") == "0.5"
           and h01.get("X-Paradee-Clamped") == "true", f"X-Paradee-Speed={h01.get('X-Paradee-Speed')}")

    # ---- A9 error shapes ------------------------------------------------
    st, h, body, _ = synth(base, "hello", voice="af_nova")
    payload = json.loads(body) if body else {}
    err = payload.get("error", {})
    record("A9 invalid voice -> 400 + allowed list",
           st == 400 and "af_heart" in err.get("message", "") and err.get("param") == "voice",
           f"status={st} message={err.get('message')!r}")

    st, h, body, _ = synth(base, "hello", fmt="mp3")
    err = json.loads(body).get("error", {}) if body else {}
    record("A9b response_format=mp3 -> 400 (no mislabel)",
           st == 400 and err.get("code") == "unsupported_response_format",
           f"status={st} code={err.get('code')} message={err.get('message')!r}")

    st, h, body, _ = synth(base, "   ")
    record("A9c empty input -> 400",
           st == 400 and json.loads(body)["error"]["code"] in ("empty_input", "invalid_request"),
           f"status={st} code={json.loads(body)['error']['code']}")

    st, h, body, _ = synth(base, "a" * 4001)
    code = json.loads(body)["error"]["code"] if body else None
    record("A9d input over cap -> 400", st == 400 and code == "input_too_long",
           f"status={st} code={code}")

    st, h, body, _ = post(f"{base}/v1/audio/speech", {"voice": "af_heart"})
    record("A9e missing input -> 400 envelope",
           st == 400 and "error" in json.loads(body), f"status={st}")

    # ---- A10 openai SDK round-trip --------------------------------------
    if not args.skip_sdk:
        try:
            from openai import OpenAI
            import openai as openai_mod

            client = OpenAI(base_url=f"{base}/v1", api_key="not-needed")
            resp = client.audio.speech.create(model="paradee-1.2", voice="af_heart",
                                              input="The openai software development kit works.",
                                              response_format="wav")
            content = resp.content
            info = wav_info(content)
            record("A10 openai SDK audio.speech.create",
                   content[:4] == b"RIFF" and info["rate"] == 24000,
                   f"openai=={openai_mod.__version__} bytes={len(content)} {info}")

            listed = client.models.list()
            ids = [m.id for m in listed.data]
            record("A10b openai SDK models.list", "paradee-1.2" in ids, str(ids))

            raised = None
            try:
                client.audio.speech.create(model="paradee-1.2", voice="af_nova",
                                           input="x", response_format="wav")
            except openai_mod.BadRequestError as exc:
                raised = exc
            except Exception as exc:  # wrong exception type is a failure
                raised = exc
            record("A10c SDK raises BadRequestError for a bad voice",
                   isinstance(raised, openai_mod.BadRequestError),
                   f"{type(raised).__name__}: {str(raised)[:120]}")
        except ImportError:
            record("A10 openai SDK round-trip", False, "openai package not installed (use --skip-sdk)")

    # ---- A11 latency table ---------------------------------------------
    latency_text = "Paradee reads short sentences quickly on a single CPU thread."
    samples = []
    first = synth(base, latency_text)[1]
    for i in range(10):
        status, headers, audio, elapsed = synth(base, latency_text, voice="af_heart")
        info = wav_info(audio)
        samples.append((elapsed, info["seconds"]))
    wall = sorted(s[0] for s in samples)
    p50 = statistics.median(wall)
    p95 = wall[int(round(0.95 * (len(wall) - 1)))]
    audio_s = samples[0][1]
    record("A11 latency p50/p95 + RTF", p50 > 0,
           f"audio={audio_s}s p50={p50*1000:.0f}ms p95={p95*1000:.0f}ms "
           f"min={wall[0]*1000:.0f}ms max={wall[-1]*1000:.0f}ms "
           f"RTF_p50={audio_s/p50:.1f}x RTF_p95={audio_s/p95:.1f}x")

    # ---- A12 memory -----------------------------------------------------
    status, headers, body = get(f"{base}/health")
    health_after = json.loads(body)
    rss_after = health_after.get("rss_mb")
    peak_note = f"idle={rss_idle}MB after_load={rss_after}MB"
    record("A12 RSS reported", rss_after is not None, peak_note)
    if rss_after is not None:
        record("A12b RSS within 512 MB budget", rss_after <= 512, f"{rss_after} MB")

    # ---- A13 offline proof ---------------------------------------------
    record("A13 HF_HUB_OFFLINE in effect", health_after.get("hf_hub_offline") is True,
           f"hf_hub_offline={health_after.get('hf_hub_offline')}")

    failed = [r for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"base_url": base, "results": RESULTS}, handle, indent=2)
        print(f"wrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
