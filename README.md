# Paradee TTS API

An OpenAI-compatible text-to-speech service on **Paradee 1.2** — an 8M-parameter English
TTS model distilled from Kokoro-82M. One 8.5 MB int8 ONNX file holds four voices, runs on
CPU with no GPU and no API keys, and speaks 24 kHz audio with word-level timestamps.

This repository is the Railway template's source: a small FastAPI wrapper that turns the
model into `POST /v1/audio/speech`, with the weights baked into the image (sha256-verified),
a build-time gate that fails the build if the model cannot load offline, and an acceptance
battery for the deployed service.

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/paradee-tts-api)

One service, no variables, no credentials, no volume. Weights are baked into the image at
build time and verified by sha256; the running container makes no outbound calls.

```bash
curl -X POST https://<your-service>.up.railway.app/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Paradee is a small voice that runs anywhere.","voice":"af_heart","response_format":"wav"}' \
  --output speech.wav
```

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-service>.up.railway.app/v1", api_key="not-needed")
resp = client.audio.speech.create(
    model="paradee-1.2",            # selects nothing; one model per image
    voice="af_heart",               # af_heart, am_puck, bf_isabella, bm_fable
    input="Paradee is a small voice that runs anywhere.",
    response_format="wav",          # WAV only in v1 — mp3 is refused, not mislabeled
    speed=1.0,                      # clamped to 0.5–2.0
)
open("speech.wav", "wb").write(resp.content)
```

## Endpoints

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/audio/speech` | `{input, voice, response_format, speed, model?}` → `audio/wav` (16-bit PCM, 24 kHz, mono) |
| `POST` | `/v1/audio/speech/with-timestamps` | same input → `{audio_b64, format, sample_rate, duration_s, words:[{word,start,end}]}` |
| `GET` | `/voices` | the four voices, accents, formats, caps |
| `GET` | `/v1/models` | `paradee-1.2` |
| `GET` | `/health` | `{status, model_loaded, rss_mb, voices, g2p_loaded, hf_hub_offline, uptime_s}` — 200 only when the model is loaded |
| `GET` | `/` | endpoint index (`/docs` has the generated OpenAPI UI) |

Voices: `af_heart`, `am_puck` (American), `bf_isabella`, `bm_fable` (British — these use
misaki's British lexicon, matching what Kokoro's `b*` voices use).

### Limits and error behaviour

- `input` is capped at **4,000 characters** (about four minutes of audio); longer input is a
  `400 input_too_long`. OpenAI's own endpoint caps at 4,096.
- `speed` is clamped to **0.5–2.0**; the clamped value comes back in `X-Paradee-Speed`, with
  `X-Paradee-Clamped: true|false`.
- `response_format` must be `wav` (omitting it means `wav`). Any other value is a
  `400 unsupported_response_format` with the reason spelled out — this service never returns
  a WAV body labelled as mp3.
- An unknown `voice` is a `400 invalid_voice` listing the four allowed names.
- All errors use OpenAI's envelope: `{"error":{"message","type","param","code"}}`, so SDK
  clients raise `BadRequestError` with the message intact.
- Requests are generated **one at a time** inside a single worker (the ONNX session is shared
  and guarded by a lock). Concurrency scales by replicas, not by threads.

## How it is built

```
python:3.12-slim (builder)            python:3.12-slim (runtime)
  git + ca-certificates                 no git, no compilers
  venv: requirements.txt                 same venv, copied over
      + paradee @ git <commit>           weights + config from the builder
      + en_core_web_sm 3.8.0 wheel       non-root user `app`
  weights: sha256-verified download     uvicorn, 1 worker, $PORT
  gate: scripts/warmup.py  ─────────────► /health gates traffic
```

Why each step exists:

- **`paradee` has no PyPI release**, so it is installed from a pinned commit
  (`c2789ef9abf54a9384eef89cc7de3f1afe5ad01d`, tag `v1.2.1`). Version 1.0 → 1.2.1 shipped in
  weeks and added `with_timestamps`; the commit pin is what keeps the API shape stable.
- **`en_core_web_sm` must be pre-installed.** misaki calls `spacy.cli.download()` when the
  model package is missing, which would be a network call at runtime. It is installed as a
  wheel in the builder and the gate would fail if it were absent.
- **eSpeak NG ships inside `espeakng-loader`** (libespeak-ng 1.52.0 + 18 MB of data in the
  manylinux wheel); no `apt install espeak-ng` and no system data directory are needed.
- **Weights are baked and verified**: `onnx/paradee_int8.onnx` (8,462,119 B,
  sha256 `a6989796…c97da1`) and `config.json` (1,827 B) from Hugging Face revision
  `ed2934d4df8a4fd891e29584c5e1104649409765` (tag `v1.2`). A tag can move; a commit cannot.
- **The build gate** (`scripts/warmup.py`) loads the model with `hf_hub_download` monkeypatched
  to raise, synthesizes all four voices, checks that British and American phonemes actually
  differ, forces the eSpeak fallback for both accents, and asserts the WAV and timestamp
  contracts. A packaging regression fails the build instead of shipping.
- **Runtime is offline**: `HF_HUB_OFFLINE=1`, explicit `PARADEE_MODEL_PATH` /
  `PARADEE_CONFIG_PATH`, no volume, no credentials.

### Configuration

None. The template deploys with **zero variables**: the port comes from the platform
(`PORT`, default 8000), and every other setting is baked into the image. `PARADEE_MODEL_PATH`
and `PARADEE_CONFIG_PATH` exist for local testing only.

### Running locally

```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.11 or 3.12 (misaki requires <3.13)
pip install -r requirements.txt
pip install "paradee @ git+https://github.com/sahilmahendrakar/paradee@c2789ef9abf54a9384eef89cc7de3f1afe5ad01d"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
pip install openai   # only for tests/battery.py A10

MODELS_DIR="$PWD/models" bash scripts/fetch_models.sh
PARADEE_MODEL_PATH="$PWD/models/paradee_int8.onnx" \
PARADEE_CONFIG_PATH="$PWD/models/config.json" \
HF_HUB_OFFLINE=1 python scripts/warmup.py

PARADEE_MODEL_PATH="$PWD/models/paradee_int8.onnx" \
PARADEE_CONFIG_PATH="$PWD/models/config.json" \
HF_HUB_OFFLINE=1 uvicorn app.main:app --port 8011 --workers 1
python tests/battery.py --base-url http://127.0.0.1:8011
```

## Measured performance

Measured by `tests/battery.py` against a running instance — not copied from the model's
README. The vendor's ~62× realtime figure is **one CPU thread of an Apple M4 Pro** and is not
repeated here. Two columns: a local desktop (Windows, Python 3.11) and the deployed Railway
service (shared vCPU, single replica, one thread), which is the configuration a template user
actually gets.

| What | Local desktop (1 thread) | Railway (shared vCPU) |
|---|---|---|
| Latency, short sentence (10 calls) | p50 **231 ms**, p95 239 ms for 4.6 s of audio | p50 **2.1–3.9 s**, p95 2.2–7.1 s |
| Implied realtime factor | ~20× at p50 (19.9×) | **1.2–2.2×** at p50 (host load dependent) |
| 1,500-character paragraph | 102.6 s of audio in 4.5 s wall (**22.7×**) | 102.6 s of audio in 6.7–12.7 s (**8–15×**) |
| Speed scaling | `speed=2.0` → 1.92× shorter, `speed=0.5` → 2.02× longer | same (1.92× / 2.02×) |
| RSS after boot | 441 MB (both accents warmed) | **332 MB** (American lexicon warm) |
| RSS after the first British request | — | ~505 MB |
| RSS after a 1,500-character call | ~503 MB peak | **~766–795 MB plateau** |
| Cold start | ~28 s first boot, ~7 s on restart | about a minute including the healthcheck window; restart to `/health` in ~15–20 s |

Request latency on Railway swings with the shared host: the same 4.6 s sentence measured
2.1 s and 4.2 s p50 in different runs, while the long-paragraph throughput stayed at 8–15×
realtime. Quote the range, not a single sample.

Memory: the British lexicon is loaded lazily — `app/engine.py` warms only the American G2P at
boot and the first British request pays a one-off ~2 s while the lexicon comes in. That keeps a
fresh process at ~332 MB instead of ~440 MB. The bigger number is the per-request working set
for a maximum-length (1,500-character / ~102 s) utterance, which lifts RSS to a stable
~766–795 MB plateau; repeating the request does not raise it further, and lowering it is not
something the API layer can do (a `malloc_trim` per request was measured and moved it by only
~15 MB, so it was not kept). Give the instance ~1 GB of memory if you expect maximum-length
input.


## Disclosures

- **Output is stochastic, not bit-exact.** The exported graph contains `RandomNormal` noise
  nodes: the decoder's breathiness comes from randomised noise. The same text and voice always
  produce the **same length** of audio (the predicted durations are deterministic) but not the
  same samples — the run-to-run difference measured here is 0.85% of full scale on average.
  Do not build byte-level caching on top of it; cache by text + voice + speed with your own TTL.
- **English only.** Text in other languages is phonemized as if it were English.
- **24 kHz mono, WAV only** in v1. There is no mp3/opus/flac encoder in the image; requesting
  one is a clear 400.
- **Text is split by sentence**, and any sentence longer than 510 phonemes is cut at the last
  space that fits (the model's text side has 512 positions including two pads). Very long
  unpunctuated input therefore gains small pauses at chunk boundaries.
- **Quality is below the teacher.** The author's 200-sentence evaluation: UTMOS 4.40 for
  `af_heart` vs 4.52 for Kokoro-82M, 4.06 for `bf_isabella`, 4.32 for `am_puck`, 4.18 for
  `bm_fable`; word error rate within a few tenths of a point of Kokoro on all four voices.
  `bf_isabella` is the furthest from its teacher.
- **Early release.** Paradee 1.2.x moves fast (1.0 → 1.2.1 in a few weeks). This image pins
  the upstream commit and the model revision; upgrades are deliberate.
- **No server-side URL fetching.** The API takes text only, so there is no SSRF surface.

## Attribution and licenses

- **Paradee** — model and code by Sahil Mahendrakar, Apache-2.0:
  <https://github.com/sahilmahendrakar/paradee>, weights at
  <https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2>.
- **Kokoro-82M** — the teacher model, by hexgrad, Apache-2.0:
  <https://huggingface.co/hexgrad/Kokoro-82M>. Paradee is distilled from it and reuses its
  vocabulary and voice names.
- **Training sentences** — WikiText-103, **CC BY-SA 3.0**.
- **misaki** — grapheme-to-phoneme (Kokoro's own G2P), by hexgrad.
- **eSpeak NG** — bundled in `espeakng-loader` (GPL-3.0 library, loaded at runtime as a
  separate work; the espeak-ng phoneme data ships inside the wheel).
- **spaCy / en_core_web_sm** — MIT.
- **This wrapper** — Apache-2.0, see `LICENSE`.

## Verification

`tests/battery.py` is the acceptance battery (A1–A14): health, four distinct voices, British
vs American audio, the WAV header contract, timestamp ordering and duration consistency,
duration stability under the model's stochastic noise, a 1,500-character paragraph, speed
0.5/2.0 scaling plus clamping, the four 400 shapes, an `openai` SDK round-trip
(`audio.speech.create` + `models.list` + `BadRequestError`), a p50/p95 latency table with
implied realtime factor, RSS, and the offline flag. Restart persistence is checked by
restarting the service and re-reading `/health`.

`requirements.txt` holds the direct pins; `requirements.lock` is the full resolved closure
(every transitive package, with the exact `en_core_web_sm` wheel and its sha256) recorded from
the verified install. The Dockerfile installs `requirements.txt`; the lock is there to prove
what the tested build actually contained and to make an exact rebuild possible.
