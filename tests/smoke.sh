#!/usr/bin/env bash
# Quick smoke test against a running instance (local or Railway).
#   ./tests/smoke.sh https://<service>.up.railway.app
# The full acceptance battery is tests/battery.py; this is the 20-second check.
set -euo pipefail

BASE="${1:-http://127.0.0.1:8011}"
BASE="${BASE%/}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

say() { printf '%s\n' "$*"; }

say "== /health"
curl -fsS "$BASE/health" | tee "$TMP/health.json"
say

say "== /voices"
curl -fsS "$BASE/voices" | head -c 400
say

say "== /v1/audio/speech (af_heart)"
code=$(curl -sS -o "$TMP/a.wav" -w '%{http_code} %{size_download}' \
  -X POST "$BASE/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"model":"paradee-1.2","input":"Paradee is a small voice that runs anywhere.","voice":"af_heart","response_format":"wav"}')
say "http=$code"
python - "$TMP/a.wav" <<'PY'
import sys, wave
with wave.open(sys.argv[1], "rb") as w:
    print(f"wav channels={w.getnchannels()} width={w.getsampwidth()} rate={w.getframerate()} frames={w.getnframes()}")
PY

say "== /v1/audio/speech/with-timestamps (bm_fable)"
curl -fsS -X POST "$BASE/v1/audio/speech/with-timestamps" -H 'Content-Type: application/json' \
  -d '{"input":"One two three. Four five.","voice":"bm_fable"}' \
  | python -c 'import json,sys; d=json.load(sys.stdin); print("duration_s=",d["duration_s"],"words=",d["words"][:4])'

say "== error shapes"
for body in '{"input":"hi","voice":"af_nova"}' '{"input":"hi","response_format":"mp3"}' '{"input":"   "}'; do
  printf '%s -> ' "$body"
  curl -sS -o /dev/null -w '%{http_code}\n' -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' -d "$body"
done

say "OK"
