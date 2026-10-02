#!/usr/bin/env bash
# A small pilot: serve the writer + two verifiers, generate a handful of items with each
# generator, and print the two-verifier keep-rate per export. Run it ON the GPU host, from the
# repository root. Uses the "local" backend (data/generators/llm_client.py) and serve-local.sh.
#
#   training/aws/scripts/pilot-local.sh
#
# Default trio (Gemma-4): writer gemma-4-12B-it on GPU 0, verifiers gemma-4-26B-A4B-it on GPU 1
# and gemma-4-31B-it on GPUs 2-3 (TP=2). GPU 4 is left clear (in use by another job). Override any
# serve-local.sh knob through the environment.
#
# Knobs:
#   DOCS=10   docs per skill set (v16, mixed, weak)   BATCHES=5  adequacy (6 items each) + flips
#   WORKERS=32   OUT=/opt/hobson/scratch/local-pilot   KEEP_SERVING=0  (1 leaves servers up)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"
SERVE="$HERE/serve-local.sh"

DOCS="${DOCS:-10}"; BATCHES="${BATCHES:-5}"; WORKERS="${WORKERS:-32}"
OUT="${OUT:-/opt/hobson/scratch/local-pilot}"
KEEP_SERVING="${KEEP_SERVING:-0}"
ROUTER_PORT="${ROUTER_PORT:-4000}"
SERVE_VENV="${SERVE_VENV:-/opt/hobson/venv-serve}"
PY="${PY:-$SERVE_VENV/bin/python}"; [[ -x "$PY" ]] || PY=python3

log() { printf '%s pilot: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
cleanup() { [[ "$KEEP_SERVING" == "1" ]] || { log "stopping servers"; "$SERVE" stop || true; }; }
trap cleanup EXIT

log "serving writer + two verifiers (GPU 4 left clear)"
"$SERVE" up

export HOBSON_LLM_BACKEND=local
export HOBSON_LLM_BASE_URL="http://127.0.0.1:$ROUTER_PORT/v1"
cd "$REPO"

common=(--reasoning-effort none --writer local-writer
        --verify-models local-verifier-1,local-verifier-2 --workers "$WORKERS")

log "documents (v16 skill set), $DOCS docs"
"$PY" data/generators/gen_documents_openrouter.py --out "$OUT/gen_v16"   --skill-set v16   --docs "$DOCS" "${common[@]}"
log "documents (mixed skill set), $DOCS docs"
"$PY" data/generators/gen_documents_openrouter.py --out "$OUT/gen_mixed" --skill-set mixed --docs "$DOCS" "${common[@]}"
log "documents (weak skill set), $DOCS docs"
"$PY" data/generators/gen_documents_openrouter.py --out "$OUT/gen_weak"  --skill-set weak  --docs "$DOCS" "${common[@]}"
log "adequacy, $BATCHES batches"
"$PY" data/generators/gen_adequacy_openrouter.py  --out "$OUT/gen_adequacy" --batches "$BATCHES" "${common[@]}"
log "flips, $BATCHES batches"
"$PY" data/generators/gen_flips_openrouter.py     --out "$OUT/gen_flips"    --batches "$BATCHES" "${common[@]}"
# Paraphrases are left out of the pilot: gen_paraphrases reads the committed question exports
# and takes a single --checker, not two verifiers.

echo
log "===== two-verifier keep-rate (gemma-4-12B writer; 26B-A4B + 31B verifiers) ====="
"$PY" - "$OUT" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
for name in ("gen_v16", "gen_mixed", "gen_weak", "gen_adequacy", "gen_flips"):
    sp = root / name / "stats.json"
    if not sp.exists():
        print(f"  {name:14s} (no stats.json -- generation did not complete)"); continue
    s = json.loads(sp.read_text())
    produced = s.get("questions") or s.get("items") or s.get("pairs") or 0
    kept = s.get("kept", 0)
    pct = (100 * kept / produced) if produced else 0.0
    print(f"  {name:14s} produced {produced:4d}  kept {kept:4d}  ({pct:4.1f}%)")
print("\nCommitted single-judge (Qwen) keep-rates, for reference:")
print("  gen_v16 ~80%, gen_mixed ~82%, gen_weak ~78%, gen_adequacy ~81%, gen_flips ~52%")
print("Two distinct verifiers are a stricter (unanimous) gate, so expect lower keep-rates.")
PY
log "pilot exports in $OUT"
