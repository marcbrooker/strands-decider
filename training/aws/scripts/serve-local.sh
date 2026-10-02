#!/usr/bin/env bash
# Serve a writer and TWO verifiers for the data generators, on one multi-GPU host.
#
# Three vLLM servers (one model each) sit behind a LiteLLM router that exposes ONE
# OpenAI-compatible endpoint and dispatches on the model name. That matches the generators'
# client (data/generators/llm_client.py), which routes by the request's "model" field against
# a single base URL, and passes two verifiers as a comma-separated --verify-models list.
#
#     export HOBSON_LLM_BACKEND=local
#     export HOBSON_LLM_BASE_URL=http://127.0.0.1:4000/v1
#     python data/generators/gen_documents_openrouter.py --skill-set v16 --docs 2000 \
#         --reasoning-effort none --writer local-writer \
#         --verify-models local-verifier-1,local-verifier-2 --workers 64
#
# A question is kept only if BOTH verifiers agree with the writer (the generators' unanimity
# rule). The default trio is Gemma-4 (Apache-2.0, US open weights): a 12B dense writer and two
# cross-architecture judges (a 26B-A4B MoE and the 31B dense model, which is also the g4
# distillation teacher -- a mild selection coupling, accepted for the stronger judge).
#
# Usage:
#   training/aws/scripts/serve-local.sh up      # launch the three vLLM servers + the router
#   training/aws/scripts/serve-local.sh stop     # kill everything this script started
#   training/aws/scripts/serve-local.sh status   # show what is listening
#
# Knobs (environment; defaults for the hobson-g4 host with GPU 4 in use elsewhere):
#   WRITER_MODEL=google/gemma-4-12B-it         WRITER_GPUS=0      WRITER_PORT=8001
#   VERIFIER1_MODEL=google/gemma-4-26B-A4B-it  VERIFIER1_GPUS=1   VERIFIER1_PORT=8002
#   VERIFIER2_MODEL=google/gemma-4-31B-it      VERIFIER2_GPUS=2,3 VERIFIER2_PORT=8003
#     (TP=2: on one H100 the 59 GiB of bf16 weights leave 8.6 GiB of KV cache, under the
#      13.8 GiB one 16k-token request needs)
#   ROUTER_PORT=4000  MAX_MODEL_LEN=16384  GPU_MEM_UTIL=0.90  DTYPE=bfloat16  QUANT=
#   SERVE_VENV=/opt/hobson/venv-serve  (its bin/ is prepended to PATH for vllm + litellm)
#   LOG_DIR=~/vllm-logs  HOST=127.0.0.1  (loopback only)
# GPU 4 is never assigned by the defaults. QUANT empty => no --quantization (bf16 fits).
set -euo pipefail

WRITER_MODEL="${WRITER_MODEL:-google/gemma-4-12B-it}";        WRITER_GPUS="${WRITER_GPUS:-0}";        WRITER_PORT="${WRITER_PORT:-8001}"
VERIFIER1_MODEL="${VERIFIER1_MODEL:-google/gemma-4-26B-A4B-it}"; VERIFIER1_GPUS="${VERIFIER1_GPUS:-1}"; VERIFIER1_PORT="${VERIFIER1_PORT:-8002}"
VERIFIER2_MODEL="${VERIFIER2_MODEL:-google/gemma-4-31B-it}";  VERIFIER2_GPUS="${VERIFIER2_GPUS:-2,3}";  VERIFIER2_PORT="${VERIFIER2_PORT:-8003}"
ROUTER_PORT="${ROUTER_PORT:-4000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
QUANT="${QUANT:-}"
HOST="${HOST:-127.0.0.1}"
SERVE_VENV="${SERVE_VENV:-/opt/hobson/venv-serve}"
LOG_DIR="${LOG_DIR:-$HOME/vllm-logs}"

[[ -d "$SERVE_VENV/bin" ]] && PATH="$SERVE_VENV/bin:$PATH"
PIDFILE="$LOG_DIR/serve-local.pids"
ROUTER_CONFIG="$LOG_DIR/litellm.yaml"

log() { printf '%s serve-local: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
tp_size() { awk -F, '{print NF}' <<<"$1"; }

wait_healthy() {  # port, name
  local port="$1" name="$2" i
  for i in $(seq 1 120); do  # up to ~10 min: weights take a while to download/load
    curl -fsS "http://$HOST:$port/v1/models" >/dev/null 2>&1 && { log "$name healthy on $port"; return 0; }
    sleep 5
  done
  die "$name did not become healthy on $port (see $LOG_DIR/$name.log)"
}

launch_vllm() {  # served-name, model, gpus, port
  local name="$1" model="$2" gpus="$3" port="$4"
  local quant=(); [[ -n "$QUANT" ]] && quant=(--quantization "$QUANT")
  log "$name = $model on GPUs $gpus (TP=$(tp_size "$gpus")) port $port"
  CUDA_VISIBLE_DEVICES="$gpus" vllm serve "$model" \
    --served-model-name "$name" \
    --tensor-parallel-size "$(tp_size "$gpus")" --dtype "$DTYPE" "${quant[@]}" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --host "$HOST" --port "$port" >"$LOG_DIR/$name.log" 2>&1 &
  echo "$!" >>"$PIDFILE"
}

cmd_up() {
  command -v vllm >/dev/null || die "vllm not found (pip install vllm into $SERVE_VENV)"
  command -v litellm >/dev/null || die "litellm not found (pip install 'litellm[proxy]' into $SERVE_VENV)"
  mkdir -p "$LOG_DIR"; : >"$PIDFILE"

  launch_vllm local-writer     "$WRITER_MODEL"     "$WRITER_GPUS"     "$WRITER_PORT"
  launch_vllm local-verifier-1 "$VERIFIER1_MODEL"  "$VERIFIER1_GPUS"  "$VERIFIER1_PORT"
  launch_vllm local-verifier-2 "$VERIFIER2_MODEL"  "$VERIFIER2_GPUS"  "$VERIFIER2_PORT"

  wait_healthy "$WRITER_PORT" local-writer
  wait_healthy "$VERIFIER1_PORT" local-verifier-1
  wait_healthy "$VERIFIER2_PORT" local-verifier-2

  cat >"$ROUTER_CONFIG" <<EOF
model_list:
  - model_name: local-writer
    litellm_params: {model: openai/local-writer, api_base: http://$HOST:$WRITER_PORT/v1, api_key: local}
  - model_name: local-verifier-1
    litellm_params: {model: openai/local-verifier-1, api_base: http://$HOST:$VERIFIER1_PORT/v1, api_key: local}
  - model_name: local-verifier-2
    litellm_params: {model: openai/local-verifier-2, api_base: http://$HOST:$VERIFIER2_PORT/v1, api_key: local}
EOF
  log "router config at $ROUTER_CONFIG"
  litellm --config "$ROUTER_CONFIG" --host "$HOST" --port "$ROUTER_PORT" >"$LOG_DIR/litellm.log" 2>&1 &
  echo "$!" >>"$PIDFILE"
  wait_healthy "$ROUTER_PORT" litellm

  log "ready. Point the generators at the router:"
  echo
  echo "    export HOBSON_LLM_BACKEND=local"
  echo "    export HOBSON_LLM_BASE_URL=http://$HOST:$ROUTER_PORT/v1"
  echo "    --writer local-writer --verify-models local-verifier-1,local-verifier-2"
  echo
  log "logs in $LOG_DIR"
}

cmd_stop() {
  [[ -f "$PIDFILE" ]] || { log "no pidfile at $PIDFILE; nothing to stop"; return 0; }
  while read -r pid; do [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && { log "killing $pid"; kill "$pid" 2>/dev/null || true; }; done <"$PIDFILE"
  sleep 2
  while read -r pid; do [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && { log "force-killing $pid"; kill -9 "$pid" 2>/dev/null || true; }; done <"$PIDFILE"
  rm -f "$PIDFILE"; log "stopped"
}

cmd_status() {
  local pair port name
  for pair in "$WRITER_PORT:local-writer" "$VERIFIER1_PORT:local-verifier-1" "$VERIFIER2_PORT:local-verifier-2" "$ROUTER_PORT:litellm"; do
    port="${pair%%:*}"; name="${pair##*:}"
    curl -fsS "http://$HOST:$port/v1/models" >/dev/null 2>&1 && log "$name UP on $port" || log "$name DOWN on $port"
  done
}

case "${1:-up}" in
  up) cmd_up ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) die "usage: $0 {up|stop|status}" ;;
esac
