#!/usr/bin/env bash
# The v19 recipe, end to end -- the reference recipe. Run it under WSL2 (see
# training/README.md#setup): the Qwen3.5 torso's fused kernels need Linux.
#
#   1. build     the v5 corpus and the held-out calibration file. A fresh build today
#                gives the sha256 recorded in data/SHA256SUMS. Identity with the
#                original v5 file is not verified.
#   2. fetch     the ContractNLI, MuSiQue and HelpSteer2 releases into data/raw/ (~350 MB)
#   3. multistep the 12,909 multi-step training rows and the evaluation sets
#   4. generated the generated document questions and their held-out-domain eval sets --
#                v16's 2,148 and v18's 1,667 -- copied from data/synthetic/, the exact
#                files every model since v16 used, and the same rows with v20's checked
#                paraphrases attached (plus the paired consistency evals and v20's
#                instruction-flip rows). No generator is rerun.
#   5. adequacy  the answer-adequacy rows and eval sets: HelpSteer2 (4,866 rows, built from
#                the download) and the generated items (1,300 rows, data/synthetic/)
#   6. teacher   frozen Qwen3.5-4B distributions for the multi-step rows (~1 h)
#   7. parent    the parent, v14's recipe: configs/train-parent.yaml (~5 h)
#   8. replay    the parent's own distributions for the multi-step rows (~40 min)
#   9. train     configs/train.yaml, toward those distributions (~6 h on one RTX 3090)
#  10. calibrate, then eval: held-out short tasks, the multi-step sets (HotpotQA among
#      them as the held-out transfer test), the generated held-out-domain sets and the
#      two answer-adequacy sets
#
# Writes checkpoints/hobson-2b-recipe-parent and $CKPT (default checkpoints/hobson-2b-recipe):
# `parent` always writes checkpoints/hobson-2b-recipe-parent, where `replay` reads it, and
# `train` always writes to $CKPT, whatever the configs say.
# Usage: training/recipe.sh STEP [STEP ...], each one of
#   build fetch multistep generated adequacy catchall teacher teacher31b distill parent
#   replay train calibrate eval all
#
# Synthetic data is committed; public data is not. data/synthetic/ holds every generated
# row and every model-produced label the recipes train on, as the exact files used: the
# generated questions and adequacy items (built from the data/generators/gen_*/ exports by
# strands_decider.data.generated), v20's instruction flips, the frozen Qwen3.5-4B's distributions
# on the short-task corpus and the multi-step rows, and v14's replay distributions.
# Public sources are downloaded and converted by `build`, `fetch`, `multistep` and
# `adequacy`, and v20's catch-all rows are derived from them by `catchall`.
#
# Retraining v20 (configs/experiments/v20.yaml) from the repo:
#   training/recipe.sh build fetch multistep generated adequacy catchall distill
#   TRAIN_CONFIG=configs/experiments/v20.yaml CKPT=checkpoints/hobson-2b-v20-retrain \
#     training/recipe.sh train calibrate eval
# `distill` uses the committed teacher and replay labels, so no parent is trained.
#
# Training g4 (configs/experiments/g4.yaml: a Gemma 4 E2B torso toward gemma-4-31B-it's
# distributions; research/preregistrations/PREREGISTRATION-g4.md) from the repo:
#   training/recipe.sh build fetch multistep generated adequacy teacher31b
#   TRAIN_CONFIG=configs/experiments/g4.yaml CKPT=checkpoints/hobson-e2b-g4-retrain \
#     training/recipe.sh train calibrate eval
# `teacher31b` uses the committed 31B labels, so no 31B model is loaded unless RELABEL=1.
# g4 trains on one GPU (host_embeddings), not under NGPU > 1.
# NGPU=8 uses eight GPUs: teacher and replay one shard per GPU, then a merge; parent and
# train under torchrun (training/README.md#training-on-several-gpus). PARENT_CONFIG and TRAIN_CONFIG
# replace the two training configs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # every path below is relative to the repo root
CKPT="${CKPT:-checkpoints/hobson-2b-recipe}"
export PY="${PY:-python}"  # the active environment; the AWS runner sets PY explicitly
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
NGPU="${NGPU:-1}"
PARENT_CONFIG="${PARENT_CONFIG:-configs/train-parent.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train.yaml}"
strands-decider() { "$PY" -u -m strands_decider.cli "$@"; }

# label MODULE ARGS...: one process, or with NGPU > 1 one shard per GPU and then --merge.
label() {
  [ "$NGPU" -gt 1 ] || { "$PY" -m "$@"; return; }
  local i bad=0 pids=()
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i "$PY" -m "$@" --num-shards "$NGPU" --shard-index "$i" & pids+=("$!")
  done
  for i in "${pids[@]}"; do wait "$i" || bad=1; done
  [ "$bad" = 0 ] || { echo "$1: a shard failed" >&2; return 1; }
  "$PY" -m "$@" --merge --num-shards "$NGPU"
}

fit() {  # fit CONFIG [strands-decider train options]
  local cfg="$1"; shift
  if [ "$NGPU" -gt 1 ]; then
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" -m strands_decider.cli train --config "$cfg" "$@"
  else strands-decider train --config "$cfg" "$@"; fi
}

# verify FILE...: stop unless data/SHA256SUMS has an entry for each FILE and FILE has that
# sha256. Teacher and replay targets attach to corpus rows by position, so they are correct
# only for these exact bytes. A changed upstream download also stops here. To check every
# recorded file by hand: sha256sum -c data/SHA256SUMS
verify() {
  local f line sums=""
  for f; do
    line="$(awk -v f="$f" '$2 == f' data/SHA256SUMS)"
    [ -n "$line" ] || { echo "$f: no entry in data/SHA256SUMS" >&2; return 1; }
    sums+="$line"$'\n'
  done
  printf '%s' "$sums" | sha256sum -c --quiet --strict - ||
    { echo "stop: the files above do not match data/SHA256SUMS" >&2; return 1; }
}

build() { bash training/recipe_v7.sh build; }

# download FILE URL: get URL into FILE, unless FILE exists. curl fails on an HTTP error and
# writes to FILE.part. Only a complete download becomes FILE, so a rerun never keeps a failed
# or partial file. fetch checks the downloads against data/SHA256SUMS before it extracts them.
download() { [ -f "$1" ] || { curl -fsSL -o "$1.part" "$2" && mv "$1.part" "$1"; }; }

fetch() {
  mkdir -p data/raw/helpsteer2 && cd data/raw
  download contract-nli.zip \
    https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip
  # MuSiQue's authors distribute through Google Drive (see their download_data.sh).
  download musique_data_v1.0.zip \
    "https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t"
  for f in train validation; do
    download "helpsteer2/$f.jsonl.gz" \
      "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/$f.jsonl.gz"
  done
  cd - >/dev/null
  verify data/raw/contract-nli.zip data/raw/musique_data_v1.0.zip \
    data/raw/helpsteer2/train.jsonl.gz data/raw/helpsteer2/validation.jsonl.gz
  cd data/raw
  "$PY" - <<'EOF'
import zipfile
z = zipfile.ZipFile("contract-nli.zip")
z.extractall(".", [m for m in z.namelist() if m.endswith((".json", "LICENSE", "TERMS", "README.md"))])
z = zipfile.ZipFile("musique_data_v1.0.zip")
for m in ("data/musique_full_v1.0_train.jsonl", "data/musique_full_v1.0_dev.jsonl"):
    z.extract(m, "musique")
print("extracted ContractNLI and MuSiQue (full)")
EOF
  cd - >/dev/null
}

multistep() { "$PY" -m strands_decider.data.multistep; }

# The committed files were built from the exports by
#   python -m strands_decider.data.generated [--src data/generators/gen_v16]              (v16)
#   python -m strands_decider.data.generated --src data/generators/gen_pilot_qwen data/generators/gen_mixed_pilot \
#     data/generators/gen_weak --out data/generated_v18.jsonl --eval-out data/generated_v18_eval.jsonl
#   python -m strands_decider.data.generated --src data/generators/gen_adequacy --out data/adequacy_gen.jsonl \
#     --eval-out data/adequacy_gen_eval.jsonl
#   python -m strands_decider.data.generated --src data/generators/gen_flips --out data/flips_v20.jsonl \
#     --eval-out data/flips_v20_eval.jsonl
# A rebuild today need not match them byte for byte, which is why they are committed.
generated() {
  local v
  for v in generated_v16 generated_v16_eval generated_v18 generated_v18_eval flips_v20 flips_v20_eval; do
    cp "data/synthetic/$v.jsonl" "data/$v.jsonl"
  done
  for v in v16 v18; do
    "$PY" -m strands_decider.data.generated --attach "data/generated_$v.jsonl" \
      --attach-eval "data/generated_${v}_eval.jsonl" --paraphrases data/generators/gen_paraphrases/paraphrases.jsonl \
      --out "data/generated_${v}p.jsonl" --pairs-out "data/para_pairs_${v}_eval.jsonl"
  done
}

adequacy() {
  "$PY" -m strands_decider.data.adequacy --raw data/raw/helpsteer2 \
    --out data/adequacy_hs2.jsonl --eval-out data/adequacy_hs2_eval.jsonl
  cp data/synthetic/adequacy_gen.jsonl data/synthetic/adequacy_gen_eval.jsonl data/
}

catchall() { "$PY" -m strands_decider.data.catchall; }  # v20: from data/train_v5.jsonl and the held-out set

teacher() {
  verify data/train_v5.jsonl data/multistep_v14.jsonl
  label strands_decider.data.teacher --src data/multistep_v14.jsonl \
    --out data/teacher_multistep_v14.jsonl --shift-by data/train_v5.jsonl
}

# v20's teacher file: the committed 4B distributions on the short-task corpus (labelled
# for v12 by `python -m strands_decider.data.teacher --src data/train_v5.jsonl --out
# data/teacher_v5_qwen35-4b.jsonl`; the committed copy omits the score rows, which
# `distill` never kept), kept where they agree with gold, then v14's replay
# distributions on the multi-step rows (`python -m strands_decider.data.replay
# checkpoints/hobson-2b-v14 ...`, as in `replay`).
distill() {
  verify data/train_v5.jsonl data/synthetic/teacher_v5_qwen35-4b.jsonl \
    data/synthetic/replay_v14_multistep.jsonl
  cp data/synthetic/teacher_v5_qwen35-4b.jsonl data/synthetic/replay_v14_multistep.jsonl data/
  "$PY" -m strands_decider.data.distill --corpus data/train_v5.jsonl --teacher data/teacher_v5_qwen35-4b.jsonl \
    --append data/replay_v14_multistep.jsonl --out data/teacher_v20.jsonl
}

# g4's teacher file: gemma-4-31B-it's committed distributions on the short-task, generated
# and adequacy rows, plus v14's replay distributions on the multi-step rows, merged into
# data/teacher_g4.jsonl without the rating-scale rows (training/merge_teacher_g4.py).
# RELABEL=1 relabels with the 31B model instead (~2 h on one 96 GB GPU, 62 GB of weights).
TEACHER31B_REV=842da3794eaa0b77d5f08bae87a17459d91ff475
teacher31b() {
  local f
  verify data/train_v5.jsonl data/multistep_v14.jsonl data/synthetic/replay_v14_multistep.jsonl
  mkdir -p data/teacher31b
  cp data/synthetic/replay_v14_multistep.jsonl data/
  for f in train_v5 generated_v16 generated_v18 adequacy_hs2 adequacy_gen; do
    if [ "${RELABEL:-0}" = 1 ]; then
      label strands_decider.data.teacher --src "data/$f.jsonl" --out "data/teacher31b/$f.jsonl" \
        --model google/gemma-4-31B-it --revision "$TEACHER31B_REV" --max-batch-tokens 16000
    else
      verify "data/synthetic/teacher_gemma4-31b-it_$f.jsonl"
      cp "data/synthetic/teacher_gemma4-31b-it_$f.jsonl" "data/teacher31b/$f.jsonl"
    fi
  done
  "$PY" training/merge_teacher_g4.py --labels-dir data/teacher31b --out data/teacher_g4.jsonl
}

parent() {
  verify data/train_v5.jsonl data/multistep_v14.jsonl
  fit "$PARENT_CONFIG" --output-dir checkpoints/hobson-2b-recipe-parent
}

replay() {
  verify data/train_v5.jsonl data/multistep_v14.jsonl
  label strands_decider.data.replay checkpoints/hobson-2b-recipe-parent --src data/multistep_v14.jsonl \
    --out data/replay_parent_multistep.jsonl --shift-by data/train_v5.jsonl
}

train() {
  verify data/train_v5.jsonl data/multistep_v14.jsonl
  fit "$TRAIN_CONFIG" --output-dir "$CKPT"
}

calibrate() { verify data/holdout_v5_norule.jsonl; strands-decider calibrate "$CKPT" --data data/holdout_v5_norule.jsonl; }

evaluate() {
  verify data/holdout_v5_norule.jsonl data/multistep_v14_eval.jsonl
  mkdir -p reports
  strands-decider eval "$CKPT" --data data/holdout_v5_norule.jsonl --limit 6000 --out reports/recipe_heldout.json
  "$PY" evaluation/multistep_eval.py "$CKPT"
  "$PY" evaluation/multistep_eval.py "$CKPT" --data data/generated_v16_eval.jsonl
  "$PY" evaluation/multistep_eval.py "$CKPT" --data data/generated_v18_eval.jsonl
  "$PY" evaluation/multistep_eval.py "$CKPT" --data data/adequacy_hs2_eval.jsonl
  "$PY" evaluation/multistep_eval.py "$CKPT" --data data/adequacy_gen_eval.jsonl
  # v20's evaluations, where their files have been built (`catchall`, `generated`)
  if [ -f data/catchall_v20_eval.jsonl ]; then
    "$PY" evaluation/multistep_eval.py "$CKPT" --data data/catchall_v20_eval.jsonl
  fi
  if [ -f data/flips_v20_eval.jsonl ]; then
    "$PY" evaluation/pair_eval.py "$CKPT" data/para_pairs_v16_eval.jsonl data/para_pairs_v18_eval.jsonl \
      data/flips_v20_eval.jsonl
  fi
}

[ $# -gt 0 ] || set -- all
for STEP in "$@"; do
  case "$STEP" in
    build|fetch|multistep|generated|adequacy|catchall|teacher|teacher31b|distill|parent|replay|train|calibrate) "$STEP" ;;
    eval) evaluate ;;
    all) build; fetch; multistep; generated; adequacy; teacher; parent; replay; train; calibrate; evaluate ;;
    *) echo "unknown step: $STEP" >&2; exit 2 ;;
  esac
done
