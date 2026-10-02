# Preregistrations

Each file here was written before its run, and the outcome was appended after the run.
The files are kept byte for byte, with one exception: before publication, the exact API
spend in v16, v18, v19 and v20 was rounded to a reproduction cost, and the restart, retry,
per-part and budget details next to it were removed; nothing else changed. They name paths
from the layout they were written in, and those paths have since moved:

| Path in the files | Path now |
|---|---|
| `recipe.sh`, `./recipe.sh`, `recipe_v7.sh` | `training/recipe.sh`, `training/recipe_v7.sh` |
| `infra/run_recipe.sh` | `training/run_recipe.sh` |
| `infra/eval/jevbench.sh`, `infra/eval/paired.py` | `evaluation/jevbench/` |
| `infra/` (the rest) | `training/aws/` |
| `tests/multistep_eval.py`, `tests/pair_eval.py`, `tests/pair_accuracy.py`, `tests/question_sensitivity.py`, `tests/calibrate_mix.py` | `evaluation/` |
| `tests/verify_synth_labels.py`, `tests/verify_synth_pairs.py` | `data/checks/` |
| `scripts/bench_local.py` | `evaluation/bench_local.py` |
| `scripts/jevbench_cold_warm.py` | `evaluation/jevbench/jevbench_cold_warm.py` |
| `scripts/` (the generators and their `gen_*/` exports) | `data/generators/` |
| `docs/inference.md`, `docs/training.md`, `docs/evaluation.md`, `docs/data.md` | `inference/README.md`, `training/README.md`, `evaluation/README.md`, `data/README.md` |
| `docs/experiments/README.md`, `docs/experiments/history.md` | `research/README.md`, `research/history.md` |
| `docs/experiments/PREREGISTRATION-*.md` | this folder |
| `report/HISTORY.md`, `report/data/`, `report/figures/`, `report/scripts/` | `research/generations.md`, `research/data/`, `research/figures/`, `research/scripts/` |
| `scripts/merge_teacher_g4.py` (g4) | `training/merge_teacher_g4.py` |

## Configurations

The training configuration of each pre-registered run from v11 on is in
`configs/experiments/`. Its header comment names its preregistration. These files are
records: their settings are not changed, only their comments are corrected, and some
name an input file that no step of `recipe.sh` writes today. The Inputs column gives the `training/recipe.sh` steps, and any command outside
the recipe, that write every file a configuration reads (`train_files`, `teacher_file`,
`kl_only_files`).

| Run | Configuration | Inputs |
| --- | --- | --- |
| v11a, v11b | [v11a.yaml](../../configs/experiments/v11a.yaml), [v11b.yaml](../../configs/experiments/v11b.yaml) | `build`, then `python -m hobson.data.question_transforms`, which writes `data/train_v11a.jsonl` (v11a) and `data/kl_v11b.jsonl` (v11b) |
| v12 | [v12.yaml](../../configs/experiments/v12.yaml) | `build`, then the labels command in the header of v12.yaml (`python -m hobson.data.teacher ... --out data/teacher_v5_qwen35-4b.jsonl`). Do not run `distill` after it: it copies the committed file, which lacks the 21,000 score rows, over that path |
| v13 | [v13.yaml](../../configs/experiments/v13.yaml) | `build` |
| v14 | [v14.yaml](../../configs/experiments/v14.yaml) | `build fetch multistep teacher`. The record names `data/teacher_v14_train.jsonl`; `teacher` writes `data/teacher_multistep_v14_train.jsonl`, so copy it to the record's name |
| v15 | [v15.yaml](../../configs/experiments/v15.yaml) | v14's steps, then `python -m hobson.data.policy`, which writes `data/policy_v15.jsonl` from ShARC and ConditionalQA in `data/raw/sharc/` and `data/raw/conditionalqa/`; `fetch` does not download these two |
| v16 | [v16.yaml](../../configs/experiments/v16.yaml) | `build fetch multistep generated teacher`, with v14's note on the teacher file's name |
| v17 | [v17.yaml](../../configs/experiments/v17.yaml) | `build fetch multistep generated distill`; `distill` copies the committed `data/synthetic/replay_v14_multistep.jsonl` into `data/` |
| v18 | [v18.yaml](../../configs/experiments/v18.yaml) | as v17 |
| v19 | [v19.yaml](../../configs/experiments/v19.yaml) | `build fetch multistep generated adequacy distill` |
| v19-seed1 | [v19-seed1.yaml](../../configs/experiments/v19-seed1.yaml) | as v19 |
| v20 | [v20.yaml](../../configs/experiments/v20.yaml) | the route in the header of `recipe.sh`: `training/recipe.sh build fetch multistep generated adequacy catchall distill`, then `TRAIN_CONFIG=configs/experiments/v20.yaml CKPT=checkpoints/hobson-2b-v20-retrain training/recipe.sh train calibrate eval` |
| g4 | [g4.yaml](../../configs/experiments/g4.yaml) | `build fetch multistep generated adequacy teacher31b`; `teacher31b` merges the committed gemma-4-31B-it labels with v14's replay distributions into `data/teacher_g4.jsonl` |
| g4-e4b (exploratory, not pre-registered) | [g4-e4b.yaml](../../configs/experiments/g4-e4b.yaml) | as g4; outcome in [research/g4-e4b.md](../g4-e4b.md) |

g4 was run in the hobson-gemma4 fork, which moved the torso from Qwen3.5-2B-Base to
Gemma 4 E2B. Its preregistration builds on that fork's g1, g2 and g3, which are not copied
here; it says what g4 changes from g3. It did not meet its own bar (BoardgameQA and
generated adequacy), and v19 remains this repository's reference recipe.

v19-calmix trains nothing. It refits temperatures with `evaluation/calibrate_mix.py`. The v19
reference recipe is `configs/train.yaml` with its parent `configs/train-parent.yaml`.
[training/README.md](../../training/README.md) describes how to run it.
