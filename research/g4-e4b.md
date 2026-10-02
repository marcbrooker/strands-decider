# g4-e4b: g4's recipe on a Gemma 4 E4B torso

**Exploratory, not pre-registered.** No predictions were committed before training, so
this record states what was measured and cannot promote a model by the rule in
[README.md](README.md#preregistration-practice). It is one run.

## What changed

Only the torso. `configs/experiments/g4-e4b.yaml` is `configs/experiments/g4.yaml` with
`base_model: google/gemma-4-E4B-it` (snapshot `ee0ef60`; 42 layers, hidden 2560, against
E2B's 35 and 1536) and its own `output_dir`; `tests/test_configs.py` holds the two files to
that difference. Same 31B teacher file (`teacher31b`, committed labels), no frozen anchor,
same hyperparameters, `force_bos` and `host_embeddings`. 7.50B parameters in the loaded
torso (with the per-layer embedding table), 36.2M trainable.

## The run

2026-10-02, one H100 80GB of a p5.48xlarge shared with a concurrent g4 retrain, code
`4458a37` (tag `g4-e4b-trained`), no restart. The route of the
[training README](../training/README.md):

```bash
training/recipe.sh build fetch multistep generated adequacy teacher31b
TRAIN_CONFIG=configs/experiments/g4-e4b.yaml CKPT=checkpoints/hobson-e4b-g4 \
  training/recipe.sh train calibrate eval
evaluation/jevbench/jevbench.sh checkpoints/hobson-e4b-g4 <out> v19
```

Training took 4 h 40 min (3,738 steps at 0.22 step/s; g4 took 3 h 13 min on an RTX PRO
6000), peaking at 24.2 GiB allocated. Validation 0.386 / 0.833 (g4 0.427 / 0.810).
Temperatures: choice 1.328, noul 0.818, score 2.534.

## Results

The g4 and v19 columns are from [PREREGISTRATION-g4.md](preregistrations/PREREGISTRATION-g4.md).
g4-e4b's JevBench and all its sets were measured on the H100; g4's and v19's multi-step,
generated and adequacy sets were re-run on the local RTX 3090. Small cross-machine
differences are possible there; the JevBench comparison against v19 is paired on one
task file.

| | v19 | g4 | g4-e4b |
| --- | --- | --- | --- |
| JevBench (231) at 4096: tasks | 168 | 183 | **186** |
| easy / standard / hard | 48 / 63 / 57 | 48 / 68 / 67 | 48 / **71** / 67 |
| Brier / ECE / paraphrase consistency | 0.342 / 0.051 / 0.861 | 0.292 / 0.048 / 0.944 | **0.250 / 0.042 / 0.972** |
| JevBench latency p50 / p95 | 115 / 296 ms (3090) | - / 159 ms (RTX PRO 6000) | 122 / 204 ms (H100) |
| MuSiQue [g4 floor 0.859] | 0.879 | 0.902 | **0.913** |
| ContractNLI [0.842] | 0.862 | 0.861 | **0.881** |
| BoardgameQA [0.790] | 0.810 | 0.781 | **0.832** |
| HotpotQA, held out [0.706] | 0.726 | 0.759 | **0.840** |
| adequacy, HelpSteer2 [0.719] | 0.739 | 0.722 | **0.795** |
| adequacy, generated [0.768] | 0.788 | 0.801 | **0.831** |
| held-out short tasks: accuracy / ECE | 0.647 / 0.054 | 0.655 / 0.065 | **0.680 / 0.053** |
| held-out sarcasm (yes/no) | - | 0.632 | **0.716** |

**JevBench against v19** (paired, `paired.txt`; v19's row here is 167): +30 / -11, net +19,
McNemar **p = 0.004**; standard +9 / -1 (p = 0.02), hard +21 / -10 (p = 0.07). Against g4,
+3 tasks, inside the ~4.5-task run-to-run noise; no paired test (g4's per-task results are
not in `research/data/`). By family: `adequacy`, `intent`, `routing`, `extraction`, `trap`,
`adversarial`, `tool_selection` and `ordinal` all right; weakest `temporal_numeric` 3/15,
`probability` 5/10, `multi_hop` 9/18, `ambiguous` 4/7.

**Held out by task:** emotion 0.590, hate severity 0.479, intent 0.910, sarcasm 0.716 (its
ECE 0.155, the worst calibrated slice). Confidence >= 0.9: 1,717 rows at 0.962 accuracy.
**Paraphrase pairs** both right 0.873 / 0.855 (g4 0.850 / 0.769); **flip pairs** 0.567
(g4 0.450).

Every floor g4 was held to is met, including the two it missed (BoardgameQA, and generated
adequacy above g3's 0.804).

## Reading

The larger torso moved every measurement we have the same way: the best JevBench, Brier,
ECE and paraphrase consistency recorded here, and gains on every internal set, largest on
held-out HotpotQA (+0.08 over g4) and the adequacy sets (+0.07 / +0.03), where g4's
teacher led (0.803 / 0.920) and g4 did not follow. One reading, untested here: the E2B
student's capacity, not the teacher signal, limited what g4 took from the teacher. The
BoardgameQA loss g4 took when the anchor went away is gone without restoring the anchor.

What it costs: a larger torso (7.50B loaded parameters with the per-layer embedding table)
and longer training (4 h 40 min on an H100, against g4's 3 h 13 min on a different GPU, so
not a like-for-like ratio). Serving latency against E2B on the same GPU was not measured.

## What this cannot show

- **Not pre-registered and one run**, so it cannot claim a bar. +3 tasks over g4 is noise.
- **Cross-machine set comparisons.** The HotpotQA gain in particular should be confirmed by
  re-running g4's and v19's sets on the H100, or g4-e4b's on the 3090.
- **Which part of E4B helps** (depth, width, or the larger per-layer embeddings).

## Artifacts

`results/g4-e4b-20261002/` in the project bucket: checkpoint, data, reports, JevBench. A
Hugging Face export (`python -m strands_decider.hf_export`, verified) names the repo
`StrandsAgents/strands-decider-4B-g4-e4b`; its `training/stages.jsonl` was rebuilt from
the run log's stage markers and file times, as the run had no per-stage timer.
