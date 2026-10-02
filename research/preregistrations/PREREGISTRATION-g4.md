# Pre-registration: g4, a gemma-4-31B-it teacher in place of the frozen anchor

Committed before labelling and training. Same discipline as before: predictions fixed
here, the outcome appended below without editing anything above it.

## Why

g3 (gemma-4-E2B-it with `<bos>`, v19's recipe) scored 171/231 on JevBench, finished
level with v19 on validation and ahead on seven of nine sets, and missed the default rule
narrowly on three counts — JevBench by two tasks, HelpSteer2 adequacy by two rows, held-out
ECE by 0.003 — all pointing at yes/no handling (PREREGISTRATION-g3.md). Its standard-tier
misses were 7 yes/no tasks out of 8, every one of them also wrong, confidently, in the
frozen reading g3's anchor pulls toward.

Step 0 (2026-10-02, one g7e, `data/teacher.py` with the soft-cap fix of caa6969) read the
larger Gemma 4 instruction-tuned models frozen, the way a teacher labels rows:

| frozen, read as a teacher | E2B-it | 12B-it | 26B-A4B-it | 31B-it | g3 (trained) |
| --- | --- | --- | --- | --- | --- |
| JevBench (SemIf's reading) | 146 | 194 | 204 | **212** | 171 |
| HelpSteer2 adequacy | 0.581 | 0.752 | 0.752 | **0.803** | 0.714 |
| generated adequacy, balanced | 0.703 | 0.870 | 0.895 | **0.920** | 0.804 |
| held-out sarcasm (yes/no) | 0.561 | 0.677 | 0.724 | **0.786** | 0.599 |
| generated documents, v16's / v18's | 0.714 / 0.591 | 0.883 / 0.834 | 0.851 / 0.858 | **0.914 / 0.903** | 0.851 / 0.810 |
| training sample, yes/no rows (gold) | 0.673 | 0.830 | 0.828 | **0.857** | |
| training sample, rating-scale rows (gold) | 0.407 | 0.498 | 0.525 | 0.532 | |

v12 replaced the frozen anchor with a teacher that read JevBench at 0.805 and lost eight
tasks: its edge was on long multi-step documents, absent from the prompts it labelled, and
it was no better than the student on the training rows. The 31B model is better on the
training rows' yes/no questions and on the held-out tasks where g3 is weakest (sarcasm
+0.19, HelpSteer2 +0.09), and no better on emotion or the rating scales. Its batched
labels match its own head (max difference 0.0000 on the parity rows); it is Apache-2.0
(commit 842da3794eaa0b77d5f08bae87a17459d91ff475).

## What is being changed

The KL target, from the frozen torso to the 31B teacher (`configs/experiments/g4.yaml`):

| | g3 | g4 |
| --- | --- | --- |
| KL toward the frozen torso | 0.3 | **0** |
| teacher (`teacher_weight` 1.0) on multi-step rows | v14's replay | v14's replay (unchanged) |
| teacher on short-task, generated and adequacy rows | none | **gemma-4-31B-it** |
| rating-scale (score) rows | gold | gold only (no teacher) |

`data/teacher.py` labels `train_v5`, both generated files and both adequacy files with
gemma-4-31B-it at the commit above; `scripts/merge_teacher_g4.py` drops score rows, adds
v14's replay rows and writes `data/teacher_g4.jsonl` in concatenated indices. Dry-run on
placeholder labels: 93,298 teacher rows of 123,339 (54,857 yes/no, 38,441 choice); the
21,000 score rows and about 9,000 rows with more than 16 options (no answer letter)
train on gold alone; no width mismatch. Everything else is g3's, including `force_bos`.

Labelling and training on one AWS g7e.2xlarge (RTX PRO 6000; torch 2.7.1+cu128),
automatic upgrades off, `save_every: 500`, `expandable_segments`. Labels go to S3 as
each file finishes. Evaluation as for g3 (fixed `multistep_eval`; the six sets re-run on
the local RTX 3090 alongside v19's and g3's), plus JF100 and Typed Decisions, exploratory.

## Baselines (fixed evaluation)

| evaluation | v19 | g3 |
| --- | --- | --- |
| JevBench (231) at 4096: tasks / standard / hard | 168 / 63 / 57 | 171 / 64 / 59 |
| JevBench Brier / ECE / paraphrase consistency | 0.342 / 0.051 / 0.861 | 0.344 / 0.057 / 0.944 |
| validation loss / accuracy | 0.421 / 0.843 | 0.424 / 0.838 |
| held-out short tasks: accuracy / ECE; sarcasm accuracy | 0.647 / 0.054; - | 0.650 / 0.077; 0.599 |
| MuSiQue / ContractNLI / BoardgameQA / HotpotQA | 0.879 / 0.862 / 0.810 / 0.726 | 0.908 / 0.848 / 0.806 / 0.734 |
| generated documents, v16's / v18's | 0.846 / 0.757 | 0.851 / 0.810 |
| adequacy: HelpSteer2 / generated (balanced) | 0.739 / 0.788 | 0.714 / 0.804 |
| JF100 (300 rotations) / Typed Decisions accuracy | 162 / 0.614 | 182 / 0.641 |

Noise: two runs of one recipe differ by about 4.5 JevBench tasks.

## Predictions

1. **JevBench at least 173** of 231 at the 4096 window (unchanged).
2. **The teacher transfers where it leads:** g4 is above g3 on HelpSteer2 adequacy (0.714),
   generated adequacy balanced (0.804) and held-out sarcasm accuracy (0.599).
3. **Nothing is lost against v19**, each within 0.02 (unchanged floors): MuSiQue at least
   0.859, ContractNLI at least 0.842, BoardgameQA at least 0.790, HotpotQA at least
   0.706; generated sets at least 0.826 and 0.737; adequacy at least 0.719 (HelpSteer2)
   and 0.768 (generated, balanced); held-out short-task accuracy at least 0.627.
4. **Calibration holds:** JevBench ECE at most 0.07 and held-out ECE at most 0.074.
5. **Serving stays practical:** JevBench p95 latency at most 600 ms (reported).

Exploratory, no numeric prediction: JF100 (g3 182, Jev 232) and Typed Decisions (g3
0.641) by domain and question type; JevBench's standard-tier yes/no tasks (`policy`,
`adequacy`), the probe's negation capture (g3 0.738, v19 0.935); validation, which
measures fit to gold and may fall as the targets move toward the teacher.

## Which model is recommended afterwards

g4 becomes hobson-gemma4's reference model, and the case for moving hobson's default to
Gemma, if predictions 1, 3 and 4 hold. If 2 holds and 1, 3 or 4 does not, distillation
helps where the teacher leads and the next step is its weight and coverage (score rows,
the multi-step rows, rows past 16 options). If 2 fails, v12's result repeats with a teacher
better on the training rows: distillation at weight 1.0 does not carry what the teacher
knows into this student.

## What would count as failure

- **Prediction 2 fails.**
- **JevBench below g3 by more than noise (under 167):** without the anchor, the student
  drifts on what the benchmark asks.

## What this test cannot show

- **Two changes in one.** The anchor is removed and the teacher added together; this
  cannot say how much each contributes.
- **One run** (about 4.5 tasks of noise between two runs).
- **What the 31B model's post-training saw.** HelpSteer2 is a widely used preference set;
  the teacher labels HelpSteer2 training rows, so a gain on its evaluation split could
  carry exposure through the teacher's labels. The generated adequacy set, written for
  this project, is the cleaner test of the same skill.

## Outcome (added after the run)

Nothing above this section was edited after labelling or training. **By the rule fixed
above, g4 does not become the reference model and hobson keeps v19:** prediction 1 held
by a wide margin, 4 and 5 held, but 3 failed on one floor (BoardgameQA) and 2 failed on one
of its three sets (generated adequacy, level with g3 rather than above). **g4 is
nonetheless the strongest model here on JevBench, and the first to beat v19 beyond noise.**

The run: one g7e.2xlarge, 2026-10-02, no restart. gemma-4-31B-it labelled 101,389 of
110,430 rows in 2 h (agreement with gold: train_v5 0.792 on the 91,408 rows it could
label, generated 0.908 / 0.896, adequacy as in step 0); the merge wrote 93,298 teacher
rows, exactly the dry run's count. Training took 3 h 13 m (no frozen forward). Labels and
the teacher file are kept with the results. The six sets were re-run with the fixed eval on
the local RTX 3090 alongside v19's and g3's; JevBench, calibration and held-out come from
the host.

| | prediction | v19 | g3 | g4 | |
| --- | --- | --- | --- | --- | --- |
| 1 | JevBench >= 173 | 168 | 171 | **183** | pass |
| 2 | above g3 on HelpSteer2, generated adequacy (balanced), sarcasm | | 0.714 / 0.804 / 0.599 | 0.722 / 0.801 / 0.632 | FAIL (generated adequacy level) |
| 3 | nothing lost against v19 (nine floors) | | one below | BoardgameQA 0.781 against 0.790 | FAIL |
| 4 | JevBench ECE <= 0.07; held-out ECE <= 0.074 | 0.051; 0.054 | 0.057; 0.077 | 0.048; 0.065 | pass |
| 5 | p95 latency <= 600 ms | 296 ms (3090) | 159 ms | 159 ms (RTX PRO 6000) | pass |

**JevBench: 183/231** (easy 48, standard **68**, hard **67** — both tiers the highest
recorded here). Against v19 +27 / -12 (**p = 0.02**): `adequacy`, `long_policy` and
`probability` +3, `intent` and `multi_hop` +2; `policy` and `ambiguous` -1. Against g3
+17 / -5 (p = 0.02), led by `adequacy` +3 and `judge_hard`, `tradeoff`, `multi_hop` +2.
Brier **0.292** (v19 0.342, g3 0.344), the best here; ECE 0.048; paraphrase consistency
0.944.

**Sets** (fixed eval, one machine; floor in brackets):

| set | v19 | g3 | g4 | vs v19 | vs g3 | floor |
| --- | --- | --- | --- | --- | --- | --- |
| MuSiQue [0.859] | 0.879 | 0.908 | 0.902 | +66 / -39, p = 0.01 | +21 / -29, p = 0.32 | pass |
| ContractNLI [0.842] | 0.862 | 0.848 | 0.861 | +35 / -36, p = 1.0 | +27 / -14, p = 0.06 | pass |
| BoardgameQA [0.790] | 0.810 | 0.806 | **0.781** | +51 / -77, p = 0.03 | +20 / -42, p = 0.007 | FAIL |
| HotpotQA, held out [0.706] | 0.726 | 0.734 | **0.759** | +114 / -82, p = 0.03 | +67 / -43, p = 0.03 | pass |
| generated, v16's [0.826] | 0.846 | 0.851 | 0.863 | +20 / -14, p = 0.39 | +14 / -10, p = 0.54 | pass |
| generated, v18's [0.737] | 0.757 | 0.810 | 0.794 | +27 / -18, p = 0.23 | +9 / -13, p = 0.52 | pass |
| adequacy, HelpSteer2 [0.719] | 0.739 | 0.714 | 0.722 | +21 / -25, p = 0.66 | +18 / -16, p = 0.86 | pass |
| adequacy, generated, balanced [0.768] | 0.788 | 0.804 | 0.801 | | | pass |
| held-out short tasks [0.627] | 0.647 | 0.650 | 0.655 | | | pass |

Held-out by task: sarcasm 0.632 (g3 0.599), its ECE 0.209 (g3 0.232); emotion 0.581,
hate severity 0.480, intent 0.886. Validation 0.427 / 0.810 (g3 0.424 / 0.838): accuracy
against gold fell, as expected with targets moved toward the teacher.

**Exploratory.** Probe: negation on trained yes/no tasks back to 0.897 (g3 0.738, v19
0.935); held-out choice 0.733 (v19 0.718). Paraphrase pairs both right 0.850 / 0.769;
flip pairs 0.450 (g3 0.583). Catch-all set close to g3's. **Typed Decisions** (zero-shot,
400 cases): accuracy 0.669 (g3 0.641, v19 0.614), KL 0.262, Brier 0.131 (g3 0.301 / 0.155),
yes/no 0.803 (g3 0.715). **JF100**: 172/300 (g3 182, v19 162, Jev 232) — down on
customer service (25 against 30), evidence integration (15 against 20) and temporal (12
against 14), up on relations (9 against 6) and code (17 against 14).

**Reading.** Replacing the frozen anchor with the 31B teacher produced the largest
JevBench gain in this series (+12 over g3, +15 over v19, both beyond noise), the best
Brier, and broad gains on Typed Decisions and held-out multi-hop. It did not transfer the
teacher's lead on the adequacy sets themselves: the teacher reads them at 0.803 / 0.920,
g4 at 0.722 / 0.801, barely above g3 — so prediction 2's specific mechanism is not what
moved. The one clear loss is BoardgameQA (0.806 -> 0.781), a multi-step set whose rows had
no 31B labels and, with the anchor gone, no frozen-torso pull either; and JF100 fell 10
of 300. The failure case "v12 repeats" named above does not fit: distillation clearly
carried something into the student, just not on the sets predicted. Under the rule g4 is
not the reference; the evidence says the next run should keep the 31B teacher and address
BoardgameQA — a small frozen anchor alongside the teacher, or 31B labels on the
multi-step rows — and a seed replicate of g4 would show how much of +12 is the recipe.

## Decision, 2026-10-02

**g4 is made hobson-gemma4's default, by the owner's decision overriding the rule above**,
as v14's, v16's, v17's and v18's promotions in hobson were. The rule's verdict stands as
recorded: g4 missed prediction 3 on BoardgameQA (0.781 against a 0.790 floor) and
prediction 2 on generated adequacy (level with g3). The grounds are the rest: JevBench
183 against v19's 168, beyond noise (p = 0.02), the best Brier, standard and hard tiers
recorded, every other floor held, and calibration within bounds. `configs/train.yaml` is
now g4's recipe (v19's is kept as `configs/train-v19.yaml`), `./recipe.sh` reproduces its
teacher file byte for byte from the committed 31B labels, and the README states the
BoardgameQA loss where g4 is recommended.
