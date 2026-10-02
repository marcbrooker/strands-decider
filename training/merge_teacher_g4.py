"""Build g4's teacher file: gemma-4-31B-it's distributions on every train file except the
multi-step rows, which keep v14's replay distributions; rating-scale (score) rows get none,
so they train on the gold label alone.

The committed train_v5 labels omit the 21,000 score rows, as the committed Qwen3.5-4B file
does (data/README.md); they would be dropped here anyway, so the output is unchanged.

Inputs are teacher.py outputs with per-file row indices; the output uses indices into the
concatenation of g4's `train_files`, in that order, which is what TrainConfig.teacher_file
expects. Every row's distribution width is checked against the row's option count.

    python training/merge_teacher_g4.py --labels-dir data/teacher31b --out data/teacher_g4.jsonl
"""

import argparse
import json
from collections import Counter

from strands_decider.data.format import read_jsonl

TRAIN_FILES = [
    "data/train_v5.jsonl",
    "data/multistep_v14.jsonl",
    "data/generated_v16.jsonl",
    "data/generated_v18.jsonl",
    "data/adequacy_hs2.jsonl",
    "data/adequacy_gen.jsonl",
]
REPLAY = "data/replay_v14_multistep.jsonl"  # already in concatenated indices


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--labels-dir", required=True, help="teacher.py outputs, <stem>.jsonl per train file"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out, counts, offset = [], Counter(), 0
    for path in TRAIN_FILES:
        rows = list(read_jsonl(path))
        stem = path.split("/")[-1][: -len(".jsonl")]
        if stem == "multistep_v14":
            for line in open(REPLAY, encoding="utf-8"):
                d = json.loads(line)
                local = d["i"] - offset
                assert 0 <= local < len(rows) and len(d["probs"]) == rows[local].n_options
                out.append(d)
                counts["multistep (v14 replay)"] += 1
        else:
            for line in open(f"{args.labels_dir}/{stem}.jsonl", encoding="utf-8"):
                d = json.loads(line)
                ex = rows[d["i"]]
                assert len(d["probs"]) == ex.n_options, (path, d["i"])
                if ex.kind == "score":
                    counts[f"{stem} score (gold only)"] += 1
                    continue
                out.append({"i": d["i"] + offset, "probs": d["probs"]})
                counts[f"{stem} {ex.kind}"] += 1
        offset += len(rows)
    out.sort(key=lambda d: d["i"])
    with open(args.out, "w", encoding="utf-8") as fh:
        for d in out:
            fh.write(json.dumps(d) + "\n")
    print(f"{len(out):,} teacher rows of {offset:,} training rows -> {args.out}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<36} {v:,}")


if __name__ == "__main__":
    main()
