"""Score a served checkpoint on Typed Decisions (LocalLLaMA/typed-decisions), zero-shot.

Each `test` case (400, four workflows) is one `POST /v1/systemone` with the case's own
`state` and all five `questions`, as the dataset card scores every model. Gold is a soft
distribution (the mean of three teacher samples). Per decision (2,000):

- accuracy: the predicted top option is the gold's top option;
- KL from gold: sum_k g_k log(g_k / p_k);
- Brier: sum_k (p_k - g_k)^2;
- ECE: 10 equal-width bins of the top probability against accuracy.

KL and Brier reproduce the card's Uniform reference row (0.444, 0.238). ECE is defined
differently by different submitters (the card says so), so compare it only within this
script. `noul` answers become {true: p, false: 1 - p}; `score` levels are keyed "0".."n-1",
as the gold is.

    strands-decider serve <ckpt> --port 8100
    python evaluation/typed_decisions.py --endpoint http://127.0.0.1:8100 --out runs/td-<ckpt>

The test parquet is downloaded at a pinned revision. Rows are appended as they complete
and a rerun resumes. Needs pandas and pyarrow (the `train` extra's datasets brings both).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = "LocalLLaMA/typed-decisions"
REVISION = "main"


def _post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any, float]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read()), (time.time() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:500], (time.time() - t0) * 1000


def _probs(answer: dict[str, Any], gold: dict[str, float]) -> dict[str, float]:
    if answer.get("type") == "noul":
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    probs = answer["probabilities"]
    return {k: float(probs.get(k, 0.0)) for k in gold}


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[tuple[float, float, float, float, bool]]] = collections.defaultdict(list)
    for r in rows:
        for q, g in r["gold"].items():
            p = r["pred"].get(q)
            if p is None:
                continue
            top = max(p, key=lambda k: p[k])
            hit = top == max(g, key=lambda k: g[k])
            kl = sum(g[k] * math.log(g[k] / max(p.get(k, 0.0), 1e-12)) for k in g if g[k] > 0)
            brier = sum((p.get(k, 0.0) - g[k]) ** 2 for k in g)
            for key in ("all", r["type"][q], r["workflow"]):
                by[key].append((float(hit), kl, brier, p[top], hit))

    def summarize(xs: list[tuple[float, float, float, float, bool]]) -> dict[str, Any]:
        n = len(xs)
        ece = 0.0
        for b in range(10):
            sel = [x for x in xs if (b / 10 < x[3] <= (b + 1) / 10) or (b == 0 and x[3] == 0)]
            if sel:
                acc = sum(x[4] for x in sel) / len(sel)
                conf = sum(x[3] for x in sel) / len(sel)
                ece += len(sel) / n * abs(acc - conf)
        return {
            "decisions": n,
            "accuracy": round(sum(x[0] for x in xs) / n, 4),
            "kl_from_gold": round(sum(x[1] for x in xs) / n, 4),
            "brier": round(sum(x[2] for x in xs) / n, 4),
            "ece10": round(ece, 4),
        }

    return {k: summarize(v) for k, v in sorted(by.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", default="http://127.0.0.1:8100")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="strands-decider", help="label recorded in each row")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        REPO, "all/test-00000-of-00001.parquet", repo_type="dataset", revision=REVISION
    )
    cases = pd.read_parquet(path)
    if len(cases) != 400:
        raise SystemExit(f"expected 400 test cases, got {len(cases)}")

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "outcomes.jsonl"
    done = set()
    if target.exists():
        done = {json.loads(line)["id"] for line in target.read_text().splitlines()}
    url = args.endpoint.rstrip("/") + "/v1/systemone"
    for n, (_, case) in enumerate(cases.iterrows(), 1):
        if case["id"] in done:
            continue
        state = json.loads(case["state"])
        questions = json.loads(case["questions"])
        gold_raw = json.loads(case["gold"])
        gold = {q: v["probabilities"] for q, v in gold_raw.items()}
        payload = {"state": state, "questions": questions, "model": args.model}
        for attempt in range(2):
            status, body, ms = _post(url, payload, args.timeout)
            if status not in (0, 429, 500, 502, 503, 504):
                break
            if attempt == 0:
                time.sleep(2)
        if status != 200 or not isinstance(body, dict):
            raise SystemExit(f"{case['id']}: HTTP {status}: {body}")
        answers = body["answers"]
        row = {
            "id": case["id"],
            "workflow": case["workflow"],
            "type": {q: v["type"] for q, v in questions.items()},
            "gold": gold,
            "pred": {q: _probs(answers[q], gold[q]) for q in questions if q in answers},
            "elapsed_ms": round(ms, 1),
            "input_tokens": (body.get("usage") or {}).get("input_tokens"),
        }
        with target.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if n % 50 == 0:
            print(f"{n}/400", flush=True)

    rows = [json.loads(line) for line in target.read_text().splitlines()]
    summary = score(rows)
    latencies = sorted(r["elapsed_ms"] for r in rows)
    summary["latency_p50_ms"] = latencies[len(latencies) // 2]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    a = summary["all"]
    print(
        f"{args.model}: accuracy {a['accuracy']}  KL {a['kl_from_gold']}  Brier {a['brier']}"
        f"  ECE10 {a['ece10']}  ({a['decisions']} decisions, p50 {summary['latency_p50_ms']} ms)"
    )
    for k in ("noul", "choice", "score"):
        if k in summary:
            print(f"  {k:7s} accuracy {summary[k]['accuracy']}  KL {summary[k]['kl_from_gold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
