"""Run JF100 (github.com/softpudding/jev-frontier-100) against a local `strands-decider serve`.

JF100's own Jev runner builds a `/v1/systemone` request per item and trial and posts
it to api.typesafe.ai. This script builds the same request with JF100's own
`jev_payload` (so option rotation and gold come from upstream code), posts it to a
local endpoint instead, writes rows in upstream's outcome schema, and scores them with
upstream's `aggregate.summarize`: accuracy, the paired-template bootstrap interval,
pair-joint accuracy and cross-rotation consistency. It then compares item by item with
upstream's published Jev 1.13.0 outcomes.

    git clone https://github.com/softpudding/jev-frontier-100 && git -C jev-frontier-100 checkout 9abacec
    strands-decider serve <ckpt> --port 8100
    python evaluation/jf100_run.py --jf100 jev-frontier-100 \
        --endpoint http://127.0.0.1:8100 --out runs/jf100-<ckpt>

Rows are appended as they complete and a rerun resumes, never re-asking a finished
(item, trial). Infrastructure failures are retried once, as upstream does; wrong
answers never are. Nothing here reads gold before the response is recorded.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path


def _load_upstream(root: Path):
    sys.path.insert(0, str(root / "src"))
    from jf100 import aggregate, core, runner  # imported after the path is set

    return aggregate, core, runner


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jf100", required=True, type=Path, help="checkout of jev-frontier-100")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8099")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="strands-decider", help="label recorded in each row")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    aggregate, core, runner = _load_upstream(args.jf100)
    items, manifest = core.load_items()  # verifies the frozen dataset checksum
    byid = {i["id"]: i for i in items}
    trials = len(core.SEEDS)

    health = runner.request(args.endpoint.rstrip("/") + "/health", timeout=5)
    if health["http_status"] != 200:
        raise SystemExit(f"no server at {args.endpoint}: {health}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_config.json").write_text(
        json.dumps(
            {
                "endpoint": args.endpoint,
                "model": args.model,
                "server_health": health["body"],
                "dataset_sha256": manifest["sha256"],
                "trials": trials,
                "seeds": core.SEEDS,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )

    target = args.out / "outcomes.jsonl"
    done = set()
    if target.exists():
        for line in target.read_text().splitlines():
            r = json.loads(line)
            done.add((r["item_id"], r["trial"]))
    url = args.endpoint.rstrip("/") + "/v1/systemone"
    n = 0
    for t in range(trials):
        order = list(items)
        random.Random(core.SEEDS[t]).shuffle(order)  # upstream's Jev order
        for item in order:
            if (item["id"], t) in done:
                continue
            payload, gold = runner.jev_payload(item, t)
            payload["model"] = args.model
            attempts = []
            for attempt in range(2):
                result = runner.request(url, payload, timeout=args.timeout)
                attempts.append({k: result[k] for k in ("http_status", "elapsed_ms")})
                if result["http_status"] not in (0, 429, 500, 502, 503, 504):
                    break
                if attempt == 0:
                    time.sleep(2)
            body = result["body"] if isinstance(result["body"], dict) else {}
            ans = body.get("answers", {}).get("answer", {})
            answer = ans.get("choice")
            status = (
                "service_error"
                if result["http_status"] != 200
                else "ok"
                if answer in list("ABCD")
                else "invalid_output"
            )
            row = {
                "system": args.model,
                "model": args.model,
                "budget": None,
                "item_id": item["id"],
                "pair_id": item["pair_id"],
                "domain": item["domain"],
                "difficulty": item["difficulty"],
                "trial": t,
                "seed": None,
                "request_sha256": core.digest(runner.jev_payload(item, t)[0]),
                "gold": gold,
                "answer": answer,
                "confidence": ans.get("confidence"),
                "probabilities": ans.get("probabilities"),
                "correct": status == "ok" and answer == gold,
                "status": status,
                "elapsed_ms": result["elapsed_ms"],
                "server_latency_ms": body.get("latency_ms"),
                "input_tokens": (body.get("usage") or {}).get("input_tokens"),
                "attempts": attempts,
            }
            with target.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"{len(done) + n}/{len(items) * trials}", flush=True)

    rows = [json.loads(line) for line in target.read_text().splitlines()]
    summary = aggregate.summarize(rows, items, trials)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    ci = summary.get("accuracy_95ci")
    print(
        f"\n{args.model}: {summary['correct']}/{summary['records']} = {summary['accuracy']:.3f}"
        f"  95% CI {ci}  status {summary['status_counts']}"
    )
    print(
        f"pair-joint {summary['pair_joint_accuracy']:.3f}  "
        f"rotation-consistent {summary['semantic_consistency']:.3f}  "
        f"median latency {summary['latency_median_ms']} ms"
    )

    # Paired comparison with upstream's published Jev 1.13.0 outcomes (same requests).
    ref_path = args.jf100 / "results/v0.2-budget/outcomes.jsonl"
    ref = [json.loads(line) for line in ref_path.read_text().splitlines()]
    systems = collections.defaultdict(dict)
    for r in ref:
        systems[f"{r['model']}/{r['budget']}" if r["budget"] is not None else r["model"]][
            (r["item_id"], r["trial"])
        ] = r["correct"]
    ours = {(r["item_id"], r["trial"]): r["correct"] for r in rows}
    print(f"\n{'system':24s} {'correct':>8s} {'ours only':>10s} {'theirs only':>12s}")
    for name, theirs in sorted(systems.items(), key=lambda kv: -sum(kv[1].values())):
        a = sum(ours[k] and not theirs[k] for k in ours)
        b = sum(theirs[k] and not ours[k] for k in ours)
        print(f"{name:24s} {sum(theirs.values()):>5d}/300 {a:>10d} {b:>12d}")
    print("\nby domain (ours vs jev):")
    jev = systems["jev"]
    for d in sorted({i["domain"] for i in items}):
        keys = [k for k in ours if byid[k[0]]["domain"] == d]
        print(
            f"  {d:22s} {sum(ours[k] for k in keys):>2d}/30   jev {sum(jev[k] for k in keys):>2d}/30"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
