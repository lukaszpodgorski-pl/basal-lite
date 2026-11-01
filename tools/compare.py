"""basal-lite against basal-serve on the same requests, for any option orders: every request of basal's
data/format2_vectors.jsonl (whatever orders it was written for), data/tasks.jsonl and data/max_error.jsonl; with --set
media the image and audio cases of data/media/przypadki.jsonl (basal-omni / Gemma 4; "file" entries sent as base64 "data");
--set long: one request of every type with a state over the context (--context-overflow truncate); --set think: the
text requests with "think": "force" and the reasoning text (basal-serve --think auto, lite --think-url).

    python tools/compare.py ref  --url http://localhost:8001 --out results/ref_o2.jsonl      # basal-serve --orders 2
    python tools/compare.py lite --url http://localhost:8000 --model-dir <model dir> --orders 2 --out results/lite_o2.jsonl
    python tools/compare.py compare results/ref_o2.jsonl results/lite_o2.jsonl
    python tools/compare.py tempo --target lite --url http://localhost:8000 --model-dir <model dir>   # media throughput

compare: the decisions (every non-number field of every answer, recursively: choice, action, selected, tool, labels,
spans, max_error, calibration, ...) must be equal; numbers are measurements; max |d| over the probabilities and confidences.
"""

import argparse
import asyncio
import base64
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def requests(which="text"):
    """[(id, request as recorded, request as sent)]."""
    if which == "media":
        d = DATA / "media"
        out = []
        for it in map(json.loads, (d / "przypadki.jsonl").read_text(encoding="utf-8").splitlines()):
            req = {k: it[k] for k in it if k != "id"}
            sent = dict(
                req,
                media=[
                    {"type": m["type"], "data": base64.b64encode((d / m["file"]).read_bytes()).decode()}
                    for m in req["media"]
                ],
            )
            out.append((it["id"], req, sent))
        return out
    out = [
        (v["id"], v["request"])
        for v in map(json.loads, (DATA / "format2_vectors.jsonl").read_text(encoding="utf-8").splitlines())
    ]
    for f in ("tasks.jsonl", "max_error.jsonl"):
        for it in map(json.loads, (DATA / f).read_text(encoding="utf-8").splitlines()):
            out.append((it["id"], {k: it[k] for k in it if k != "id"}))
    if which == "think":  # think mode: every question that can think does (force), with its reasoning text
        return [(i, r, dict(r, think="force", think_text=True)) for i, r in out if isinstance(r["state"], str)]
    if which == "long":  # --context-overflow truncate: one request of every question type, its state over 60k chars
        seen, longs = set(), []
        for i, r in out:
            t = {q.get("type", "choice") for q in r["questions"].values()}
            if isinstance(r["state"], str) and not t <= seen and not r.get("media") and not r.get("early_exit"):
                seen |= t
                longs.append((i + "-long", dict(r, state=(r["state"] + "\n") * (60000 // (len(r["state"]) + 1) + 1))))
        out = longs
    return [(i, r, r) for i, r in out]


async def collect(a):
    import httpx

    if a.cmd == "lite":
        from basal_lite import BasalLite

        orders = int(a.orders) if a.orders in ("1", "2") else a.orders
        b = BasalLite(a.url, "basal", model_dir=a.model_dir, orders=orders, think_url=a.think_url)
    http = httpx.AsyncClient(timeout=300)
    rows = []
    for rid, req, sent in requests(a.set):
        rec = {"id": rid, "request": req}
        try:
            if a.cmd == "lite":
                rec.update(await b.decide(sent))
            else:
                r = await http.post(a.url.rstrip("/") + "/v1/basal", json=sent)
                rec.update(r.json())
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"requests": len(rows), "errors": sum("error" in r for r in rows)}))


async def tempo(a):
    """Decisions per second on the media cases, per kind (image only, audio only): each case sent until --n requests,
    --concurrency at a time, to basal-serve (ref: --url) or basal-lite (--target lite)."""
    import time

    import httpx
    from basal_lite import BasalLite

    b = BasalLite(a.url, "basal", model_dir=a.model_dir, orders=int(a.orders or 1)) if a.target == "lite" else None
    http = httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=256))
    sem = asyncio.Semaphore(a.concurrency)

    async def one(body):
        async with sem:
            if b:
                return (await b.decide(body))["usage"]["questions"]
            r = await http.post(a.url.rstrip("/") + "/v1/basal", json=body)
            return r.json()["usage"]["questions"] if r.status_code == 200 else None

    cases = [s for _, _, s in requests("media") if not any(q.get("evidence") for q in s["questions"].values())]
    for kind in ("image", "audio"):
        group = [s for s in cases if {m["type"] for m in s["media"]} == {kind}]
        bodies = [group[i % len(group)] for i in range(a.n)]
        await asyncio.gather(*(one(x) for x in bodies[: a.concurrency]))  # warm-up
        t0 = time.perf_counter()
        res = await asyncio.gather(*(one(x) for x in bodies), return_exceptions=True)
        wall = time.perf_counter() - t0
        ok = [r for r in res if isinstance(r, int)]
        print(
            json.dumps(
                {
                    "target": a.target,
                    "kind": kind,
                    "cases": len(group),
                    "requests": a.n,
                    "concurrency": a.concurrency,
                    "decisions_per_s": round(sum(ok) / wall, 2),
                    "requests_per_s": round(len(ok) / wall, 2),
                    "errors": len(res) - len(ok),
                }
            )
        )


def walk(x, y, path, diffs, worst):
    """diffs: paths whose decision differs; worst: [max |d|, path] over the numbers."""
    # floats are measurements; ints (span offsets, counts) are decisions
    if not (isinstance(x, float) or isinstance(y, float)) or not all(isinstance(v, (int, float)) for v in (x, y)):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in set(x) | set(y):
                walk(x.get(k), y.get(k), f"{path}/{k}", diffs, worst)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]", diffs, worst)
        elif x != y:
            diffs.append(
                f"{path}: {json.dumps(x, ensure_ascii=False)[:80]} vs {json.dumps(y, ensure_ascii=False)[:80]}"
            )
    elif ("probabilit" in path or "confidence" in path or path.endswith("/noul")) and abs(x - y) > worst[0]:
        worst[:] = [abs(x - y), path]


def compare(a):
    load = lambda p: {r["id"]: r for r in map(json.loads, Path(p).read_text(encoding="utf-8").splitlines())}  # noqa
    ref, new = load(a.ref), load(a.new)
    same, worst, errs, tok = 0, [0.0, None], [], []
    for rid in ref:
        r, n = ref[rid], new.get(rid, {"error": "missing"})
        if "error" in r or "error" in n:
            ok = "error" in r and "error" in n  # both refuse (e.g. evidence): the same behaviour
            errs.append(f"{rid}: ref {r.get('error', 'ok')} | basal-lite {n.get('error', 'ok')}") if not ok else None
            same += ok
            continue
        tok.append(r["usage"].get("input_tokens") == n["usage"].get("input_tokens"))  # the same prompts' length
        if not tok[-1]:
            print(f"input_tokens: {rid}: {r['usage'].get('input_tokens')} vs {n['usage'].get('input_tokens')}")
        diffs = []
        for k in ("truncated", "second_order_branches"):  # --context-overflow truncate, --orders auto
            walk(r["usage"].get(k), n["usage"].get(k), f"{rid}/usage/{k}", diffs, worst)
        for q in r["answers"]:
            walk(r["answers"][q], n["answers"].get(q), f"{rid}/{q}", diffs, worst)
        same += not diffs
        for d in diffs:
            print("differs:", d)
    for e in errs:
        print("error:", e)
    print(
        json.dumps(
            {
                "requests": len(ref),
                "same_decisions": f"{same}/{len(ref)}",
                "max_d": round(worst[0], 4),
                "max_d_at": worst[1],
                "same_input_tokens": f"{sum(tok)}/{len(tok)}",
            }
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ref", "lite", "compare", "tempo"])
    ap.add_argument("--target", choices=["ref", "lite"], default="ref", help="tempo: basal-serve or basal-lite")
    ap.add_argument("--n", type=int, default=128, help="tempo: requests per kind")
    ap.add_argument("--concurrency", type=int, default=16, help="tempo: requests in flight")
    ap.add_argument("ref", nargs="?")
    ap.add_argument("new", nargs="?")
    ap.add_argument("--url")
    ap.add_argument("--model-dir")
    ap.add_argument("--orders", default=None)
    ap.add_argument("--out")
    ap.add_argument("--set", choices=["text", "media", "long", "think"], default="text")
    ap.add_argument(
        "--think-url", dest="think_url", default=None, help="lite: the think server (vLLM on <model>/think)"
    )
    a = ap.parse_args()
    compare(a) if a.cmd == "compare" else asyncio.run(tempo(a) if a.cmd == "tempo" else collect(a))


if __name__ == "__main__":
    main()
