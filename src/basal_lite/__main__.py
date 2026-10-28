"""basal-lite command line.

    basal-lite serve basal-omni-1.0-mini [--port 8000] [--gpu 0.85] [--think]     # vLLM in Docker
    basal-lite ask --model basal-omni-1.0-mini < request.json                      # one /v1/systemone request
    basal-lite flags basal-2.0-4.5B                                                # vllm serve flags for a model
    basal-lite stop
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import serve as srv
from .client import BasalLite, RequestError


def ask(a):
    md = srv.snapshot(a.model_dir or a.model)
    if md is None:
        raise SystemExit(f"{a.model}: not in the Hugging Face cache {srv.hub_dir()}; pass --model-dir")
    orders = int(a.orders) if a.orders in ("1", "2") else a.orders
    b = BasalLite(a.url, "basal", model_dir=md, orders=orders, think_url=a.think_url)
    raw = Path(a.file).read_bytes() if a.file else sys.stdin.buffer.read()
    try:
        out = asyncio.run(b.decide(json.loads(raw.decode("utf-8-sig"))))
    except RequestError as e:
        raise SystemExit(f"400: {e}") from None
    print(json.dumps(out, ensure_ascii=False, indent=1))


def flags(a):
    md = srv.snapshot(a.model)
    if md is None:
        raise SystemExit(f"{a.model}: not in the Hugging Face cache {srv.hub_dir()}")
    print(" ".join(srv.flags(md, a.template_dir)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="basal-lite", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask", help="one request (the body of basal-serve's POST /v1/systemone) from stdin or --file")
    p.add_argument("--url", default="http://localhost:8000", help="the vLLM server")
    p.add_argument("--model", default="basal-omni-1.0-mini", help="the served model (Remek/<model> in the HF cache)")
    p.add_argument("--model-dir", help="the model's directory instead (basal.json, CALIBRATION*.json)")
    p.add_argument("--orders", help='1, 2, "2le<N>" or "auto" (default: 1 for format 2 models, 2 for format 1)')
    p.add_argument("--think-url", help="the think server (basal-2.0-4.5B, serve --think)")
    p.add_argument("--file", "-f", help="the request JSON (default: stdin)")
    p.set_defaults(fn=ask)

    p = sub.add_parser("serve", help="start vLLM in Docker for a basal model")
    p.add_argument("model", nargs="?", default="basal-omni-1.0-mini", help="Remek/<model> or a model directory")
    p.add_argument("--port", type=int, default=8000, help="host port (the think server: port + 1)")
    p.add_argument("--gpu", type=float, default=0.85, help="vLLM --gpu-memory-utilization")
    p.add_argument("--max-len", type=int, default=8192, help="vLLM --max-model-len")
    p.add_argument("--think", action="store_true", help="also the think server on the model's think/ weights")
    p.add_argument("--think-gpu", type=float, default=0.46, help="--gpu-memory-utilization of the think server")
    p.add_argument("--name", default="basal-lite", help="container name")
    p.add_argument("--network", help="a Docker network to join (other containers: http://<name>:8000)")
    p.add_argument("--always", action="store_true", help="restart with Docker (--restart unless-stopped)")
    p.add_argument("--template-dir", help="where the Gemma 4 server template is written")
    p.add_argument("--image", default=srv.IMAGE)
    p.set_defaults(fn=srv.serve)

    p = sub.add_parser("flags", help="print the vllm serve flags for a model in the HF cache or a directory")
    p.add_argument("model")
    p.add_argument("--template-dir", help="where the Gemma 4 server template is written")
    p.set_defaults(fn=flags)

    p = sub.add_parser("stop", help="remove the server containers")
    p.add_argument("--name", default="basal-lite")
    p.set_defaults(fn=srv.stop)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
