"""Refresh the vendored basal modules (src/basal_lite/_basal) from a basal checkout.

basal-lite reuses basal's pure-Python modules (prompt format, tasks, tools, label sets, facts, media decoding; no torch)
instead of re-implementing them. They are copied here verbatim, except one import: tools.py takes `certified` from
basal's server module (which imports torch); the client has its own copy of that function.

    git clone --branch v2.4.1 https://github.com/rkinas/basal /tmp/basal
    python tools/vendor.py /tmp/basal
    python tools/check_copies.py /tmp/basal/basal/server.py     # then the functions copied into client.py
"""

import sys
from pathlib import Path

MODULES = ("prompt", "tasks", "tools", "sets", "facts", "media")
OUT = Path(__file__).resolve().parents[1] / "src" / "basal_lite" / "_basal"


def main(basal_root):
    src_dir = Path(basal_root) / "basal"
    srcs = {}
    for m in MODULES:  # every module checked before any is written: a refusal leaves _basal as it was
        src = (src_dir / f"{m}.py").read_text(encoding="utf-8")
        src = src.replace(
            "from .server import NO_CERTIFICATE, certified", "from ..client import NO_CERTIFICATE, certified"
        )
        if ".server" in src:
            raise SystemExit(f"{m}.py imports basal's server module: check the copy by hand")
        srcs[m] = src
    for m, src in srcs.items():
        (OUT / f"{m}.py").write_text(src, encoding="utf-8")
    print(f"{len(MODULES)} modules -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
