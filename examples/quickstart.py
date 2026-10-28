"""basal-lite from Python: start the server first (basal-lite serve basal-omni-1.0-mini), then

    python examples/quickstart.py
"""
import asyncio
import json
from pathlib import Path

from basal_lite import BasalLite
from basal_lite.serve import snapshot


async def main():
    # the model's directory gives the prompt format (basal.json) and the calibration (CALIBRATION*.json)
    b = BasalLite("http://localhost:8000", "basal", model_dir=snapshot("basal-omni-1.0-mini"))
    request = json.loads(Path(__file__).with_name("request_en.json").read_text(encoding="utf-8"))
    out = await b.decide(request)
    for name, ans in out["answers"].items():
        decision = ans.get("choice") or ans.get("selected") or ans.get("noul") or ans.get("score")
        print(f"{name:8} {decision!s:24} confidence {ans.get('confidence', ans.get('set_confidence')):.2f}")
    print(out["usage"])


asyncio.run(main())
