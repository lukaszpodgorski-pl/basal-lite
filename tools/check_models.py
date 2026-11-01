"""Which basal models the basal-lite client on a stock vLLM server can serve. For every Remek/basal-* repository vLLM can
load (safetensors: not -GGUF, -MLX), only its small files (config, tokenizer, chat template, basal.json, CALIBRATION*),
then: check_prompt (the same prompt tokens as basal), the architecture and quantization in this vLLM, the calibration the
client picks, media, fast prefill, evidence head and the weight size.

    uv run --with transformers --with huggingface_hub --with vllm tools/check_models.py /tmp/hf-cache results/models.json
    (or inside the vLLM image: pip install -e . transformers huggingface_hub; CPU is enough)
"""

import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

import basal_lite.client as bn
from basal_lite._basal.media import kinds_of
from basal_lite._basal.prompt import model_format, model_layout
from basal_lite.template import SPACE

GEMMA_SPACE = SPACE[0]  # the Gemma 4 template pattern template.py changes for media


def vllm_support():
    from vllm import ModelRegistry
    from vllm.model_executor.layers.quantization import get_quantization_config

    archs = set(ModelRegistry.get_supported_archs())

    def min_cap(method):
        try:
            return get_quantization_config(method).get_min_capability()
        except Exception as e:  # noqa: BLE001
            return f"? {e}"

    return archs, min_cap


def main(cache, out):
    api = HfApi()
    archs, min_cap = vllm_support()
    repos = sorted(
        m.id for m in api.list_models(author="Remek", search="basal") if not any(s in m.id for s in ("GGUF", "MLX"))
    )
    rows = []
    for repo in repos:
        r = {"repo": repo}
        rows.append(r)
        try:
            files = {s.rfilename: s.size or 0 for s in api.model_info(repo, files_metadata=True).siblings}
            md = Path(
                snapshot_download(
                    repo, cache_dir=cache, allow_patterns=["*.json", "*.jinja", "tokenizer*", "*.model", "*.txt"]
                )
            )
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {e}"[:200]
            continue
        cfg = json.loads((md / "config.json").read_text(encoding="utf-8"))
        text_cfg = cfg.get("text_config", cfg)
        qc = cfg.get("quantization_config") or {}
        quant = {"FP8": "modelopt", "NVFP4": "modelopt_fp4"}.get(qc.get("quant_algo"), qc.get("quant_method"))
        tpl = (md / "chat_template.jinja").read_text(encoding="utf-8") if (md / "chat_template.jinja").exists() else ""
        cal, why = bn.load_calibration(md, 1 if model_format(md) == 2 else 2, model_layout(md))
        r.update(
            arch=(cfg.get("architectures") or ["?"])[0],
            arch_in_vllm=(cfg.get("architectures") or ["?"])[0] in archs,
            quant=quant,
            quant_min_capability=min_cap(quant) if quant else None,
            weights_gb=round(sum(v for k, v in files.items() if k.endswith(".safetensors")) / 1e9, 1),
            format=model_format(md),
            layout=model_layout(md),
            calibration=sorted(k for k in files if k.startswith("CALIBRATION")),
            calibration_picked=bool(cal),
            calibration_unvalidated=why,
            evidence_head="evidence_head.pt" in files,
            media=sorted(kinds_of(cfg)),
            kv_shared_layers=text_cfg.get("num_kv_shared_layers"),
            gemma_media_template=GEMMA_SPACE in tpl if "gemma" in cfg.get("model_type", "") else None,
            max_len=text_cfg.get("max_position_embeddings"),
        )
        p = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("check_prompt.py")), str(md)], capture_output=True, text=True
        )
        r["check_prompt"] = (p.stdout.strip().splitlines() or [p.stderr.strip()[-300:]])[-1]
        r["check_prompt_ok"] = p.returncode == 0
        print(json.dumps(r, ensure_ascii=False), flush=True)
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:3])
