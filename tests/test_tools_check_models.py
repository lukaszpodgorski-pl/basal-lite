"""tools/check_models.py with fake huggingface_hub and vllm modules (neither is installed) and a fake check_prompt run.

uv run pytest tests/test_tools_check_models.py
"""

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from basal_lite.template import SPACE

TOOLS = Path(__file__).resolve().parents[1] / "tools"

# repo: (config.json, other small files, sibling sizes on the hub)
REPOS = {
    "Remek/basal-2.0-4.5B-FP8": (
        {
            "architectures": ["LlamaForCausalLM"],
            "quantization_config": {"quant_algo": "FP8"},
            "max_position_embeddings": 8192,
        },
        {"basal.json": {"prompt": 2}, "CALIBRATION.vllm.json": {"temperature": 1.2}},
        {
            "model.safetensors": 4_500_000_000,
            "think/model.safetensors": 4_000_000_000,
            "evidence_head.pt": None,
            "CALIBRATION.vllm.json": 300,
        },
    ),
    "Remek/basal-omni-1.0-mini": (
        {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "text_config": {"num_kv_shared_layers": 18, "max_position_embeddings": 32768},
            "vision_config": {"x": 1},
            "audio_config": {"x": 1},
        },
        {"basal.json": {"prompt": 2}, "chat_template.jinja": "<|turn>" + SPACE[0]},
        {"model-00001.safetensors": None, "config.json": 100},
    ),
    "Remek/basal-q-AWQ": ({"architectures": ["NewArch"], "quantization_config": {"quant_method": "awq"}}, {}, {}),
    "Remek/basal-gone": None,
    "Remek/basal-1.0-GGUF": None,
    "Remek/basal-1.0-MLX": None,
}


class FakeApi:
    def list_models(self, author, search):
        assert (author, search) == ("Remek", "basal")
        return [types.SimpleNamespace(id=r) for r in REPOS]

    def model_info(self, repo, files_metadata):
        assert files_metadata is True
        if REPOS[repo] is None:
            raise OSError(f"{repo}: 404 Not Found")
        return types.SimpleNamespace(
            siblings=[types.SimpleNamespace(rfilename=k, size=v) for k, v in REPOS[repo][2].items()]
        )


@pytest.fixture
def check_models(monkeypatch, tmp_path):
    downloads = []

    def snapshot_download(repo, cache_dir, allow_patterns):
        downloads.append((repo, cache_dir, allow_patterns))
        cfg, files, _ = REPOS[repo]
        md = Path(cache_dir) / repo.replace("/", "--")
        md.mkdir(parents=True)
        (md / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        for name, content in files.items():
            (md / name).write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return str(md)

    class Quant:
        def __init__(self, method):
            if method == "awq":
                raise ValueError("awq is not supported")

        def get_min_capability(self):
            return 89

    hub = types.SimpleNamespace(HfApi=FakeApi, snapshot_download=snapshot_download)
    registry = types.SimpleNamespace(get_supported_archs=lambda: ["LlamaForCausalLM", "Gemma4ForConditionalGeneration"])
    quantization = types.SimpleNamespace(get_quantization_config=Quant)
    for name, m in {
        "huggingface_hub": hub,
        "vllm": types.SimpleNamespace(ModelRegistry=registry),
        "vllm.model_executor": types.SimpleNamespace(),
        "vllm.model_executor.layers": types.SimpleNamespace(),
        "vllm.model_executor.layers.quantization": quantization,
    }.items():
        monkeypatch.setitem(sys.modules, name, m)

    runs = []

    def fake_run(cmd, capture_output, text):
        runs.append(cmd)
        if "omni" in cmd[-1]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="Traceback ...\nOSError: no tokenizer")
        return types.SimpleNamespace(returncode=0, stdout="loading\nprompts: 43/43 identical\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = importlib.util.spec_from_file_location("tools_check_models", TOOLS / "check_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, downloads, runs


def test_vllm_support_reads_the_architectures_and_the_quantization_capability(check_models):
    mod, _, _ = check_models
    archs, min_cap = mod.vllm_support()
    assert archs == {"LlamaForCausalLM", "Gemma4ForConditionalGeneration"}
    assert min_cap("modelopt") == 89
    assert min_cap("awq") == "? awq is not supported"


def test_main_reports_every_loadable_basal_repository(check_models, tmp_path, capsys):
    mod, downloads, runs = check_models
    out = tmp_path / "models.json"
    mod.main(str(tmp_path / "cache"), str(out))
    rows = {r["repo"]: r for r in json.loads(out.read_text(encoding="utf-8"))}
    assert list(rows) == sorted(r for r in REPOS if "GGUF" not in r and "MLX" not in r)
    assert [d[0] for d in downloads] == [r for r in rows if r != "Remek/basal-gone"]
    assert downloads[0][2] == ["*.json", "*.jinja", "tokenizer*", "*.model", "*.txt"]

    assert rows["Remek/basal-gone"] == {"repo": "Remek/basal-gone", "error": "OSError: Remek/basal-gone: 404 Not Found"}

    llama = rows["Remek/basal-2.0-4.5B-FP8"]
    assert llama == dict(
        repo="Remek/basal-2.0-4.5B-FP8",
        arch="LlamaForCausalLM",
        arch_in_vllm=True,
        quant="modelopt",
        quant_min_capability=89,
        weights_gb=8.5,
        format=2,
        layout="canonical",
        calibration=["CALIBRATION.vllm.json"],
        calibration_picked=True,
        calibration_unvalidated=None,
        evidence_head=True,
        media=[],
        kv_shared_layers=None,
        gemma_media_template=None,
        max_len=8192,
        check_prompt="prompts: 43/43 identical",
        check_prompt_ok=True,
    )

    omni = rows["Remek/basal-omni-1.0-mini"]
    assert omni["quant"] is None and omni["quant_min_capability"] is None and omni["weights_gb"] == 0
    assert omni["media"] == ["audio", "image"] and omni["kv_shared_layers"] == 18 and omni["max_len"] == 32768
    assert omni["gemma_media_template"] is True and omni["calibration_picked"] is False
    assert omni["check_prompt"] == "Traceback ...\nOSError: no tokenizer" and omni["check_prompt_ok"] is False

    awq = rows["Remek/basal-q-AWQ"]
    assert awq["arch_in_vllm"] is False and awq["quant"] == "awq"
    assert awq["quant_min_capability"] == "? awq is not supported" and awq["format"] == 1

    assert runs[0][:2] == [sys.executable, str(TOOLS / "check_prompt.py")]
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [p["repo"] for p in printed] == [r for r in rows if r != "Remek/basal-gone"]
