"""serve.py: the vllm serve flags and the docker commands, on fake Hugging Face caches; no Docker, GPU or network.

uv run pytest tests/test_serve.py
"""

import json
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from basal_lite import serve
from basal_lite.template import SPACE

GEMMA = "<|turn>system\n" + SPACE[0] + "\n{{- '<|turn>' + role + '\\n' }}"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    h = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(h))
    monkeypatch.setattr(serve.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return h


def cached(hub, name, rev="abc123", config=None, template=None, weights=True, think=False):
    """models--Remek--<name>/snapshots/<rev> with config.json, weights and optionally a chat template, think/."""
    d = hub / f"models--Remek--{name}" / "snapshots" / rev
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config or {"architectures": ["LlamaForCausalLM"]}), encoding="utf-8")
    if weights:
        (d / "model.safetensors").write_bytes(b"")
    if template:
        (d / "chat_template.jinja").write_text(template, encoding="utf-8")
    if think:
        (d / "think").mkdir()
        (d / "think" / "config.json").write_text("{}", encoding="utf-8")
    return d


def test_hub_dir_follows_hf_hub_cache_then_hf_home_then_the_home_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "a"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "b"))
    assert serve.hub_dir() == tmp_path / "a"
    monkeypatch.delenv("HF_HUB_CACHE")
    assert serve.hub_dir() == tmp_path / "b" / "hub"
    monkeypatch.delenv("HF_HOME")
    monkeypatch.setattr(serve.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert serve.hub_dir() == tmp_path / "home" / ".cache" / "huggingface" / "hub"


def test_a_bare_model_name_is_a_remek_repo():
    assert serve.repo_id("basal-2.0-4.5B") == "Remek/basal-2.0-4.5B"
    assert serve.repo_id("someone/model") == "someone/model"


def test_snapshot_takes_a_model_directory_as_it_is(tmp_path):
    assert serve.snapshot(str(tmp_path)) == tmp_path


def test_snapshot_picks_the_first_cached_revision_with_weights(hub):
    cached(hub, "m", rev="a-no-weights", weights=False)
    d = cached(hub, "m", rev="b-full")
    assert serve.snapshot("m") == d


def test_snapshot_is_none_when_the_model_is_not_cached(hub):
    assert serve.snapshot("missing") is None
    cached(hub, "partial", weights=False)
    assert serve.snapshot("partial") is None


@pytest.mark.parametrize("think, ignore", [(False, "['think/*']"), (True, "[]")])
def test_download_runs_huggingface_hub_in_the_vllm_image(hub, monkeypatch, think, ignore):
    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        cached(hub, "m")

    monkeypatch.setattr(serve, "run", fake_run)
    md = serve.download("m", think=think, image="img:1")
    assert md == hub / "models--Remek--m" / "snapshots" / "abc123"
    (cmd,) = calls
    assert cmd[:4] == ["docker", "run", "--rm", "-v"] and cmd[4] == f"{hub}:/hf"
    assert cmd[cmd.index("--entrypoint") + 1 : cmd.index("-c")] == ["python3", "img:1"]
    assert (
        cmd[-1]
        == f"from huggingface_hub import snapshot_download; snapshot_download('Remek/m', ignore_patterns={ignore})"
    )


def test_flags_for_a_non_gemma_model_keep_its_own_template_with_string_content(hub, tmp_path):
    md = cached(hub, "llama")
    out = serve.flags(md, tmp_path / "tpl")
    assert out == [
        "--max-logprobs",
        "255",
        "--logprobs-mode",
        "processed_logprobs",
        "--generation-config",
        "vllm",
        "--enable-prompt-tokens-details",
        "--chat-template-content-format",
        "string",
    ]
    assert not (tmp_path / "tpl").exists()


def test_flags_turn_on_fast_prefill_for_kv_shared_layers(hub, tmp_path):
    md = cached(hub, "e4b", config={"text_config": {"num_kv_shared_layers": 18}})
    assert "--kv-sharing-fast-prefill" in serve.flags(md, tmp_path)
    md = cached(hub, "flat", config={"num_kv_shared_layers": 0, "model_type": "x"})
    assert "--kv-sharing-fast-prefill" not in serve.flags(md, tmp_path)


def test_flags_for_gemma_write_the_server_template_named_after_the_repo(hub, tmp_path):
    cfg = {"architectures": ["Gemma4ForConditionalGeneration"], "vision_config": {"x": 1}, "audio_config": {"x": 1}}
    md = cached(hub, "basal-omni-1.0-mini", config=cfg, template=GEMMA)
    out = serve.flags(md, tmp_path / "tpl", container_template_dir="/t")
    written = tmp_path / "tpl" / "Remek--basal-omni-1.0-mini.jinja"
    assert SPACE[1] in written.read_text(encoding="utf-8")
    assert out[out.index("--chat-template-content-format") :] == [
        "--chat-template-content-format",
        "openai",
        "--chat-template",
        "/t/Remek--basal-omni-1.0-mini.jinja",
        "--limit-mm-per-prompt.image=8",
        "--limit-mm-per-prompt.audio=8",
        "--limit-mm-per-prompt.video=0",
        "--trust-request-mm-kwargs",
    ]


def test_flags_for_gemma_unified_take_no_media_and_a_plain_directory_name(tmp_path, monkeypatch):
    monkeypatch.setattr(serve.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    md = tmp_path / "basal-omni-1.1-max"
    md.mkdir()
    cfg = {"architectures": ["Gemma4UnifiedForConditionalGeneration"], "vision_config": {"x": 1}}
    (md / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (md / "chat_template.jinja").write_text(GEMMA, encoding="utf-8")
    out = serve.flags(md)  # template_dir defaults to ~/.cache/basal-lite/templates
    assert (tmp_path / "home" / ".cache" / "basal-lite" / "templates" / "basal-omni-1.1-max.jinja").exists()
    assert "/templates/basal-omni-1.1-max.jinja" in out
    assert "--limit-mm-per-prompt.image=0" in out and "--limit-mm-per-prompt.audio=0" in out


def test_think_flags_are_greedy_with_the_readout_logprobs():
    out = serve.think_flags()
    assert out[out.index("--seed") + 1] == "0" and out[out.index("--max-model-len") + 1] == "8192"
    assert out[out.index("--logprobs-mode") + 1] == "processed_logprobs"


@pytest.mark.parametrize("check, capture", [(True, False), (False, True)])
def test_run_captures_the_output_only_when_it_does_not_check(monkeypatch, check, capture):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.append((cmd, kw)) or "done")
    assert serve.run(["docker", "ps"], check=check) == "done"
    assert seen == [(["docker", "ps"], {"check": check, "capture_output": capture, "text": True})]


def capture_start(monkeypatch, **kw):
    removed, started = [], []
    monkeypatch.setattr(serve, "run", lambda cmd, check=True: removed.append((cmd, check)))
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: started.append((cmd, k)))
    args = dict(gpu=0.9, max_len=None, hub=Path("/h"), template_dir=Path("/t"), network=None, always=False, image="i")
    serve.start("basal", 8000, "/hf/m", ["--x"], **(args | kw))
    assert removed == [(["docker", "rm", "-f", "basal"], False)]
    ((cmd, k),) = started
    assert k == {"check": True, "stdout": subprocess.DEVNULL}
    return cmd


def test_start_replaces_the_container_and_runs_vllm_serve(monkeypatch):
    cmd = capture_start(monkeypatch)
    assert cmd[:4] == ["docker", "run", "-d", "--name"]
    assert cmd[cmd.index("-p") + 1] == "8000:8000"
    assert cmd[cmd.index("--restart") + 1] == "no"
    assert f"{Path('/h')}:/hf:ro" in cmd and f"{Path('/t')}:/templates:ro" in cmd
    assert "--network" not in cmd and "--max-model-len" not in cmd
    assert cmd[cmd.index("--entrypoint") :] == [
        "--entrypoint",
        "vllm",
        "i",
        "serve",
        "/hf/m",
        "--served-model-name",
        "basal",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--gpu-memory-utilization",
        "0.9",
        "--x",
    ]


def test_start_with_a_network_a_length_limit_and_always_restart(monkeypatch):
    cmd = capture_start(monkeypatch, network="net", max_len=4096, always=True)
    assert cmd[cmd.index("--network") + 1] == "net"
    assert cmd[cmd.index("--max-model-len") + 1] == "4096"
    assert cmd[cmd.index("--restart") + 1] == "unless-stopped"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, s):
        self.now += s


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(serve.time, "time", c.time)
    monkeypatch.setattr(serve.time, "sleep", c.sleep)
    return c


def docker(running="true", logs="boom"):
    def fake_run(cmd, check=True):
        if cmd[1] == "inspect":
            return SimpleNamespace(stdout=running + "\n")
        return SimpleNamespace(stderr=logs)

    return fake_run


def test_wait_returns_once_the_server_answers(monkeypatch, clock, capsys):
    answers = iter([httpx.ConnectError("refused"), SimpleNamespace(status_code=503), SimpleNamespace(status_code=200)])
    urls = []

    def get(url, timeout):
        urls.append(url)
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(serve, "run", docker())
    monkeypatch.setattr(serve.httpx, "get", get)
    serve.wait("basal", 8000)
    assert urls == ["http://localhost:8000/v1/models"] * 3
    assert "basal ready after 15 s: http://localhost:8000" in capsys.readouterr().out


def test_wait_stops_with_the_log_when_the_container_dies(monkeypatch, clock, capsys):
    monkeypatch.setattr(serve, "run", docker(running="false", logs="CUDA out of memory"))
    with pytest.raises(SystemExit, match="basal: the server stopped"):
        serve.wait("basal", 8000)
    assert "CUDA out of memory" in capsys.readouterr().out


def test_wait_gives_up_after_the_time_limit(monkeypatch, clock):
    monkeypatch.setattr(serve, "run", docker())
    monkeypatch.setattr(serve.httpx, "get", lambda url, timeout: SimpleNamespace(status_code=503))
    with pytest.raises(SystemExit, match="not ready after 1 min"):
        serve.wait("basal", 8000, minutes=1)
    assert clock.now == 1060.0


def serve_args(**kw):
    a = dict(
        model="m",
        think=False,
        image="img",
        template_dir=None,
        network=None,
        name="basal",
        port=8000,
        gpu=0.9,
        think_gpu=0.4,
        max_len=None,
        always=False,
    )
    return Namespace(**(a | kw))


@pytest.fixture
def docker_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(serve, "run", lambda cmd, check=True: calls.append(("run", cmd)))
    monkeypatch.setattr(serve, "start", lambda *a, **kw: calls.append(("start", a, kw)))
    monkeypatch.setattr(serve, "wait", lambda name, port: calls.append(("wait", name, port)))
    return calls


def test_serve_starts_a_cached_model_from_its_path_inside_the_container(hub, tmp_path, docker_calls, capsys):
    cached(hub, "m")
    serve.serve(serve_args())
    (start, wait) = docker_calls
    assert start[1][:3] == ("basal", 8000, "/hf/models--Remek--m/snapshots/abc123")
    assert start[1][3][-1] == "string"
    assert start[2] == dict(
        gpu=0.9,
        max_len=None,
        hub=hub,
        template_dir=tmp_path / "home" / ".cache" / "basal-lite" / "templates",
        network=None,
        always=False,
        image="img",
    )
    assert wait == ("wait", "basal", 8000)
    out = capsys.readouterr().out
    assert "downloading" not in out
    assert "ask: basal-lite ask --url http://localhost:8000 --model m < request.json" in out


def test_serve_downloads_a_model_that_is_not_cached(hub, monkeypatch, docker_calls, capsys):
    monkeypatch.setattr(serve, "download", lambda model, think, image: cached(hub, model))
    serve.serve(serve_args(template_dir=str(hub.parent / "tpl")))
    assert "downloading Remek/m" in capsys.readouterr().out
    assert docker_calls[0][2]["template_dir"] == hub.parent / "tpl"


def test_serve_with_think_starts_the_think_server_first_on_the_next_port(hub, monkeypatch, docker_calls, capsys):
    cached(hub, "m")  # without think/: downloaded again with it
    got = []
    monkeypatch.setattr(
        serve, "download", lambda model, think, image: got.append(think) or cached(hub, "m", "b", think=True)
    )
    serve.serve(serve_args(think=True, network="basal-net", always=True))
    assert got == [True]
    net, think_start, think_wait, start, wait = docker_calls
    assert net == ("run", ["docker", "network", "create", "basal-net"])
    assert think_start[1] == ("basal-think", 8001, "/hf/models--Remek--m/snapshots/b/think", serve.think_flags())
    assert (
        think_start[2]["gpu"] == 0.4 and think_start[2]["max_len"] is None and think_start[2]["network"] == "basal-net"
    )
    assert think_wait == ("wait", "basal-think", 8001)
    assert start[1][:3] == ("basal", 8000, "/hf/models--Remek--m/snapshots/b") and start[2]["always"] is True
    assert wait == ("wait", "basal", 8000)
    assert "--think-url http://localhost:8001 < request.json" in capsys.readouterr().out


def test_serve_with_think_skips_the_download_when_think_weights_are_cached(hub, monkeypatch, docker_calls):
    cached(hub, "m", think=True)
    monkeypatch.setattr(serve, "download", lambda *a, **kw: pytest.fail("downloaded"))
    serve.serve(serve_args(think=True))
    assert [c[0] for c in docker_calls] == ["start", "wait", "start", "wait"]


def test_serve_refuses_a_model_directory_outside_the_hub_cache(hub, tmp_path, docker_calls):
    md = tmp_path / "elsewhere"
    md.mkdir()
    with pytest.raises(SystemExit, match="serve takes models from the Hugging Face cache"):
        serve.serve(serve_args(model=str(md)))
    assert docker_calls == []


def test_stop_removes_both_containers_and_reports_the_ones_that_ran(monkeypatch, capsys):
    calls = []

    def fake_run(cmd, check=True):
        calls.append((cmd, check))
        return SimpleNamespace(stdout="basal\n" if cmd[-1] == "basal" else "")

    monkeypatch.setattr(serve, "run", fake_run)
    serve.stop(Namespace(name="basal"))
    assert calls == [(["docker", "rm", "-f", "basal"], False), (["docker", "rm", "-f", "basal-think"], False)]
    assert capsys.readouterr().out == "stopped: basal\n"
