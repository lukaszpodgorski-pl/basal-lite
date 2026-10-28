"""A stock vLLM server for the client, in Docker: the flags every basal model needs, the Gemma 4 chat template, the model
from Hugging Face when it is not in the local cache, and optionally the think server (basal-2.0-4.5B's think/ weights).

    basal-lite serve basal-omni-1.0-mini                    # http://localhost:8000
    basal-lite serve basal-2.0-4.5B --think --gpu 0.46 --think-gpu 0.46
    basal-lite flags basal-2.0-4.5B                         # only print the vllm serve flags
    basal-lite stop
"""
import json
import os
import subprocess
import time
from pathlib import Path

import httpx

from .template import serve_template

IMAGE = "vllm/vllm-openai:v0.32.0"  # the version the compatibility results were measured on


def hub_dir():
    """The local Hugging Face hub cache (HF_HUB_CACHE, HF_HOME/hub, else ~/.cache/huggingface/hub)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def repo_id(model):
    return model if "/" in model else f"Remek/{model}"


def snapshot(model):
    """A model directory, or the snapshot of a Hugging Face repo (basal-2.0-4.5B = Remek/basal-2.0-4.5B) in the local
    cache with its weights, else None."""
    if Path(model).is_dir():
        return Path(model)
    snaps = hub_dir() / f"models--{repo_id(model).replace('/', '--')}" / "snapshots"
    return next((s for s in sorted(snaps.glob("*")) if any(s.glob("*.safetensors"))), None)


def download(model, think=False, image=IMAGE):
    """The model into the local cache, with huggingface_hub from the vLLM image (nothing to install on the host). Without
    think, the think/ weights (basal-2.0-4.5B: 9.5 GB more) are skipped."""
    hub = hub_dir()
    hub.mkdir(parents=True, exist_ok=True)
    ignore = "[]" if think else "['think/*']"
    run(["docker", "run", "--rm", "-v", f"{hub}:/hf", "-e", "HF_HUB_CACHE=/hf", "--entrypoint", "python3", image, "-c",
         f"from huggingface_hub import snapshot_download; snapshot_download('{repo_id(model)}', ignore_patterns={ignore})"])
    return snapshot(model)


def flags(md, template_dir=None, container_template_dir="/templates"):
    """vllm serve flags for the client and the model in `md`. Gemma 4: the server chat template (template.py) is written
    to template_dir, which the server sees as container_template_dir."""
    a = ["--max-logprobs", "255",                        # up to 255 options; the client reads the logprob of each one
         "--logprobs-mode", "processed_logprobs",        # log-probabilities after the restriction to the option codes
         "--generation-config", "vllm",                  # no default temperature / top-p from the model's files
         "--enable-prompt-tokens-details"]               # cached prompt tokens in the usage
    cfg = json.loads((Path(md) / "config.json").read_text(encoding="utf-8"))
    text = cfg.get("text_config") or cfg
    if text.get("num_kv_shared_layers"):
        a.append("--kv-sharing-fast-prefill")  # Gemma 4 E2B / E4B: 1.64x on an RTX 3090, the same answers
    tpl = serve_template(md)
    if tpl is None:  # Llama, Qwen3.5: the model's own template; string content (no list rendering)
        return a + ["--chat-template-content-format", "string"]
    name = Path(md).parent.parent.name.removeprefix("models--") if Path(md).parent.name == "snapshots" \
        else Path(md).name
    template_dir = Path(template_dir or Path.home() / ".cache" / "basal-lite" / "templates")
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / f"{name}.jinja").write_text(tpl, encoding="utf-8")
    # Gemma 4 Unified (basal-omni-1.0-max, 1.1-max): vLLM reads their media differently from the model's processor
    unified = "Gemma4UnifiedForConditionalGeneration" in (cfg.get("architectures") or ())
    image_n = 8 if cfg.get("vision_config") and not unified else 0
    audio_n = 8 if cfg.get("audio_config") and not unified else 0
    return a + ["--chat-template-content-format", "openai", "--chat-template", f"{container_template_dir}/{name}.jinja",
                f"--limit-mm-per-prompt.image={image_n}", f"--limit-mm-per-prompt.audio={audio_n}",
                "--limit-mm-per-prompt.video=0",
                "--trust-request-mm-kwargs"]  # only for "detail": "high"


def think_flags():
    """The think server (basal/think.py Sidecar): greedy reasoning and the second readout on the think/ weights."""
    return ["--dtype", "bfloat16", "--max-model-len", "8192", "--seed", "0", "--generation-config", "vllm",
            "--logprobs-mode", "processed_logprobs", "--max-logprobs", "255"]


def run(cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=not check, text=True)


def start(name, port, md_in_container, vllm_flags, *, gpu, max_len, hub, template_dir, network, always, image):
    run(["docker", "rm", "-f", name], check=False)
    cmd = ["docker", "run", "-d", "--name", name, "-p", f"{port}:8000", "--gpus", "all", "--ipc", "host",
           "--restart", "unless-stopped" if always else "no", "-e", "HF_HUB_OFFLINE=1",
           "-v", f"{hub}:/hf:ro", "-v", f"{template_dir}:/templates:ro"]
    if network:
        cmd += ["--network", network]
    cmd += ["--entrypoint", "vllm", image, "serve", md_in_container, "--served-model-name", "basal",
            "--host", "0.0.0.0", "--port", "8000", "--gpu-memory-utilization", str(gpu)]
    if max_len:
        cmd += ["--max-model-len", str(max_len)]
    subprocess.run(cmd + vllm_flags, check=True, stdout=subprocess.DEVNULL)  # (stdout: the container id)


def wait(name, port, minutes=20):
    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        time.sleep(5)
        if run(["docker", "inspect", "-f", "{{.State.Running}}", name], check=False).stdout.strip() != "true":
            print(run(["docker", "logs", "--tail", "30", name], check=False).stderr)
            raise SystemExit(f"{name}: the server stopped (log above)")
        try:
            if httpx.get(f"http://localhost:{port}/v1/models", timeout=5).status_code == 200:
                print(f"{name} ready after {time.time() - t0:.0f} s: http://localhost:{port}")
                return
        except httpx.HTTPError:
            pass
    raise SystemExit(f"{name}: not ready after {minutes} min")


def serve(a):
    md = snapshot(a.model)
    if md is None or (a.think and not (md / "think" / "config.json").exists()):
        print(f"downloading {repo_id(a.model)}")
        md = download(a.model, think=a.think, image=a.image)
    hub = hub_dir()
    try:
        inside = "/hf/" + md.resolve().relative_to(hub.resolve()).as_posix()
    except ValueError:
        raise SystemExit(f"{md}: serve takes models from the Hugging Face cache {hub} (HF_HUB_CACHE)") from None
    template_dir = Path(a.template_dir or Path.home() / ".cache" / "basal-lite" / "templates")
    vllm_flags = flags(md, template_dir)
    if a.network:
        run(["docker", "network", "create", a.network], check=False)
    common = dict(hub=hub, template_dir=template_dir, network=a.network, always=a.always, image=a.image)
    if a.think:  # first: vLLM's memory profile of the main server then sees the think server's memory as taken
        start(f"{a.name}-think", a.port + 1, inside + "/think", think_flags(), gpu=a.think_gpu, max_len=None, **common)
        wait(f"{a.name}-think", a.port + 1)
    start(a.name, a.port, inside, vllm_flags, gpu=a.gpu, max_len=a.max_len, **common)
    print(f"{a.model}: {' '.join(vllm_flags)}")
    wait(a.name, a.port)
    think = f" --think-url http://localhost:{a.port + 1}" if a.think else ""
    print(f"ask: basal-lite ask --url http://localhost:{a.port} --model {a.model}{think} < request.json")


def stop(a):
    for n in (a.name, f"{a.name}-think"):
        if run(["docker", "rm", "-f", n], check=False).stdout.strip():  # (rm -f also exits 0 when there is none)
            print(f"stopped: {n}")
