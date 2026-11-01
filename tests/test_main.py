"""The command line (basal_lite.__main__): ask against the fake vLLM server, flags, serve and stop dispatch.

No Docker and no network: serve.py's functions are replaced."""

import io
import json
import sys
from pathlib import Path

import pytest
from fake_vllm import Server, client_for, first_wins

import basal_lite.__main__ as cli
from basal_lite import BasalLite

BODY = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Urgent?"}}}


@pytest.fixture
def made(monkeypatch, tmp_path):
    """BasalLite as ask builds it, on the fake server; the model directory resolves to tmp_path."""
    (tmp_path / "basal.json").write_text('{"prompt": 2}', encoding="utf-8")
    seen, server = [], Server()

    def build(*args, **kw):
        seen.append((args, kw))
        return BasalLite(*args, client=client_for(server), **kw)

    monkeypatch.setattr(cli, "BasalLite", build)
    monkeypatch.setattr(
        cli.srv, "snapshot", lambda m: tmp_path if m in ("basal-omni-1.0-mini", str(tmp_path)) else None
    )
    return seen, server, tmp_path


def stdin(monkeypatch, data):
    monkeypatch.setattr(sys, "stdin", type("Stdin", (), {"buffer": io.BytesIO(data)})())


def test_ask_reads_stdin_with_a_bom_and_prints_the_answers(made, monkeypatch, capsys):
    seen, server, md = made
    stdin(monkeypatch, b"\xef\xbb\xbf" + json.dumps(BODY).encode())
    cli.main(["ask"])
    out = json.loads(capsys.readouterr().out)
    assert out["answers"]["q"]["noul"] == pytest.approx(first_wins(2)[0])
    args, kw = seen[0]
    assert args == ("http://localhost:8000", "basal")
    assert kw == {"model_dir": md, "orders": None, "think_url": None}


def test_ask_from_a_file_with_orders_url_and_think_server(made, tmp_path_factory, capsys):
    seen, server, md = made
    f = tmp_path_factory.mktemp("req") / "request.json"
    f.write_text(json.dumps(BODY, ensure_ascii=False), encoding="utf-8")
    cli.main(["ask", "--model-dir", str(md), "-f", str(f), "--orders", "2", "--url", "http://gpu:8000/v1"])
    assert json.loads(capsys.readouterr().out)["answers"]["q"]["noul"] == pytest.approx(0.5)
    cli.main(["ask", "--file", str(f), "--orders", "2le4", "--think-url", "http://gpu:8001"])
    assert seen[0][0][0] == "http://gpu:8000/v1" and seen[0][1]["orders"] == 2
    assert seen[1][1]["orders"] == "2le4" and seen[1][1]["think_url"] == "http://gpu:8001"
    assert len(server.sent("/v1/completions", host="gpu")) == 2


def test_ask_refusal_exits_with_the_400_message(made, monkeypatch):
    stdin(monkeypatch, json.dumps({"state": "x"}).encode())
    with pytest.raises(SystemExit, match="400: missing field 'questions'"):
        cli.main(["ask"])


def test_ask_without_the_model_points_at_the_hub_cache(made, monkeypatch):
    monkeypatch.setattr(cli.srv, "hub_dir", lambda: Path("/hub"))
    with pytest.raises(SystemExit, match=r"basal-x: not in the Hugging Face cache .*hub; pass --model-dir"):
        cli.main(["ask", "--model", "basal-x"])


def test_flags_prints_the_vllm_flags_of_a_cached_model(monkeypatch, capsys, tmp_path):
    calls = []
    monkeypatch.setattr(cli.srv, "snapshot", lambda m: tmp_path if m == "basal-2.0-4.5B" else None)
    monkeypatch.setattr(cli.srv, "flags", lambda md, td: calls.append((md, td)) or ["--max-logprobs", "255"])
    monkeypatch.setattr(cli.srv, "hub_dir", lambda: Path("/hub"))
    cli.main(["flags", "basal-2.0-4.5B", "--template-dir", "/t"])
    assert capsys.readouterr().out == "--max-logprobs 255\n"
    assert calls == [(tmp_path, "/t")]
    with pytest.raises(SystemExit, match=r"nope: not in the Hugging Face cache .*hub$"):
        cli.main(["flags", "nope"])


def test_serve_and_stop_get_their_parsed_arguments(monkeypatch):
    got = []
    monkeypatch.setattr(cli.srv, "serve", lambda a: got.append(("serve", a)))
    monkeypatch.setattr(cli.srv, "stop", lambda a: got.append(("stop", a)))
    cli.main(["serve", "basal-2.0-4.5B", "--port", "9000", "--gpu", "0.5", "--think", "--always"])
    a = got[0][1]
    assert (a.model, a.port, a.gpu, a.max_len, a.think, a.think_gpu) == ("basal-2.0-4.5B", 9000, 0.5, 8192, True, 0.46)
    assert (a.name, a.network, a.always, a.template_dir, a.image) == ("basal-lite", None, True, None, cli.srv.IMAGE)
    cli.main(["serve"])
    cli.main(["stop", "--name", "other"])
    assert [c for c, _ in got] == ["serve", "serve", "stop"]
    assert got[1][1].model == "basal-omni-1.0-mini" and got[1][1].port == 8000 and got[2][1].name == "other"


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2 and "required" in capsys.readouterr().err
