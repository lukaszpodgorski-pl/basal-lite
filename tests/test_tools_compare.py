"""tools/compare.py: the request sets, the decision comparison, and collect / tempo against mock servers (basal-serve's
/v1/basal and tests/fake_vllm.py); no GPU, no model.

    uv run pytest tests/test_tools_compare.py
"""

import asyncio
import base64
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import httpx
import pytest
from fake_vllm import handler as vllm_handler

import basal_lite

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def cmp():
    spec = importlib.util.spec_from_file_location("tools_compare", TOOLS / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def noul(q="Urgent?"):
    return {"q": {"type": "noul", "instructions": q}}


@pytest.fixture
def data(cmp, tmp_path, monkeypatch):
    """A small tools/data: vectors, tasks, max_error and media cases."""
    d = tmp_path / "data"
    (d / "media").mkdir(parents=True)
    jsonl(
        d / "format2_vectors.jsonl",
        [
            {"id": "v-noul", "request": {"state": "abc", "questions": noul()}},
            {"id": "v-noul-again", "request": {"state": "abc", "questions": noul("Late?")}},
            {
                "id": "v-json",
                "request": {"state": {"order": 1}, "questions": {"c": {"instructions": "?", "criteria": ["a", "b"]}}},
            },
            {
                "id": "v-choice",
                "request": {"state": "abcd", "questions": {"c": {"instructions": "?", "criteria": ["a", "b"]}}},
            },
        ],
    )
    jsonl(
        d / "tasks.jsonl", [{"id": "t-early", "state": "x", "early_exit": True, "questions": {"s": {"type": "score"}}}]
    )
    jsonl(d / "max_error.jsonl", [{"id": "me-media", "state": "x", "media": [1], "questions": {"a": {"type": "act"}}}])
    (d / "media" / "a.png").write_bytes(b"png-bytes")
    (d / "media" / "b.wav").write_bytes(b"wav-bytes")
    jsonl(
        d / "media" / "przypadki.jsonl",
        [
            {"id": "img", "state": "s", "media": [{"type": "image", "file": "a.png"}], "questions": noul()},
            {"id": "aud", "state": "s", "media": [{"type": "audio", "file": "b.wav"}], "questions": noul()},
            {
                "id": "img-evidence",
                "state": "s",
                "media": [{"type": "image", "file": "a.png"}],
                "questions": {"q": {"type": "noul", "instructions": "?", "evidence": True}},
            },
        ],
    )
    monkeypatch.setattr(cmp, "DATA", d)
    return d


def test_the_text_set_is_every_vector_task_and_max_error_request_of_tools_data(cmp):
    reqs = cmp.requests()
    assert len(reqs) == 43 + 5 + 5
    assert all(rec is sent for _, rec, sent in reqs)
    assert reqs[0][0] == "pl-noul" and "id" not in reqs[-1][1]


def test_the_media_set_sends_the_files_as_base64(cmp):
    reqs = cmp.requests("media")
    assert len(reqs) == 16
    rid, rec, sent = reqs[0]
    assert rid == "img-faktura-noul" and rec["media"] == [{"type": "image", "file": "faktura.png"}]
    data = (TOOLS / "data" / "media" / "faktura.png").read_bytes()
    assert sent["media"] == [{"type": "image", "data": base64.b64encode(data).decode()}]
    assert sent["questions"] == rec["questions"]


def test_the_think_set_forces_thinking_on_text_states_only(cmp, data):
    reqs = cmp.requests("think")
    assert [i for i, _, _ in reqs] == ["v-noul", "v-noul-again", "v-choice", "t-early", "me-media"]
    _, rec, sent = reqs[0]
    assert "think" not in rec and sent == dict(rec, think="force", think_text=True)


def test_the_long_set_takes_one_text_request_per_question_type_over_60k_chars(cmp, data):
    reqs = cmp.requests("long")
    assert [i for i, _, _ in reqs] == ["v-noul-long", "v-choice-long"]  # choice: the default type
    for _, rec, sent in reqs:
        assert rec is sent and len(rec["state"]) > 60000
    assert reqs[0][1]["state"].startswith("abc\nabc\n")


def test_walk_compares_decisions_exactly_and_measures_probabilities(cmp):
    diffs, worst = [], [0.0, None]
    x = {
        "choice": "a",
        "probabilities": {"a": 0.7, "b": 0.3},
        "confidence": 0.9,
        "spans": [[0, 3]],
        "labels": ["x"],
        "latency": 1.0,
        "noul": 0.5,
    }
    y = {
        "choice": "b",
        "probabilities": {"a": 0.6, "b": 0.4},
        "confidence": 0.95,
        "spans": [[0, 4]],
        "labels": ["x", "y"],
        "latency": 9.0,
        "noul": 0.51,
        "extra": True,
    }
    cmp.walk(x, y, "r", diffs, worst)
    assert sorted(diffs) == [
        'r/choice: "a" vs "b"',
        "r/extra: null vs true",
        'r/labels: ["x"] vs ["x", "y"]',
        "r/spans[0][1]: 3 vs 4",
    ]
    assert worst[0] == pytest.approx(0.1) and worst[1] in ("r/probabilities/a", "r/probabilities/b")


def test_walk_treats_a_float_against_a_non_number_as_a_decision(cmp):
    diffs, worst = [], [0.0, None]
    cmp.walk(0.5, None, "r/noul", diffs, worst)
    cmp.walk(1, 1.0, "r/count", diffs, worst)  # an int against a float: a measurement, not on a measured path
    assert diffs == ["r/noul: 0.5 vs null"] and worst == [0.0, None]


def answer(choice="a", p=0.7, tokens=10, **usage):
    return {"answers": {"q": {"choice": choice, "probabilities": {"a": p}}}, "usage": {"input_tokens": tokens, **usage}}


def test_compare_counts_the_same_decisions_and_reports_the_rest(cmp, tmp_path, capsys):
    ref, new = tmp_path / "ref.jsonl", tmp_path / "new.jsonl"
    jsonl(
        ref,
        [
            {"id": "same", **answer(p=0.7)},
            {"id": "other-choice", **answer(choice="a", truncated=True)},
            {"id": "longer-prompt", **answer(tokens=10)},
            {"id": "both-refuse", "error": "evidence"},
            {"id": "lite-refuses", **answer()},
            {"id": "missing", **answer()},
        ],
    )
    jsonl(
        new,
        [
            {"id": "same", **answer(p=0.75)},
            {"id": "other-choice", **answer(choice="b", truncated=False)},
            {"id": "longer-prompt", **answer(tokens=12)},
            {"id": "both-refuse", "error": "evidence"},
            {"id": "lite-refuses", "error": "RequestError: no"},
        ],
    )
    cmp.compare(Namespace(ref=str(ref), new=str(new)))
    *lines, last = capsys.readouterr().out.splitlines()
    assert lines == [
        "differs: other-choice/usage/truncated: true vs false",
        'differs: other-choice/q/choice: "a" vs "b"',
        "input_tokens: longer-prompt: 10 vs 12",
        "error: lite-refuses: ref ok | basal-lite RequestError: no",
        "error: missing: ref ok | basal-lite missing",
    ]
    assert json.loads(last) == {
        "requests": 6,
        "same_decisions": "3/6",
        "max_d": 0.05,
        "max_d_at": "same/q/probabilities/a",
        "same_input_tokens": "2/3",
    }


def basal_serve(request):
    body = json.loads(request.content)
    if request.url.path != "/v1/basal":
        return httpx.Response(404)
    if body["state"] == "boom":
        return httpx.Response(500, text="not json")
    return httpx.Response(200, json={"answers": {"q": {"noul": 0.5}}, "usage": {"questions": len(body["questions"])}})


@pytest.fixture
def servers(monkeypatch):
    """Every httpx.AsyncClient answers from basal_serve (/v1/basal) or fake_vllm (/tokenize, /v1/chat/completions)."""
    real = httpx.AsyncClient
    route = lambda r: basal_serve(r) if r.url.path == "/v1/basal" else vllm_handler(r)  # noqa: E731
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(route), **kw))


def collect_args(tmp_path, **kw):
    a = dict(cmd="ref", url="http://serve:8001/", set="text", out=str(tmp_path / "out.jsonl"))
    return Namespace(**(dict(model_dir=None, orders=None, think_url=None) | a | kw))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_collect_ref_records_basal_serves_answers_and_errors(cmp, data, servers, tmp_path, capsys):
    jsonl(data / "tasks.jsonl", [{"id": "t-boom", "state": "boom", "questions": noul()}])
    jsonl(data / "max_error.jsonl", [])
    a = collect_args(tmp_path)
    asyncio.run(cmp.collect(a))
    rows = read_rows(a.out)
    assert [r["id"] for r in rows] == ["v-noul", "v-noul-again", "v-json", "v-choice", "t-boom"]
    assert rows[0] == {
        "id": "v-noul",
        "request": {"state": "abc", "questions": noul()},
        "answers": {"q": {"noul": 0.5}},
        "usage": {"questions": 1},
    }
    assert rows[-1]["error"].startswith("JSONDecodeError")
    assert json.loads(capsys.readouterr().out) == {"requests": 5, "errors": 1}


@pytest.mark.parametrize("orders, want", [("2", 2), ("auto", "auto")])
def test_collect_lite_decides_with_basal_lite_on_the_vllm_server(cmp, data, servers, tmp_path, capsys, orders, want):
    made = []
    real = basal_lite.BasalLite

    def make(*args, **kw):
        made.append((args, kw))
        return real(*args, **kw)

    jsonl(data / "format2_vectors.jsonl", [{"id": "ok", "request": {"state": "abc", "questions": noul()}}])
    one_option = {"q": {"type": "choice", "instructions": "?", "criteria": {"only": "one"}}}
    jsonl(data / "tasks.jsonl", [{"id": "bad", "state": "abc", "questions": one_option}])
    jsonl(data / "max_error.jsonl", [])
    a = collect_args(tmp_path, cmd="lite", url="http://vllm:8000", orders=orders)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(basal_lite, "BasalLite", make)
        asyncio.run(cmp.collect(a))
    assert made == [(("http://vllm:8000", "basal"), dict(model_dir=None, orders=want, think_url=None))]
    ok, bad = read_rows(a.out)
    assert 0 < ok["answers"]["q"]["noul"] < 1 and ok["usage"]["questions"] == 1
    assert bad["error"].startswith("RequestError")
    assert json.loads(capsys.readouterr().out) == {"requests": 2, "errors": 1}


def tempo_args(**kw):
    return Namespace(
        **(dict(target="ref", url="http://serve:8001", model_dir=None, orders=None, n=6, concurrency=2) | kw)
    )


def test_tempo_ref_counts_decisions_per_kind_and_failed_requests(cmp, data, servers, capsys):
    jsonl(
        data / "media" / "przypadki.jsonl",
        [
            {"id": "img", "state": "s", "media": [{"type": "image", "file": "a.png"}], "questions": noul()},
            {"id": "aud", "state": "boom", "media": [{"type": "audio", "file": "b.wav"}], "questions": noul()},
        ],
    )
    asyncio.run(cmp.tempo(tempo_args()))
    image, audio = map(json.loads, capsys.readouterr().out.splitlines())
    for row, kind in ((image, "image"), (audio, "audio")):
        assert row["target"] == "ref" and row["kind"] == kind and row["cases"] == 1
        assert row["requests"] == 6 and row["concurrency"] == 2
    assert image["errors"] == 0 and image["decisions_per_s"] > 0 and image["requests_per_s"] > 0
    assert audio["errors"] == 6 and audio["decisions_per_s"] == 0


class FakeLite:
    """Answers images; audio passes the warm-up, then the server refuses it."""

    made = []

    def __init__(self, url, model, model_dir, orders):
        FakeLite.made.append((url, model, model_dir, orders))
        self.audio = 0

    async def decide(self, body):
        if body["media"][0]["type"] == "audio":
            self.audio += 1
            if self.audio > 2:
                raise RuntimeError("/v1/chat/completions: HTTP 400")
        return {"usage": {"questions": len(body["questions"])}}


@pytest.mark.parametrize("orders, want", [(None, 1), ("2", 2)])
def test_tempo_lite_sends_the_media_cases_without_evidence_to_basal_lite(cmp, data, monkeypatch, capsys, orders, want):
    FakeLite.made = []
    monkeypatch.setattr(basal_lite, "BasalLite", FakeLite)
    asyncio.run(cmp.tempo(tempo_args(target="lite", url="http://vllm:8000", model_dir="/m", orders=orders, n=3)))
    assert FakeLite.made == [("http://vllm:8000", "basal", "/m", want)]
    image, audio = map(json.loads, capsys.readouterr().out.splitlines())
    assert image["cases"] == 1 and image["errors"] == 0  # the evidence case is left out
    assert audio["errors"] == 3 and audio["requests_per_s"] == 0


def test_main_dispatches_to_compare_collect_and_tempo(cmp, monkeypatch, tmp_path):
    calls = []

    async def collect(a):
        calls.append(("collect", a.cmd, a.url, a.orders, a.out, a.set, a.think_url))

    async def tempo(a):
        calls.append(("tempo", a.target, a.n, a.concurrency))

    monkeypatch.setattr(cmp, "compare", lambda a: calls.append(("compare", a.ref, a.new)))
    monkeypatch.setattr(cmp, "collect", collect)
    monkeypatch.setattr(cmp, "tempo", tempo)
    for argv in (
        ["compare", "r.jsonl", "n.jsonl"],
        ["lite", "--url", "u", "--orders", "2", "--out", "o", "--set", "think", "--think-url", "t"],
        ["ref", "--url", "u", "--out", "o"],
        ["tempo", "--target", "lite", "--n", "4", "--concurrency", "1"],
    ):
        monkeypatch.setattr(sys, "argv", ["compare.py", *argv])
        cmp.main()
    assert calls == [
        ("compare", "r.jsonl", "n.jsonl"),
        ("collect", "lite", "u", "2", "o", "think", "t"),
        ("collect", "ref", "u", None, "o", "text", None),
        ("tempo", "lite", 4, 1),
    ]
