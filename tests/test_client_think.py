"""Think mode (basal-serve --think auto): reasoning on a second server, then readout 2 over the state and the reasoning.

The fake think server reasons " The parcel is late. " (4 tokens) and its readout prefers the last option."""

import asyncio
import json

import httpx
import pytest
from fake_vllm import Server, client_for, detokenize, first_wins, last_wins

from basal_lite import BasalLite, RequestError
from basal_lite.client import temper

PREFILL = '{"answer": "'


def lite(server, model_dir=None, **kw):
    client = client_for(server)
    return BasalLite("http://fake", "basal", model_dir=model_dir, client=client, think_url="http://think/v1/", **kw)


def decide(body, server, **kw):
    return asyncio.run(lite(server, **kw).decide(body))


def completions(server):
    return server.sent("/v1/completions", host="think")


def test_uncertain_questions_think_and_keep_their_first_readout():
    s = Server()
    qs = {"n": {"type": "noul", "instructions": "Late?"}, "m": {"type": "multi", "criteria": ["a", "b"]}}
    out = decide({"state": "Order 12 is late.", "questions": qs, "think_text": True}, s)
    n, m = out["answers"]["n"], out["answers"]["m"]
    assert n["noul"] == pytest.approx(last_wins(2)[0])
    assert n["think"]["tokens"] == 4 and n["think"]["calibration"] == "missing" and n["think"]["ms"] >= 0
    assert n["think"]["text"] == " The parcel is late. "
    no_think = n["think"]["no_think"]
    assert list(no_think["probabilities"].values()) == pytest.approx(first_wins(2))
    assert no_think["confidence"] == pytest.approx(first_wins(2)[0])
    assert n["meta"]["think"] == {"used": True, "tokens": 4} and m["meta"]["think"] == {"used": False, "tokens": 0}
    assert out["usage"]["output_tokens"] == 4 and out["usage"]["think_questions"] == 1
    reason, read = completions(s)
    assert detokenize(reason["prompt"]).endswith("Options:\nA. Yes\nB. No<think>\n")  # the prefill dropped
    assert reason["max_tokens"] == 256 and reason["stop"] == ["</think>"]
    assert reason["model"] == "basal" and reason["temperature"] == 0.0 and reason["seed"] == 0
    assert detokenize(read["prompt"]).endswith("B. No<think>\nThe parcel is late.\n</think>\n" + PREFILL)
    assert read["max_tokens"] == 1 and read["logprobs"] == 2 and len(read["allowed_token_ids"]) == 2


def test_think_off_in_the_request_skips_the_think_server():
    s = Server()
    out = decide({"state": "x", "questions": {"n": {"type": "noul"}}, "think": "off"}, s)
    assert out["answers"]["n"]["meta"]["think"] == {"used": False, "tokens": 0}
    assert out["usage"]["think_questions"] == 0 and completions(s) == []


def test_think_mode_and_body_are_checked():
    with pytest.raises(RequestError, match='think must be "off", "auto" or "force"'):
        decide({"state": "x", "questions": {}, "think": "always"}, Server())
    with pytest.raises(RequestError, match="the request body must be a JSON object"):
        decide([], Server())


def test_confident_questions_think_only_when_forced(tmp_path):
    (tmp_path / "basal.json").write_text('{"prompt": 2}', encoding="utf-8")
    cal = {"thresholds": {"0.05": {"confidence": 0.5}}}
    (tmp_path / "CALIBRATION.json").write_text(json.dumps(cal), encoding="utf-8")
    (tmp_path / "CALIBRATION.think.json").write_text('{"temperature_per_prim": {"noul": 2.0}}', encoding="utf-8")
    body = {"state": "x", "questions": {"n": {"type": "noul"}}}
    s = Server()
    out = decide(body, s, model_dir=tmp_path)
    assert "think" not in out["answers"]["n"] and completions(s) == []
    out = decide(body | {"think": "force"}, s, model_dir=tmp_path)
    ans = out["answers"]["n"]
    assert ans["think"]["calibration"] == "CALIBRATION.think.json" and "text" not in ans["think"]
    assert list(ans["probabilities"].values()) == pytest.approx(temper(last_wins(2), 2.0))


def test_a_prompt_too_long_for_the_think_server_is_skipped():
    s = Server(think_len=300)
    out = decide({"state": "x" * 100, "questions": {"n": {"type": "noul"}}}, s)
    ans = out["answers"]["n"]
    assert ans["think"] == {"skipped": "context"} and ans["noul"] == pytest.approx(first_wins(2)[0])
    assert ans["meta"]["think"] == {"used": False, "tokens": 0} and completions(s) == []


def test_a_truncated_state_does_not_think():
    s = Server(max_model_len=400)
    out = decide({"state": "x" * 1000, "questions": {"n": {"type": "noul"}}}, s)
    assert out["usage"]["truncated"] and out["usage"]["think_questions"] == 0 and completions(s) == []


def test_think_reads_the_state_with_its_facts():
    s = Server()
    body = {"state": "Zamówione 03.02.2026, dziś 20.02.2026.", "facts": "auto", "questions": {"n": {"type": "noul"}}}
    decide(body, s)
    assert "Fakty pomocnicze" in detokenize(completions(s)[0]["prompt"])


def test_the_think_servers_context_is_asked_once():
    s = Server()
    b = lite(s)
    for _ in range(2):
        asyncio.run(b.decide({"state": "x", "questions": {"n": {"type": "noul"}}}))
    assert len(s.sent("/v1/models", host="think")) == 1 and len(completions(s)) == 4


def test_think_server_errors_and_foreign_chat_templates_are_runtime_errors():
    with pytest.raises(RuntimeError, match="think server: 500 server error"):
        decide({"state": "x", "questions": {"n": {"type": "noul"}}}, Server(fail={"/v1/completions": 500}))

    class Mangled(Server):
        def __call__(self, request):
            if request.url.path == "/detokenize":
                return httpx.Response(200, json={"prompt": "<s>something else"})
            return super().__call__(request)

    with pytest.raises(RuntimeError, match="the decoded prompt does not end in the answer prefill"):
        decide({"state": "x", "questions": {"n": {"type": "noul"}}}, Mangled())
