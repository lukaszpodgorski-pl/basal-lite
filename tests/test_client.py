"""The client against a fake vLLM server (httpx.MockTransport): no GPU, no model.

The fake tokenizer gives every character of a chat one token and an option code after the answer prefill one more, so
the client's checks of the option codes pass. The fake readout always prefers the first shown option (logprob -0.5 per
position), which makes the effect of option orders and multi branches easy to predict.

    pip install -e .[dev] && pytest
"""
import asyncio
import json
import math
import sys

import httpx
import pytest

from basal_lite import BasalLite, RequestError
from basal_lite._basal.prompt import PREFILL

CODES = {}


def fake_tokens(messages):
    text = "".join(m["content"] for m in messages[:-1])
    last = messages[-1]["content"]
    head, code = last[: len(PREFILL)], last[len(PREFILL):]
    ids = [ord(c) for c in text + head]
    return ids + [CODES.setdefault(code, 100000 + len(CODES))] if code else ids


def handler(request):
    body = json.loads(request.content)
    if request.url.path == "/tokenize":
        return httpx.Response(200, json={"tokens": fake_tokens(body["messages"])})
    if request.url.path == "/v1/chat/completions":
        top = [{"token": f"token_id:{t}", "logprob": -0.5 * i} for i, t in enumerate(body["allowed_token_ids"])]
        n = len(fake_tokens(body["messages"]))
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}],
                                         "usage": {"prompt_tokens": n, "completion_tokens": 1}})
    return httpx.Response(404)


def decide(body, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return asyncio.run(BasalLite("http://fake", "basal", client=client, **kw).decide(body))


def first_wins(k):
    e = [math.exp(-0.5 * i) for i in range(k)]
    return [v / sum(e) for v in e]


def test_choice_reads_the_option_codes():
    out = decide({"state": "Customer: I cannot log in.", "questions": {"dept": {
        "type": "choice", "instructions": "Which team?", "criteria": {"cards": "Cards", "online": "Online", "loans": "Loans"}}}})
    ans = out["answers"]["dept"]
    assert ans["choice"] == "cards"
    assert list(ans["probabilities"].values()) == pytest.approx(first_wins(3))
    assert out["usage"]["questions"] == 1 and out["usage"]["branches"] == 1


def test_two_option_orders_average_out_a_position_bias():
    out = decide({"state": "x", "questions": {"q": {"type": "noul", "instructions": "Urgent?"}}}, orders=2)
    assert out["answers"]["q"]["noul"] == pytest.approx(0.5)  # "yes first" and "no first" both pick the first shown


def test_multi_asks_one_yes_no_branch_per_label():
    out = decide({"state": "x", "questions": {"m": {"type": "multi", "instructions": "Topics?",
                                                     "criteria": ["billing", "delivery"]}}})
    assert out["answers"]["m"]["selected"] == ["billing", "delivery"]  # p(yes) = 0.62 > 0.5 for each
    assert out["usage"]["branches"] == 2


def test_score_and_two_questions_in_one_request():
    out = decide({"state": "x", "questions": {
        "s": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "upset", "furious"]},
        "n": {"type": "noul", "instructions": "Urgent?"}}})
    p = first_wins(3)
    assert out["answers"]["s"]["score"] == pytest.approx(p[1] + 2 * p[2])
    assert out["answers"]["n"]["noul"] == pytest.approx(first_wins(2)[0])


def test_bad_requests_are_refused_as_basal_serve_does():
    with pytest.raises(RequestError):
        decide({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"only": "one"}}}})
    with pytest.raises(RequestError):
        decide({"questions": {}})


def test_no_torch():
    assert "torch" not in sys.modules
