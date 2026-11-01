"""A fake vLLM server (httpx.MockTransport) for the tests: no GPU, no model.

The fake tokenizer gives every character of a chat one token and an option code after the answer prefill one more, so
the client's checks of the option codes pass. The fake readout always prefers the first shown option (logprob -0.5 per
position), which makes the effect of option orders and multi branches easy to predict.
"""

import asyncio
import json
import math

import httpx

from basal_lite import BasalLite
from basal_lite._basal.prompt import PREFILL

CODES = {}


def text_of(content):
    """A turn's content as text: a string, or the text parts of a list (media chats)."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "\x00") for p in content)


def fake_tokens(messages):
    text = "".join(text_of(m["content"]) for m in messages[:-1])
    last = messages[-1]["content"]
    head, code = last[: len(PREFILL)], last[len(PREFILL) :]
    ids = [ord(c) for c in text + head]
    return ids + [CODES.setdefault(code, 100000 + len(CODES))] if code else ids


def readout(body):
    top = [{"token": f"token_id:{t}", "logprob": -0.5 * i} for i, t in enumerate(body["allowed_token_ids"])]
    n = len(fake_tokens(body["messages"]))
    return {
        "choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}],
        "usage": {"prompt_tokens": n, "completion_tokens": 1},
    }


def handler(request):
    body = json.loads(request.content)
    if request.url.path == "/tokenize":
        return httpx.Response(200, json={"tokens": fake_tokens(body["messages"])})
    if request.url.path == "/v1/chat/completions":
        return httpx.Response(200, json=readout(body))
    return httpx.Response(404)


def client_for(fn=handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(fn))


def decide(body, handler_fn=handler, **kw):
    return asyncio.run(BasalLite("http://fake", "basal", client=client_for(handler_fn), **kw).decide(body))


def first_wins(k):
    e = [math.exp(-0.5 * i) for i in range(k)]
    return [v / sum(e) for v in e]


def detokenize(tokens):
    return "".join(chr(t) for t in tokens)


class Server:
    """A configurable fake vLLM server, and its think server (any host but "fake"), that records every request.

    /tokenize also takes {"prompt": text} (one token per character) and answers max_model_len when it is given;
    /detokenize inverts it. usage: fields added to the usage of every readout. fail: {path: HTTP status}. The think
    server: /v1/models with think_len, /v1/completions reasoning `reasoning` (completion_tokens: its word count) and a
    readout that prefers the last option (logprob -0.5 per position from the end)."""

    def __init__(self, max_model_len=None, usage=None, fail=None, think_len=4096, reasoning=" The parcel is late. "):
        self.max_model_len, self.usage, self.fail = max_model_len, usage or {}, fail or {}
        self.think_len, self.reasoning = think_len, reasoning
        self.requests = []

    def sent(self, path, host="fake"):
        return [b for h, p, b in self.requests if p == path and h == host]

    def __call__(self, request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.host, path, body))
        if path in self.fail:
            return httpx.Response(self.fail[path], text="server error")
        if path == "/tokenize":
            tokens = fake_tokens(body["messages"]) if "messages" in body else [ord(c) for c in body["prompt"]]
            out = {"tokens": tokens}
            if self.max_model_len:
                out["max_model_len"] = self.max_model_len
            return httpx.Response(200, json=out)
        if path == "/detokenize":
            return httpx.Response(200, json={"prompt": detokenize(body["tokens"])})
        if path == "/v1/chat/completions":
            out = readout(body)
            out["usage"].update(json.loads(json.dumps(self.usage)))
            return httpx.Response(200, json=out)
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "basal", "max_model_len": self.think_len}]})
        if path == "/v1/completions":
            if body["max_tokens"] == 1:
                ids = body["allowed_token_ids"]
                top = {f"token_id:{t}": -0.5 * (len(ids) - 1 - i) for i, t in enumerate(ids)}
                return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": [top]}}], "usage": {}})
            usage = {"completion_tokens": len(self.reasoning.split())}
            return httpx.Response(200, json={"choices": [{"text": self.reasoning}], "usage": usage})
        return httpx.Response(404)


def last_wins(k):
    return first_wins(k)[::-1]
