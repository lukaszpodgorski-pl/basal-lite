"""--context-overflow on a stock vLLM server: the limit from /tokenize's max_model_len, the state cut on the server's
tokenizer (/tokenize with a prompt, /detokenize), or the refusal."""

import httpx
import pytest
from fake_vllm import Server, decide, fake_tokens, first_wins

from basal_lite import RequestError

MARK = "\n[…]\n"
LIMIT = 399  # max_model_len 400 minus the generated token


def noul(state, question="Pilne?"):
    return {"state": state, "questions": {"q": {"type": "noul", "instructions": question}}}


def prompt_lengths(server):
    return [len(fake_tokens(b["messages"])) for b in server.sent("/v1/chat/completions")]


def test_short_prompts_need_no_extra_tokenize():
    s = Server(max_model_len=400)
    out = decide(noul("x"), s)
    assert "truncated" not in out["usage"]
    assert len(s.sent("/tokenize")) == 1 + 255 + 1  # the option codes only


def test_long_state_loses_its_middle_and_keeps_the_language_of_the_whole():
    s = Server(max_model_len=400)
    state = "A" * 600 + "ą" + "B" * 600  # Polish only in the middle
    q = {"instructions": "Priorytet?", "criteria": {"a": "Pilne", "b": "Zwykłe"}, "option_keys": "hide"}
    out = decide({"state": state, "questions": {"q": q}}, s, orders=2)
    t = out["usage"]["truncated"]
    assert t["state_tokens"] == 1201 and t["head_share"] == pytest.approx(1 / 3)
    assert t["head_tokens"] == round((t["head_tokens"] + t["tail_tokens"]) / 3)
    assert t["dropped_tokens"] == 1201 - t["head_tokens"] - t["tail_tokens"]
    chats = s.sent("/v1/chat/completions")
    assert len(chats) == 2 and max(prompt_lengths(s)) <= LIMIT
    sent_state = "A" * t["head_tokens"] + MARK + "B" * t["tail_tokens"]
    assert (
        chats[0]["messages"][1]["content"] == f"Stan:\n{sent_state}\n\nPytanie: Priorytet?\nOpcje:\nA. Pilne\nB. Zwykłe"
    )
    assert chats[1]["messages"][1]["content"].endswith("A. Zwykłe\nB. Pilne")
    assert chats[0]["messages"][0]["content"].startswith("Oceniasz")
    assert out["answers"]["q"]["probabilities"] == pytest.approx({"a": 0.5, "b": 0.5})


def test_truncate_head_share_one_keeps_the_start_and_zero_the_end():
    for share, start, end in ((1.0, "A", MARK), (0.0, MARK, "B")):
        s = Server(max_model_len=400)
        out = decide(noul("A" * 500 + "B" * 500), s, truncate_head=share)
        user = s.sent("/v1/chat/completions")[0]["messages"][1]["content"]
        state = user.split("\n\nQuestion:")[0].removeprefix("State:\n")
        assert state.startswith(start) and state.endswith(end) and len(state) > 200
        assert out["usage"]["truncated"]["head_share"] == share


def test_overflow_refuse_names_the_question_and_the_limit():
    msg = r"question 'q': the prompt is \d+ tokens .* more than the 399 tokens of vLLM's --max-model-len 400"
    with pytest.raises(RequestError, match=msg):
        decide(noul("x" * 1000), Server(max_model_len=400), overflow="refuse")


def test_a_state_too_short_to_cut_refuses_the_longest_question():
    qs = {"short": {"type": "noul"}, "long": {"type": "noul", "instructions": "Q" * 500}}
    body = {"state": "x" * 60, "questions": qs}
    with pytest.raises(RequestError, match="question 'long': the prompt is"):
        decide(body, Server(max_model_len=400))


class CostlyMark(Server):
    """A tokenizer on which the cut's mark costs 50 tokens more than its characters."""

    def __call__(self, request):
        r = super().__call__(request)
        if request.url.path == "/tokenize" and "…" in request.content.decode("utf-8"):
            return httpx.Response(200, json=dict(r.json(), tokens=r.json()["tokens"] + [0] * 50))
        return r


def test_a_cut_that_frees_too_few_tokens_is_cut_again():
    s = CostlyMark(max_model_len=400)
    out = decide(noul("x" * 1000), s)
    assert len(s.sent("/detokenize")) == 4  # head and tail, twice
    assert out["answers"]["q"]["noul"] == pytest.approx(first_wins(2)[0])
    assert out["usage"]["longest_prompt_tokens"] == LIMIT - 50
