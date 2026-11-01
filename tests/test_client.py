"""The client against a fake vLLM server (tests/fake_vllm.py): no GPU, no model.

uv run pytest
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
from fake_vllm import Server, client_for, decide, fake_tokens, first_wins, handler

from basal_lite import BasalLite, RequestError
from basal_lite._basal.prompt import PREFILL
from basal_lite.client import NO_CERTIFICATE, certified

DATA = Path(__file__).parent.parent / "tools" / "data"
WEATHER = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
TIME = {"type": "function", "function": {"name": "get_time", "description": "Local time.", "parameters": {}}}


def jsonl(name):
    return {d.pop("id"): d for d in map(json.loads, (DATA / name).read_text(encoding="utf-8").splitlines())}


def user_turns(server):
    return [b["messages"][1]["content"] for b in server.sent("/v1/chat/completions")]


def test_choice_reads_the_option_codes():
    out = decide(
        {
            "state": "Customer: I cannot log in.",
            "questions": {
                "dept": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"cards": "Cards", "online": "Online", "loans": "Loans"},
                }
            },
        }
    )
    ans = out["answers"]["dept"]
    assert ans["choice"] == "cards"
    assert list(ans["probabilities"].values()) == pytest.approx(first_wins(3))
    assert out["usage"]["questions"] == 1 and out["usage"]["branches"] == 1


def test_two_option_orders_average_out_a_position_bias():
    out = decide({"state": "x", "questions": {"q": {"type": "noul", "instructions": "Urgent?"}}}, orders=2)
    assert out["answers"]["q"]["noul"] == pytest.approx(0.5)  # "yes first" and "no first" both pick the first shown


def test_multi_asks_one_yes_no_branch_per_label():
    out = decide(
        {
            "state": "x",
            "questions": {"m": {"type": "multi", "instructions": "Topics?", "criteria": ["billing", "delivery"]}},
        }
    )
    assert out["answers"]["m"]["selected"] == ["billing", "delivery"]  # p(yes) = 0.62 > 0.5 for each
    assert out["usage"]["branches"] == 2


def test_score_and_two_questions_in_one_request():
    out = decide(
        {
            "state": "x",
            "questions": {
                "s": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "upset", "furious"]},
                "n": {"type": "noul", "instructions": "Urgent?"},
            },
        }
    )
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


# ---- question types ---------------------------------------------------------------------------------------------


def test_choice_criteria_as_a_list_show_the_keys_alone():
    s = Server()
    out = decide({"state": "x", "questions": {"q": {"instructions": "Pick", "criteria": ["red", "blue"]}}}, s)
    assert out["answers"]["q"]["choice"] == "red"
    assert user_turns(s)[0].endswith("Options:\nA. red\nB. blue")


def test_hidden_option_keys_stay_shown_where_descriptions_repeat():
    s = Server()
    q = {"instructions": "Pick", "option_keys": "hide", "criteria": {"a": "Same", "b": "Same", "c": "Other"}}
    decide({"state": "x", "questions": {"q": q}}, s)
    assert user_turns(s)[0].endswith("Options:\nA. a: Same\nB. b: Same\nC. Other")


def test_score_levels_as_an_object_have_a_legend_of_their_shown_texts():
    q = {"type": "score", "instructions": "Risk?", "criteria": {"low": "Low risk", "high": "High risk"}}
    ans = decide({"state": "x", "questions": {"q": q}})["answers"]["q"]
    assert ans["legend"] == {"low": "low: Low risk", "high": "high: High risk"}
    assert ans["score"] == pytest.approx(first_wins(2)[1])


def test_noul_criteria_rename_yes_and_no():
    s = Server()
    q = {"type": "noul", "instructions": "Paid?", "criteria": {"true": "Paid in full", "false": {"status": "open"}}}
    ans = decide({"state": "x", "questions": {"q": q}}, s)["answers"]["q"]
    assert ans["noul"] == pytest.approx(first_wins(2)[0])
    assert user_turns(s)[0].endswith('A. Paid in full\nB. {"status": "open"}')


def test_multi_criteria_object_selects_by_its_keys():
    q = {"type": "multi", "instructions": "Topics?", "criteria": {"bill": "Billing", "post": None}, "max": 1}
    ans = decide({"state": "x", "questions": {"m": q}})["answers"]["m"]
    assert ans["selected"] == ["bill"]
    assert ans["threshold"] == 0.5 and "constraints_applied" not in ans


def test_multi_with_an_objective_reports_its_expected_score():
    q = {"type": "multi", "instructions": "Topics?", "criteria": ["a", "b"], "objective": "f1"}
    ans = decide({"state": "x", "questions": {"m": q}})["answers"]["m"]
    assert ans["objective"] == "f1" and ans["selected"] == ["a", "b"]
    assert 0 < ans["expected_score"] < 1 and "threshold" not in ans


def test_multi_fallback_reports_the_constraints_and_where_its_threshold_came_from():
    q = {"type": "multi", "instructions": "?", "criteria": ["a", "b", "none"], "fallback": "none", "threshold": 0.7}
    ans = decide({"state": "x", "questions": {"m": q}})["answers"]["m"]
    assert ans["selected"] == []
    assert ans["constraints_applied"] == {
        "exclusive": [],
        "fallback": "none",
        "chosen": "empty",
        "fallback_threshold": 0.7,
        "fallback_threshold_source": "default",
    }
    q["fallback_threshold"] = 0.6
    ans = decide({"state": "x", "questions": {"m": q}})["answers"]["m"]
    assert ans["selected"] == ["none"]
    assert ans["constraints_applied"]["chosen"] == "fallback"
    assert ans["constraints_applied"]["fallback_threshold_source"] == "request"


def test_multi_exclusive_without_fallback_reports_no_threshold_source():
    q = {"type": "multi", "instructions": "?", "criteria": ["a", "b", "any"], "exclusive": ["any"], "max": 1}
    ans = decide({"state": "x", "questions": {"m": q}})["answers"]["m"]
    assert ans["selected"] == ["a"]
    assert ans["constraints_applied"] == {"exclusive": ["any"], "fallback": None, "chosen": "labels"}


def test_tool_question_picks_a_tool_with_the_default_question_and_extra_options():
    s = Server()
    q = {"type": "tool", "tools": [WEATHER, TIME], "rank": 2}
    ans = decide({"state": "User: weather in Gdansk?", "questions": {"q": q}}, s)["answers"]["q"]
    assert ans["tool"] == "get_weather" and ans["call"] is True
    assert list(ans["probabilities"]) == ["get_weather", "get_time", "answer_directly", "ask_user", "none"]
    assert [r["tool"] for r in ans["ranking"]] == ["get_weather", "get_time"]
    user = user_turns(s)[0]
    assert "Question: What should the assistant do next: which tool should it call?" in user
    assert "A. get_weather(city) — Current weather for a city.\nB. get_time() — Local time." in user


def test_tool_max_error_without_a_certified_threshold_defers_to_a_person():
    q = jsonl("max_error.jsonl")["me-tool-001"]
    ans = decide(q)["answers"]["q"]
    assert ans["tool"] is None and ans["call"] is False
    assert ans["deferred"] is True and ans["refused"] == "get_weather"
    assert ans["max_error"] == NO_CERTIFICATE


def test_tool_multi_asks_one_yes_no_per_tool_and_returns_the_set():
    s = Server()
    q = {"type": "tool", "tools": [WEATHER, TIME], "multi": True, "rank": 1}
    ans = decide({"state": "User: weather and time?", "questions": {"q": q}}, s)["answers"]["q"]
    assert ans["type"] == "tool" and ans["multi"] is True
    assert ans["tools"] == ["get_weather", "get_time"] and ans["call"] is True
    assert ans["ranking"] == [{"tool": "get_weather", "probability": pytest.approx(first_wins(2)[0])}]
    users = user_turns(s)
    assert len(users) == 2
    assert 'Which tools should the assistant call now?\nDoes this apply: "get_time() — Local time."' in users[1]


def test_tool_question_with_one_option_is_refused():
    q = {"type": "tool", "tools": [WEATHER], "extra": []}
    with pytest.raises(RequestError, match=r"1 options \(tools and extra options\) \(supported: 2..255\)"):
        decide({"state": "x", "questions": {"q": q}})


def test_act_short_form_defers_without_a_certificate():
    body = jsonl("max_error.jsonl")["me-act-noul-clear"]
    ans = decide(body)["answers"]["q"]
    p = first_wins(2)[0]
    assert ans["expected_costs"] == pytest.approx({"true": (1 - p) * 50, "false": p * 50, "defer": 2.0}, abs=1e-6)
    assert ans["action"] == "defer" and ans["deferred"] is True and "refused" not in ans
    assert ans["answer"] == "true" and ans["max_error"] == NO_CERTIFICATE
    assert ans["calibration"] == "unvalidated"


def test_act_cheaper_automatic_action_is_refused_below_the_certificate():
    q = {"type": "act", "instructions": "Refund?", "costs": {"wrong": 1, "defer": 0.5}, "max_error": 0.05}
    ans = decide({"state": "x", "questions": {"q": q}})["answers"]["q"]
    assert ans["action"] == "defer" and ans["refused"] == "true" and ans["deferred"] is True


def test_act_without_max_error_takes_the_cheapest_action():
    q = {"type": "act", "instructions": "?", "criteria": ["approve", "reject"], "costs": {"wrong": 1}}
    ans = decide({"state": "x", "questions": {"q": q}})["answers"]["q"]
    assert ans["action"] == "approve" and set(ans["expected_costs"]) == {"approve", "reject"}
    assert "deferred" not in ans


def test_act_full_cost_matrix_defers_to_its_human_row():
    ans = decide(jsonl("max_error.jsonl")["me-act-choice-005"])["answers"]["q"]
    assert ans["action"] == "human" and ans["deferred"] is True
    assert ans["expected_costs"]["reject"] == pytest.approx(first_wins(2)[0] * 10, abs=1e-6)


def test_act_defer_action_names_the_row_a_refused_action_goes_to():
    costs = {"approve": {"approve": 0, "reject": 20}, "reject": {"approve": 10, "reject": 0}}
    costs["escalate"] = {"approve": 8, "reject": 8}
    q = {"type": "act", "instructions": "?", "criteria": {"approve": "Yes", "reject": "No"}, "costs": costs}
    q |= {"defer_action": "escalate", "max_error": 0.05}
    ans = decide({"state": "x", "questions": {"q": q}})["answers"]["q"]
    assert ans["action"] == "escalate" and ans["refused"] == "reject"


def test_act_with_a_noul_base_asks_yes_or_no():
    s = Server()
    q = {"type": "act", "base": "noul", "instructions": "Refund?", "criteria": ["a", "b"], "costs": {"wrong": 1}}
    ans = decide({"state": "x", "questions": {"q": q}}, s)["answers"]["q"]
    assert list(ans["probabilities"]) == ["true", "false"]
    assert user_turns(s)[0].endswith("A. Yes\nB. No")


def test_act_requests_without_usable_costs_are_refused():
    def ask(**q):
        decide({"state": "x", "questions": {"q": {"type": "act", "instructions": "?", **q}}})

    with pytest.raises(RequestError, match="act needs costs"):
        ask()
    with pytest.raises(RequestError, match=r"costs\['a'\] must give a cost for every outcome"):
        ask(criteria=["a", "b"], costs={"a": {"a": 0}})
    with pytest.raises(RequestError, match="max_error needs a defer action"):
        ask(costs={"wrong": 1}, max_error=0.05)


def test_request_shape_errors_name_the_problem():
    cases = [
        ([], "the request body must be a JSON object"),
        ({"state": "x"}, "missing field 'questions'"),
        ({"state": "x", "questions": {"q": "?"}}, '"questions" must be an object'),
        ({"state": "x", "questions": {}, "early_exit": 0.9}, "early_exit needs basal-serve"),
        ({"state": "x", "questions": {"q": {"type": "noul", "evidence": True}}}, "evidence needs basal-serve"),
        ({"state": "x", "questions": {}, "facts": "on"}, 'facts must be "off" or "auto"'),
        ({"state": "x", "questions": {}, "lang": "de"}, 'lang must be "pl" or "en"'),
        ({"state": "x", "questions": {"q": {"criteria": ["a", "b"], "option_keys": "no"}}}, "option_keys must be"),
        ({"state": "x", "questions": {"q": {"type": "multi", "criteria": {}}}}, "multi needs criteria"),
    ]
    for body, msg in cases:
        with pytest.raises(RequestError, match=msg):
            decide(body)


def test_facts_auto_adds_the_computed_facts_to_the_state():
    s = Server()
    out = decide(jsonl("tasks.jsonl")["facts-mixed"], s)
    assert out["answers"]["late"]["noul"] == pytest.approx(first_wins(2)[0])
    assert all("Fakty pomocnicze" in u and "17 lutego 2026" in u for u in user_turns(s))


def test_lang_forces_the_prompt_language():
    s = Server()
    decide({"state": "Klient: paczka nie dotarła.", "lang": "en", "questions": {"q": {"type": "noul"}}}, s)
    assert user_turns(s)[0].startswith("State:\nKlient")
    assert user_turns(s)[0].endswith("A. Yes\nB. No")


# ---- tasks --------------------------------------------------------------------------------------------------------


def test_classify_and_verify_are_answered_from_their_decisions():
    tasks = jsonl("tasks.jsonl")
    ans = decide(tasks["classify"])["answers"]["intent"]
    assert ans["type"] == "classify" and ans["label"] == "card_payment_duplicate"
    assert ans["automated"] == {"0.01": None, "0.05": None}
    out = decide(tasks["verify"])
    assert set(out["answers"]["check"]["claims"]) == {"total", "terms", "bank"}
    assert out["usage"]["branches"] == 3 and out["usage"]["questions"] == 1


def test_extract_adds_the_span_round_to_the_usage():
    s = Server()
    out = decide(jsonl("tasks.jsonl")["extract"], s)
    ans = out["answers"]["inv"]
    assert ans["type"] == "extract" and ans["fields"]["invoice_number"]["source"] == "span"
    chats = s.sent("/v1/chat/completions")
    assert out["usage"]["branches"] == len(chats) == 11
    assert out["usage"]["input_tokens"] == sum(len(fake_tokens(b["messages"])) for b in chats)


def test_compress_keeps_the_units_it_chose():
    ans = decide(jsonl("tasks.jsonl")["compress"])["answers"]["ctx"]
    assert ans["type"] == "compress" and ans["kept"] == [0, 1, 2, 3]


# ---- option orders, scheduling, the server ------------------------------------------------------------------------


def test_orders_2le_reverses_only_questions_with_few_options():
    s = Server()
    qs = {"n": {"type": "noul"}, "c": {"criteria": ["a", "b", "c"]}}
    out = decide({"state": "x", "questions": qs}, s, orders="2le2")
    assert out["answers"]["n"]["noul"] == pytest.approx(0.5)
    assert list(out["answers"]["c"]["probabilities"].values()) == pytest.approx(first_wins(3))
    assert len(s.sent("/v1/chat/completions")) == 3


def test_orders_auto_reads_the_reversed_order_only_for_uncertain_branches():
    qs = {
        "n": {"type": "noul"},  # 0.62: inside the yes/no band
        "c2": {"criteria": ["a", "b"]},  # 0.62 >= tau 0.6: certain
        "c3": {"criteria": ["a", "b", "c"]},  # 0.51 < tau
        "m": {"type": "multi", "criteria": ["a", "b"]},
    }
    out = decide({"state": "x", "questions": qs}, orders="auto")
    a = out["answers"]
    assert out["usage"]["second_order_branches"] == 4
    assert a["n"]["noul"] == pytest.approx(0.5)
    assert a["c2"]["confidence"] == pytest.approx(first_wins(2)[0])
    p = first_wins(3)
    assert list(a["c3"]["probabilities"].values()) == pytest.approx([(x + y) / 2 for x, y in zip(p, p[::-1])])
    assert a["m"]["probabilities"] == pytest.approx({"a": 0.5, "b": 0.5})


class InFlight:
    """An async fake server that records how many readouts were in flight when each one started."""

    def __init__(self):
        self.flight, self.starts = 0, []

    async def __call__(self, request):
        if request.url.path == "/v1/chat/completions":
            self.starts.append(self.flight)
            self.flight += 1
            await asyncio.sleep(0.01)
            self.flight -= 1
        return handler(request)


def test_warm_first_sends_the_first_prompt_alone():
    qs = {k: {"type": "noul"} for k in "abc"}
    for warm, starts in ((False, [0, 1, 2]), (True, [0, 0, 1])):
        srv = InFlight()
        decide({"state": "x", "questions": qs}, srv, warm_first=warm)
        assert srv.starts == starts


def test_max_requests_limits_the_requests_on_the_server_at_once():
    body = {"state": "x", "questions": {"a": {"type": "noul"}, "b": {"type": "noul"}}}
    for limit, most in ((None, 3), (1, 1)):
        srv = InFlight()
        b = BasalLite("http://fake", "basal", client=client_for(srv), max_requests=limit)

        async def both():
            return await asyncio.gather(b.decide(body), b.decide(body))

        outs = asyncio.run(both())
        assert [o["answers"]["a"]["noul"] for o in outs] == pytest.approx([first_wins(2)[0]] * 2)
        assert max(srv.starts) == most


def test_option_code_ids_are_read_once_per_client():
    s = Server()
    b = BasalLite("http://fake/v1/", "basal", client=client_for(s))
    body = {"state": "x", "questions": {"q": {"type": "noul"}}}
    asyncio.run(b.decide(body))
    asyncio.run(b.decide(body))
    assert len(s.sent("/tokenize")) == 1 + 255 + 1  # the prompt, every code after it, the empty template
    assert len(b.ids) == 255 and b.overhead == len('{"answer": "') + 16


def test_usage_sums_prompt_and_cached_tokens():
    s = Server(usage={"prompt_tokens_details": {"cached_tokens": 40}})
    out = decide({"state": "x", "questions": {"a": {"type": "noul"}, "b": {"type": "noul"}}}, s)
    n = [len(fake_tokens(b["messages"])) for b in s.sent("/v1/chat/completions")]
    assert out["usage"]["input_tokens"] == sum(n) and out["usage"]["longest_prompt_tokens"] == max(n)
    assert out["usage"]["cached_tokens"] == 80 and out["usage"]["output_tokens"] == 0


def test_readout_request_restricts_the_token_to_the_option_codes():
    s = Server()
    decide({"state": "x", "questions": {"q": {"criteria": ["a", "b", "c"]}}}, s)
    r = s.sent("/v1/chat/completions")[0]
    assert r["max_tokens"] == 1 and r["top_logprobs"] == 3 and len(r["allowed_token_ids"]) == 3
    assert r["continue_final_message"] is True and r["add_generation_prompt"] is False
    assert r["messages"][-1] == {"role": "assistant", "content": '{"answer": "'}


def test_server_errors_are_runtime_errors_with_the_path_and_status():
    with pytest.raises(RuntimeError, match="/v1/chat/completions: HTTP 500: server error"):
        decide({"state": "x", "questions": {"q": {"type": "noul"}}}, Server(fail={"/v1/chat/completions": 500}))


def test_a_readout_without_option_codes_points_at_the_server_flags():
    def no_codes(request):
        if request.url.path == "/v1/chat/completions":
            top = [{"token": "token_id:7", "logprob": 0.0}]
            return httpx.Response(200, json={"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]})
        return handler(request)

    with pytest.raises(RuntimeError, match="--logprobs-mode processed_logprobs"):
        decide({"state": "x", "questions": {"q": {"type": "noul"}}}, no_codes)


def test_option_codes_must_be_single_distinct_tokens():
    def tokenizer(extra):
        def fn(request):
            body = json.loads(request.content)
            if request.url.path == "/tokenize" and body["messages"][-1]["content"] != PREFILL:
                tokens = fake_tokens(body["messages"][:-1] + [{"content": PREFILL}]) + extra
                return httpx.Response(200, json={"tokens": tokens})
            return handler(request)

        return fn

    body = {"state": "x", "questions": {"q": {"type": "noul"}}}
    with pytest.raises(RuntimeError, match="option code 'A' is not one token at the answer position"):
        decide(body, tokenizer([65, 66]))
    with pytest.raises(RuntimeError, match="two option codes share a token"):
        decide(body, tokenizer([65]))


def test_certified_reads_per_type_thresholds_of_format_2_calibrations():
    flat = {"thresholds": {"0.05": {"confidence": 0.7}}}
    per_type = {"prompt": 2, "thresholds": {"0.05": {"per_type": {"noul": {"confidence": 0.6}}}}}
    assert certified(flat, 0.05, "noul") == 0.7 and certified(flat, 0.01, "noul") is None
    assert certified(per_type, 0.05, "noul") == 0.6 and certified(per_type, 0.05, "choice") is None
    assert certified({"prompt": 2, "thresholds": {"0.05": {"confidence": 0.7}}}, 0.05, "noul") is None
