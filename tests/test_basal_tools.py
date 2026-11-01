"""basal's `tool` question (src/basal_lite/_basal/tools.py): the option texts of the tool vectors in
tools/data/format2_vectors.jsonl, every definition form and refusal, and the single / set answers with max_error.

    uv run pytest tests/test_basal_tools.py
"""

import json
from pathlib import Path

import pytest

import basal_lite  # noqa: F401  (tools.answer imports basal_lite.client)
from basal_lite._basal import tools
from basal_lite.client import NO_CERTIFICATE

DATA = Path(__file__).parents[1] / "tools" / "data"
VECTORS = [json.loads(line) for line in (DATA / "format2_vectors.jsonl").read_text(encoding="utf-8").splitlines()]
TOOL_VECTORS = [v for v in VECTORS if v["request"]["questions"].get("q", {}).get("type") == "tool"]
MAX_ERROR = [json.loads(line) for line in (DATA / "max_error.jsonl").read_text(encoding="utf-8").splitlines()]
TOOLS = TOOL_VECTORS[0]["request"]["questions"]["q"]["tools"]  # get_order (OpenAI), refund (MCP), send_message


@pytest.mark.parametrize("vec", TOOL_VECTORS, ids=[v["id"] for v in TOOL_VECTORS])
def test_parse_gives_the_options_and_keys_of_every_tool_vector(vec):
    q = vec["request"]["questions"]["q"]
    (p, *rest) = vec["prompts"]
    keys, texts, names = tools.parse("q", q, p["lang"])
    assert names == [tools._definition("q", i, t)[0] for i, t in enumerate(q["tools"])]
    assert keys[: len(names)] == names
    if q.get("multi"):
        assert [x["label"] for x in vec["prompts"]] == keys
        assert keys[len(names) :] == q.get("extra", [])
    else:
        assert [texts[i] for i in p["option_order"]] == p["options"]


@pytest.mark.parametrize("req", [r for r in MAX_ERROR if r["questions"]["q"]["type"] == "tool"], ids=lambda r: r["id"])
def test_parse_accepts_max_error_on_single_tool_questions(req):
    keys, texts, names = tools.parse("q", req["questions"]["q"], "en")
    assert keys == names + list(tools.EXTRA) and len(texts) == len(keys)


def test_every_definition_form_gives_name_description_and_schema():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    forms = [
        {"type": "function", "function": {"name": "w", "description": "d", "parameters": schema}},
        {"type": "function", "name": "w", "description": "d", "parameters": schema},
        {"name": "w", "description": "d", "inputSchema": schema},
        {"name": "w", "description": "d", "input_schema": schema},
    ]
    for t in forms:
        assert tools._definition("q", 0, t) == ("w", "d", schema)
    assert tools._definition("q", 0, {"name": "ping"}) == ("ping", None, None)


@pytest.mark.parametrize(
    "t, message",
    [
        ("get_order", "tools\\[1\\] must be an object \\(a tool definition\\), got str"),
        (
            {"type": "retrieval", "name": "x"},
            "tools\\[1\\]: type 'retrieval' is not supported \\(function tools only\\)",
        ),
        ({"type": "function", "function": "x"}, "tools\\[1\\].function must be an object"),
        ({"name": ""}, "tools\\[1\\]: name must be a non-empty string without spaces, got ''"),
        ({"name": "get order"}, "name must be a non-empty string without spaces, got 'get order'"),
        ({"description": "no name"}, "name must be a non-empty string without spaces, got None"),
        ({"name": "x", "description": ["a"]}, "tools\\[1\\] \\(x\\): description must be a string"),
        ({"name": "x", "inputSchema": []}, "tools\\[1\\] \\(x\\).inputSchema must be a JSON schema object, got list"),
        ({"name": "x", "parameters": {"type": "array"}}, "\\(x\\).parameters: type must be \"object\", got 'array'"),
        ({"name": "x", "parameters": {"properties": []}}, "properties must be an object of parameter schemas"),
        ({"name": "x", "parameters": {"properties": {"a": "string"}}}, "properties: \\['a'\\] are not schemas"),
        (
            {"name": "x", "parameters": {"properties": {"a": {}}, "required": "a"}},
            "required must be a list of distinct",
        ),
        (
            {"name": "x", "parameters": {"properties": {"a": {}}, "required": [1]}},
            "required must be a list of distinct",
        ),
        ({"name": "x", "parameters": {"properties": {"a": {}}, "required": ["a", "a"]}}, "required must be a list"),
        (
            {"name": "x", "parameters": {"properties": {"a": True}, "required": ["b"]}},
            "required: \\['b'\\] are not among",
        ),
    ],
)
def test_malformed_tool_definitions_are_refused(t, message):
    with pytest.raises(ValueError, match=message) as e:
        tools.parse("q", {"tools": [TOOLS[0], t]}, "en")
    assert str(e.value).startswith("question 'q': tools[1]")


def test_a_schema_without_type_or_properties_is_accepted():
    assert tools._check_schema({}, "w") == {}
    assert tools._check_schema({"properties": {"flag": False}}, "w") == {"properties": {"flag": False}}


@pytest.mark.parametrize(
    "q, message",
    [
        ({"criteria": ["a"], "tools": TOOLS}, 'tool takes "tools" \\(tool definitions\\), not criteria'),
        ({}, 'tool needs "tools", a non-empty list of tool definitions'),
        ({"tools": []}, 'tool needs "tools"'),
        ({"tools": TOOLS[0]}, 'tool needs "tools"'),
        ({"tools": [TOOLS[0], TOOLS[0]]}, "duplicate tool names \\['get_order'\\]"),
        ({"tools": TOOLS, "extra": "none"}, "extra must be a list of distinct names"),
        ({"tools": TOOLS, "extra": ["none", "none"]}, "extra must be a list of distinct names"),
        ({"tools": TOOLS, "extra": [None]}, "extra must be a list of distinct names"),
        ({"tools": TOOLS, "extra": ["escalate"]}, "unknown extra options \\['escalate'\\]"),
        ({"tools": [{"name": "ask_user"}], "extra": ["ask_user"]}, "tool names \\['ask_user'\\] clash with the extra"),
        ({"tools": TOOLS, "multi": "yes"}, "multi must be true or false"),
        ({"tools": TOOLS, "rank": 0}, "rank must be an integer in 1..6 \\(the number of options\\)"),
        ({"tools": TOOLS, "rank": 7}, "rank must be an integer in 1..6"),
        ({"tools": TOOLS, "rank": True}, "rank must be an integer in 1..6"),
        ({"tools": TOOLS, "multi": True, "rank": 4}, "rank must be an integer in 1..3"),
        ({"tools": TOOLS, "rank": "2"}, "rank must be an integer"),
        ({"tools": TOOLS, "description_chars": -1}, "description_chars must be a non-negative integer"),
        ({"tools": TOOLS, "description_chars": False}, "description_chars must be a non-negative integer"),
        ({"tools": TOOLS, "description_chars": 1.5}, "description_chars must be a non-negative integer"),
        ({"tools": TOOLS, "multi": True, "max_error": 0.01}, "max_error is not supported with multi"),
        ({"tools": TOOLS, "max_error": 0}, "max_error must be a number between 0 and 1"),
        ({"tools": TOOLS, "max_error": 1}, "max_error must be a number between 0 and 1"),
        ({"tools": TOOLS, "max_error": True}, "max_error must be a number between 0 and 1"),
        ({"tools": TOOLS, "max_error": "0.01"}, "max_error must be a number between 0 and 1"),
    ],
)
def test_malformed_tool_questions_are_refused(q, message):
    with pytest.raises(ValueError, match=f"question 'q': {message}"):
        tools.parse("q", q, "en")


def test_extras_default_to_all_three_and_to_none_with_multi():
    assert tools.parse("q", {"tools": TOOLS}, "pl")[0][3:] == ["answer_directly", "ask_user", "none"]
    assert tools.parse("q", {"tools": TOOLS, "multi": True}, "en")[0] == ["get_order", "refund", "send_message"]
    keys, texts, _ = tools.parse("q", {"tools": TOOLS, "extra": [], "rank": 3, "max_error": 0.05}, "en")
    assert keys == ["get_order", "refund", "send_message"] and len(texts) == 3


@pytest.mark.parametrize(
    "desc, cap, text",
    [
        (None, 300, "t(a, b?)"),
        ("  Look   up\n an order.  ", 300, "t(a, b?) — Look up an order."),
        ("Look up an order by its number", 12, "t(a, b?) — Look up an…"),  # cut inside "order": back a word
        ("Look up an order by its number", 10, "t(a, b?) — Look up an…"),  # cut on a space: the word is whole
        ("Look up, then refund", 8, "t(a, b?) — Look up…"),  # trailing punctuation dropped
        ("Supercalifragilistic", 5, "t(a, b?) — Super…"),  # one long word: cut inside it
        ("Look up an order by its number", 0, "t(a, b?) — Look up an order by its number"),
    ],
)
def test_option_text_lists_parameters_and_caps_the_description(desc, cap, text):
    schema = {"properties": {"a": {}, "b": {}}, "required": ["a"]}
    assert tools.option_text("t", desc, schema, cap=cap) == text


def test_ranking_orders_by_probability_with_ties_in_option_order():
    probs = {"a": 0.2, "b": 0.5, "c": 0.2, "d": 0.1}
    assert tools.ranking(probs, 3) == [
        {"tool": "b", "probability": 0.5},
        {"tool": "a", "probability": 0.2},
        {"tool": "c", "probability": 0.2},
    ]


Q = {"tools": ["get_order", "refund"], "rank": None, "max_error": None}
PROBS = {"get_order": 0.7, "refund": 0.2, "ask_user": 0.1}


def test_answer_picks_the_most_probable_option_and_whether_it_calls_a_tool():
    assert tools.answer(Q, PROBS, 0.7, {}) == {
        "type": "tool",
        "tool": "get_order",
        "call": True,
        "probabilities": PROBS,
        "confidence": 0.7,
    }
    ans = tools.answer(dict(Q, rank=2), {"get_order": 0.1, "refund": 0.2, "ask_user": 0.7}, 0.7, {})
    assert ans["tool"] == "ask_user" and ans["call"] is False
    assert [r["tool"] for r in ans["ranking"]] == ["ask_user", "refund"]


CAL = {"prompt": 2, "thresholds": {"0.01": {"per_type": {"choice": {"confidence": 0.8}}}}}


@pytest.mark.parametrize("conf, deferred", [(0.85, False), (0.8, False), (0.79, True)])
def test_answer_defers_below_the_certified_choice_threshold(conf, deferred):
    ans = tools.answer(dict(Q, max_error=0.01), PROBS, conf, CAL)
    assert ans["deferred"] is deferred and "max_error" not in ans
    if deferred:
        assert ans["tool"] is None and ans["call"] is False and ans["refused"] == "get_order"
    else:
        assert ans["tool"] == "get_order" and ans["call"] is True and "refused" not in ans


def test_answer_without_a_certificate_always_defers():
    ans = tools.answer(dict(Q, max_error=0.05), PROBS, 0.99, CAL)
    assert ans["deferred"] is True and ans["max_error"] == NO_CERTIFICATE and ans["refused"] == "get_order"


def test_set_answer_turns_a_multi_answer_into_the_tool_set():
    multi = {
        "type": "multi",
        "selected": ["refund", "get_order"],
        "probabilities": {"get_order": 0.6, "refund": 0.9},
        "set_confidence": 0.5,
    }
    out = tools.set_answer(dict(Q, rank=1), dict(multi))
    assert out == {
        "type": "tool",
        "multi": True,
        "tools": ["refund", "get_order"],
        "call": True,
        "probabilities": {"get_order": 0.6, "refund": 0.9},
        "set_confidence": 0.5,
        "ranking": [{"tool": "refund", "probability": 0.9}],
    }
    none = tools.set_answer(Q, dict(multi, selected=["ask_user"]))
    assert none["tools"] == ["ask_user"] and none["call"] is False and "ranking" not in none
