"""basal's task types (src/basal_lite/_basal/tasks.py): classify, verify, compress and extract expanded into decision
questions and finished from hand-made answers, the requests of tools/data/tasks.jsonl among them, plus the text units,
span candidates, number parsing and request errors.

    uv run pytest tests/test_basal_tasks.py
"""

import json
from pathlib import Path

import pytest

from basal_lite._basal import tasks
from basal_lite._basal.prompt import RequestError

TASKS = {
    r["id"]: r
    for r in map(
        json.loads,
        (Path(__file__).parents[1] / "tools" / "data" / "tasks.jsonl").read_text(encoding="utf-8").splitlines(),
    )
}
CAL = {
    "task_thresholds": {
        "classify": {"0.01": 0.9, "0.05": 0.7},
        "verify": {"0.05": 0.6},
        "extract": {"0.01": 0.95, "0.05": 0.5},
    }
}


def choice(key, probs):
    return {"choice": key, "probabilities": probs, "confidence": max(probs.values())}


def noul(p):
    return {"noul": p}


def response(answers, n=1):
    return {"answers": answers, "usage": {"questions": n}}


# --- expansion of the requests in tools/data/tasks.jsonl ------------------------------------------------------------


def test_classify_becomes_one_choice_with_rubrics_and_a_none_option():
    body, plans = tasks.expand(TASKS["classify"])
    q = body["questions"]["intent::labels"]
    assert list(body["questions"]) == ["intent::labels"] and body["state"] == TASKS["classify"]["state"]
    assert q == {
        "type": "choice",
        "instructions": "Which label best describes this text?",
        "option_keys": "show",
        "criteria": {
            "card_payment_duplicate": "a card payment was taken more than once",
            "card_lost": "the card is lost or stolen",
            "refund_status": "asks about a refund in progress",
            "none": "None of the labels above",
        },
    }
    assert plans == {"intent": {"type": "classify", "multi": False, "risks": [0.01, 0.05]}}


def test_verify_asks_one_three_way_choice_per_claim():
    body, plans = tasks.expand(TASKS["verify"])
    assert list(body["questions"]) == ["check::c::total", "check::c::terms", "check::c::bank"]
    q = body["questions"]["check::c::terms"]
    assert q["type"] == "choice" and q["option_keys"] == "hide"
    assert q["instructions"] == "Claim: Payment is due in 30 days.\nWhat does the source say about this claim?"
    assert list(q["criteria"]) == ["supported", "contradicted", "not_stated"]
    assert plans["check"] == {
        "type": "verify",
        "keys": ["total", "terms", "bank"],
        "binary": False,
        "risks": [0.01, 0.05],
    }


def test_compress_asks_a_yes_no_question_per_line_in_polish():
    body, plans = tasks.expand(TASKS["compress"])
    state = TASKS["compress"]["state"]
    assert len(body["questions"]) == 8 == len(state.splitlines())
    assert body["questions"]["ctx::u::1"] == {
        "type": "noul",
        "instructions": "Fragment: „Najemca płaci czynsz 4 500 zł miesięcznie.”\nCzy ten fragment "
        "zawiera informację potrzebną do odpowiedzi na pytanie: Jakie są warunki płatności?",
    }
    p = plans["ctx"]
    assert p["unit"] == "line" and p["keep_ratio"] == 0.5 and p["threshold"] == 0.5 and p["min_keep"] == 1
    assert [state[a:b] for a, b in p["spans"]] == state.splitlines()


def test_a_request_without_tasks_passes_through():
    body, plans = tasks.expand(TASKS["facts-mixed"])
    assert body == TASKS["facts-mixed"] and plans == {}


INVOICE = {
    "invoice_number": "FV/309/02/2026",
    "issue_date": "03.02.2026",
    "gross_total": "1 230,00 zł",
    "seller.name": "Med Energia Sp. z o.o.",
    "seller.tax_id": "5251234567",
}


def test_extract_end_to_end_reads_exact_source_spans_from_the_invoice():
    req = TASKS["extract"]
    state = req["state"]
    body, plans = tasks.expand(req)
    p = plans["inv"]
    assert [x[:2] for x in p["fields"]] == [
        ("invoice_number", "string"),
        ("issue_date", "string"),
        ("gross_total", "number"),
        ("payment_method", "enum"),
        ("seller.name", "string"),
        ("seller.tax_id", "string"),
    ]
    assert p["lang"] == "pl" and [state[a:b] for a, b in p["regions"]] == state.split("\n")
    pm = body["questions"]["inv::f::payment_method"]
    assert pm["criteria"] == {
        "transfer": "transfer",
        "card": "card",
        "cash": "cash",
        "__missing__": "Nie podano w dokumencie",
    }
    assert (
        body["questions"]["inv::f::seller.tax_id"]["instructions"]
        == "W której linii dokumentu jest wartość pola: seller / tax id?"
    )
    assert body["questions"]["inv::f::issue_date"]["instructions"].endswith("pola: issue date (the date of issue)?")

    pm_probs = {"transfer": 0.9, "card": 0.05, "cash": 0.03, "__missing__": 0.02}
    answers = {"inv::f::payment_method": choice("transfer", pm_probs)}
    for path, value in INVOICE.items():
        line = next(i for i, (a, b) in enumerate(p["regions"]) if value in state[a:b])
        probs = {k: 0.0 for k in body["questions"][f"inv::f::{path}"]["criteria"]}
        probs[f"L{line}"], probs["__missing__"] = 0.96, 0.04
        answers[f"inv::f::{path}"] = choice(f"L{line}", probs)
    out = response(answers, 6)

    second = tasks.second_round(out, req, plans)
    assert set(second) == {"state", "questions"} and second["state"] == state
    assert set(second["questions"]) == {f"inv::s::{k}" for k in INVOICE}
    span_q = second["questions"]["inv::s::issue_date"]
    assert span_q["instructions"].startswith("Linia: „Data wystawienia: 03.02.2026”\nKtóry fragment tej linii")
    a2 = {}
    for path, value in INVOICE.items():
        crit = second["questions"][f"inv::s::{path}"]["criteria"]
        key = next(k for k, v in crit.items() if v == value)
        a2[f"inv::s::{path}"] = choice(key, {key: 0.9, "__missing__": 0.1})

    res = tasks.finish(out, req, plans, CAL, a2)["answers"]["inv"]
    assert res["data"] == {
        "invoice_number": "FV/309/02/2026",
        "issue_date": "03.02.2026",
        "gross_total": 1230.0,
        "payment_method": "transfer",
        "seller": {"name": "Med Energia Sp. z o.o.", "tax_id": "5251234567"},
    }
    assert res["experimental"] is True and out["usage"]["questions"] == 1
    total = res["fields"]["gross_total"]
    assert state[total["start"] : total["end"]] == total["text"] == "1 230,00 zł"
    assert total["confidence"] == pytest.approx(0.96 * 0.9) and total["line_probability"] == 0.96
    assert total["automated"] == {"0.01": False, "0.05": True} and total["calibration"] == "per decision"
    assert res["fields"]["payment_method"] == {
        "value": "transfer",
        "probabilities": pm_probs,
        "confidence": 0.9,
        "source": "choice",
        "automated": {"0.01": False, "0.05": True},
        "calibration": "validated",
    }


# --- expansion: options and refusals --------------------------------------------------------------------------------


def test_classify_multi_label_keeps_set_options_and_drops_none():
    q = {
        "type": "classify",
        "criteria": ["billing", "delivery"],
        "multi_label": True,
        "none": True,
        "threshold": 0.3,
        "min": 1,
        "objective": "f1",
        "option_keys": "hide",
        "instructions": "Topics?",
        "max_error": 0.02,
    }
    body, plans = tasks.expand({"state": "x", "questions": {"t": q}})
    assert body["questions"]["t::labels"] == {
        "type": "multi",
        "criteria": {"billing": None, "delivery": None},
        "instructions": "Topics?",
        "option_keys": "hide",
        "threshold": 0.3,
        "min": 1,
        "objective": "f1",
    }
    assert plans["t"] == {"type": "classify", "multi": True, "risks": [0.02]}


@pytest.mark.parametrize(
    "spec, labels",
    [
        ({"a": "rubric a", 2: "two"}, {"a": "rubric a", "2": "two"}),
        (["a", 2], {"a": None, "2": None}),
        ([{"name": "a", "description": "d"}, {"name": 1}], {"a": "d", "1": None}),
    ],
)
def test_labels_come_as_rubrics_keys_or_named_entries(spec, labels):
    assert tasks._labels(spec, "t") == labels


@pytest.mark.parametrize("spec", [None, [], {}, "a"])
def test_labels_refuse_empty_or_other_values(spec):
    with pytest.raises(ValueError, match="question 't': labels must be a non-empty object"):
        tasks._labels(spec, "t")


def test_verify_binary_with_list_claims_and_instructions():
    q = {
        "type": "verify",
        "claims": ["Kwota to 1230 zł.", {"amount": 1230}],
        "binary": True,
        "instructions": "Źródło: faktura.",
    }
    body, plans = tasks.expand({"state": "Invoice", "lang": "pl", "questions": {"v": q}})
    assert body["questions"] == {
        "v::c::0": {
            "type": "noul",
            "instructions": "Źródło: faktura.\nTwierdzenie: Kwota to 1230 zł.\nCzy źródło potwierdza to twierdzenie?",
        },
        "v::c::1": {
            "type": "noul",
            "instructions": 'Źródło: faktura.\nTwierdzenie: {"amount": 1230}\nCzy źródło potwierdza to twierdzenie?',
        },
    }
    assert plans["v"]["keys"] == ["0", "1"] and plans["v"]["binary"] is True


def test_language_comes_from_the_request_then_a_polish_state_then_the_instructions():
    def lang(body):
        return tasks.expand(body)[0]["questions"]["c::labels"]["instructions"]

    q = {"type": "classify", "labels": ["a", "b"]}
    assert lang({"state": "Zażółć", "questions": {"c": q}}) == tasks.W["pl"]["classify"]
    assert lang({"state": "Zażółć", "lang": "en", "questions": {"c": q}}) == tasks.W["en"]["classify"]
    assert lang({"state": {"json": 1}, "questions": {"c": dict(q, instructions="Która?")}}) == "Która?"
    body, _ = tasks.expand(
        {"state": "plain", "questions": {"c": dict(q, type="verify", claims={"k": "c"}, instructions="Czy źródło?")}}
    )
    assert "Twierdzenie" in body["questions"]["c::c::k"]["instructions"]


def test_compress_without_query_asks_about_important_information_and_keeps_threshold_options():
    body, plans = tasks.expand(
        {
            "state": "First point. Second point.",
            "questions": {"c": {"type": "compress", "threshold": "0.7", "min_keep": 0}},
        }
    )
    assert (
        body["questions"]["c::u::0"]["instructions"]
        == 'Passage: "First point."\nDoes this passage hold important information of the document?'
    )
    assert plans["c"]["threshold"] == 0.7 and plans["c"]["keep_ratio"] is None and plans["c"]["min_keep"] == 0


@pytest.mark.parametrize(
    "body, message",
    [
        (
            {"state": "x", "questions": {"a::b": {"type": "verify", "claims": ["c"]}}},
            "question 'a::b': a task question's name may not contain '::'",
        ),
        (
            {"state": "x", "questions": {"v": {"type": "verify", "claims": []}}},
            "question 'v': claims must be a non-empty object or list",
        ),
        (
            {"state": "x", "questions": {"v": {"type": "verify", "claims": "c"}}},
            "question 'v': claims must be a non-empty object or list",
        ),
        ({"state": "", "questions": {"c": {"type": "compress"}}}, "question 'c': compress needs a text state"),
        (
            {"state": " \n\n ", "questions": {"c": {"type": "compress"}}},
            "question 'c': the state has no text to compress",
        ),
        (
            {"state": "x\n" * 601, "questions": {"c": {"type": "compress", "unit": "line"}}},
            "question 'c': 601 lines \\(at most 600\\); use a larger unit or split the document",
        ),
        (
            {"state": "x", "questions": {"c": {"type": "compress", "unit": "word"}}},
            'unit must be "sentence", "line" or "paragraph"',
        ),
        (
            {"state": "x", "questions": {"c": {"type": "compress", "keep_ratio": 0}}},
            "question 'c': keep_ratio must be in \\(0, 1\\]",
        ),
        (
            {"state": "x", "questions": {"c": {"type": "compress", "keep_ratio": 1.5}}},
            "question 'c': keep_ratio must be in \\(0, 1\\]",
        ),
        (
            {
                "state": "",
                "questions": {"e": {"type": "extract", "schema": {"properties": {"name": {"type": "string"}}}}},
            },
            "question 'e': field 'name' is taken from the state's text, which is empty",
        ),
        (
            {"state": "x", "questions": {"e": {"type": "extract", "schema": {"type": "object"}}}},
            "extract: object 'root' has no properties",
        ),
        ({"state": "x"}, "missing field 'questions'"),
        ({"state": "x", "questions": ["c"]}, "invalid request: 'list' object has no attribute 'items'"),
        ({"state": "x", "questions": {"c": {"type": "compress", "threshold": "high"}}}, "could not convert string"),
    ],
)
def test_expand_refuses_malformed_task_questions_as_request_errors(body, message):
    with pytest.raises(RequestError, match=message):
        tasks.expand(body)


def test_extract_fields_from_media_are_enum_and_boolean_only():
    schema = {
        "properties": {
            "paid": {"type": "boolean", "title": "is it paid"},
            "kind": {
                "oneOf": [
                    {"const": "invoice", "description": "an invoice"},
                    {"const": "receipt", "title": "a receipt"},
                    {"const": 3},
                ]
            },
            "tags": {"type": "array", "items": {"anyOf": [{"const": "urgent"}, {"const": "vip"}]}},
        }
    }
    body, plans = tasks.expand(
        {
            "state": "",
            "lang": "en",
            "questions": {"e": {"type": "extract", "schema": schema, "instructions": "From the photo."}},
        }
    )
    qs = body["questions"]
    assert qs["e::f::paid"] == {
        "type": "choice",
        "option_keys": "hide",
        "instructions": "From the photo.\nWhat is the value of the field: paid (is it paid)?",
        "criteria": {"true": "Yes", "false": "No", "__missing__": "Not stated in the document"},
    }
    assert qs["e::f::kind"]["criteria"] == {
        "invoice": "invoice (an invoice)",
        "receipt": "receipt (a receipt)",
        "3": "3",
        "__missing__": "Not stated in the document",
    }
    assert qs["e::f::tags"] == {
        "type": "multi",
        "criteria": {"urgent": "urgent", "vip": "vip"},
        "instructions": "From the photo.\nWhich values does the field have: tags?",
    }
    assert plans["e"]["regions"] is None and plans["e"]["pre"] == "From the photo.\n"


# --- extract schema -------------------------------------------------------------------------------------------------


def test_leaves_flatten_a_schema_into_typed_fields():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": ["null", "integer"]},
            "b": {"type": ["null"]},
            "c": {},
            "d": {"enum": ["x", None]},
            "e": {"type": "array", "items": {"enum": ["p", "q"]}},
            "f": {"properties": {"g": {"type": "number"}}},
            "h": {"type": "boolean"},
        },
    }
    assert [(p, k) for p, k, _ in tasks.leaves(schema)] == [
        ("a", "integer"),
        ("b", "string"),
        ("c", "string"),
        ("d", "enum"),
        ("e", "enum_list"),
        ("f.g", "number"),
        ("h", "boolean"),
    ]


@pytest.mark.parametrize(
    "schema, message",
    [
        ([], "schema at 'root' must be an object"),
        ({"properties": {"a": "string"}}, "schema at 'a' must be an object"),
        ({"type": "object"}, "object 'root' has no properties"),
        ({"properties": {"o": {"type": "object", "properties": {}}}}, "object 'o' has no properties"),
        ({"properties": {"l": {"type": "array", "items": {"type": "string"}}}}, "field 'l': arrays are supported only"),
        ({"properties": {"l": {"type": "array"}}}, "field 'l': arrays are supported only with enum items"),
        ({"properties": {"n": {"type": "null"}}}, "field 'n': type 'null' is not supported"),
    ],
)
def test_leaves_refuse_unsupported_schemas(schema, message):
    with pytest.raises(ValueError, match=f"extract: {message}"):
        list(tasks.leaves(schema))


@pytest.mark.parametrize(
    "schema, consts",
    [
        ({"oneOf": [{"const": 1, "title": "one"}, {"const": 2}]}, [(1, "one"), (2, None)]),
        ({"anyOf": [{"const": "a"}, {"type": "string"}], "enum": ["b", None]}, [("b", None)]),
        ({"oneOf": []}, None),
        ({"type": "string"}, None),
    ],
)
def test_consts_read_one_of_any_of_or_enum(schema, consts):
    assert tasks._consts(schema) == consts


@pytest.mark.parametrize(
    "path, spec, text",
    [
        ("seller.tax_id", {}, "seller / tax id"),
        ("total", {"description": "gross", "title": "T"}, "total (gross)"),
        ("total", {"title": "T"}, "total (T)"),
    ],
)
def test_field_text_names_the_path_with_its_description(path, spec, text):
    assert tasks._field_text(path, spec) == text


def test_set_path_builds_nested_objects():
    data = {}
    tasks._set_path(data, "seller.address.city", "Gdańsk")
    tasks._set_path(data, "seller.name", "Med")
    tasks._set_path(data, "total", 1)
    assert data == {"seller": {"address": {"city": "Gdańsk"}, "name": "Med"}, "total": 1}


# --- text units -----------------------------------------------------------------------------------------------------


def spans(text, unit="sentence"):
    return [text[a:b] for a, b in tasks.units(text, unit)]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Dr. Kowalski przyszedł. Np. tak. Pan J. Nowak też! Co? Tak.",
            ["Dr. Kowalski przyszedł.", "Np. tak.", "Pan J. Nowak też!", "Co?", "Tak."],
        ),
        ("Cena to ok. 5 zł netto. Termin: 3 dni.", ["Cena to ok. 5 zł netto.", "Termin: 3 dni."]),
        ("Kosztuje 5 zł. Termin minął.", ["Kosztuje 5 zł. Termin minął."]),  # "zł." is an abbreviation
        ("Uwaga (!) Dalej. „Cytat.” Koniec", ["Uwaga (!)", "Dalej.", "„Cytat.”", "Koniec"]),
        (
            "Lista:\n- jeden\n- dwa\nciąg dalszy\n\nNowy akapit",
            ["Lista:", "- jeden", "- dwa\nciąg dalszy", "Nowy akapit"],
        ),
        ("  \n  ", []),
    ],
)
def test_sentences_split_on_ends_and_breaks_but_not_abbreviations(text, expected):
    assert spans(text) == expected


def test_lines_and_paragraphs_are_trimmed_and_skip_blanks():
    text = "\n  one \n\n two\nthree \n \n four"
    assert spans(text, "line") == ["one", "two", "three", "four"]
    assert spans(text, "paragraph") == ["one", "two\nthree", "four"]


def test_join_units_keeps_neighbour_separators_and_bridges_gaps():
    text = "A one.\nB two. C three.\nD four."
    s = tasks.units(text)
    assert tasks.join_units(text, [(0, s[0]), (1, s[1]), (3, s[3])]) == "A one.\nB two.\nD four."
    assert tasks.join_units(text, [(1, s[1]), (3, s[3])]) == "B two.\nD four."
    assert tasks.join_units(text, [(0, s[0]), (2, s[2])]) == "A one.\nC three."
    assert tasks.join_units(text, [(1, s[1]), (2, s[2])]) == "B two. C three."
    one = "Ab. Cd? Ef!"
    u = tasks.units(one)
    assert tasks.join_units(one, [(0, u[0]), (2, u[2])]) == "Ab. Ef!"  # a gap on one line: a space
    assert tasks.join_units(one, []) == ""


def test_line_regions_group_lines_beyond_the_cap():
    text = "\n".join(f"line {i}" for i in range(10))
    assert len(tasks.line_regions(text)) == 10
    assert [text[a:b] for a, b in tasks.line_regions(text, 3)] == [
        "line 0\nline 1\nline 2\nline 3",
        "line 4\nline 5\nline 6\nline 7",
        "line 8\nline 9",
    ]


def test_shown_collapses_whitespace_and_cuts_long_text():
    assert tasks._shown("a \n  b") == "a b"
    assert tasks._shown("x" * 401) == "x" * 400 + " …"
    assert tasks._shown("x" * 5, limit=3) == "xxx …"


def test_span_candidates_split_on_separators_and_after_colons_then_word_runs():
    text = "Sprzedawca: Med Energia; NIP: 525 | (tel. 600)"
    out = [text[a:b] for a, b in tasks.span_candidates(text, 0, len(text))]
    assert out[:6] == ["Sprzedawca: Med Energia", "Med Energia", "NIP: 525", "525", "tel. 600", "Sprzedawca"]
    assert len(out) == len(set(out)) and all(any(c.isalnum() for c in x) for x in out)
    assert [text[a:b] for a, b in tasks.span_candidates(text, 0, len(text), cap=3)] == out[:3]
    assert tasks.span_candidates("a | -- | b", 4, 6) == []


def test_span_candidates_shorten_word_runs_to_fit_the_cap():
    text = " ".join(f"w{i}" for i in range(30))
    many = tasks.span_candidates(text, 0, len(text))
    assert text[slice(*many[0])] == text and max(text[a:b].count(" ") for a, b in many[1:]) == 7  # up to 8 words
    few = tasks.span_candidates(text, 0, len(text), cap=40)
    assert len(few) == 31 and max(text[a:b].count(" ") for a, b in few[1:]) == 0  # the cap allows one word each


# --- numbers --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, integer, value",
    [
        ("1 230,00 zł", False, 1230.0),
        ("1.234,56", False, 1234.56),
        ("1,234.56", False, 1234.56),
        ("12,5 kg", False, 12.5),
        ("1,234", False, 1234),
        ("1.234.567", False, 1234567),
        ("-5 °C", False, -5),
        ("7.0", False, 7.0),
        ("+3 450", False, 3450),
        ("12'500", False, 12500),
        ("42 szt.", True, 42),
        ("7.0", True, 7),
        ("4,5", True, None),
        ("1,5.5,5", False, None),
        ("brak", False, None),
        (None, False, None),
    ],
)
def test_parse_number_reads_polish_and_english_notation(text, integer, value):
    out = tasks.parse_number(text, integer)
    assert out == value and type(out) is type(value)


# --- assembly -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cal, conf, risks, expected",
    [
        (CAL, 0.8, [0.01, 0.05], {"0.01": False, "0.05": True}),
        (CAL, 0.95, [0.01, 0.02], {"0.01": True, "0.02": None}),
        ({}, 0.99, [0.01], {"0.01": None}),
        ({"task_thresholds": None}, 0.99, [0.05], {"0.05": None}),
    ],
)
def test_automated_needs_a_threshold_certified_for_the_task(cal, conf, risks, expected):
    assert tasks._automated(cal, "classify", conf, risks) == expected


def test_risks_default_to_both_targets_and_take_a_single_value():
    assert tasks._risks({}) == [0.01, 0.05]
    assert tasks._risks({"max_error": 0.02}) == [0.02]
    assert tasks._risks({"max_error": [0.1, "0.2"]}) == [0.1, 0.2]


def test_is_task_only_for_the_four_task_types():
    assert all(tasks.is_task({"type": t}) for t in tasks.TASKS)
    assert not tasks.is_task({"type": "choice"}) and not tasks.is_task("classify") and not tasks.is_task({})


def test_finish_classify_single_and_multi_label():
    body = {"state": "x", "questions": {"one": {"type": "classify"}, "many": {"type": "classify"}, "other": {}}}
    plans = {
        "one": {"type": "classify", "multi": False, "risks": [0.01, 0.05]},
        "many": {"type": "classify", "multi": True, "risks": [0.01]},
    }
    out = response(
        {
            "one::labels": choice("card_lost", {"card_lost": 0.92, "none": 0.08}),
            "many::labels": {"selected": ["a"], "probabilities": {"a": 0.8, "b": 0.1}, "set_confidence": 0.72},
            "other": {"choice": "x"},
        },
        5,
    )
    ans = tasks.finish(out, body, plans, CAL)["answers"]
    assert ans["one"] == {
        "type": "classify",
        "label": "card_lost",
        "top_label": "card_lost",
        "scores": {"card_lost": 0.92, "none": 0.08},
        "confidence": 0.92,
        "automated": {"0.01": True, "0.05": True},
        "experimental": True,
    }
    assert ans["many"] == {
        "type": "classify",
        "labels": ["a"],
        "scores": {"a": 0.8, "b": 0.1},
        "set_confidence": 0.72,
        "experimental": True,
    }
    assert ans["other"] == {"choice": "x"} and set(ans) == {"one", "many", "other"}
    assert out["usage"]["questions"] == 3


def test_finish_verify_flags_every_claim_that_is_not_supported():
    body, plans = tasks.expand(TASKS["verify"])
    probs = {"supported": 0.85, "contradicted": 0.1, "not_stated": 0.05}
    out = response(
        {
            "check::c::total": choice("supported", probs),
            "check::c::terms": choice("contradicted", {"supported": 0.05, "contradicted": 0.9, "not_stated": 0.05}),
            "check::c::bank": choice("not_stated", {"supported": 0.1, "contradicted": 0.2, "not_stated": 0.7}),
        }
    )
    res = tasks.finish(out, TASKS["verify"], plans, CAL)["answers"]["check"]
    assert res["all_supported"] is False and res["not_supported"] == ["terms", "bank"]
    assert res["claims"]["total"] == {
        "verdict": "supported",
        "probabilities": probs,
        "confidence": 0.85,
        "automated": {"0.01": None, "0.05": True},
    }


@pytest.mark.parametrize("p, verdict", [(0.5, "supported"), (0.49, "not_supported")])
def test_finish_verify_binary_reads_yes_as_supported(p, verdict):
    plans = {"v": {"type": "verify", "keys": ["0"], "binary": True, "risks": [0.05]}}
    res = tasks.finish(response({"v::c::0": noul(p)}), {"questions": {"v": {}}}, plans, CAL)["answers"]["v"]
    claim = res["claims"]["0"]
    assert claim["verdict"] == verdict and claim["probabilities"] == pytest.approx(
        {"supported": p, "not_supported": 1 - p}
    )
    assert claim["confidence"] == pytest.approx(max(p, 1 - p)) and res["all_supported"] is (verdict == "supported")


COMPRESS_P = [0.1, 0.9, 0.95, 0.2, 0.05, 0.6, 0.3, 0.02]


def compress(q, probs=COMPRESS_P):
    body = dict(TASKS["compress"], questions={"ctx": dict(TASKS["compress"]["questions"]["ctx"], **q)})
    _, plans = tasks.expand(body)
    out = response({f"ctx::u::{i}": noul(p) for i, p in enumerate(probs)})
    return tasks.finish(out, body, plans, CAL)["answers"]["ctx"]


def test_finish_compress_keeps_the_most_probable_lines_up_to_the_ratio():
    res = compress({})
    state = TASKS["compress"]["state"]
    assert res["kept"] == [1, 2, 5, 6]  # 149 of 337 characters is below half: the fourth line too
    assert res["compressed"] == (
        "Najemca płaci czynsz 4 500 zł miesięcznie.\nCzynsz płatny jest z góry do 10. dnia "
        "każdego miesiąca.\nZa każdy dzień zwłoki naliczane są odsetki ustawowe.\n"
        "Kaucja wynosi dwukrotność czynszu."
    )
    assert res["original_chars"] == len(state) and res["compressed_chars"] == len(res["compressed"])
    assert res["ratio"] == round(len(res["compressed"]) / len(state), 4) and res["unit"] == "line"
    assert res["units"][2] == {"start": state.index("Czynsz"), "end": state.index("Lokal") - 1, "probability": 0.95}


@pytest.mark.parametrize(
    "q, kept",
    [
        ({"keep_ratio": None, "threshold": 0.5}, [1, 2, 5]),
        ({"keep_ratio": None, "threshold": 0.99}, [2]),  # min_keep 1: the most probable line
        ({"keep_ratio": None, "threshold": 0.99, "min_keep": 0}, []),
        ({"keep_ratio": None, "threshold": 0.25, "min_keep": 2}, [1, 2, 5, 6]),
        ({"keep_ratio": 1}, list(range(8))),
    ],
)
def test_finish_compress_by_threshold_and_min_keep(q, kept):
    res = compress(q)
    assert res["kept"] == kept
    if not kept:
        assert res["compressed"] == "" and res["ratio"] == 0


def extract_plan(schema, state):
    body = {
        "state": state,
        "questions": {"e": {"type": "extract", "schema": {"properties": schema}, "max_error": [0.05]}},
    }
    return body, tasks.expand(body)[1]


def line_answer(body, path, key, p=0.9):
    crit = tasks.expand(body)[0]["questions"][f"e::f::{path}"]["criteria"]
    probs = {k: (p if k == key else (1 - p) / (len(crit) - 1)) for k in crit}
    return choice(key, probs)


STATE = "Numer: FV/1\n---\nKwota: brak danych\nIlość: około 3 szt.\nRazem 99"


def test_extract_second_round_asks_only_where_a_line_has_several_candidates():
    schema = {
        "num": {"type": "string"},
        "sep": {"type": "string"},
        "amount": {"type": "number"},
        "qty": {"type": "integer"},
        "total": {"type": "integer"},
        "gone": {"type": "string"},
        "flag": {"type": "boolean"},
    }
    body, plans = extract_plan(schema, STATE)
    out = response(
        {
            "e::f::num": line_answer(body, "num", "L0"),
            "e::f::sep": line_answer(body, "sep", "L1"),
            "e::f::amount": line_answer(body, "amount", "L2"),
            "e::f::qty": line_answer(body, "qty", "L3"),
            "e::f::total": line_answer(body, "total", "L4"),
            "e::f::gone": line_answer(body, "gone", "__missing__", 0.7),
            "e::f::flag": choice("__missing__", {"true": 0.1, "false": 0.1, "__missing__": 0.8}),
        }
    )
    second = tasks.second_round(out, dict(body, lang="pl", facts="auto", media=[1]), plans, max_options=6)
    assert set(second) == {"state", "lang", "facts", "questions"}
    assert set(second["questions"]) == {"e::s::num", "e::s::amount", "e::s::qty", "e::s::total"}
    cands = plans["e"]["cands"]
    assert cands["sep"] == [] and "gone" not in cands and "flag" not in cands
    assert len(second["questions"]["e::s::amount"]["criteria"]) == 6  # 5 candidates (max_options - 1) + missing
    crit = second["questions"]["e::s::qty"]["criteria"]
    j = next(k for k, v in crit.items() if v == "około 3 szt.")
    a2 = {
        "e::s::num": choice("S1", {"S0": 0.2, "S1": 0.8}),
        "e::s::amount": choice("S1", {"S1": 0.7}),
        "e::s::qty": choice(j, {j: 0.6}),
        "e::s::total": choice("__missing__", {"S0": 0.1, "__missing__": 0.9}),
    }
    res = tasks.finish(out, body, plans, CAL, a2)["answers"]["e"]
    f = res["fields"]
    assert res["data"] == {
        "num": "FV/1",
        "sep": None,
        "amount": None,
        "qty": 3,
        "total": None,
        "gone": None,
        "flag": None,
    }
    assert f["sep"] == {
        "value": None,
        "present_probability": pytest.approx(0.98),
        "source": "span",
        "confidence": pytest.approx(0.02),
        "automated": {"0.05": False},
        "calibration": "per decision",
    }  # "---" has no candidate: read as not stated
    assert f["amount"]["error"] == "not a number" and f["amount"]["text"] == "brak danych"
    assert f["amount"]["automated"] == {"0.05": False}
    assert f["qty"]["confidence"] == pytest.approx(0.9 * 0.6) and f["qty"]["span_probability"] == 0.6
    assert f["total"] == {
        "value": None,
        "present_probability": pytest.approx(0.98),
        "source": "span",
        "confidence": 0.9,
        "automated": {"0.05": True},
        "calibration": "per decision",
    }
    assert f["gone"]["confidence"] == 0.7 and f["gone"]["present_probability"] == pytest.approx(0.3)
    assert f["flag"]["value"] is None and f["flag"]["source"] == "choice"


def test_extract_one_candidate_is_the_value_without_a_second_round():
    body, plans = extract_plan({"city": {"type": "string"}, "zip": {"type": "string"}}, "Gdańsk\n80-001")
    out = response(
        {"e::f::city": line_answer(body, "city", "L0", 0.8), "e::f::zip": line_answer(body, "zip", "L1", 0.6)}
    )
    assert tasks.second_round(out, body, plans) is None
    f = tasks.finish(out, body, plans, CAL)["answers"]["e"]["fields"]
    assert f["city"]["value"] == "Gdańsk" and f["city"]["span_probability"] == 1.0
    assert f["city"]["confidence"] == 0.8 and f["city"]["automated"] == {"0.05": True}
    assert f["zip"]["text"] == "80-001" and (f["zip"]["start"], f["zip"]["end"]) == (7, 13)


def test_extract_without_second_round_candidates_reads_the_line_as_not_stated():
    body, plans = extract_plan({"city": {"type": "string"}}, "Gdańsk")
    out = response({"e::f::city": line_answer(body, "city", "L0", 0.8)})
    f = tasks.finish(out, body, plans, CAL)["answers"]["e"]["fields"]["city"]  # second_round was not run
    assert f["value"] is None and f["confidence"] == pytest.approx(0.2)


def test_second_round_skips_other_tasks_and_non_text_fields():
    plans = {
        "c": {"type": "classify"},
        "e": {
            "type": "extract",
            "lang": "en",
            "pre": "",
            "regions": None,
            "fields": [("ok", "boolean", {}), ("t", "enum", {})],
        },
    }
    assert tasks.second_round(response({}), {"state": None}, plans) is None
    assert plans["e"]["cands"] == {}


def test_finish_extract_enum_list_and_boolean_values():
    schema = {
        "tags": {"type": "array", "items": {"enum": ["urgent", 7]}},
        "paid": {"type": "boolean"},
        "kind": {"enum": ["a", 2]},
    }
    body, plans = extract_plan(schema, "")
    out = response(
        {
            "e::f::tags": {
                "selected": ["7", "urgent"],
                "probabilities": {"urgent": 0.9, "7": 0.6},
                "set_confidence": 0.54,
            },
            "e::f::paid": choice("false", {"true": 0.2, "false": 0.75, "__missing__": 0.05}),
            "e::f::kind": choice("2", {"a": 0.1, "2": 0.85, "__missing__": 0.05}),
        }
    )
    res = tasks.finish(out, body, plans, CAL)["answers"]["e"]
    assert res["data"] == {"tags": [7, "urgent"], "paid": False, "kind": 2}
    assert res["fields"]["tags"]["automated"] == {"0.05": None} and res["fields"]["tags"]["confidence"] == 0.54
    assert res["fields"]["paid"]["automated"] == {"0.05": True}
