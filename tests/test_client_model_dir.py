"""The client with a model directory: prompt format and layout (basal.json), calibration files, media encoders."""

import json

import pytest
from fake_vllm import Server, decide, first_wins

from basal_lite.client import load_calibration, temper
from basal_lite._basal.sets import label_set_id


def model_dir(tmp_path, basal=None, **files):
    """basal.json and the other JSON files: CALIBRATION_vllm_json=... writes CALIBRATION.vllm.json, config_json=...
    config.json."""
    if basal is not None:
        (tmp_path / "basal.json").write_text(json.dumps(basal), encoding="utf-8")
    for name, data in files.items():
        name = name.replace("_", ".") if name.startswith("CALIBRATION") else name.removesuffix("_json") + ".json"
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


NOUL = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Urgent?"}}}
CHOICE3 = {"state": "x", "questions": {"q": {"criteria": ["a", "b", "c"]}}}


def test_format_1_model_uses_letters_and_two_option_orders(tmp_path):
    s = Server()
    out = decide(NOUL, s, model_dir=model_dir(tmp_path))
    assert out["answers"]["q"]["noul"] == pytest.approx(0.5)
    chats = s.sent("/v1/chat/completions")
    assert len(chats) == 2 and len(s.sent("/tokenize")) == 1 + 10 + 1
    assert chats[0]["messages"][1]["content"].endswith('Answer in the format {"answer": "<letter>"}.')


def test_explicit_format_and_orders_override_the_model_directory(tmp_path):
    s = Server()
    out = decide(NOUL, s, model_dir=model_dir(tmp_path), fmt=2, orders=1)
    assert out["answers"]["q"]["noul"] == pytest.approx(first_wins(2)[0])
    assert len(s.sent("/tokenize")) == 1 + 255 + 1


def test_layout_of_basal_json_puts_the_question_first(tmp_path):
    s = Server()
    decide(NOUL, s, model_dir=model_dir(tmp_path, {"prompt": 2, "layout": "qfirst"}))
    assert s.sent("/v1/chat/completions")[0]["messages"][1]["content"].startswith("Question: Urgent?\n\nState:\nx")


def test_per_type_temperature_and_buckets_rescale_the_probabilities(tmp_path):
    cal = {"temperature_per_prim": {"choice": 2.0}, "temperature_per_bucket": {"choice": {"4-10": 3.0}}}
    md = model_dir(tmp_path, {"prompt": 2}, CALIBRATION_json=cal)
    p = list(decide(CHOICE3, model_dir=md)["answers"]["q"]["probabilities"].values())
    assert p == pytest.approx(temper(first_wins(3), 2.0))
    cal["temperature_per_bucket"]["choice"]["2-3"] = 0.5
    md = model_dir(tmp_path, CALIBRATION_json=cal)
    p = list(decide(CHOICE3, model_dir=md)["answers"]["q"]["probabilities"].values())
    assert p == pytest.approx(temper(first_wins(3), 0.5))


def test_certified_threshold_lets_a_confident_action_through(tmp_path):
    cal = {"thresholds": {"0.05": {"confidence": 0.6}}, "temperature_per_prim": {"noul": 1.0}}
    md = model_dir(tmp_path, {"prompt": 2}, CALIBRATION_json=cal)
    q = {"type": "act", "instructions": "Refund?", "costs": {"wrong": 1, "defer": 0.5}, "max_error": 0.05}
    ans = decide({"state": "x", "questions": {"q": q}}, model_dir=md)["answers"]["q"]
    assert ans["action"] == "true" and ans["deferred"] is False and "max_error" not in ans
    assert ans["calibration"] == "validated"
    cal["thresholds"]["0.05"]["confidence"] = 0.7
    md = model_dir(tmp_path, CALIBRATION_json=cal)
    ans = decide({"state": "x", "questions": {"q": q}}, model_dir=md)["answers"]["q"]
    assert ans["action"] == "defer" and ans["refused"] == "true" and "max_error" not in ans


def test_act_with_hidden_option_keys_counts_as_validated(tmp_path):
    md = model_dir(tmp_path, {"prompt": 2}, CALIBRATION_json={"temperature_per_prim": {"choice": 1.0}})
    q = {"type": "act", "instructions": "?", "criteria": {"a": "A", "b": "B"}, "costs": {"wrong": 1}}
    shown = decide({"state": "x", "questions": {"q": q}}, model_dir=md)["answers"]["q"]
    hidden = decide({"state": "x", "questions": {"q": q | {"option_keys": "hide"}}}, model_dir=md)["answers"]["q"]
    assert shown["calibration"] == "unvalidated" and hidden["calibration"] == "validated"


def test_multi_fallback_takes_the_threshold_fitted_for_its_label_set(tmp_path):
    keys = ["a", "b", "none"]
    sid = label_set_id(keys, keys, "none")
    cal = {"multi": {sid: {"fallback_threshold": 0.6}, "other": {"fallback_threshold": 0.1}}}
    md = model_dir(tmp_path, {"prompt": 2}, CALIBRATION_json=cal)
    q = {"type": "multi", "instructions": "?", "criteria": keys, "fallback": "none", "threshold": 0.7}
    ans = decide({"state": "x", "questions": {"m": q}}, model_dir=md)["answers"]["m"]
    assert ans["selected"] == ["none"]
    assert ans["constraints_applied"]["fallback_threshold"] == 0.6
    assert ans["constraints_applied"]["fallback_threshold_source"] == "calibration"
    q["criteria"] = ["a", "c", "none"]  # another label set: no stored threshold
    ans = decide({"state": "x", "questions": {"m": q}}, model_dir=md)["answers"]["m"]
    assert ans["selected"] == [] and ans["constraints_applied"]["fallback_threshold_source"] == "default"


def test_calibration_fitted_on_other_orders_marks_every_answer_unvalidated(tmp_path):
    cal = {"orders": 2, "thresholds": {"0.05": {"confidence": 0.1}}}
    md = model_dir(tmp_path, {"prompt": 2}, CALIBRATION_json=cal)
    q = {"type": "act", "instructions": "?", "costs": {"wrong": 1, "defer": 0.5}, "max_error": 0.05}
    out = decide({"state": "x", "questions": {"q": q, "n": {"type": "noul"}}}, model_dir=md)
    assert out["answers"]["q"]["action"] == "defer" and out["answers"]["q"]["max_error"]  # thresholds dropped
    assert {a["calibration"] for a in out["answers"].values()} == {"unvalidated"}


def test_no_calibration_file_is_an_empty_calibration(tmp_path):
    assert load_calibration(None, 1) == ({}, None)
    assert load_calibration(model_dir(tmp_path), 1) == ({}, None)


def test_calibration_file_preference_vllm_and_layout_first(tmp_path):
    md = model_dir(tmp_path, CALIBRATION_json={"n": "plain"}, CALIBRATION_vllm_json={"n": "vllm"})
    assert load_calibration(md, 1)[0] == {"n": "vllm"}
    md = model_dir(tmp_path, CALIBRATION_qfirst_json={"n": "qfirst", "fitted_on": {"layout": "qfirst"}})
    assert load_calibration(md, 1, "qfirst")[0]["n"] == "qfirst"
    md = model_dir(tmp_path, CALIBRATION_vllm_qfirst_json={"n": "vllm-qfirst", "layout": "qfirst"})
    assert load_calibration(md, 1, "qfirst") == ({"n": "vllm-qfirst", "layout": "qfirst"}, None)


def test_calibration_of_another_layout_keeps_its_temperatures_only(tmp_path):
    cal = {"temperature_per_prim": {"noul": 1.5}, "thresholds": {}, "multi": {}, "task_thresholds": {}}
    cal["fitted_on"] = {"layout": "state_twice", "orders": None}
    data, why = load_calibration(model_dir(tmp_path, CALIBRATION_json=cal), 1)
    assert data == {"temperature_per_prim": {"noul": 1.5}, "fitted_on": cal["fitted_on"]}
    assert why == "CALIBRATION.json was fitted on the state_twice layout, served with --layout canonical"


def test_calibration_of_other_orders_switches_to_its_o_file(tmp_path):
    md = model_dir(tmp_path, CALIBRATION_json={"orders": 2, "n": "o2"}, CALIBRATION_o1_json={"orders": 1, "n": "o1"})
    assert load_calibration(md, 1) == ({"orders": 1, "n": "o1"}, None)
    assert load_calibration(md, 2) == ({"orders": 2, "n": "o2"}, None)
    data, why = load_calibration(md, "2le4")
    assert data == {"orders": 2, "n": "o2"}
    assert why == "CALIBRATION.json was fitted on 2 option order(s), served with 2le4"


def test_calibration_o_file_fitted_on_yet_other_orders_is_unvalidated(tmp_path):
    md = model_dir(
        tmp_path,
        CALIBRATION_json={"fitted_on": {"orders": "auto"}, "thresholds": {"0.05": {}}},
        CALIBRATION_o1_json={"orders": "2le3", "thresholds": {"0.05": {}}},
    )
    assert load_calibration(md, "auto")[1] is None
    data, why = load_calibration(md, 1)
    assert data == {"orders": "2le3"}
    assert why == "CALIBRATION.o1.json was fitted on 2le3 option order(s), served with 1"


def test_think_calibration_is_read_from_the_model_directory(tmp_path):
    from basal_lite import BasalLite

    md = model_dir(tmp_path, {"prompt": 2})
    assert BasalLite("http://fake", "basal", model_dir=md).think_cal_name == "missing"
    md = model_dir(tmp_path, CALIBRATION_think_json={"temperature_per_prim": {"noul": 2.0}})
    b = BasalLite("http://fake", "basal", model_dir=md)
    assert (b.think_cal, b.think_cal_name) == ({"temperature_per_prim": {"noul": 2.0}}, "CALIBRATION.think.json")


def test_config_json_sets_the_media_the_model_takes(tmp_path):
    from basal_lite import BasalLite

    md = model_dir(tmp_path, {"prompt": 2}, config_json={"vision_config": {"x": 1}})
    b = BasalLite("http://fake", "basal", model_dir=md)
    assert b.media_kinds == ("image",) and b.max_audio_s is None
    cfg = {"vision_config": {"x": 1}, "audio_config": {"x": 1}}
    md = model_dir(tmp_path, config_json=cfg, processor_config_json={"audio_seq_length": 750, "audio_ms_per_token": 40})
    b = BasalLite("http://fake", "basal", model_dir=md)
    assert b.media_kinds == ("image", "audio") and b.max_audio_s == 30
    md = model_dir(tmp_path, config_json=cfg | {"architectures": ["Gemma4UnifiedForConditionalGeneration"]})
    b = BasalLite("http://fake", "basal", model_dir=md)
    assert b.media_kinds == () and "Gemma 4 Unified" in b.no_media
