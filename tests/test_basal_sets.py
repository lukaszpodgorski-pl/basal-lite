"""basal's set selection for `multi` (src/basal_lite/_basal/sets.py): the expected scores against brute-force
enumeration of every outcome, the four objectives, min / max, exclusive and fallback labels, and request checks.

    uv run pytest tests/test_basal_sets.py
"""

import itertools

import numpy as np
import pytest

from basal_lite._basal import sets


def brute(p, objective):
    """Expected score of every prefix k = 0..n by enumerating the 2^n outcomes."""
    n, out = len(p), []
    for k in range(n + 1):
        e = 0.0
        for y in itertools.product((0, 1), repeat=n):
            w = np.prod([pi if yi else 1 - pi for pi, yi in zip(p, y)])
            hit, pos = sum(y[:k]), sum(y)
            if objective == "exact":
                s = float(hit == k and pos == k)
            elif k == 0 and pos == 0:
                s = 1.0
            elif objective == "iou":
                s = hit / (k + pos - hit)
            else:
                s = 2 * hit / (k + pos)
            e += w * s
        out.append(e)
    return out


P10 = [0.97, 0.9, 0.81, 0.7, 0.55, 0.42, 0.3, 0.18, 0.07, 0.01]


@pytest.mark.parametrize("objective", ["exact", "iou", "f1"])
@pytest.mark.parametrize(
    "p", [[0.6], [0.9, 0.4], [0.8, 0.5, 0.2], P10, [1.0, 0.7, 0.0]], ids=["one", "two", "three", "ten", "certain"]
)
def test_expected_scores_match_enumeration_of_every_outcome(objective, p):
    assert sets.expected_scores(p, objective) == pytest.approx(brute(p, objective), abs=1e-9)


def test_expected_scores_clip_probabilities_and_handle_no_labels():
    assert sets.expected_scores([1.2, -0.1], "exact") == pytest.approx(sets.expected_scores([1.0, 0.0], "exact"))
    assert list(sets.expected_scores([], "iou")) == [1.0]
    assert list(sets.expected_scores([], "f1")) == [1.0]
    assert list(sets.expected_scores([], "exact")) == [1.0]


def test_expected_scores_stay_in_range_for_255_labels():
    p = np.linspace(0.999, 0.001, 255)
    for objective in ("iou", "f1"):
        s = sets.expected_scores(p, objective)
        assert len(s) == 256 and (s >= 0).all() and (s <= 1).all()


def test_expected_scores_refuse_unknown_objectives_and_too_many_labels():
    with pytest.raises(ValueError, match="unknown objective 'threshold'"):
        sets.expected_scores([0.5], "threshold")
    with pytest.raises(ValueError, match="257 labels \\(at most 256\\)"):
        sets.expected_scores([0.5] * 257, "f1")


PROBS = {"billing": 0.9, "delivery": 0.62, "returns": 0.3, "other": 0.1}


@pytest.mark.parametrize(
    "kw, selected",
    [
        ({}, ["billing", "delivery"]),
        ({"threshold": 0.25}, ["billing", "delivery", "returns"]),
        ({"max": 1}, ["billing"]),
        ({"threshold": 0.95}, []),
        ({"threshold": 0.95, "min": 1}, ["billing"]),
        ({"min": 3}, ["billing", "delivery", "returns"]),
    ],
)
def test_threshold_selects_labels_at_or_above_it_within_min_and_max(kw, selected):
    assert sets.select(PROBS, **kw) == (selected, None, None)


def test_threshold_exclusive_label_wins_when_more_probable_than_the_set():
    probs = dict(PROBS, general=0.95)
    sel, score, applied = sets.select(probs, exclusive=["general"])
    assert sel == ["general"] and score is None
    assert applied == {"exclusive": ["general"], "fallback": None, "chosen": "exclusive"}
    assert sets.select(dict(PROBS, general=0.7), exclusive=["general"])[0] == ["billing", "delivery"]
    assert sets.select(dict(PROBS, general=0.4), exclusive=["general"])[0] == ["billing", "delivery"]
    assert sets.select({"a": 0.2, "general": 0.6}, exclusive=["general"])[0] == ["general"]  # the set is empty
    assert sets.select(probs, exclusive=["general"], min=2)[0] == ["billing", "delivery"]  # a single label is too few


def test_threshold_fallback_only_when_nothing_else_is_selected():
    low = {"billing": 0.2, "delivery": 0.1, "nothing": 0.7}
    assert sets.select(low, fallback="nothing") == (
        ["nothing"],
        None,
        {"exclusive": [], "fallback": "nothing", "chosen": "fallback", "fallback_threshold": 0.5},
    )
    assert sets.select(low, fallback="nothing", fallback_threshold=0.8) == (
        [],
        None,
        {"exclusive": [], "fallback": "nothing", "chosen": "empty", "fallback_threshold": 0.8},
    )
    assert sets.select(dict(PROBS, nothing=0.99), fallback="nothing")[0] == ["billing", "delivery"]
    assert sets.select(low, fallback="nothing", max=0)[0] == []  # max 0: not even the fallback


@pytest.mark.parametrize("objective", ["exact", "iou", "f1"])
def test_other_objectives_return_the_best_prefix_and_its_expected_score(objective):
    sel, score, applied = sets.select(PROBS, objective)
    order = sorted(PROBS, key=PROBS.get, reverse=True)
    scores = brute([PROBS[k] for k in order], objective)
    best = 2 if objective == "exact" else int(np.argmax(scores))
    assert sel == order[:best] and score == pytest.approx(scores[best]) and applied is None


def test_min_and_max_bound_the_prefix_under_an_objective():
    assert sets.select(PROBS, "iou", min=3)[0] == ["billing", "delivery", "returns"]
    assert sets.select(PROBS, "f1", max=1)[0] == ["billing"]
    assert sets.select(PROBS, "exact", max=1)[0] == ["billing"]
    assert sets.select(PROBS, "exact", min=4)[0] == list(PROBS)
    assert sets.select({"a": 0.9}, "iou", min=5)[0] == ["a"]  # min beyond the labels: all of them


def test_objective_exclusive_label_competes_alone_with_the_best_set():
    probs = {"a": 0.55, "b": 0.5, "general": 0.9}
    sel, score, applied = sets.select(probs, "f1", exclusive=["general"])
    assert sel == ["general"] and applied["chosen"] == "exclusive"
    assert score == pytest.approx(sets._alone(probs, "general", "f1"))
    sel, _, applied = sets.select({"a": 0.95, "b": 0.9, "general": 0.1}, "iou", exclusive=["general"])
    assert sel == ["a", "b"] and applied == {"exclusive": ["general"], "fallback": None, "chosen": "labels"}
    assert sets.select(probs, "f1", exclusive=["general"], min=2)[0] == ["a", "b"]


def test_objective_fallback_scores_alone_or_uses_its_threshold():
    probs = {"a": 0.05, "b": 0.04, "nothing": 0.8}
    sel, score, applied = sets.select(probs, "iou", fallback="nothing")
    assert sel == ["nothing"] and applied == {
        "exclusive": [],
        "fallback": "nothing",
        "chosen": "fallback",
        "fallback_threshold": None,
    }
    assert score == pytest.approx(sets._alone(probs, "nothing", "iou"))
    assert sets.select(probs, "iou", fallback="nothing", fallback_threshold=0.9)[0] == []
    weak = {"a": 0.05, "b": 0.04, "nothing": 0.1}
    assert sets.select(weak, "f1", fallback="nothing")[0] == []  # alone it scores below the empty set
    sel, score, applied = sets.select(weak, "f1", fallback="nothing", fallback_threshold=0.05)
    assert sel == ["nothing"] and applied["fallback_threshold"] == 0.05


def test_alone_is_the_score_of_the_one_label_set():
    probs = {"a": 0.7, "b": 0.2, "c": 0.4}
    assert sets._alone(probs, "b", "iou") == pytest.approx(brute([0.2, 0.7, 0.4], "iou")[1])


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"objective": "best"}, "objective must be one of \\['threshold', 'exact', 'iou', 'f1'\\], got 'best'"),
        ({"exclusive": "a"}, "exclusive must be a list of label keys"),
        ({"exclusive": ["a", 1]}, "exclusive must be a list of label keys"),
        ({"exclusive": ["a", "z"]}, "exclusive labels \\['z'\\] are not among the criteria"),
        ({"fallback": "z"}, "fallback 'z' is not among the criteria"),
        ({"fallback_threshold": 0.5}, "fallback_threshold needs a fallback label"),
        ({"fallback": "a", "fallback_threshold": True}, "fallback_threshold must be a number in 0..1"),
        ({"fallback": "a", "fallback_threshold": "0.5"}, "fallback_threshold must be a number in 0..1"),
        ({"fallback": "a", "fallback_threshold": 1.5}, "fallback_threshold must be a number in 0..1"),
        ({"objective": "iou", "threshold": 0.5}, 'threshold applies only to objective "threshold"'),
        ({"objective": "iou", "min": -1}, "min must be a non-negative integer, got -1"),
        ({"objective": "f1", "max": True}, "max must be a non-negative integer, got True"),
        ({"objective": "f1", "max": 1.5}, "max must be a non-negative integer, got 1.5"),
        ({"objective": "exact", "min": 4}, "min 4 is more than the 3 labels"),
        ({"objective": "exact", "min": 2, "max": 1}, "min 2 > max 1"),
    ],
)
def test_validate_refuses_bad_constraints(kw, message):
    kw.setdefault("objective", "threshold")
    with pytest.raises(ValueError, match=f"question 'topics': {message}"):
        sets.validate("topics", keys=["a", "b", "c"], **kw)


@pytest.mark.parametrize(
    "kw",
    [
        {"objective": "threshold", "threshold": 0.3, "min": 7, "max": -1},  # threshold keeps the 1.5 handling
        {"objective": "iou", "min": 1, "max": 3, "exclusive": ["a"], "fallback": "c", "fallback_threshold": 0},
        {"objective": "f1", "exclusive": [], "fallback": "b", "fallback_threshold": 1},
        {"objective": "exact", "min": 3},
    ],
)
def test_validate_accepts_valid_constraints(kw):
    assert sets.validate("topics", keys=["a", "b", "c"], **kw) is None


def test_label_set_id_is_stable_and_bound_to_labels_and_fallback():
    a = sets.label_set_id(["x", "y"], ["Faktura", "Umowa"], None)
    assert a == sets.label_set_id(["y", "x"], ["Umowa", "Faktura"], None) and len(a) == 16
    assert a != sets.label_set_id(["x", "y"], ["Faktura", "Umowa"], "y")
    assert a != sets.label_set_id(["x", "y"], ["Faktura", "Pismo"], None)
