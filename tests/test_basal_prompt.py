"""basal's prompt formats (src/basal_lite/_basal/prompt.py) with a fake ChatML tokenizer: the vectors of
tools/data/format2_vectors.jsonl rebuilt exactly, option codes, layouts, media messages and request errors.

    uv run pytest tests/test_basal_prompt.py
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from basal_lite._basal import prompt
from basal_lite._basal.prompt import CODES, PREFILL, RequestError

VECTORS = [
    json.loads(line)
    for line in (Path(__file__).parents[1] / "tools" / "data" / "format2_vectors.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
]
HEADS = {"pl": ("Stan:\n", "\n\nPytanie: ", "\nOpcje:\n"), "en": ("State:\n", "\n\nQuestion: ", "\nOptions:\n")}


class ChatML:
    """The Bielik chat template (`<s>`, `<|im_start|>role ... <|im_end|>`) without the tokenizer."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, msgs, **kw):
        self.calls.append(kw)
        turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return "<s>" + turns + "<|im_start|>assistant\n"


class CodeTok:
    """A tokenizer of one id per character of the prompt; the text after the answer prefill is one token when it is in
    `vocab`, else one per character. `glue`: tails that merge with the prompt's last character (a changed prefix)."""

    def __init__(self, vocab=None, glue=()):
        self.vocab = {c: 1000 + i for i, c in enumerate(CODES)} if vocab is None else vocab
        self.glue = set(glue)

    def ids(self, text):
        i = text.rindex(PREFILL) + len(PREFILL)
        head, tail = [ord(c) for c in text[:i]], text[i:]
        if tail in self.glue:
            return head[:-1] + [9999]
        return head + ([self.vocab[tail]] if tail in self.vocab else [ord(c) for c in tail])

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[self.ids(t) for t in text] if isinstance(text, list) else self.ids(text))


def split_prompt(text, lang):
    """The state and question of a rendered canonical prompt."""
    st, q, o = HEADS[lang]
    body = text.split(st, 1)[1]
    state, rest = body.rsplit(q, 1)
    return state, rest.rsplit(o, 1)[0]


@pytest.mark.parametrize("vec", VECTORS, ids=[v["id"] for v in VECTORS])
def test_render_rebuilds_every_format2_vector_exactly(vec):
    for p in vec["prompts"]:
        state, question = split_prompt(p["prompt"], p["lang"])
        assert prompt.render(ChatML(), state, question, p["options"], lang=p["lang"], fmt=2) == p["prompt"]


def test_render_codes_the_options_and_prefills_the_answer():
    tok = ChatML()
    out = prompt.render(tok, "Paczka nie dotarła.", "Czy zgłosić?", ["Tak", "Nie"])
    assert out.endswith("<|im_start|>assistant\n" + PREFILL)
    assert "Opcje:\nA. Tak\nB. Nie\n\nOdpowiedz w formacie" in out  # format 1: the answer instruction
    assert tok.calls == [dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)]


def test_render_picks_the_language_from_state_and_question():
    assert "Question: Late?" in prompt.render(ChatML(), "Order 5512 is late.", "Late?", ["Yes", "No"], fmt=2)
    assert "Pytanie: Czy spóźnione?" in prompt.render(ChatML(), "Order 5512", "Czy spóźnione?", ["Tak", "Nie"], fmt=2)
    assert "State:" in prompt.render(ChatML(), "Zamówienie", "Opóźnione?", ["A", "B"], lang="en", fmt=2)


def test_format1_has_no_option_count_check_but_ten_letters():
    out = prompt.render(ChatML(), "s", "q", ["one"])  # format 1 leaves the count to the caller
    assert "A. one" in out
    assert prompt.max_options(1) == 10 and prompt.max_options() == 10 and prompt.max_options(2) == 255


def test_format2_uses_two_character_codes_after_z():
    out = prompt.render(ChatML(), "s", "q", [f"o{i}" for i in range(28)], fmt=2)
    assert "Z. o25\nAA. o26\nAB. o27" in out


@pytest.mark.parametrize("n", [1, 256])
def test_format2_refuses_fewer_than_two_or_more_than_255_options(n):
    with pytest.raises(ValueError, match=rf"{n} options \(supported: 2..255\)"):
        prompt.render(ChatML(), "s", "q", ["x"] * n, fmt=2)
    with pytest.raises(ValueError, match=rf"{n} options"):
        prompt.media_messages("s", "q", ["x"] * n, ["image"], fmt=2)


@pytest.mark.parametrize(
    "layout, expected",
    [
        ("state_twice", "State:\nS\n\nState (again):\nS\n\nQuestion: Q?\nOptions:\nA. y\nB. n"),
        ("qfirst", "Question: Q?\n\nState:\nS\n\nQuestion: Q?\nOptions:\nA. y\nB. n"),
    ],
)
def test_layouts_repeat_the_state_or_ask_the_question_first(layout, expected):
    out = prompt.render(ChatML(), "S", "Q?", ["y", "n"], lang="en", fmt=2, layout=layout)
    assert f"<|im_start|>user\n{expected}<|im_end|>" in out


def test_user_template_of_each_format_language_and_layout():
    assert prompt.user_template(1, "en").endswith('Answer in the format {{"answer": "<letter>"}}.')
    assert prompt.user_template(2, "pl") == "Stan:\n{state}\n\nPytanie: {q}\nOpcje:\n{opts}"
    assert prompt.user_template(2, "pl", "qfirst").startswith("Pytanie: {q}\n\nStan:")
    with pytest.raises(ValueError, match="unknown layout 'sideways' \\(one of canonical, state_twice, qfirst\\)"):
        prompt.user_template(2, "en", "sideways")
    with pytest.raises(ValueError, match="layout qfirst needs prompt format 2"):
        prompt.user_template(1, "en", "qfirst")


@pytest.mark.parametrize(
    "text, lang", [("Zażółć gęślą jaźń", "pl"), ("Zamowienie bez ogonkow", "en"), ("", "en"), ("ŁÓDŹ", "pl")]
)
def test_lang_of_is_polish_only_with_diacritics(text, lang):
    assert prompt.lang_of(text) == lang


@pytest.mark.parametrize(
    "config, fmt, layout",
    [
        (None, 1, "canonical"),
        ({}, 1, "canonical"),
        ({"prompt": 2}, 2, "canonical"),
        ({"prompt": 2, "layout": "qfirst"}, 2, "qfirst"),
    ],
)
def test_model_format_and_layout_read_basal_json(tmp_path, config, fmt, layout):
    if config is not None:
        (tmp_path / "basal.json").write_text(json.dumps(config), encoding="utf-8")
    assert prompt.model_format(tmp_path) == fmt
    assert prompt.model_layout(str(tmp_path)) == layout


def test_model_format_and_layout_refuse_unknown_values(tmp_path):
    (tmp_path / "basal.json").write_text('{"prompt": 3, "layout": "upside_down"}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"unknown prompt format 3 \(supported: \[1, 2\]\)"):
        prompt.model_format(tmp_path)
    with pytest.raises(ValueError, match="unknown prompt layout 'upside_down'"):
        prompt.model_layout(tmp_path)


def test_media_messages_put_the_media_parts_where_the_state_goes():
    msgs = prompt.media_messages(
        "Dokument od klienta.", "Czy kwota przekracza 1000 zł?", ["Tak", "Nie"], ["image", "audio"]
    )
    assert msgs[0] == {"role": "system", "content": prompt.TEMPLATES_2["pl"]["system"]}
    assert msgs[1]["content"] == [
        {"type": "text", "text": "Stan:\n"},
        {"type": "image"},
        {"type": "audio"},
        {
            "type": "text",
            "text": "\nDokument od klienta.\n\nPytanie: Czy kwota przekracza 1000 zł?\nOpcje:\nA. Tak\nB. Nie",
        },
    ]


def test_media_messages_without_state_text_follow_the_media_directly():
    msgs = prompt.media_messages("", "Is it a laptop?", ["Yes", "No"], ["image"], lang="en")
    assert msgs[1]["content"][-1]["text"] == "\n\nQuestion: Is it a laptop?\nOptions:\nA. Yes\nB. No"
    fmt1 = prompt.media_messages("", "Q", ["only"], ["audio"], lang="en", fmt=1)
    assert fmt1[1]["content"][-1]["text"].endswith('A. only\n\nAnswer in the format {"answer": "<letter>"}.')


def test_letter_ids_reads_one_token_per_code_at_the_answer_position():
    text = prompt.render(ChatML(), "s", "q", ["a", "b", "c"], fmt=2)
    assert prompt.letter_ids(CodeTok(), text, 3, fmt=2) == [1000, 1001, 1002]
    assert prompt.code_ids(CodeTok(), text, 255, fmt=2) == list(range(1000, 1255))
    assert prompt.letter_ids(CodeTok(), text, 10) == list(range(1000, 1010))  # format 1: A-J


@pytest.mark.parametrize(
    "tok",
    [
        CodeTok(vocab={c: 1000 + i for i, c in enumerate(CODES) if c != "AB"}),  # "AB" is two tokens
        CodeTok(glue={"AA"}),  # "AA" changes the prompt's last token
        CodeTok(vocab={c: 1000 + i % 26 for i, c in enumerate(CODES)}),  # "AA" and "A" share an id
    ],
    ids=["split", "merged", "shared"],
)
def test_letter_ids_is_none_when_a_code_is_not_its_own_token(tok):
    text = prompt.render(ChatML(), "s", "q", ["x"] * 30, fmt=2)
    assert prompt.letter_ids(tok, text, 30, fmt=2) is None
    assert prompt.letter_ids(tok, text, 26, fmt=2) == list(range(1000, 1026))


@prompt.reads_request
def read_body(body, err=None):
    if err:
        raise err
    return body["questions"]["q"]["type"]


@pytest.mark.parametrize(
    "body, err, message",
    [
        ({}, None, "missing field 'questions'"),
        ({"questions": None}, None, "invalid request: 'NoneType' object is not subscriptable"),
        ({"questions": {"q": []}}, None, "invalid request: list indices must be integers or slices, not str"),
        ({}, ValueError("bad value"), "bad value"),
        ({}, AttributeError("no attr"), "invalid request: no attr"),
        ({}, IndexError("out of range"), "invalid request: out of range"),
        ({}, RequestError("as is"), "as is"),
    ],
)
def test_reads_request_turns_body_faults_into_request_errors(body, err, message):
    with pytest.raises(RequestError) as e:
        read_body(body, err)
    assert str(e.value) == message and isinstance(e.value, ValueError)


def test_reads_request_passes_results_and_other_errors_through():
    assert read_body({"questions": {"q": {"type": "noul"}}}) == "noul"
    assert read_body.__name__ == "read_body"
    with pytest.raises(ZeroDivisionError):
        read_body({}, ZeroDivisionError())
