"""tools/check_prompt.py with a fake tokenizer (transformers is not installed): one token per character, an option code
after the answer prefill one token, and a chat template that renders a prefilled assistant turn as the generation prompt
plus the prefill, as Gemma's and Llama's templates do.

    uv run pytest tests/test_tools_check_prompt.py
"""

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

from basal_lite._basal.prompt import CODES, LETTERS, PREFILL
from basal_lite.template import SPACE

TOOLS = Path(__file__).resolve().parents[1] / "tools"
VOCAB = {c: 1_000_000 + i for i, c in enumerate(dict.fromkeys(CODES + tuple(LETTERS)))}


class Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    """codes_single: an option code after the prefill is one token (else one per character); drift: the chat endpoint's
    rendering of a prefilled turn differs from basal's prompt by one character."""

    def __init__(self, codes_single=True, drift=False):
        self.codes_single, self.drift = codes_single, drift
        self.inverse = {v: k for k, v in VOCAB.items()}

    def encode(self, text):
        head, sep, code = text.rpartition(PREFILL)
        if sep and self.codes_single and code in VOCAB:
            return [ord(c) for c in head + sep] + [VOCAB[code]]
        return [ord(c) for c in text]

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return Encoding([self.encode(t) for t in text] if isinstance(text, list) else self.encode(text))

    def decode(self, ids):
        return "".join(self.inverse.get(i) or chr(i) for i in ids)

    def apply_chat_template(
        self,
        msgs,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True,
        chat_template=None,
        continue_final_message=False,
    ):
        assert tokenize is False and enable_thinking is False
        text = lambda c: c if isinstance(c, str) else "".join(p["text"] for p in c)  # noqa: E731
        out = "".join(f"<|{m['role']}|>{text(m['content'])}<|end|>" for m in msgs)
        if continue_final_message:
            out = out.removesuffix("<|end|>") + ("!" if self.drift else "")
        return out + ("<|assistant|>" if add_generation_prompt else "")


def load(monkeypatch, tok):
    """check_prompt.py with a fake transformers whose AutoTokenizer gives `tok`."""
    seen = []
    auto = types.SimpleNamespace(from_pretrained=lambda d: seen.append(d) or tok)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=auto))
    spec = importlib.util.spec_from_file_location("tools_check_prompt", TOOLS / "check_prompt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, seen


def model(tmp_path, fmt=2, template=None):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    if fmt == 2:
        (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2}), encoding="utf-8")
    if template:
        (tmp_path / "chat_template.jinja").write_text(template, encoding="utf-8")
    return str(tmp_path)


def test_as_parts_hands_every_turn_as_one_text_part(monkeypatch):
    mod, _ = load(monkeypatch, FakeTokenizer())
    msgs = [{"role": "user", "content": "hi"}]
    assert mod.as_parts(msgs) == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert msgs == [{"role": "user", "content": "hi"}]


def test_format_2_prompts_and_think_prompts_match_basal(monkeypatch, tmp_path, capsys):
    mod, seen = load(monkeypatch, FakeTokenizer())
    md = model(tmp_path)
    assert mod.main(md) == 0
    assert seen == [md]
    m = re.fullmatch(
        r"prompts: (\d+)/(\d+) identical; option codes: 255 single tokens; think prompts: (\d+)/(\d+) identical\n",
        capsys.readouterr().out,
    )
    same, n, same_t, nt = map(int, m.groups())
    assert same == n and same_t == nt and 0 < nt < n  # think: one-branch questions only (multi: a branch per label)


def test_format_1_skips_the_requests_over_ten_options(monkeypatch, tmp_path, capsys):
    mod, _ = load(monkeypatch, FakeTokenizer())
    assert mod.main(model(tmp_path, fmt=1)) == 0
    out = capsys.readouterr().out
    assert "option codes: 10 single tokens" in out and " requests over the option limit" in out


def test_gemma_renders_the_parts_with_the_server_template_and_skips_think(monkeypatch, tmp_path, capsys):
    mod, _ = load(monkeypatch, FakeTokenizer())
    assert mod.main(model(tmp_path, template="<|turn>" + SPACE[0])) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith("single tokens; server chat template: template.py")


def test_option_codes_split_into_several_tokens_fail(monkeypatch, tmp_path, capsys):
    mod, _ = load(monkeypatch, FakeTokenizer(codes_single=False))
    assert mod.main(model(tmp_path)) == 1
    assert "option codes: 255 NOT single tokens" in capsys.readouterr().out


def test_different_prompts_fail_and_show_the_first_three(monkeypatch, tmp_path, capsys):
    mod, _ = load(monkeypatch, FakeTokenizer(drift=True))
    assert mod.main(model(tmp_path)) == 1
    out = capsys.readouterr().out
    assert out.count("DIFFERENT prompt:") == 3
    assert "prompts: 0/" in out and "think prompts: 0/" in out


@pytest.mark.parametrize("bad", [0, 1])
def test_a_think_prompt_that_does_not_decode_back_fails(monkeypatch, tmp_path, capsys, bad):
    tok = FakeTokenizer()
    if bad:
        tok.decode = lambda ids: "lossy"
    mod, _ = load(monkeypatch, tok)
    assert mod.main(model(tmp_path)) == bad
