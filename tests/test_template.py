"""template.py: the Gemma 4 chat template a stock vLLM server needs, on fake model directories.

uv run pytest tests/test_template.py
"""

import json

import pytest

from basal_lite.template import HEAD, HEAD_THOUGHT, SPACE, THOUGHT, serve_template

GEMMA = "{{ bos_token }}<|turn>system\n" + SPACE[0] + "\n" + HEAD + "{{ content }}"


def model(tmp_path, template=None, config=True):
    if config:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
    if template is not None:
        (tmp_path / "chat_template.jinja").write_text(template, encoding="utf-8")
    return tmp_path


def test_a_directory_without_config_json_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no config.json"):
        serve_template(tmp_path)


def test_a_model_without_a_chat_template_keeps_its_own(tmp_path):
    assert serve_template(model(tmp_path)) is None


def test_a_non_gemma_template_is_left_alone(tmp_path):
    assert serve_template(str(model(tmp_path, "<|im_start|>{{ m['content'] }}<|im_end|>"))) is None


def test_gemma_drops_the_space_after_a_list_system_prompt(tmp_path):
    out = serve_template(model(tmp_path, GEMMA))
    assert SPACE[0] not in out and SPACE[1] in out
    assert HEAD in out and HEAD_THOUGHT not in out  # no empty thought block in this template: header unchanged


def test_gemma_with_a_thought_block_prefills_it_on_the_last_assistant_turn(tmp_path):
    out = serve_template(model(tmp_path, GEMMA + THOUGHT))
    assert out.count(HEAD_THOUGHT) == 1 and SPACE[1] in out
    assert "loop.last and not add_generation_prompt and not enable_thinking" in out


def test_gemma_without_the_space_pattern_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="list system space pattern"):
        serve_template(model(tmp_path, "<|turn>system\n{{ content }}"))


@pytest.mark.parametrize("heads", [0, 2])
def test_gemma_with_the_turn_header_missing_or_repeated_is_refused(tmp_path, heads):
    tpl = "<|turn>" + SPACE[0] + HEAD * heads + THOUGHT
    with pytest.raises(SystemExit, match="turn header"):
        serve_template(model(tmp_path, tpl))
