"""The chat template a stock vLLM server needs for a Gemma 4 basal model (basal-omni-*), so that the chat the client sends
renders exactly basal's prompt. None for the other models: their own template is right (--chat-template-content-format
string). Two changes, both only where the model's template has the pattern:

1. no space after a system prompt given as a list (vLLM with --chat-template-content-format openai, needed for media,
   sends every turn as a list of parts);
2. 12B / 31B (1.0-max, 1.1-max, 1.0-ultra): their generation prompt has an empty thought block without thinking, and the
   template strips any thought block from an assistant turn, so the prefilled answer turn (continue_final_message) never
   gets it. It is added when the assistant turn is the last one.
"""

from pathlib import Path

SPACE = ("{{- item['text'] | trim + ' '-}}", "{{- item['text'] | trim -}}")
THOUGHT = "{{- '<|channel>thought\\n<channel|>' -}}"
HEAD = "{{- '<|turn>' + role + '\\n' }}"
HEAD_THOUGHT = HEAD + (
    "{%- if role == 'model' and loop.last and not add_generation_prompt and not enable_thinking -%}"
    + THOUGHT
    + "{%- endif -%}"
)


def serve_template(model_dir):
    """The server's chat template for this model, or None when the model's own template is right."""
    if not (Path(model_dir) / "config.json").exists():
        raise SystemExit(f"{model_dir}: no config.json")
    p = Path(model_dir) / "chat_template.jinja"
    tpl = p.read_text(encoding="utf-8") if p.exists() else ""
    if "<|turn>" not in tpl:
        return None
    if SPACE[0] not in tpl:
        raise SystemExit(f"{p}: a Gemma 4 template without the list system space pattern, check it by hand")
    tpl = tpl.replace(*SPACE)
    if THOUGHT in tpl:
        if tpl.count(HEAD) != 1:
            raise SystemExit(f"{p}: the turn header is not where expected, check it by hand")
        tpl = tpl.replace(HEAD, HEAD_THOUGHT)
    return tpl
