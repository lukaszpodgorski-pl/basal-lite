"""Check, with a model's own tokenizer, that a stock vLLM server sees exactly basal's prompt.

    docker run --rm -v <hf cache>:/hf:ro -v .:/repo:ro -w /repo -e PYTHONPATH=/repo/src --entrypoint python3 \\
        vllm/vllm-openai:v0.32.0 tools/check_prompt.py /hf/models--Remek--<model>/snapshots/<revision>

Run it with the server's transformers (the vLLM image's, CPU is enough): the server tokenizes with that version, and
another one can split the option codes differently (transformers 5.21: "NOT single tokens"; 5.19 in the image: fine).

1. For every prompt of every request of tools/data/format2_vectors.jsonl (basal's own, all question types), the chat of
   basal_lite, rendered as vLLM's chat endpoint does (apply_chat_template, continue_final_message, no special tokens added), gives
   the same token ids as basal's own prompt (basal/prompt.py render + add_special_tokens=False).
2. Every option code is one token at the answer position (basal/prompt.py letter_ids).
3. Think mode: basal_lite.think_one decodes the chat's token ids (/detokenize = tok.decode) back to basal's prompt
   text, cuts the answer prefill and tokenizes it with the reasoning, as basal/think.py does with render's text.
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

import basal_lite.client as bn
from basal_lite._basal.prompt import FORMATS, RequestError, letter_ids, render
from basal_lite.template import serve_template

DATA = Path(__file__).resolve().parent / "data"


def as_parts(msgs):
    """--chat-template-content-format openai: vLLM hands every string turn to the template as one text part."""
    return [dict(m, content=[{"type": "text", "text": m["content"]}]) for m in msgs]


def main(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    b = bn.BasalLite("http://unused", "basal", model_dir=model_dir, orders=1)
    served = serve_template(model_dir)  # Gemma 4: the server's --chat-template, else the model's own
    n = bad = skipped = nt = bad_t = 0
    for line in (DATA / "format2_vectors.jsonl").read_text(encoding="utf-8").splitlines():
        try:  # format 1 (basal-1.x): 10 option codes, the larger vectors are refused as by basal-serve
            items = bn.to_items(
                json.loads(line)["request"]["state"],
                json.loads(line)["request"]["questions"],
                len(FORMATS[b.fmt]["codes"]),
            )
        except RequestError:
            skipped += 1
            continue
        for q in items:
            for question, options in q["branches"]:
                want = tok(
                    render(tok, q["state"], question, options, q["lang"], b.fmt, b.layout), add_special_tokens=False
                ).input_ids
                msgs = b.messages(q["state"], question, options, q["lang"])
                text = tok.apply_chat_template(
                    as_parts(msgs) if served else msgs,
                    tokenize=False,
                    chat_template=served,
                    continue_final_message=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                got = tok(text, add_special_tokens=False).input_ids
                n += 1
                if got != want:
                    bad += 1
                    if bad <= 3:
                        print(
                            f"DIFFERENT prompt:\n  basal: {tok.decode(want[-12:])!r}\n  chat:  {tok.decode(got[-12:])!r}"
                        )
                if len(q["branches"]) == 1 and not served:  # 3. think (basal-2.0: Llama; Gemma has no think weights)
                    # /detokenize is tok.decode: the prompt text back from the chat's ids must be basal's own text
                    nt += 1
                    bad_t += tok.decode(got) != render(tok, q["state"], question, options, q["lang"], b.fmt, b.layout)
    codes = FORMATS[b.fmt]["codes"]
    ids = letter_ids(tok, render(tok, "x", "y", ["a", "b"], "en", b.fmt), len(codes), b.fmt)
    print(
        f"prompts: {n - bad}/{n} identical; option codes: {len(codes)} "
        f"{'single tokens' if ids else 'NOT single tokens'}"
        f"{f'; {skipped} requests over the option limit' if skipped else ''}"
        f"{'; server chat template: template.py' if served else ''}"
        f"{f'; think prompts: {nt - bad_t}/{nt} identical' if nt else ''}"
    )
    return 0 if ids and not bad and not bad_t else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
