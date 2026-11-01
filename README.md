# basal-lite

**basal decisions from a stock vLLM server.** basal-lite is a small Python client that does everything
[basal](https://github.com/rkinas/basal)'s own server (`basal-serve`) does around the model: prompt format, option
orders, calibration, every question type, tasks, media and think mode. The server is a plain `vllm serve` with the
standard OpenAI API and no basal code at all.

```
your app ── basal request (JSON) ──► basal-lite ── /v1/chat/completions ──► vllm serve ──► GPU
                                     prompt, option orders,                  1 token, restricted
                                     calibration, answers                    to the option codes
```

Measured on an RTX 3090 against `basal-serve --mode vllm` on 10 basal models: the same decisions (differences only on
exact ties), the same prompt tokens, and up to 2.2x the throughput on images and audio.

## What basal is

basal (models [`Remek/basal-*`](https://huggingface.co/Remek) on Hugging Face) is a family of language models that do
not write text: they **make decisions**. You give a state (a customer message, a document, an image, a recording) and
questions with options; basal returns a calibrated probability for every option.

Under the hood every decision is one forward pass. The prompt lists the options with short codes (`A`, `B`, ...) and
the answer starts with `{"answer": "`. basal reads the probability of each option code at the next position, takes the
softmax over the codes only, and rescales it with a temperature fitted per question type (`CALIBRATION.json` in the
model repository), so that "0.9" means right nine times out of ten.

Question types: `choice`, `noul` (yes / no), `score`, `multi` (any number of labels), `act` (choose an action from a
cost matrix, defer when unsure), `tool` (which tool to call), and the tasks built on them: `classify`, `verify`,
`compress`, `extract`.

## Why this project exists

The official way to serve basal is `basal-serve`: about 12,000 lines of Python, 12 inference engines, pinned library
versions and a custom API (`POST /v1/systemone`). It works, but:

- **It owns the GPU process.** The vLLM engine runs inside basal's server, so the GPU serves basal and nothing else,
  and vLLM can only be upgraded together with basal.
- **Most of it is not needed for vLLM.** A basal decision is a single generated token with log-probabilities over a
  restricted set of tokens. vLLM's OpenAI server already does that (`allowed_token_ids`, `logprobs`,
  `--logprobs-mode processed_logprobs`), and it can batch, cache prefixes, and be monitored and scaled like any vLLM
  deployment.
- **The logic around the model is small and pure.** Building prompts, ordering options, averaging, calibrating and
  assembling answers needs no torch.

basal-lite moves that logic into a client of about 800 lines (plus basal's own pure modules, vendored) and leaves the
server stock. One vLLM server can then serve basal next to other applications, and vLLM can be upgraded on its own,
checked with one script (`tools/check_prompt.py`).

## Results

Live comparison on one RTX 3090 (24 GB), basal 2.4.1, vLLM 0.32.0: the same requests sent to `basal-serve --mode vllm`
and to basal-lite, every non-numeric field of every answer compared (`tools/compare.py`).

| model | orders 1 | orders 2 | orders 2le4 | media |
|---|---|---|---|---|
| basal-2.0-4.5B | 53/53 | 51/53 | 52/53 | no encoder |
| basal-2.0-4.5B-FP8 | 53/53 | 51/53 | 53/53 | no encoder |
| basal-2.0-mini-FP8 | 53/53 | 53/53 | 53/53 | no encoder |
| basal-2.0-max-FP8 | 52/53 | 52/53 | 52/53 | no encoder |
| basal-omni-pico | 53/53 | 53/53 | 53/53 | no encoder |
| basal-omni-1.0-mini | 52/53 | 51/53 | 53/53 | 16/16 |
| basal-omni-1.0-mini-FP8 | 53/53 | 51/53 | | 16/16 |
| basal-omni-1.0-max-FP8 | 52/53 | 52/53 | | refused on both sides |
| basal-1.5-4.5B (format 1) | 53/53 | 53/53 | | no encoder |

Every difference is a tie: options a few thousandths apart (synthetic questions with up to 255 near-identical
options), or a label sitting exactly on a decision threshold. The prompt length matched on every request, which shows
that both sides sent the same prompt. Details per model: [docs/MODELS.md](docs/MODELS.md).

Other features, compared the same way:

| feature | result |
|---|---|
| `--orders auto` (second option order only for uncertain branches) | 52/53 (one label on the band edge) |
| state longer than the context (`--context-overflow truncate`) | 10/10, truncation identical to the token |
| think mode, `think: auto` (basal-2.0-4.5B) | 53/53 |
| think mode, `think: force` | 50/52 (two reasonings diverge after 50-100 tokens) |

Throughput (requests in flight at once, decisions per second):

| workload | basal-serve | basal-lite |
|---|---|---|
| text, omni-1.0-mini, 1,500-token state | 8.6 | 8.3 |
| images, omni-1.0-mini, 16 in flight | 36.0 | 67.1 (1.9x) |
| audio, omni-1.0-mini, 16 in flight | 40.5 | 90.2 (2.2x) |

`basal-serve` runs media requests one at a time under a lock; a stock vLLM server batches them.

## How it works

A request is the body of basal-serve's `POST /v1/systemone`:

```json
{"state": "Customer: I cannot log in and I have to pay a bill tomorrow.",
 "questions": {
   "team":   {"type": "choice", "instructions": "Which team?",
              "criteria": {"cards": "Payment cards", "online": "Online banking", "loans": "Loans"}},
   "urgent": {"type": "noul", "instructions": "Is this urgent?"}}}
```

For each question the client:

1. **Builds basal's prompt** with basal's own prompt module (format 1 or 2 and the layout from the model's
   `basal.json`), as a chat: system turn, user turn, and an assistant turn prefilled with `{"answer": "`.
2. **Finds the option code tokens once** through the server's `/tokenize`, and checks that each code is exactly one
   token at the answer position.
3. **Asks vLLM for one token**: `max_tokens: 1`, `allowed_token_ids` = the option codes, `logprobs` for all of them,
   `continue_final_message: true`. With `--logprobs-mode processed_logprobs` the log-probabilities come after that
   restriction, so their softmax is exactly the readout every basal engine computes.
4. **Reads the options in one or two orders** (`orders`: 1, 2, `2le<N>` or `auto`) and averages the probabilities back
   in the canonical order, which removes the model's bias towards the first or last position.
5. **Calibrates and answers**: the temperature of the question type, certified thresholds for `max_error`, the cost
   matrix of `act`, label sets of `multi`, and the two-round logic of the tasks. The answer has basal-serve's shape.

Several questions over one state are separate requests sent in parallel. They share the state as a common prefix, so
vLLM's prefix cache computes it once (5 questions over 1,500 tokens: 78% of prompt tokens from the cache).

Beyond that:

- **State longer than the context.** As `basal-serve --context-overflow truncate`: the middle of the state is cut
  (1/3 of the kept tokens from the start, the rest from the end) on the server's own tokenizer through `/tokenize`
  and `/detokenize`.
- **Images and audio** (basal-omni, Gemma 4). Decoded and checked with basal's media module, sent to vLLM as PNG and
  float32 WAV in the places basal puts them. The server needs a fixed chat template, which `basal-lite serve` writes
  (`template.py`).
- **Think mode** (basal-2.0-4.5B, which ships `think/` weights). A second stock vLLM server on those weights: greedy
  reasoning up to `</think>`, then a second readout over the state and the reasoning with `CALIBRATION.think.json`,
  only for questions below their certified confidence (`auto`) or for all (`force`).

Not supported, because they need the model's hidden states, which vLLM's API does not return: `evidence` (basal-1.5
`evidence_head.pt`) and `early_exit`. The client refuses them the way `basal-serve --mode vllm` refuses what it cannot do.

## Quick start

Requirements: an NVIDIA GPU, Docker with GPU support, Python 3.10+. The server image `vllm/vllm-openai:v0.32.0` and the
model are pulled on first start.

```bash
uv sync                                      # or uv sync --extra media for images and audio
uv run basal-lite serve basal-omni-1.0-mini  # vLLM in Docker on http://localhost:8000, about 2-4 min to start
uv run basal-lite ask --model basal-omni-1.0-mini -f examples/request_en.json
```

The answer for `examples/request_en.json` from basal-omni-pico, the smallest model (rounded, trimmed):

```json
{"model": "basal",
 "answers": {
  "team":   {"type": "choice", "choice": "online",
             "probabilities": {"cards": 0.020, "online": 0.977, "loans": 0.002}, "confidence": 0.977},
  "urgent": {"type": "noul", "noul": 0.928, "probabilities": {"true": 0.928, "false": 0.072}, "confidence": 0.928},
  "mood":   {"type": "score", "score": 2.28, "legend": {"0": "calm", "1": "slightly worried", "2": "upset",
             "3": "very upset"}, "probabilities": {"0": 0.048, "1": 0.116, "2": 0.341, "3": 0.495}, "confidence": 0.495},
  "topics": {"type": "multi", "selected": ["login", "payment"], "threshold": 0.5, "set_confidence": 0.750,
             "probabilities": {"login": 0.964, "payment": 0.872, "card": 0.086, "complaint": 0.024}}},
 "usage": {"input_tokens": 688, "cached_tokens": 0, "questions": 4, "branches": 7, "longest_prompt_tokens": 103}}
```

From Python:

```python
import asyncio
from basal_lite import BasalLite
from basal_lite.serve import snapshot

b = BasalLite("http://localhost:8000", "basal", model_dir=snapshot("basal-omni-1.0-mini"))
out = asyncio.run(b.decide({"state": "...", "questions": {"q": {"type": "noul", "instructions": "Urgent?"}}}))
```

`model_dir` is the model's directory (the Hugging Face cache snapshot is fine): the client reads the prompt format from
`basal.json` and the calibration from `CALIBRATION*.json`. It must be the same model the server runs.

### Commands

| command | what it does |
|---|---|
| `basal-lite serve <model>` | starts `vllm serve` in Docker with the right flags; `--port`, `--gpu`, `--max-len`, `--network`, `--always` (restart with Docker) |
| `basal-lite serve basal-2.0-4.5B --think` | also the think server on port + 1 (two 4.5B models: about 22 GB) |
| `basal-lite ask` | one request from stdin or `-f`; `--orders`, `--think-url`, `--model-dir` |
| `basal-lite flags <model>` | prints the `vllm serve` flags for a model, to run vLLM yourself |
| `basal-lite stop` | removes the containers |

`<model>` is a name from `Remek/` (`basal-2.0-4.5B`), a full repo id, or a local directory.

### Client options

They mirror `basal-serve`'s flags:

| `BasalLite(...)` | `basal-serve` | default |
|---|---|---|
| `orders` | `--orders` (1, 2, `"2le<N>"`, `"auto"`) | 1 for format 2 models, 2 for format 1 |
| `auto_tau`, `auto_band` | `--auto-tau`, `--auto-band` | 0.6, (0.2, 0.8) |
| `overflow`, `truncate_head` | `--context-overflow`, `--truncate-head` | `"truncate"`, 1/3 |
| `think_url`, `think_tokens` | `--think auto`, `--think-url`, `--think-tokens` | off, 256 |
| `max_requests` | (none: basal-serve's queue keeps a request's questions together) | no limit |
| `warm_first` | (none) | off |

**`max_requests` under heavy load.** The questions of a request share their state through the prefix cache only while
vLLM schedules them together. With more requests in flight than the KV cache holds, vLLM's queue splits them and the
state is computed again: basal-2.0-4.5B-FP8 with 84k KV tokens, 5 questions over 1,500 tokens per request, gave
3.6 requests/s at 8-64 in flight but 1.7 at 128 (cached share 78% down to 53%). `max_requests=32` restored the full
rate. A rule of thumb: KV tokens (from the vLLM log) / (state + questions per request) / 1.5.

## Running vLLM yourself

`basal-lite flags <model>` prints what to add to `vllm serve <model> --served-model-name basal`. For Llama
(basal-1.x, basal-2.0) and Qwen3.5 (basal-omni-pico):

```
--max-logprobs 255 --logprobs-mode processed_logprobs --generation-config vllm \
--enable-prompt-tokens-details --chat-template-content-format string
```

| flag | why |
|---|---|
| `--max-logprobs 255` | basal-2.0 and omni take up to 255 options; the client reads the log-probability of each |
| `--logprobs-mode processed_logprobs` | log-probabilities after the restriction to the option codes, as basal reads them |
| `--generation-config vllm` | ignore the default temperature / top-p in the model's `generation_config.json` |
| `--chat-template-content-format string` | Gemma 4's template would otherwise put a space after a list system prompt |
| `--enable-prompt-tokens-details` | cached tokens in the usage (optional) |

Gemma 4 (basal-omni-*) additionally needs the server template (`--chat-template`, written by `basal-lite flags`),
`--chat-template-content-format openai`, `--limit-mm-per-prompt.*` for media, `--trust-request-mm-kwargs` for
`"detail": "high"`, and on E2B / E4B `--kv-sharing-fast-prefill` (1.64x faster on an RTX 3090, the same answers).

## Checks after a vLLM, model or basal update

| command | what it checks |
|---|---|
| `pytest` | the client against a fake server (no GPU) |
| `python tools/check_prompt.py <model dir>` | with the model's tokenizer: the chat the client sends renders exactly basal's prompt tokens (all 69 of basal's test vectors), option codes are single tokens, think prompts match |
| `python tools/check_copies.py <basal>/basal/server.py` | the functions copied from basal's server are unchanged |
| `python tools/vendor.py <basal checkout>` | refresh the vendored basal modules |
| `python tools/check_models.py <cache> out.json` | every `Remek/basal-*` model: prompt, architecture and quantization in this vLLM, calibration, media |
| `python tools/compare.py ref / lite / compare` | live decisions against `basal-serve` (text, media, long states, think mode) and media throughput (`tempo`) |

### Code copied from basal

basal-lite reuses basal's code instead of re-implementing it, in two places, and both are kept as exact text copies so
that a new basal release is a plain diff away:

- `src/basal_lite/_basal/` holds basal's pure modules byte for byte (`tools/vendor.py` writes them; one import line
  differs). Ruff skips the directory (`extend-exclude`).
- `src/basal_lite/client.py` copies the functions of `basal/server.py` that it needs (that module imports torch), between
  the `copied from basal/server.py` and `end of the copy` markers. The block sits inside `# fmt: off` / `# fmt: on`, so
  `ruff format` keeps basal's own line breaks, and E501 is ignored for the same reason. The one line that is not
  basal's text is `TRUNC_MARK, TRUNC_MARGIN`: a class attribute of basal's `Server`, a module constant here.

Why text and not only behaviour: `tools/check_copies.py` compares the syntax trees and passes however the code is
wrapped, but a reformatted copy turns every re-wrapped line into a diff line against the next basal release and hides
its real changes. To check a release, keep a basal checkout in `source/` (ignored by git):

    git clone https://github.com/rkinas/basal source/basal
    python tools/vendor.py source/basal && git diff src/basal_lite/_basal      # the modules
    python tools/check_copies.py source/basal/basal/server.py                    # the copied functions

Run `check_prompt.py` and `check_models.py` inside the vLLM image, so they use the server's own `transformers` (another
version can tokenize the option codes differently), with the repository mounted and `PYTHONPATH=/repo/src`:

```bash
docker run --rm -v ~/.cache/huggingface/hub:/hf:ro -v "$PWD":/repo:ro -w /repo -e PYTHONPATH=/repo/src \
  --entrypoint python3 vllm/vllm-openai:v0.32.0 tools/check_prompt.py /hf/models--Remek--basal-omni-pico/snapshots/<revision>
# prompts: 69/69 identical; option codes: 255 single tokens; think prompts: 41/41 identical
```

`compare.py ref` needs a running `basal-serve` from the basal repository.

## Project layout

```
src/basal_lite/
  client.py       BasalLite: request -> prompts -> vLLM -> answer (with the functions copied from basal's server)
  serve.py        vLLM in Docker, its flags, model download
  template.py     the Gemma 4 server chat template
  __main__.py     the basal-lite command
  _basal/         basal 2.4.1's pure modules, vendored (prompt, tasks, tools, sets, facts, media)
tools/            maintenance checks and the live comparison, data/ = their test requests and media
tests/            offline tests
examples/         requests (English, Polish) and a Python example
docs/MODELS.md    every basal model on basal-lite
```

## Versions

The version says which basal release it follows: the first three numbers are basal's, the fourth counts basal-lite
releases for it (git tags `v<version>`, as basal's own `v2.4.1`).

| basal-lite | basal (prompt format, calibration files, vendored code) | measured on vLLM |
|---|---|---|
| 2.4.1.1 | 2.4.1 | 0.32.0 |

## Status and limits

- Pinned to the basal 2.4.1 prompt format and calibration files, measured on vLLM 0.32.0. A new basal release means
  `tools/vendor.py` and `tools/check_copies.py`; a new vLLM release means `tools/check_prompt.py`.
- Each question branch is its own HTTP request. That is what lets vLLM batch them with other traffic, but it relies on
  the prefix cache to share the state (see `max_requests`).
- Hybrid models (basal-omni-pico) only reuse full 544-token cache blocks, so short states are not shared.
- `evidence` and `early_exit` need `basal-serve`.
- With think mode on, basal 2.4.1 itself refuses `classify`, `verify`, `compress` and `extract` (its think step reads
  the questions before the tasks expand them); basal-lite reproduces that.

## License and credits

Apache-2.0, see [LICENSE](LICENSE). basal and its models are the work of their authors
([rkinas/basal](https://github.com/rkinas/basal), [Remek on Hugging Face](https://huggingface.co/Remek)), also
Apache-2.0. `src/basal_lite/_basal/` is an unmodified copy of six basal modules (one import redirected), and
`client.py` contains functions copied from basal's `server.py`, marked in the source. basal-lite is an independent
project, not affiliated with or endorsed by basal's authors.
