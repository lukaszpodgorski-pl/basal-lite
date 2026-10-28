"""basal on a stock vLLM server: no basal server, no plugin, nothing installed into vLLM. The harness is this client.

    vllm serve Remek/basal-2.0-4.5B --served-model-name basal \\
        --max-logprobs 255 --logprobs-mode processed_logprobs --generation-config vllm --enable-prompt-tokens-details \\
        --chat-template-content-format string
    # Gemma 4 E2B / E4B (basal-omni-mini): also --kv-sharing-fast-prefill (1.64x on an RTX 3090, same answers)
    # Gemma 4 (basal-omni-*, text and media): --chat-template-content-format openai instead of string, the chat
    # template of template.py (--chat-template), --limit-mm-per-prompt and, for "detail": "high",
    # --trust-request-mm-kwargs. All flags for any model: `basal-lite flags <model>` (serve.py)

--chat-template-content-format string: the Gemma 4 template renders list content with a space after each text part;
vLLM auto-detects that template as list-content and would turn the string turns into lists (a different prompt:
basal-omni-mini, a score question over 4,300 tokens, dp 0.13 against basal-serve).

A decision is one chat completion per prompt: the system and user turn of the basal prompt format, the assistant turn
prefilled with {"answer": " (continue_final_message), one generated token restricted to the option codes
(allowed_token_ids) and their log-probabilities after that restriction (processed_logprobs), i.e. the softmax over the
option codes that every basal engine reads. The option code ids come from the server's /tokenize, once.

The request and answer are POST /v1/systemone's: every question type (choice, noul, score, multi, act, tool, classify,
verify, compress, extract), facts, lang, orders 1 / 2 / 2le<N>, media (images and audio of basal-omni). The rest of
basal-serve's logic is basal's own pure modules (prompt, tasks, tools, sets, facts, media: _basal/, numpy only) plus the
request -> branches -> answer functions of basal/server.py copied below (that module imports torch), --orders auto and the state truncation of
--context-overflow (on the server's /tokenize and /detokenize). Not here: evidence and early_exit (they need the
model's hidden states, which vLLM does not return).

State sharing: the prompts of a request share the state as their prefix and vLLM computes it once even when they are
sent together (RTX 3090, 5 questions over 1,500 tokens: 78% of prompt tokens from the cache either way). warm_first
(send the first prompt alone, then the rest) gave the same throughput and 28 ms more latency, so it is off by default.

    import asyncio
    from basal_lite import BasalLite
    b = BasalLite("http://localhost:8000", "basal", model_dir="/models/basal-2.0-4.5B")  # dir: format, calibration
    print(asyncio.run(b.decide({"state": "Klient: nie mogę się zalogować.", "questions": {"dept": {"type": "choice",
          "instructions": "Do którego działu?", "criteria": {"cards": "Karty", "online": "Bankowość"}}}})))
"""
import asyncio
import json
import math
import time
from pathlib import Path

import httpx

from ._basal import tasks
from ._basal.facts import inject as inject_facts
from ._basal.media import SAMPLE_RATE, audio_window, detail_tokens, kinds_of
from ._basal.media import decode as decode_media
from ._basal.prompt import FORMATS, MAX_OPTIONS, PREFILL, RequestError, lang_of, media_messages, model_format, \
    model_layout, reads_request, user_template
from ._basal.sets import label_set_id
from ._basal.sets import select as select_set
from ._basal.sets import validate as validate_set
from ._basal.tools import QUESTION as TOOL_Q
from ._basal.tools import QUESTION_SET as TOOL_SET_Q
from ._basal.tools import answer as tool_answer
from ._basal.tools import parse as parse_tools
from ._basal.tools import set_answer as tool_set_answer

# ---- copied from basal/server.py 2.4.1 (Apache-2.0), unchanged ------------------------------------------------------


def _text(x):
    """Strings as they are; structured values (objects, lists, numbers) as compact JSON."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def named_options(crit, option_keys="show"):
    """{key: description} -> option texts shown to the model ("key: description"; "hide": the description alone
    unless it is not unique)."""
    if option_keys not in ("show", "hide"):
        raise RequestError(f'option_keys must be "show" or "hide", got {option_keys!r}')
    texts = [str(k) if v is None else _text(v) for k, v in crit.items()]
    dup = {t for t in texts if texts.count(t) > 1}
    shown = [t if v is None or (option_keys == "hide" and t not in dup) else f"{k}: {t}"
             for (k, v), t in zip(crit.items(), texts)]
    return list(crit), shown


MULTI_Q = {"pl": "{q}\nCzy dotyczy: „{label}”?", "en": "{q}\nDoes this apply: \"{label}\"?"}


def yes_no(lang, crit=None):
    crit = crit or {}
    return (["true", "false"], [_text(crit.get("true") or ("Tak" if lang == "pl" else "Yes")),
                                _text(crit.get("false") or ("Nie" if lang == "pl" else "No"))])


def check_body(body):
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    for k in ("state", "questions"):
        if k not in body:
            raise RequestError(f"missing field {k!r}")
    if not isinstance(body["questions"], dict) or not all(isinstance(q, dict) for q in body["questions"].values()):
        raise RequestError('"questions" must be an object of {name: question object}')


@reads_request
def to_items(state, questions, max_opts=MAX_OPTIONS, force_lang=None):
    """Questions of a request -> readout items with `branches` = [(question text, options)]: one for the decision
    types, one yes/no branch per label for multi (and tool with "multi": true)."""
    state = _text(state)
    out = []
    state_pl = not force_lang and lang_of(state) == "pl"
    for name, q in questions.items():
        t = q.get("type", "choice")
        instr = _text(q.get("instructions", ""))
        lang = force_lang or ("pl" if state_pl else lang_of(instr))
        extra = {}
        if t == "multi":
            crit = q.get("criteria") or {}
            if isinstance(crit, list):
                crit = {k: None for k in crit}
            if not crit:
                raise RequestError(f"question {name!r}: multi needs criteria")
            keys = list(crit)
            labels = [str(k) if v is None else _text(v) for k, v in crit.items()]
            yn = yes_no(lang)[1]
            branches = [(MULTI_Q[lang].format(q=instr, label=lab), yn) for lab in labels]
            objective = q.get("objective", "threshold")
            validate_set(name, objective, keys, q.get("threshold"), q.get("min"), q.get("max"), q.get("exclusive"),
                         q.get("fallback"), q.get("fallback_threshold"))
            extra = dict(threshold=float(q.get("threshold", 0.5)), max=q.get("max"), min=q.get("min"), labels=labels,
                         objective=objective, exclusive=q.get("exclusive") or [], fallback=q.get("fallback"),
                         fallback_threshold=q.get("fallback_threshold"),
                         fallback_threshold_source="request" if q.get("fallback_threshold") is not None else None)
            out.append(dict(name=name, type=t, keys=keys, state=state, question=instr, options=yn, lang=lang,
                            branches=branches, **extra))
            continue
        if t == "tool":
            keys, opts, names = parse_tools(name, q, lang)
            extra = dict(base="choice", tools=names, rank=q.get("rank"), max_error=q.get("max_error"),
                         multi=bool(q.get("multi")))
            if extra["multi"]:
                instr = instr or TOOL_SET_Q[lang]
                objective = q.get("objective", "threshold")
                validate_set(name, objective, keys, q.get("threshold"), q.get("min"), q.get("max"), q.get("exclusive"),
                             q.get("fallback"), q.get("fallback_threshold"))
                yn = yes_no(lang)[1]
                excl = list(dict.fromkeys([k for k in keys if k not in names] + (q.get("exclusive") or [])))
                extra.update(base="noul", threshold=float(q.get("threshold", 0.5)), max=q.get("max"), min=q.get("min"),
                             labels=opts, objective=objective, exclusive=excl, fallback=q.get("fallback"),
                             fallback_threshold=q.get("fallback_threshold"),
                             fallback_threshold_source="request" if q.get("fallback_threshold") is not None else None)
                out.append(dict(name=name, type=t, keys=keys, state=state, question=instr, options=yn, lang=lang,
                                branches=[(MULTI_Q[lang].format(q=instr, label=o), yn) for o in opts], **extra))
                continue
            instr = instr or TOOL_Q[lang]
        elif t == "act":
            crit = q.get("criteria") or {}
            if isinstance(crit, dict) and set(crit) <= {"true", "false"} or q.get("base") == "noul":
                keys, opts = yes_no(lang, crit if isinstance(crit, dict) else None)
                base = "noul"
            else:
                base = q.get("base", "choice")
                if isinstance(crit, list):
                    crit = {k: None for k in crit}
                keys, opts = named_options(crit, q.get("option_keys", "show"))
            extra = dict(base=base, act=act_spec(name, q, keys))
        elif t == "noul":
            keys, opts = yes_no(lang, q.get("criteria"))
        elif t == "score":
            crit = q.get("criteria") or q.get("levels") or []
            if isinstance(crit, dict):
                keys, opts = named_options(crit, q.get("option_keys", "show"))
            else:
                keys, opts = [str(i) for i in range(len(crit))], [_text(v) for v in crit]
        else:
            crit = q.get("criteria") or {}
            if isinstance(crit, list):
                crit = {k: None for k in crit}
            keys, opts = named_options(crit, q.get("option_keys", "show"))
        if not 2 <= len(opts) <= max_opts:
            what = " (tools and extra options)" if t == "tool" else ""
            raise RequestError(f"question {name!r}: {len(opts)} options{what} (supported: 2..{max_opts})")
        out.append(dict(name=name, type=t, keys=keys, state=state, question=instr, options=opts, lang=lang,
                        branches=[(instr, opts)], option_keys=q.get("option_keys", "show"), **extra))
    return out


def act_spec(name, q, outcomes):
    """Cost matrix C[action][outcome] from the full form or the short form {"wrong": W, "defer": D}."""
    costs = q.get("costs")
    if not isinstance(costs, dict) or not costs:
        raise RequestError(f"question {name!r}: act needs costs")
    if set(costs) <= {"wrong", "defer"} and not isinstance(costs.get("wrong"), dict):
        w = float(costs["wrong"])
        C = {y: {z: 0.0 if z == y else w for z in outcomes} for y in outcomes}
        if "defer" in costs:
            C["defer"] = {z: float(costs["defer"]) for z in outcomes}
        return dict(costs=C, defer="defer" if "defer" in costs else None, max_error=q.get("max_error"))
    for a, row in costs.items():
        if not isinstance(row, dict) or set(row) != set(outcomes):
            raise RequestError(f"question {name!r}: costs[{a!r}] must give a cost for every outcome {outcomes}")
    C = {a: {z: float(v) for z, v in row.items()} for a, row in costs.items()}
    defer = q.get("defer_action") or next((a for a in ("defer", "human") if a in C), None)
    return dict(costs=C, defer=defer, max_error=q.get("max_error"))


NO_CERTIFICATE = "no certified threshold for this type at this max_error"


def certified(cal, max_error, q_type):
    """The certified confidence threshold of a max_error target for a question type, else None (deferred)."""
    entry = (cal.get("thresholds") or {}).get(str(float(max_error))) or {}
    if "per_type" not in entry and cal.get("prompt", 1) == 1:
        return entry.get("confidence")
    return ((entry.get("per_type") or {}).get(q_type) or {}).get("confidence")


def act_answer(q, probs, conf, cal):
    """Minimum expected cost action; with max_error, automatic actions below the certified threshold are refused."""
    spec = q["act"]
    exp = {a: sum(probs[z] * c for z, c in row.items()) for a, row in spec["costs"].items()}
    action = min(exp, key=exp.get)
    ans = {"type": "act", "action": action, "expected_costs": {a: round(v, 6) for a, v in exp.items()},
           "answer": max(probs, key=probs.get), "probabilities": probs, "confidence": conf}
    if spec["max_error"] is not None:
        thr = certified(cal, spec["max_error"], q["base"])
        if thr is None:
            ans["max_error"] = NO_CERTIFICATE
        if (thr is None or conf < thr) and action != spec["defer"]:
            if spec["defer"] is None:
                raise RequestError("max_error needs a defer action (\"defer\"/\"human\" or defer_action)")
            ans["action"], ans["refused"] = spec["defer"], action
        ans["deferred"] = ans["action"] == spec["defer"]
    validated = bool(cal.get("temperature_per_prim") or cal.get("temperature_per_bucket")) and (
        q["base"] == "noul" or q.get("option_keys") == "hide")
    ans["calibration"] = "validated" if validated else "unvalidated"
    return ans


def multi_answer(q, p_yes):
    probs = dict(zip(q["keys"], p_yes))
    objective = q.get("objective", "threshold")
    sel, score, applied = select_set(probs, objective, q["threshold"], q["min"], q["max"], q.get("exclusive"),
                                     q.get("fallback"), q.get("fallback_threshold"))
    conf = 1.0
    for k, v in probs.items():
        conf *= v if k in sel else 1 - v
    if objective == "threshold":
        ans = {"type": "multi", "selected": sel, "probabilities": probs, "threshold": q["threshold"],
               "set_confidence": conf}
    else:
        ans = {"type": "multi", "selected": sel, "probabilities": probs, "objective": objective,
               "expected_score": score, "set_confidence": conf}
    if applied is not None:
        if q.get("fallback") is not None:
            applied["fallback_threshold_source"] = q.get("fallback_threshold_source") or "default"
        ans["constraints_applied"] = applied
    return ans


def stored_fallback(cal, q):
    """A multi fallback without its own threshold takes the one fitted for the same label set, if any."""
    if q.get("fallback") is None or q.get("fallback_threshold") is not None:
        return
    entry = (cal.get("multi") or {}).get(label_set_id(q["keys"], q["labels"], q["fallback"])) or {}
    if entry.get("fallback_threshold") is not None:
        q["fallback_threshold"], q["fallback_threshold_source"] = float(entry["fallback_threshold"]), "calibration"


def temperature(cal, q_type, k):
    """Calibrated temperature of a question type with k options (bucket, else per type, else 1)."""
    for b, T in ((cal.get("temperature_per_bucket") or {}).get(q_type) or {}).items():
        lo, hi = (int(x) for x in b.split("-"))
        if lo <= k <= hi:
            return float(T)
    return float((cal.get("temperature_per_prim") or {}).get(q_type, 1.0))


def orders_for(orders, k):
    if orders == 2:
        return 2
    if isinstance(orders, str) and orders.startswith("2le"):
        return 2 if k <= int(orders[3:]) else 1
    return 1


TRUNC_MARK, TRUNC_MARGIN = "\n[…]\n", 32  # --context-overflow truncate: the cut in the state, spare tokens

# ---- end of the copy ------------------------------------------------------------------------------------------------


def answer_of(q, pl, cal):
    """Probabilities in the canonical order -> the answer of a single-readout question (Server.assemble, think.answer)."""
    probs = dict(zip(q["keys"], pl))
    conf = max(pl)
    if q["type"] == "act":
        return act_answer(q, probs, conf, cal)
    if q["type"] == "tool":
        return tool_answer(q, probs, conf, cal)
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs["true"], "probabilities": probs, "confidence": conf}
    if q["type"] == "score":
        return {"type": "score", "score": sum(i * v for i, v in enumerate(pl)),
                "legend": dict(zip(q["keys"], q["options"])), "probabilities": probs, "confidence": conf}
    return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": conf}


THINK_TYPES, OPEN, CLOSE = ("choice", "noul", "score", "act", "tool"), "<think>\n", "</think>"  # basal/think.py
THINK_MAX_ERROR = 0.05  # a question thinks below its type's certified threshold for this max_error


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) if x > -1e30 else 0.0 for x in xs]
    s = sum(e)
    return [v / s for v in e]


def temper(p, T):
    """p_cal = softmax(log(max(p, 1e-12)) / T) (FORMAT2.md 5.2)."""
    return p if T == 1.0 else softmax([math.log(max(v, 1e-12)) / T for v in p])


def _fitted(cal, key, default=None):
    """server.py calibration_layout / calibration_orders: fitted_on.<key>, else a top-level <key>."""
    fitted = cal.get("fitted_on")
    v = fitted.get(key) if isinstance(fitted, dict) else None
    return v if v is not None else cal.get(key, default)


def load_calibration(md, orders, layout="canonical"):
    """server.py load_calibration for --mode vllm -> (calibration, None or why it is unvalidated):
    CALIBRATION[.vllm][.<layout>].json, the .o<orders> file when the default one was fitted on other option orders.
    A file fitted on another layout or other orders keeps its temperatures without its thresholds (max_error defers,
    tasks report automated: null) and the answers are marked "calibration": "unvalidated"."""
    if md is None:
        return {}, None
    sfx = [f".{layout}"] if layout != "canonical" else []
    names = [f"CALIBRATION.vllm{s}.json" for s in sfx] + [f"CALIBRATION{s}.json" for s in sfx] + \
        ["CALIBRATION.vllm.json", "CALIBRATION.json"]
    cal = next((md / n for n in names if (md / n).exists()), None)
    if cal is None:
        return {}, None
    data = json.loads(cal.read_text(encoding="utf-8"))
    strip = lambda d: {k: v for k, v in d.items() if k not in ("thresholds", "multi", "task_thresholds")}  # noqa: E731
    fitted_layout = _fitted(data, "layout") or "canonical"
    if fitted_layout != layout:
        return strip(data), f"{cal.name} was fitted on the {fitted_layout} layout, served with --layout {layout}"
    norm = lambda o: None if o is None else o if o == "auto" or str(o).startswith("2le") else int(o)  # noqa: E731
    fitted = norm(_fitted(data, "orders"))
    alt = cal.with_name(f"{cal.stem}.o{orders}{cal.suffix}")
    if fitted is not None and fitted != orders and alt.exists():
        cal, data = alt, json.loads(alt.read_text(encoding="utf-8"))
        fitted = norm(_fitted(data, "orders"))
    if fitted is None or fitted == orders:
        return data, None
    return strip(data), f"{cal.name} was fitted on {fitted} option order(s), served with {orders}"


class BasalLite:
    """basal decisions from a stock vLLM OpenAI server. think_url: a second one on the model's think/ weights (the
    think mode of basal-serve --think auto, think_tokens = --think-tokens). url: the server's root (http://host:8000; a trailing /v1 is
    dropped). model_dir: the model's directory (basal.json: prompt format and layout; CALIBRATION*.json); without it
    format 2 and no calibration. orders: 1, 2, "2le<N>" or "auto" (default as basal-serve: 1 for format 2, 2 for format
    1); auto_tau, auto_band: basal-serve's --auto-tau, --auto-band. overflow: "truncate" (as basal-serve's default
    --context-overflow: the middle of a too long state dropped, truncate_head of the kept tokens from its start) or
    "refuse"; the limit is the server's --max-model-len minus the generated token."""

    def __init__(self, url, model, model_dir=None, orders=None, fmt=None, layout=None, warm_first=False,
                 client=None, auto_tau=0.6, auto_band=(0.2, 0.8), overflow="truncate", truncate_head=1 / 3,
                 think_url=None, think_tokens=256, max_requests=None):
        md = Path(model_dir) if model_dir else None
        self.fmt = fmt or (model_format(md) if md else 2)
        self.layout = layout or (model_layout(md) if md else "canonical")
        self.orders = orders if orders is not None else (1 if self.fmt == 2 else 2)
        self.cal, self.cal_unvalidated = load_calibration(md, self.orders, self.layout)
        self.codes = FORMATS[self.fmt]["codes"]
        self.model, self.warm_first = model, warm_first
        self.url = url.rstrip("/").removesuffix("/v1")
        self.http = client or httpx.AsyncClient(timeout=300, limits=httpx.Limits(max_connections=256))
        self.ids, self._ids_lock = None, asyncio.Lock()
        # requests whose prompts are on the server at once: the questions of a request share its state through the
        # prefix cache only while they are scheduled together; past the KV cache (RTX 3090, 4.5B-FP8, 128 requests of
        # 5 questions) vLLM's queue splits them and the state is computed again (cached share 0.78 -> 0.53)
        self.slots = asyncio.Semaphore(max_requests) if max_requests else None
        self.tau, self.band = auto_tau, tuple(auto_band)
        self.overflow, self.trunc_head = overflow, truncate_head
        self.limit, self.overhead = None, None  # longest prompt (max_model_len - 1); template tokens of a prompt
        # think mode (basal-serve --think auto): a second vLLM server on <model>/think, CALIBRATION.think.json
        self.think = "auto" if think_url else "off"
        self.think_url, self.think_tokens, self.think_len = (think_url or "").rstrip("/").removesuffix("/v1"), \
            think_tokens, None
        tc = md / "CALIBRATION.think.json" if md else None
        self.think_cal, self.think_cal_name = (json.loads(tc.read_text(encoding="utf-8")), tc.name) \
            if tc and tc.exists() else ({}, "missing")
        self.media_kinds, self.max_audio_s = (), None  # basal-omni (Gemma 4): the image / audio encoders it has
        self.no_media = "this model does not take media (no image / audio encoder in config.json)"
        if md and (md / "config.json").exists():
            cfg = json.loads((md / "config.json").read_text(encoding="utf-8"))
            self.media_kinds = kinds_of(cfg)
            if "Gemma4UnifiedForConditionalGeneration" in (cfg.get("architectures") or ()):
                # as basal-serve --mode vllm --media on (engine.py VLLMBackend): vLLM reads their images and audio
                # differently from the model's processor
                self.media_kinds = ()
                self.no_media = "media on vLLM: not for Gemma 4 Unified models (basal-omni-1.0-max, 1.1-max)"
            if (md / "processor_config.json").exists():
                self.max_audio_s = audio_window(json.loads((md / "processor_config.json").read_text(encoding="utf-8")))

    def media(self, body):
        """The request's "media" (basal/media.py decode: the same checks, RGB images, 16 kHz mono audio) -> (content
        parts in the request's order, max_soft_tokens, the "detail"). Images go to the server as PNG and audio as
        float32 WAV, so it reads the same pixels and samples basal-serve hands to vLLM."""
        import base64
        import io

        if not self.media_kinds:
            raise RequestError(self.no_media)
        import soundfile
        if self.layout != "canonical":
            raise RequestError(f"images and audio are read in the canonical layout only (this client: {self.layout})")
        try:
            soft = detail_tokens(body.get("detail", "default"))
            images, audio, order = decode_media(body["media"], self.media_kinds, self.max_audio_s)
        except ValueError as e:
            raise RequestError(str(e)) from None
        parts, images, audio = [], iter(images), iter(audio)
        for kind in order:
            buf = io.BytesIO()
            if kind == "image":
                next(images).save(buf, format="PNG")
                parts.append({"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
            else:
                soundfile.write(buf, next(audio), SAMPLE_RATE, format="WAV", subtype="FLOAT")
                parts.append({"type": "input_audio", "input_audio": {
                    "data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}})
        return parts, soft, body.get("detail", "default")

    def media_chat(self, state, question, options, lang, parts):
        """basal/prompt.py media_messages with the request's media parts where it puts its placeholders: the media
        before the state text, in order. Needs the server's --chat-template-content-format openai (the parts reach the
        chat template as they are) and the Gemma 4 template without the space after a list system prompt."""
        kinds = [{"image_url": "image", "input_audio": "audio"}[p["type"]] for p in parts]
        msgs = media_messages(state, question, options, kinds, lang, self.fmt)
        media = iter(parts)
        msgs[1]["content"] = [p if p["type"] == "text" else next(media) for p in msgs[1]["content"]]
        return msgs + [{"role": "assistant", "content": PREFILL}]

    async def post(self, path, body):
        r = await self.http.post(self.url + path, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"{path}: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def messages(self, state, question, options, lang, prefill=PREFILL):
        """The chat of one option order, ending in the prefilled assistant turn (FORMAT2.md 3.1, 3.4, 3.5)."""
        opts = "\n".join(f"{self.codes[k]}. {o}" for k, o in enumerate(options))
        user = user_template(self.fmt, lang, self.layout).format(state=state, q=question, opts=opts)
        return [{"role": "system", "content": FORMATS[self.fmt]["templates"][lang]["system"]},
                {"role": "user", "content": user},
                {"role": "assistant", "content": prefill}]

    async def tokenize(self, msgs):
        r = await self.post("/tokenize", {"model": self.model, "messages": msgs, "add_generation_prompt": False,
                                          "continue_final_message": True, "add_special_tokens": False,
                                          "chat_template_kwargs": {"enable_thinking": False}})
        if self.limit is None and r.get("max_model_len"):
            self.limit = r["max_model_len"] - 1  # max_model_len holds the one generated token too
        return r["tokens"]

    async def code_ids(self):
        """Token id of every option code at the answer position (basal/prompt.py letter_ids, on the server's
        tokenizer and chat template): the prefilled prompt plus the code must be the prompt plus exactly one token."""
        async with self._ids_lock:
            if self.ids is None:
                msgs = self.messages("x", "x", ["x", "x"], "pl")
                base = await self.tokenize(msgs)
                fulls = await asyncio.gather(*(self.tokenize(self.messages("x", "x", ["x", "x"], "pl", PREFILL + c))
                                               for c in self.codes))
                for c, f in zip(self.codes, fulls):
                    if f[: len(base)] != base or len(f) != len(base) + 1:
                        raise RuntimeError(f"option code {c!r} is not one token at the answer position")
                self.ids = [f[-1] for f in fulls]
                if len(set(self.ids)) != len(self.ids):
                    raise RuntimeError("two option codes share a token")
                empty = [{"role": m["role"], "content": ""} for m in msgs[:2]] + msgs[2:]
                self.overhead = len(await self.tokenize(empty)) + 16  # template and prefill tokens, merges at the joins
        return self.ids

    async def fit(self, body, qs, branches, jobs):
        """basal-serve's ask_text for --context-overflow: the longest prompt of every branch; over the limit, the state
        without its middle (twice at most), else the refusal of check_context -> (qs, branches, jobs, truncation)."""
        await self.code_ids()
        if not self.limit:
            return qs, branches, jobs, None
        # a token is at least one UTF-8 byte of the text, so short prompts need no /tokenize
        if max(sum(len(m["content"].encode()) for m in j[0][:2]) for j in jobs) + self.overhead <= self.limit:
            return qs, branches, jobs, None
        longest = await self.longest(branches, jobs)
        truncation, orig, excess = None, qs[0]["state"], 0
        for _ in range(2):
            if self.overflow != "truncate" or max(longest) <= self.limit:
                break
            excess += max(longest) - self.limit
            state, t = await self.truncate_state(orig, excess)
            if t is None:
                break
            truncation = t
            qs, branches, jobs = self.prepare(body, cut=(state, [q["lang"] for q in qs]))[2:5]
            longest = await self.longest(branches, jobs)
        if max(longest) > self.limit:
            i = max(range(len(longest)), key=longest.__getitem__)
            raise RequestError(f"question {qs[branches[i][0]]['name']!r}: the prompt is {longest[i]} tokens (state, "
                               f"question and options), more than the {self.limit} tokens of vLLM's --max-model-len "
                               f"{self.limit + 1}; shorten the state, the question or the options")
        return qs, branches, jobs, truncation

    async def longest(self, branches, jobs):
        n = [len(t) for t in await asyncio.gather(*(self.tokenize(j[0]) for j in jobs))]
        out, j = [], 0
        for _, perms in branches:
            out.append(max(n[j: j + len(perms)]))
            j += len(perms)
        return out

    async def truncate_state(self, text, excess):
        """basal-serve's truncate_state on the server's tokenizer (/tokenize, /detokenize)."""
        ids = (await self.post("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False}))["tokens"]
        keep = len(ids) - excess - TRUNC_MARGIN
        if keep < TRUNC_MARGIN:
            return None, None
        head = int(round(keep * self.trunc_head))
        tail = keep - head
        dec = [(await self.post("/detokenize", {"model": self.model, "tokens": t}))["prompt"] if t else ""
               for t in (ids[:head], ids[-tail:] if tail else [])]
        return dec[0] + TRUNC_MARK + dec[1], {"state_tokens": len(ids), "head_tokens": head, "tail_tokens": tail,
                                              "dropped_tokens": len(ids) - keep, "head_share": self.trunc_head}

    def uncertain(self, q, perms, outs):
        """basal-serve's --orders auto test: yes/no branches inside the band, the others below tau (calibrated)."""
        t = "noul" if q["type"] == "multi" else q.get("base", q["type"])
        p = self.canonical(t, perms, outs)
        if t == "noul":
            return self.band[0] <= p[0] <= self.band[1]
        return max(p) < self.tau

    async def readout(self, msgs, k, extra=None):
        """One chat completion -> (softmax over the first k option codes at the answer position, usage). extra: more
        request fields (media: mm_processor_kwargs)."""
        ids = (await self.code_ids())[:k]
        r = await self.post("/v1/chat/completions", {**(extra or {}),
            "model": self.model, "messages": msgs, "max_tokens": 1, "temperature": 0.0, "logprobs": True,
            "top_logprobs": k, "allowed_token_ids": ids, "return_tokens_as_token_ids": True,
            "continue_final_message": True, "add_generation_prompt": False, "add_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.0})
        top = {int(t["token"].removeprefix("token_id:")): t["logprob"]
               for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        if not any(i in top for i in ids):
            raise RuntimeError("no option code among the returned log-probabilities (server flags: --max-logprobs "
                               "255 --logprobs-mode processed_logprobs)")
        return softmax([top.get(i, -math.inf) for i in ids]), r.get("usage") or {}

    def canonical(self, q_type, perms, outs):
        """Position probabilities of every option order -> mean in the canonical order, then the type temperature."""
        k = len(perms[0])
        p = [0.0] * k
        for perm, out in zip(perms, outs):
            for pos, j in enumerate(perm):
                p[j] += out[pos] / len(perms)
        return temper(p, temperature(self.cal, q_type, k))

    async def ask(self, jobs):
        """[(messages, k)] -> [(probabilities, usage)]; the first alone when warm_first (it leaves the shared state in
        vLLM's prefix cache for the others); with max_requests, after a slot is free."""
        if self.slots:
            async with self.slots:
                return await self._ask(jobs)
        return await self._ask(jobs)

    async def _ask(self, jobs):
        if self.warm_first and len(jobs) > 1:
            first = await self.readout(*jobs[0])
            return [first] + list(await asyncio.gather(*(self.readout(*j) for j in jobs[1:])))
        return list(await asyncio.gather(*(self.readout(*j) for j in jobs)))

    def prepare(self, body, cut=None):
        """A request body -> (original body, task plans, readout items, branches [(question index, perms)], jobs
        [(messages, k)]: one per branch and option order). cut: (truncated state, the languages of the whole one)."""
        check_body(body)
        if body.get("early_exit") is not None:
            raise RequestError("early_exit needs basal-serve")
        orig, plans = body, None
        if any(tasks.is_task(q) for q in body["questions"].values()):  # classify / verify / compress / extract
            body, plans = tasks.expand(body, len(self.codes))
        if any(q.get("evidence") for q in body["questions"].values()):
            raise RequestError("evidence needs basal-serve (evidence_head.pt)")
        if body.get("facts", "off") not in ("off", "auto"):
            raise RequestError('facts must be "off" or "auto"')
        state = body["state"]
        if body.get("facts") == "auto":
            state = inject_facts(_text(state))
        lang = body.get("lang")
        if lang is not None and lang not in ("pl", "en"):
            raise RequestError('lang must be "pl" or "en"')
        media = self.media(body) if body.get("media") is not None else None
        qs = to_items(cut[0] if cut else state, body["questions"], len(self.codes), force_lang=lang)
        for i, q in enumerate(qs):
            if cut:
                q["lang"] = cut[1][i]  # the language of the whole state
            stored_fallback(self.cal, q)
        # every branch of every question in its option orders: (question index, perms) and one job per order
        branches, jobs = [], []
        for i, q in enumerate(qs):
            for question, options in q["branches"]:
                k = len(options)
                perms = [list(range(k)), list(range(k))[::-1]][: orders_for(self.orders, k)]
                branches.append((i, perms))
                for perm in perms:
                    opts = [options[c] for c in perm]
                    if media is None:
                        jobs.append((self.messages(q["state"], question, opts, q["lang"]), k))
                    else:
                        # "default" = the processor's own max_soft_tokens (280); "high" needs the server's
                        # --trust-request-mm-kwargs (vLLM refuses per-request processor arguments without it)
                        extra = None if media[2] == "default" else {"mm_processor_kwargs": {"max_soft_tokens": media[1]}}
                        jobs.append((self.media_chat(q["state"], question, opts, q["lang"], media[0]), k, extra))
        return orig, plans, qs, branches, jobs, media

    async def decide(self, body):
        """POST /v1/systemone: a request body -> its answers, in basal-serve's response shape. With think_url, the
        think mode of basal/think.py (ThinkServer.decide) after readout 1."""
        if self.think == "off":  # basal-serve --think off: Server, which does not read "think"
            return await self.decide_base(body)
        mode = body.get("think", self.think) if isinstance(body, dict) else self.think  # else check_body refuses
        if mode not in ("off", "auto", "force"):
            raise RequestError('think must be "off", "auto" or "force"')
        out = await self.decide_base(body)
        t0 = time.perf_counter()
        todo = []
        if mode != "off" and not out["usage"].get("truncated"):
            state = body["state"]
            if body.get("facts") == "auto":  # the state readout 1 saw
                state = inject_facts(_text(state))
            todo = [q for q in to_items(state, body["questions"], len(self.codes), force_lang=body.get("lang"))
                    if self.routed(q, out["answers"][q["name"]], mode)]
        n_think = 0
        for q, (res, why) in zip(todo, await asyncio.gather(*(self.think_one(q) for q in todo))):
            old = out["answers"][q["name"]]
            if res is None:
                old["think"] = {"skipped": why}
                continue
            pl, notes, n, ms = res
            ans = answer_of(q, pl, self.think_cal)
            ans["think"] = {"tokens": n, "ms": ms, "calibration": self.think_cal_name,
                            "no_think": {"probabilities": old["probabilities"], "confidence": old["confidence"]}}
            if body.get("think_text"):
                ans["think"]["text"] = notes
            out["answers"][q["name"]] = ans
            out["usage"]["output_tokens"] += n
            n_think += 1
        out["usage"]["think_questions"] = n_think
        for a in out["answers"].values():  # every answer of a think server says whether it used the reasoning
            th = a.get("think") or {}
            a.setdefault("meta", {})["think"] = {"used": "tokens" in th, "tokens": th.get("tokens", 0)}
        out["usage"]["latency_ms"] = round(out["usage"]["latency_ms"] + (time.perf_counter() - t0) * 1000, 2)
        return out

    def routed(self, q, ans, mode):
        if q["type"] not in THINK_TYPES or q.get("multi") or len(q["branches"]) != 1:
            return False
        if mode == "force":
            return True
        thr = certified(self.cal, THINK_MAX_ERROR, q.get("base", q["type"]))
        return thr is None or ans["confidence"] < thr

    async def think_post(self, body):
        r = await self.http.post(self.think_url + "/v1/completions",
                                 json=dict(body, model=self.model, temperature=0.0, seed=0))
        if r.status_code != 200:
            raise RuntimeError(f"think server: {r.status_code} {r.text[:300]}")
        d = r.json()
        return d["choices"][0], d["usage"]

    async def text_ids(self, text):
        return (await self.post("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False}))[
            "tokens"]

    async def think_one(self, q):
        """basal/think.py think_one: greedy reasoning after <think> on the think server, then readout 2 over the state
        and the reasoning (order 1) -> ((calibrated probabilities, reasoning, tokens, ms), None) or (None, why). The
        prompt is basal's text up to the answer prefill (the chat on this server, decoded) plus the reasoning, tokenized
        as one text: a token of the prefill alone is not always the one inside the prompt (Llama: "{")."""
        t0 = time.perf_counter()
        k = len(q["options"])
        full = await self.tokenize(self.messages(q["state"], q["question"], q["options"], q["lang"]))
        text = (await self.post("/detokenize", {"model": self.model, "tokens": full}))["prompt"]
        if not text.endswith(PREFILL):
            raise RuntimeError("the decoded prompt does not end in the answer prefill")
        text = text[: -len(PREFILL)]
        head = await self.text_ids(text + OPEN)
        if self.think_len is None:
            r = await self.http.get(self.think_url + "/v1/models")
            self.think_len = r.json()["data"][0]["max_model_len"]
        if len(head) + self.think_tokens + 16 > self.think_len:
            return None, "context"
        c, u = await self.think_post({"prompt": head, "max_tokens": self.think_tokens, "stop": [CLOSE]})
        notes, n = c["text"], int(u["completion_tokens"])
        ids = await self.text_ids(text + OPEN + notes.strip() + "\n" + CLOSE + "\n" + PREFILL)
        codes = (await self.code_ids())[:k]
        c, _ = await self.think_post({"prompt": ids, "max_tokens": 1, "logprobs": k, "allowed_token_ids": codes,
                                      "return_tokens_as_token_ids": True})
        lp = {int(t.rsplit(":", 1)[1]): v for t, v in c["logprobs"]["top_logprobs"][0].items()}
        p = softmax([lp.get(i, -1e9) for i in codes])
        pl = temper(p, temperature(self.think_cal, q.get("base", q["type"]), k))
        return (pl, notes, n, round((time.perf_counter() - t0) * 1000, 2)), None

    async def decide_base(self, body):
        """Readout 1: basal-serve's Server.decide."""
        t0 = time.perf_counter()
        orig, plans, qs, branches, jobs, media = self.prepare(body)
        truncation, second = None, []
        if media is None:  # as basal-serve: media prompts are not truncated
            qs, branches, jobs, truncation = await self.fit(body, qs, branches, jobs)
        res = await self.ask(jobs)
        per_q, j = [[] for _ in qs], 0
        for i, perms in branches:
            per_q[i].append((list(perms), [p for p, _ in res[j: j + len(perms)]]))
            j += len(perms)
        if self.orders == "auto":  # the reversed option order, only for the branches uncertain in order 1
            second = [(i, bi) for i, brs in enumerate(per_q) for bi, br in enumerate(brs) if self.uncertain(qs[i], *br)]
            jobs2 = []
            for i, bi in second:
                question, options = qs[i]["branches"][bi]
                opts = [options[c] for c in per_q[i][bi][0][0][::-1]]
                if media is None:
                    jobs2.append((self.messages(qs[i]["state"], question, opts, qs[i]["lang"]), len(opts)))
                else:
                    extra = None if media[2] == "default" else {"mm_processor_kwargs": {"max_soft_tokens": media[1]}}
                    jobs2.append((self.media_chat(qs[i]["state"], question, opts, qs[i]["lang"], media[0]), len(opts),
                                  extra))
            res2 = await self.ask(jobs2)
            for (i, bi), (p, _) in zip(second, res2):
                perms, outs = per_q[i][bi]
                perms.append(perms[0][::-1])
                outs.append(p)
            res = res + res2
        answers = {}
        for q, brs in zip(qs, per_q):
            if q["type"] == "multi" or q.get("multi"):
                ans = multi_answer(q, [self.canonical("noul", perms, o)[0] for perms, o in brs])
                answers[q["name"]] = tool_set_answer(q, ans) if q["type"] == "tool" else ans
                continue
            perms, o = brs[0]
            answers[q["name"]] = answer_of(q, self.canonical(q.get("base", q["type"]), perms, o), self.cal)
        if self.cal_unvalidated:  # the calibration was fitted on other option orders or another layout
            for ans in answers.values():
                ans["calibration"] = "unvalidated"
        elif media is not None:  # the temperatures and certified thresholds were fitted on text, not on media
            for ans in answers.values():
                ans["calibration"] = "text"
        usage = [u for _, u in res]
        out = {"model": self.model, "answers": answers,
               "usage": {"input_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
                         "cached_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                                              for u in usage),
                         "output_tokens": 0, "questions": len(qs), "branches": len(branches),
                         "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                         "longest_prompt_tokens": max((u.get("prompt_tokens", 0) for u in usage), default=0)}}
        if self.orders == "auto":
            out["usage"]["second_order_branches"] = len(second)
        if truncation:
            out["usage"]["truncated"] = truncation
        if media is not None:
            kinds = [p["type"] for p in media[0]]
            out["usage"]["media"] = {
                "images": kinds.count("image_url"), "audio": kinds.count("input_audio"), "detail": media[2],
                "media_tokens": sum(sum(((u.get("prompt_tokens_details") or {}).get("multimodal_tokens") or {}).values())
                                    for u in usage)}
        if plans:  # task answers from their decisions; extract's spans need a second round
            r2, a2 = tasks.second_round(out, orig, plans, len(self.codes)), None
            if r2 is not None:
                out2 = await self.decide(r2)
                a2 = out2["answers"]
                for k in ("input_tokens", "cached_tokens", "branches"):
                    out["usage"][k] += out2["usage"][k]
            out = tasks.finish(out, orig, plans, self.cal, a2)
            out["usage"]["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out
