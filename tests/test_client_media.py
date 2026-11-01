"""Images and audio (basal-omni): the request's media as PNG and float32 WAV content parts where the state goes."""

import base64
import io
import json
from pathlib import Path

import pytest
import soundfile
from fake_vllm import Server, decide, first_wins
from PIL import Image

from basal_lite import RequestError

MEDIA = Path(__file__).parent.parent / "tools" / "data" / "media"
CASES = {d.pop("id"): d for d in map(json.loads, (MEDIA / "przypadki.jsonl").read_text(encoding="utf-8").splitlines())}
OMNI = {"vision_config": {"hidden_size": 8}, "audio_config": {"hidden_size": 8}}
MM_USAGE = {"prompt_tokens_details": {"cached_tokens": 5, "multimodal_tokens": {"image": 280, "audio": 100}}}


def omni_dir(tmp_path, config=OMNI, basal=None, cal=None):
    (tmp_path / "basal.json").write_text(json.dumps(basal or {"prompt": 2}), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    proc = {"audio_seq_length": 750, "audio_ms_per_token": 40}
    (tmp_path / "processor_config.json").write_text(json.dumps(proc), encoding="utf-8")
    if cal is not None:
        (tmp_path / "CALIBRATION.json").write_text(json.dumps(cal), encoding="utf-8")
    return tmp_path


def request(case):
    body = json.loads(json.dumps(CASES[case]))
    for m in body["media"]:
        m["data"] = base64.b64encode((MEDIA / m.pop("file")).read_bytes()).decode()
    return body


def test_audio_and_image_go_before_the_state_in_the_request_order(tmp_path):
    s = Server(max_model_len=50, usage=MM_USAGE)  # media prompts are never truncated
    out = decide(request("mix-obraz-dzwiek"), s, model_dir=omni_dir(tmp_path))
    assert out["answers"]["q"]["choice"] == "uznac" and out["answers"]["q"]["calibration"] == "text"
    msgs = s.sent("/v1/chat/completions")[0]["messages"]
    head, audio, image, tail = msgs[1]["content"]
    assert head == {"type": "text", "text": "Stan:\n"}
    assert tail["text"].startswith("\nReklamacja: nagranie klienta i zdjęcie.\n\nPytanie: Co zrobić z reklamacją?")
    assert msgs[2] == {"role": "assistant", "content": '{"answer": "'}
    wav, sr = soundfile.read(io.BytesIO(base64.b64decode(audio["input_audio"]["data"])))
    assert audio["input_audio"]["format"] == "wav" and sr == 16000 and wav.ndim == 1 and len(wav) > 0
    url = image["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    png = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert png.mode == "RGB" and png.size == Image.open(MEDIA / "ekran.png").size
    assert out["usage"]["media"] == {"images": 1, "audio": 1, "detail": "default", "media_tokens": 380}
    assert out["usage"]["cached_tokens"] == 5 and "truncated" not in out["usage"]
    assert "mm_processor_kwargs" not in s.sent("/v1/chat/completions")[0]


def test_detail_high_asks_for_more_soft_tokens_per_image(tmp_path):
    s = Server()
    out = decide(request("img-faktura-high-score"), s, model_dir=omni_dir(tmp_path))
    assert s.sent("/v1/chat/completions")[0]["mm_processor_kwargs"] == {"max_soft_tokens": 1120}
    assert out["usage"]["media"] == {"images": 1, "audio": 0, "detail": "high", "media_tokens": 0}
    assert out["answers"]["q"]["type"] == "score"


def test_orders_auto_sends_the_media_again_for_the_reversed_order(tmp_path):
    for case, kwargs in (("img-faktura-noul", None), ("img-faktura-high-score", {"max_soft_tokens": 1120})):
        s = Server()
        out = decide(request(case), s, model_dir=omni_dir(tmp_path), orders="auto")
        chats = s.sent("/v1/chat/completions")
        assert out["usage"]["second_order_branches"] == 1 and len(chats) == 2
        assert chats[1]["messages"][1]["content"][1]["type"] == "image_url"
        assert chats[1].get("mm_processor_kwargs") == kwargs
    p = first_wins(4)
    assert list(out["answers"]["q"]["probabilities"].values()) == pytest.approx(
        [(x + y) / 2 for x, y in zip(p, p[::-1])]
    )


def test_media_with_an_empty_state_has_no_state_line(tmp_path):
    s = Server()
    out = decide(request("only-media-lang"), s, model_dir=omni_dir(tmp_path))
    assert out["answers"]["q"]["noul"] == pytest.approx(first_wins(2)[0])
    tail = s.sent("/v1/chat/completions")[0]["messages"][1]["content"][-1]["text"]
    assert tail.startswith("\n\nPytanie: Czy na zdjęciu jest laptop?") and tail.endswith("A. Tak\nB. Nie")


def test_unvalidated_calibration_wins_over_the_text_mark(tmp_path):
    md = omni_dir(tmp_path, cal={"orders": 2})
    out = decide(request("img-ekran-noul"), model_dir=md)
    assert out["answers"]["q"]["calibration"] == "unvalidated"


def test_media_refusals(tmp_path):
    with pytest.raises(RequestError, match="this model does not take media"):
        decide(request("img-ekran-noul"))
    gemma = OMNI | {"architectures": ["Gemma4UnifiedForConditionalGeneration"]}
    with pytest.raises(RequestError, match="not for Gemma 4 Unified models"):
        decide(request("img-ekran-noul"), model_dir=omni_dir(tmp_path, gemma))
    with pytest.raises(RequestError, match="canonical layout only \\(this client: qfirst\\)"):
        decide(request("img-ekran-noul"), model_dir=omni_dir(tmp_path, basal={"prompt": 2, "layout": "qfirst"}))
    md = omni_dir(tmp_path)
    with pytest.raises(RequestError, match="detail must be one of"):
        decide(request("img-ekran-noul") | {"detail": "ultra"}, model_dir=md)
    with pytest.raises(RequestError, match='media 0: type must be "image" or "audio"'):
        decide(request("err-bad-type"), model_dir=md)
    with pytest.raises(RequestError, match="this model does not take audio"):
        decide(request("aud-reklamacja-noul"), model_dir=omni_dir(tmp_path, {"vision_config": {"x": 1}}))
