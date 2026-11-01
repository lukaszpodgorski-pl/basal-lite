"""basal's media decoding (src/basal_lite/_basal/media.py): the images and recordings of tools/data/media, plus
synthetic files for resampling, mixing and every refusal.

    uv run pytest tests/test_basal_media.py
"""

import base64
import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from basal_lite._basal import media

DATA = Path(__file__).parents[1] / "tools" / "data" / "media"
CASES = [json.loads(line) for line in (DATA / "przypadki.jsonl").read_text(encoding="utf-8").splitlines()]
BOTH = ("image", "audio")


def b64(raw):
    return base64.b64encode(raw).decode()


def entries(case):
    return [{"type": m["type"], "data": b64((DATA / m["file"]).read_bytes())} for m in case["media"]]


def wav(x, sr, fmt="WAV"):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format=fmt)
    return buf.getvalue()


def png_header(w, h):
    """A PNG whose header claims w x h pixels (no pixel data: Pillow reads the size without decoding)."""

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 1, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b"") + chunk(b"IEND", b"")


VALID = [c for c in CASES if all(m["type"] in BOTH for m in c["media"])]


@pytest.mark.parametrize("case", VALID, ids=[c["id"] for c in VALID])
def test_decode_reads_every_case_of_the_media_set_in_order(case):
    images, audio, order = media.decode(entries(case), BOTH, max_audio_seconds=30.0)
    assert order == [m["type"] for m in case["media"]]
    assert len(images) == order.count("image") and len(audio) == order.count("audio")
    for im, m in zip(images, [m for m in case["media"] if m["type"] == "image"]):
        assert im.mode == "RGB" and im.size == Image.open(DATA / m["file"]).size
    for x, m in zip(audio, [m for m in case["media"] if m["type"] == "audio"]):
        assert x.dtype == np.float32 and x.ndim == 1 and x.flags["C_CONTIGUOUS"]
        assert len(x) == sf.info(DATA / m["file"]).frames  # already mono 16 kHz: unchanged length


def test_decode_refuses_a_type_from_the_media_set_it_does_not_know():
    case = next(c for c in CASES if c["id"] == "err-bad-type")
    with pytest.raises(ValueError, match='media 0: type must be "image" or "audio"'):
        media.decode(entries(case), BOTH)


def test_a_data_url_is_read_as_its_base64_payload():
    raw = (DATA / "ksztalty.png").read_bytes()
    (im,), _, _ = media.decode([{"type": "image", "data": "data:image/png;base64," + b64(raw)}], ("image",))
    assert im.size == (600, 300)


def test_audio_is_mixed_to_mono_and_resampled_to_16_khz():
    t = np.arange(8000) / 8000
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.zeros_like(t)], axis=1).astype(np.float32)
    _, (x,), order = media.decode([{"type": "audio", "data": b64(wav(stereo, 8000))}], ("audio",))
    assert order == ["audio"] and x.dtype == np.float32 and len(x) == 16000
    assert np.abs(x).max() == pytest.approx(0.5, abs=0.02)  # the mean of a full channel and a silent one


def test_audio_resamples_from_44_1_khz_by_the_reduced_ratio():
    x = media._audio(wav(np.zeros(44100, dtype=np.float32), 44100, "FLAC"), 0, None)
    assert len(x) == 16000


def test_audio_longer_than_the_model_window_is_refused():
    raw = (DATA / "pogoda_pl.wav").read_bytes()  # 2.91 s
    assert len(media._audio(raw, 0, 2.91)) == 46560
    with pytest.raises(ValueError, match="media 3: 2.9 s of audio, more than the model's 2.5 s"):
        media._audio(raw, 3, 2.5)


def test_empty_audio_is_refused():
    with pytest.raises(ValueError, match="media 0: empty audio"):
        media._audio(wav(np.zeros((0, 1), dtype=np.float32), 16000), 0, 30)


def test_bytes_that_are_not_audio_are_refused():
    with pytest.raises(ValueError, match=r"media 1: not readable audio \(WAV, FLAC, OGG or MP3; \w+Error\)"):
        media._audio((DATA / "ekran.png").read_bytes(), 1, None)


@pytest.mark.parametrize(
    "raw, kind",
    [
        (b"not an image at all", "UnidentifiedImageError"),
        ((DATA / "faktura.png").read_bytes()[:2000], "OSError"),  # truncated: fails when the pixels are loaded
        (png_header(20000, 20000), "DecompressionBombError"),  # more than twice the limit: refused by Pillow
    ],
    ids=["garbage", "truncated", "bomb"],
)
def test_unreadable_images_are_refused_with_the_reason(raw, kind):
    with pytest.raises(ValueError, match=rf"media 2: not a readable image \({kind}\)"):
        media._image(raw, 2)


def test_an_image_over_the_pixel_limit_is_refused_before_decoding():
    with pytest.warns(Image.DecompressionBombWarning):
        with pytest.raises(ValueError, match="media 0: 8000x7000 pixels, more than 50,000,000"):
            media._image(png_header(8000, 7000), 0)


@pytest.mark.parametrize(
    "data, message",
    [
        (None, "media 0: data must be a base64 string"),
        ("", "media 0: data must be a base64 string"),
        (b"aGVsbG8=", "media 0: data must be a base64 string"),
        ("not base64!", "media 0: data is not valid base64"),
        ("aGVsbG8", "media 0: data is not valid base64"),
    ],
)
def test_raw_refuses_data_that_is_not_base64(data, message):
    with pytest.raises(ValueError, match=message):
        media._raw(data, 0)


def test_raw_decodes_base64_and_refuses_more_than_the_byte_limit():
    assert media._raw("aGVsbG8=", 0) == b"hello"
    assert media._raw("data:text/plain;base64,aGVsbG8=", 0) == b"hello"
    with pytest.raises(ValueError, match="data is not valid base64"):
        media._raw("data:aGVsbG8=", 0)  # "data:" without a comma is not a data URL
    with pytest.raises(ValueError, match="media 5: more than 20 MB"):
        media._raw("A" * (media.MAX_BYTES * 4 // 3 + 4), 5)


PNG = b64((DATA / "ksztalty.png").read_bytes())


@pytest.mark.parametrize(
    "items, kinds, message",
    [
        ({"type": "image", "data": PNG}, BOTH, "media must be a non-empty list"),
        ([], BOTH, "media must be a non-empty list"),
        ([{"type": "image", "data": PNG}] * 9, BOTH, "9 media items, at most 8 per request"),
        (["ksztalty.png"], BOTH, 'media 0: type must be "image" or "audio"'),
        ([{"data": PNG}], BOTH, 'media 0: type must be "image" or "audio"'),
        (
            [{"type": "image", "data": PNG}, {"type": "audio", "data": PNG}],
            ("image",),
            r"media 1: this model does not take audio \(it takes: image\)",
        ),
        ([{"type": "image", "data": PNG}], (), r"media 0: this model does not take image \(it takes: no media\)"),
        ([{"type": "image"}], BOTH, "media 0: data must be a base64 string"),
    ],
)
def test_decode_refuses_malformed_media_lists(items, kinds, message):
    with pytest.raises(ValueError, match=message):
        media.decode(items, kinds)


def test_decode_takes_up_to_eight_items():
    images, audio, order = media.decode([{"type": "image", "data": PNG}] * 8, ("image",))
    assert len(images) == 8 and audio == [] and order == ["image"] * 8


@pytest.mark.parametrize(
    "config, kinds",
    [
        ({"vision_config": {"x": 1}, "audio_config": {"y": 2}}, ("image", "audio")),
        ({"vision_config": {"x": 1}, "audio_config": None}, ("image",)),
        ({"audio_config": {"y": 2}}, ("audio",)),
        ({"text_config": {}}, ()),
    ],
)
def test_kinds_of_reads_the_encoders_of_a_model_config(config, kinds):
    assert media.kinds_of(config) == kinds


@pytest.mark.parametrize(
    "config, seconds",
    [
        ({"audio_seq_length": 750, "audio_ms_per_token": 40}, 30.0),
        ({"audio_seq_length": 750}, None),
        ({"audio_ms_per_token": 40}, None),
        ({}, None),
    ],
)
def test_audio_window_is_tokens_times_milliseconds(config, seconds):
    assert media.audio_window(config) == seconds


def test_detail_tokens_of_each_level_and_refusal():
    assert media.detail_tokens("default") == 280 and media.detail_tokens("high") == 1120
    with pytest.raises(ValueError, match="detail must be one of \\['default', 'high'\\], got 'ultra'"):
        media.detail_tokens("ultra")
