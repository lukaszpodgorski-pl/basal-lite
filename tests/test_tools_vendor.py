"""tools/vendor.py on a fake basal checkout; the copies go to tmp_path, never to src/basal_lite/_basal.

uv run pytest tests/test_tools_vendor.py
"""

import importlib.util
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def vendor(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("tools_vendor", TOOLS / "vendor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(mod, "OUT", out)
    return mod


def checkout(root, tools_src):
    d = root / "basal"
    d.mkdir(parents=True)
    for m in ("prompt", "tasks", "sets", "facts", "media"):
        (d / f"{m}.py").write_text(f"NAME = {m!r}\n", encoding="utf-8")
    (d / "tools.py").write_text(tools_src, encoding="utf-8")
    return root


def test_vendor_copies_every_module_and_points_certified_at_the_client(vendor, tmp_path, capsys):
    src = "from .server import NO_CERTIFICATE, certified\nfrom .prompt import render\n"
    vendor.main(str(checkout(tmp_path / "basal", src)))
    out = vendor.OUT
    assert sorted(p.stem for p in out.glob("*.py")) == sorted(vendor.MODULES)
    assert (out / "tools.py").read_text(encoding="utf-8") == (
        "from ..client import NO_CERTIFICATE, certified\nfrom .prompt import render\n"
    )
    assert (out / "media.py").read_text(encoding="utf-8") == "NAME = 'media'\n"
    assert capsys.readouterr().out == f"6 modules -> {out}\n"


def test_vendor_stops_on_any_other_import_of_basals_server(vendor, tmp_path):
    with pytest.raises(SystemExit, match="tools.py imports basal's server module"):
        vendor.main(checkout(tmp_path / "basal", "from .server import load_model\n"))
    assert list(vendor.OUT.iterdir()) == []  # not even prompt.py and tasks.py, checked before tools.py
