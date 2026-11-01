"""tools/check_copies.py: client.py's copied section against fake basal server.py files.

uv run pytest tests/test_tools_check_copies.py
"""

import ast
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROWS = (ROOT / "src" / "basal_lite" / "client.py").read_text(encoding="utf-8").splitlines()
START = next(i for i, r in enumerate(ROWS) if r.startswith("# ---- copied from basal/server.py"))
END = next(i for i, r in enumerate(ROWS) if r.startswith("# ---- end of the copy"))
COPY = "\n".join(ROWS[START : END + 1]) + "\n"

# what server.py has around the copies: imports, other functions, assignments the check leaves out
EXTRA = '''
import torch
from pathlib import Path


def load_model(path):
    return torch.load(path)


async def serve_forever():
    """Not copied."""
    return None


a = b = 1
Path.cwd = None

'''


def check(monkeypatch, tmp_path, capsys, server_src):
    server = tmp_path / "server.py"
    server.write_text(server_src, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_copies.py", str(server)])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(ROOT / "tools" / "check_copies.py"), run_name="check_copies")
    return e.value.code, capsys.readouterr().out, server


def copied():
    """(function names, single-name assignments) of the copied section, as check_copies counts them."""
    body = ast.parse(COPY).body
    funcs = [n.name for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    names = [n.targets[0].id for n in body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)]
    return funcs, names


def test_copies_identical_to_server_py_pass(monkeypatch, tmp_path, capsys):
    code, out, server = check(monkeypatch, tmp_path, capsys, EXTRA + COPY)
    n = sum(map(len, copied()))
    assert code == 0 and n > 0
    assert out == f"{n}/{n} copies match {server}\n"


def test_docstrings_and_comments_do_not_count_as_differences(monkeypatch, tmp_path, capsys):
    src = COPY.replace('"""', '"""Reworded by basal. ', 1) + "# a comment basal added\n"
    code, out, _ = check(monkeypatch, tmp_path, capsys, src)
    assert code == 0 and "differs" not in out


def test_a_changed_or_missing_copy_fails_with_its_name(monkeypatch, tmp_path, capsys):
    funcs, names = copied()
    gone, changed = funcs[0], names[0]
    src = COPY.replace(f"def {gone}(", f"def {gone}_renamed(") + f"{changed} = 'changed in basal'\n"
    code, out, server = check(monkeypatch, tmp_path, capsys, src)
    n = len(funcs) + len(names)
    assert code == 1
    assert f"differs: {gone}: not in server.py\n" in out
    assert f"differs: {changed}: differs from server.py\n" in out
    assert out.endswith(f"{n - 2}/{n} copies match {server}\n")
