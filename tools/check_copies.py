"""The functions and constants client.py copies from basal/server.py (between its "copied from" and "end of the
copy" markers) against the server.py they came from: the same code (docstrings and comments aside), or a list of what
differs. Run after every new basal release; exit 1 when a copy no longer matches.

    python tools/check_copies.py <basal checkout>/basal/server.py
"""
import ast
import sys
from pathlib import Path

server = Path(sys.argv[1])
native = Path(__file__).resolve().parents[1] / "src" / "basal_lite" / "client.py"


def defs(src, lines=None):
    """{name: normalised AST dump} of the top-level functions and assignments (within `lines`, 1-based, if given)."""
    out = {}
    for node in ast.parse(src).body:
        if lines and not lines[0] <= node.lineno <= lines[1]:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body[1:] if node.body and isinstance(node.body[0], ast.Expr) and \
                isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str) else node.body
            out[node.name] = ast.dump(node.args) + "".join(map(ast.dump, body))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.dump(node.value)
    return out


text = native.read_text(encoding="utf-8")
rows = text.splitlines()
start = next(i for i, r in enumerate(rows, 1) if r.startswith("# ---- copied from basal/server.py"))
end = next(i for i, r in enumerate(rows, 1) if r.startswith("# ---- end of the copy"))
mine, theirs = defs(text, (start, end)), defs(server.read_text(encoding="utf-8"))
bad = [n for n in mine if mine[n] != theirs.get(n)]
for n in bad:
    print(f"differs: {n}: {'not in server.py' if n not in theirs else 'differs from server.py'}")
print(f"{len(mine) - len(bad)}/{len(mine)} copies match {server}")
sys.exit(1 if bad else 0)
