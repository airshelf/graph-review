"""Every request to the Jev API must carry a User-Agent.

The vendor can reject Python's default `Python-urllib/3.x` with HTTP 403 even
when the same payload and key work with another client's User-Agent.

The failure is quiet in the worst way. Every caller fails open, so the workflow still
reports success and simply judges nothing. A vendor can turn on bot protection any
morning, so this is a wall rather than a note.

It reads the HEADER MAPPING, never just the file text. A whole-file search for
"User-Agent" would accept a comment saying the header is required even when the
real header was gone. The request data, not the commentary, must satisfy the gate.

Run: python3 -m pytest tests/test_jev_user_agent.py -q
"""
import ast
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "api.typesafe.ai"
EXPECTED = "graph-review/1.0 (+https://github.com/airshelf/graph-review)"


def _callers() -> list[Path]:
    """Scan this checkout, including files not yet tracked by Git."""
    callers = []
    for directory, subdirs, names in os.walk(ROOT):
        subdirs[:] = [d for d in subdirs if d not in {
            ".git", ".venv", ".state", ".pytest_cache", "__pycache__", "node_modules", "tests"
        }]
        for name in names:
            path = Path(directory) / name
            # A test that only asserts about itself proves nothing. Dependencies
            # and generated artifacts are not callers maintained by this repo.
            if (name.startswith("test_") or path.suffix not in {".py", ".ts", ".tsx"}
                    or path.is_symlink()):
                continue
            if ENDPOINT in path.read_text(encoding="utf-8"):
                callers.append(path)
    return sorted(callers)


def _auth_headers(path: Path) -> list[dict]:
    """Every dict literal in the file that carries an Authorization header.

    That dict IS the request's headers, so asking whether it also carries a User-Agent
    is the real question. A comment elsewhere in the file cannot satisfy it.
    """
    text = path.read_text()
    if path.suffix != ".py":
        # TS/TSX: take each object literal that mentions Authorization
        return [{"__raw": m} for m in
                re.findall(r"\{[^{}]*Authorization[^{}]*\}", text, re.S)]
    found = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Dict):
            continue
        # keep the VALUES, not just the keys: checking only that a User-Agent key
        # exists let a caller set the blocked `Python-urllib/3.12` and pass.
        # An f-string value reads as None and fails below.
        pairs = {k.value: (v.value if isinstance(v, ast.Constant) else None)
                 for k, v in zip(node.keys, node.values)
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if any(k.lower() == "authorization" for k in pairs):
            found.append(pairs)
    return found


def test_at_least_one_caller_is_found():
    """If the endpoint moves, this file must be updated rather than silently pass."""
    assert _callers(), f"no caller of {ENDPOINT} found -- did the endpoint change?"


def test_every_caller_builds_a_header_mapping_we_can_read():
    """A caller whose headers this cannot parse is unaudited, not compliant."""
    blind = [p.relative_to(ROOT) for p in _callers() if not _auth_headers(p)]
    assert not blind, ("no Authorization header mapping found in:\n  "
                       + "\n  ".join(str(b) for b in blind)
                       + "\nThis gate cannot audit them; teach it their shape.")


def test_every_authorization_header_also_sends_a_user_agent():
    """The VALUE matters, not just the key. `Python-urllib/3.12` is what the vendor
    blocks, so a caller setting it explicitly would be no better off."""
    bad = []
    for p in _callers():
        for headers in _auth_headers(p):
            raw = headers.get("__raw")
            ok = (EXPECTED in raw) if raw is not None else \
                 any(k.lower() == "user-agent" and v == EXPECTED
                     for k, v in headers.items())
            if not ok:
                bad.append(p.relative_to(ROOT))
    assert not bad, ("these send Authorization without a User-Agent, which the vendor "
                     "403s, or send one that is not ours:\n  "
                     + "\n  ".join(sorted({str(b) for b in bad}))
                     + f'\nEvery caller sends exactly: "User-Agent": "{EXPECTED}"')
