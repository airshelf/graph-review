"""stale_guards.py as a command: the report a reviewer reads, its exit codes, and the
request it sends to the optional judge.

Unit-level detector tests live in test_stale_guards.py. These run main() and run() in-process
against a throwaway Git repository, and the judge against a faked HTTP layer, so nothing
leaves the machine and no key is needed.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "stale_guards.py"
spec = importlib.util.spec_from_file_location("sg_cli_tests", SCRIPT)
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)

BEFORE = """export interface Row {
  a: boolean;
  b: boolean;
}

export function keep(r: Row, f: Row): boolean {
  if (f.a && !r.a) return false;
  if (f.b && !r.b) return false;
  return true;
}
"""


def with_fields(*names):
    return BEFORE.replace("  b: boolean;\n", "  b: boolean;\n" + "".join(f"  {n}: boolean;\n" for n in names))


def _git(repo, *args):
    subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo,
                   capture_output=True, text=True, check=True)


def make_repo(tmp_path, monkeypatch, before, after):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "row.ts").write_text(before)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "row.ts").write_text(after)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "change")
    monkeypatch.chdir(repo)
    return repo


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["stale_guards.py", *argv])
    return sg.main()


# --------------------------------------------------------------- the report
def test_main_reports_the_guard_run_a_new_field_never_reached(tmp_path, monkeypatch, capsys):
    make_repo(tmp_path, monkeypatch, BEFORE, with_fields("c"))
    assert run_main(monkeypatch, "--base", "HEAD~1", "--head", "HEAD") == 0
    out = capsys.readouterr().out
    assert "row.ts  —  `c` new on Row" in out
    assert "row.ts:8-9  a 2-line guard run over `a`, `b`, with no case for `c`" in out
    assert "        8: if (f.a && !r.a) return false;" in out
    assert "        9: if (f.b && !r.b) return false;" in out
    assert out.rstrip().endswith("1 guard run(s) predate a new field in this diff")
    assert "p=" not in out, "no judge was asked, so no probability is shown"


@pytest.mark.parametrize("fields,names,wording", [
    (("c", "d"), "`c`, `d`", "either of them"),
    (("c", "d", "e"), "`c`, `d`, `e`", "any of them"),
])
def test_the_wording_follows_how_many_fields_arrived(
        tmp_path, monkeypatch, capsys, fields, names, wording):
    make_repo(tmp_path, monkeypatch, BEFORE, with_fields(*fields))
    run_main(monkeypatch, "--base", "HEAD~1")
    out = capsys.readouterr().out
    assert f"{names} new on Row" in out and f"with no case for {wording}" in out
    assert out.count("guard run over") == 1, "one run, one finding, however many fields arrived"


def test_a_diff_that_extends_the_whole_checklist_reports_nothing(tmp_path, monkeypatch, capsys):
    complete = with_fields("c").replace("  return true;", "  if (f.c && !r.c) return false;\n  return true;")
    make_repo(tmp_path, monkeypatch, BEFORE, complete)
    assert run_main(monkeypatch, "--base", "HEAD~1") == 0
    out = capsys.readouterr().out
    assert "0 guard run(s) predate a new field in this diff" in out and "guard run over" not in out


@pytest.mark.parametrize("bad", ["base", "head"])
def test_an_unresolvable_ref_exits_2_and_says_which(tmp_path, monkeypatch, capsys, bad):
    make_repo(tmp_path, monkeypatch, BEFORE, with_fields("c"))
    refs = {"base": "HEAD~1", "head": "HEAD", bad: "no-such-ref"}
    assert run_main(monkeypatch, "--base", refs["base"], "--head", refs["head"]) == 2
    captured = capsys.readouterr()
    assert "cannot resolve ref 'no-such-ref'" in captured.err and captured.out == ""


def test_judge_flag_without_a_key_exits_3_when_there_is_something_to_ask(
        tmp_path, monkeypatch, capsys):
    """review.py only passes --judge when a key exists and relies on this exit to fall back."""
    make_repo(tmp_path, monkeypatch, BEFORE, with_fields("c"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--base", "HEAD~1", "--judge")
    assert exc.value.code == 3 and "needs TYPESAFE_API_KEY" in capsys.readouterr().err


def test_judge_flag_without_a_key_exits_3_even_when_the_diff_is_clean(
        tmp_path, monkeypatch, capsys):
    """A missing key is a wiring fault. Its exit code must not depend on what the diff holds,
    or the fault stays hidden until the first finding."""
    complete = with_fields("c").replace("  return true;", "  if (f.c && !r.c) return false;\n  return true;")
    make_repo(tmp_path, monkeypatch, BEFORE, complete)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--base", "HEAD~1", "--judge")
    captured = capsys.readouterr()
    assert exc.value.code == 3 and "needs TYPESAFE_API_KEY" in captured.err
    assert captured.out == "", "no report is printed for a run that was asked to judge and could not"


# --------------------------------------------------------------- run() with a judge
TWO_TYPES = """export interface Foo {
  a: boolean;
  b: boolean;
}

export interface Bar {
  a: boolean;
  b: boolean;
}

export function keep(x: Foo, y: Bar): boolean {
  if (x.a && !y.a) return false;
  if (x.b && !y.b) return false;
  return true;
}
"""


def two_types_after():
    return (TWO_TYPES
            .replace("export interface Foo {\n  a: boolean;\n  b: boolean;\n",
                     "export interface Foo {\n  a: boolean;\n  b: boolean;\n  c: boolean;\n")
            .replace("export interface Bar {\n  a: boolean;\n  b: boolean;\n",
                     "export interface Bar {\n  a: boolean;\n  b: boolean;\n  d: boolean;\n"))


def test_judged_findings_are_ranked_by_probability_and_show_it(tmp_path, monkeypatch, capsys):
    make_repo(tmp_path, monkeypatch, TWO_TYPES, two_types_after())
    asked = []

    def judge(findings, head):
        asked.append((sorted(f["type"] for f in findings), head))
        return {f["id"]: {"Foo": 0.31, "Bar": 0.874}[f["type"]] for f in findings}

    monkeypatch.setattr(sg, "judge", judge)
    sg.run("HEAD~1", "HEAD", use_judge=True)
    out = capsys.readouterr().out
    assert asked == [(["Bar", "Foo"], "HEAD")]
    assert out.index("`d` new on Bar  p=0.87") < out.index("`c` new on Foo  p=0.31")
    assert "2 guard run(s) predate a new field in this diff" in out


def test_the_judge_is_never_called_when_there_is_nothing_to_ask(tmp_path, monkeypatch, capsys):
    """A judge asked about empty evidence once answered 0.60 on zero bytes."""
    complete = with_fields("c").replace("  return true;", "  if (f.c && !r.c) return false;\n  return true;")
    make_repo(tmp_path, monkeypatch, BEFORE, complete)
    monkeypatch.setattr(sg, "judge", lambda *a: pytest.fail("judge must not run without findings"))
    sg.run("HEAD~1", "HEAD", use_judge=True)
    assert "0 guard run(s)" in capsys.readouterr().out


# --------------------------------------------------------------- the judge request
class FakeResponse:
    def __init__(self, body):
        self.body = body

    def read(self):
        return json.dumps(self.body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def finding(id_, line_a, type_="Row"):
    return {"id": id_, "file": "row.ts", "new": ["c"], "type": type_, "a": line_a, "b": line_a + 1,
            "fields": ["a", "b"],
            "lines": [(line_a, "if (f.a === 'buyer@example.com') return false; // ping ops@example.com"),
                      (line_a + 1, "if (f.b && !r.b) return false;")]}


JUDGED_SOURCE = """export interface Row {
  a: boolean;
  b: boolean;
  c: boolean;
}

export function keep(r: Row, f: Row): boolean {
  if (f.a === 'buyer@example.com') return false; // ping ops@example.com
  if (f.b && !r.b) return false;
  return true;
}
"""


def test_the_judge_request_carries_the_key_and_user_agent_but_no_literals_or_comments(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(sg, "sh", lambda *args: JUDGED_SOURCE)
    sent = {}

    def urlopen(req, timeout=None):
        sent.update(req=req, timeout=timeout)
        return FakeResponse({"answers": {"g0_belongs": {"noul": 0.8}, "g1_belongs": {"noul": 0.2}}})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    scores = sg.judge([finding("row.ts:Row:8", 8), finding("row.ts:Other:8", 8, "Other")], "abc123")
    assert scores == {"row.ts:Row:8": 0.8, "row.ts:Other:8": 0.2}
    req = sent["req"]
    assert req.get_method() == "POST" and req.full_url == sg.ENDPOINT and sent["timeout"] == 120
    assert req.get_header("Authorization") == "Bearer test-key"
    assert req.get_header("Content-type") == "application/json"
    assert req.get_header("User-agent").startswith("graph-review/")
    body = json.loads(req.data)
    assert body["model"] == "jev-latest"
    assert set(body["state"]) == {"g0", "g1"} and set(body["questions"]) == {"g0_belongs", "g1_belongs"}
    assert body["questions"]["g0_belongs"]["type"] == "noul"
    assert body["state"]["g0"]["new_fields"] == ["c"]
    assert body["state"]["g0"]["guard_run"].startswith("if (f.a === '<str>') return false;")
    assert "buyer@" not in req.data.decode() and "ops@" not in req.data.decode()


# --------------------------------------------------------------- evidence edge case
def test_evidence_for_an_undeclared_type_says_so_instead_of_inventing_one(monkeypatch):
    blob = "\n".join(["  // filler"] * 60 + ["  if (r.a && !r.b) return false;"])
    monkeypatch.setattr(sg, "sh", lambda *args: blob)
    ev = sg.evidence_for({"file": "row.ts", "a": 61, "new": ["d", "c"], "type": "Missing",
                          "lines": [(61, "if (r.a && !r.b) return false; // ops@example.com")]}, "HEAD")
    assert ev["type_declaration"] == "Missing (declaration not found)"
    assert ev["new_fields"] == ["c", "d"]
    assert ev["guard_run"] == "if (r.a && !r.b) return false;", "comments are stripped from the run"
