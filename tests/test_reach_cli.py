"""reach.py end to end: import graph, changed exports, and the command-line report.

The resolver unit tests live in test_reach.py. These run the real script entry point
in-process against a throwaway Git repository, so the answer a reviewer actually sees
(which entry points a diff reaches that it does not contain) is pinned, along with the
exit codes the caller in review.py depends on.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "reach.py"

FILES = {
    "tsconfig.json": '{"compilerOptions": {"paths": {"@/*": ["./*"]}}}\n',
    "shared/util.ts": ("export function buildSummary() {\n  return 1;\n}\n"
                       "export function formatDate() {\n  return 2;\n}\n"),
    "lib/mid.ts": ("import { buildSummary } from '@/shared/util';\n"
                   "export function mid() { return buildSummary(); }\n"),
    "app/page.tsx": ("import { mid } from '../lib/mid';\n"
                     "export default function Page() {\n  return mid();\n}\n"),
    "app/dates/page.tsx": ("import { formatDate } from '../../shared/util';\n"
                           "export default function Dates() {\n  return formatDate();\n}\n"),
}


@pytest.fixture
def reach(monkeypatch):
    monkeypatch.delenv("GRAPH_REVIEW_ENTRY_RE", raising=False)
    spec = importlib.util.spec_from_file_location("reach_cli_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo, *args):
    subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo,
                   capture_output=True, text=True, check=True)


def make_repo(tmp_path, monkeypatch, edits):
    """Commit FILES, then commit `edits` (path -> new text) on top. cwd becomes the repo."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    for rel, text in FILES.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    for rel, text in edits.items():
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "change")
    monkeypatch.chdir(repo)
    return repo


def run_reach(reach, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["reach.py", *argv])
    return reach.main()


CHANGE_SIGNATURE = {"shared/util.ts": FILES["shared/util.ts"].replace(
    "buildSummary()", "buildSummary(options: object)")}


# --------------------------------------------------------------- the report
def test_report_names_the_entry_point_behind_a_changed_export_and_only_that_one(
        reach, tmp_path, monkeypatch, capsys):
    """app/dates/page.tsx imports the same module but only formatDate, which did not change.
    app/page.tsx reaches buildSummary through lib/mid.ts and an @/ alias, two hops away."""
    make_repo(tmp_path, monkeypatch, CHANGE_SIGNATURE)
    assert run_reach(reach, monkeypatch, "--base", "HEAD~1", "--head", "HEAD") == 0
    out = capsys.readouterr().out
    assert "reach: 1 JS/TS files changed, 2 files import them transitively" in out
    assert "ENTRY POINTS AFFECTED BUT NOT IN THE DIFF: 1" in out
    assert "      app/page.tsx" in out and "app/dates/page.tsx" not in out
    assert "Facts to CHECK, not findings." in out
    assert "already in the diff" not in out


def test_an_entry_point_the_diff_already_edits_is_not_reported_as_unseen(
        reach, tmp_path, monkeypatch, capsys):
    """The report exists to name pages the author did not touch. A reached page that this
    very diff edits needs no warning, in text or in JSON."""
    page_body_only = FILES["app/page.tsx"].replace("return mid()", "return mid() ?? null")
    make_repo(tmp_path, monkeypatch, {**CHANGE_SIGNATURE, "app/page.tsx": page_body_only})
    run_reach(reach, monkeypatch, "--base", "HEAD~1", "--head", "HEAD")
    out = capsys.readouterr().out
    assert "ENTRY POINTS AFFECTED BUT NOT IN THE DIFF: 0" in out and "      (none)" in out
    run_reach(reach, monkeypatch, "--base", "HEAD~1", "--head", "HEAD", "--json")
    doc = json.loads(capsys.readouterr().out)
    assert sorted(doc["changed"]) == ["app/page.tsx", "shared/util.ts"]
    assert doc["unseen_entry_points"] == []


def test_a_reached_entry_point_the_diff_already_edits_is_listed_as_in_the_diff(
        reach, tmp_path, monkeypatch, capsys):
    """Changed files were dropped from the reached set before this list was built, so it was
    always empty and its report line could never print."""
    page_body_only = FILES["app/page.tsx"].replace("return mid()", "return mid() ?? null")
    make_repo(tmp_path, monkeypatch, {**CHANGE_SIGNATURE, "app/page.tsx": page_body_only})
    run_reach(reach, monkeypatch, "--base", "HEAD~1", "--head", "HEAD", "--json")
    doc = json.loads(capsys.readouterr().out)
    assert doc["entry_points_in_diff"] == ["app/page.tsx"] and doc["unseen_entry_points"] == []
    run_reach(reach, monkeypatch, "--base", "HEAD~1", "--head", "HEAD")
    out = capsys.readouterr().out
    assert "entry points reached AND already in the diff: 1" in out
    assert "reach: 2 JS/TS files changed, 1 files import them transitively" in out, \
        "the count stays files OUTSIDE the diff (lib/mid.ts)"


def test_json_report_lists_the_unseen_entry_point(reach, tmp_path, monkeypatch, capsys):
    make_repo(tmp_path, monkeypatch, CHANGE_SIGNATURE)
    run_reach(reach, monkeypatch, "--base", "HEAD~1", "--head", "HEAD", "--json")
    doc = json.loads(capsys.readouterr().out)
    assert doc == {"changed": ["shared/util.ts"], "unseen_entry_points": ["app/page.tsx"],
                   "entry_points_in_diff": []}


def test_a_body_only_change_is_a_real_zero_with_its_summary(reach, tmp_path, monkeypatch, capsys):
    """No export definition changed, so there is nothing to trace. The summary line must
    still print, or a caller cannot tell this zero from a run that never happened."""
    body_only = {"shared/util.ts": FILES["shared/util.ts"].replace("return 1", "return 11")}
    make_repo(tmp_path, monkeypatch, body_only)
    assert run_reach(reach, monkeypatch, "--base", "HEAD~1", "--head", "HEAD") == 0
    out = capsys.readouterr().out
    assert "no exported symbol definitions changed" in out
    assert "ENTRY POINTS AFFECTED BUT NOT IN THE DIFF: 0" in out


def test_a_diff_without_source_files_is_a_real_zero_with_its_summary(
        reach, tmp_path, monkeypatch, capsys):
    make_repo(tmp_path, monkeypatch, {"NOTES.md": "docs only\n"})
    assert run_reach(reach, monkeypatch, "--base", "HEAD~1", "--head", "HEAD") == 0
    out = capsys.readouterr().out
    assert "no JavaScript or TypeScript files changed" in out
    assert "ENTRY POINTS AFFECTED BUT NOT IN THE DIFF: 0" in out


@pytest.mark.parametrize("bad", ["base", "head"])
def test_an_unresolvable_ref_exits_2_and_prints_no_summary(
        reach, tmp_path, monkeypatch, capsys, bad):
    """The caller falls back from origin/<base> to <base> on a non-zero exit, so a bad ref
    must never read as a legitimate zero."""
    make_repo(tmp_path, monkeypatch, CHANGE_SIGNATURE)
    refs = {"base": "HEAD~1", "head": "HEAD", bad: "no-such-ref"}
    assert run_reach(reach, monkeypatch, "--base", refs["base"], "--head", refs["head"]) == 2
    captured = capsys.readouterr()
    assert "cannot resolve ref 'no-such-ref'" in captured.err
    assert "NOT IN THE DIFF" not in captured.out


# --------------------------------------------------------------- building blocks
def test_reverse_graph_maps_each_file_to_its_importers(reach, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for rel, text in {
        "a.ts": "import { x } from './b';\nconst y = require('./c');\n",
        "b.ts": "import { z } from 'react';\nimport self from './b';\nexport const x = 1;\n",
        "c.js": "module.exports = {};\n",
        "lib/index.ts": "export {};\n",
        "d.ts": "import { l } from './lib';\nimport { q } from '@/b';\n",
    }.items():
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_text(text)
    files = ["a.ts", "b.ts", "c.js", "lib/index.ts", "d.ts", "ghost.ts"]  # ghost.ts is listed but gone
    rev = reach.build_reverse(files, {"@/": "./"})
    assert rev == {"b.ts": {"a.ts", "d.ts"}, "c.js": {"a.ts"}, "lib/index.ts": {"d.ts"}}, \
        "bare packages, self-imports and unreadable files add no edges"


def test_changed_symbols_reads_export_definitions_from_added_and_removed_lines(
        reach, tmp_path, monkeypatch):
    before = ("export function keep() {}\nexport const gone = 1;\nexport class Widget {}\n"
              "function internal() { return 1; }\n")
    after = ("export function keep() {}\nexport async function fresh() {}\nexport class Widget2 {}\n"
             "function internal() { return 2; }\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "m.ts").write_text(before)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "m.ts").write_text(after)
    (repo / "added.ts").write_text("export const same = 1;\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "change")
    monkeypatch.chdir(repo)
    got = reach.changed_symbols("HEAD~1", "HEAD", ["m.ts", "added.ts"])
    # `keep` and the non-exported helper's body change do not count; `added.ts` is a brand-new file.
    assert got == {"m.ts": {"gone", "fresh", "Widget", "Widget2"}, "added.ts": {"same"}}
