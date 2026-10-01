"""Red paths for stale_guards.py. Each test names a way an earlier cut was wrong.

Run: python3 -m pytest tests/test_stale_guards.py -q
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sg", ROOT / "stale_guards.py")
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)


def test_a_plain_read_is_not_a_guard():
    assert not sg.tests_on("const name = row.status;", "status")


def test_optional_chaining_is_not_a_guard():
    assert not sg.tests_on("const p = record.input?.prompt;", "input")


def test_nullish_defaulting_is_not_a_guard():
    """`record.formattedText ?? record.text` is a fallback, not a checklist.

    Accepting `??` would make ordinary field extraction read as a guard run."""
    assert not sg.tests_on("const c = record.formattedText ?? record.text;", "formattedText")


def test_an_if_guard_counts():
    assert sg.tests_on("if (filters.active && !row.isActive) return false;", "isActive")


def test_a_ternary_test_counts():
    assert sg.tests_on("return s.missing ? a : b;", "missing")


def test_a_negation_guard_counts():
    assert sg.tests_on("if (!row.isHidden) return false;", "isHidden")


def test_an_equality_guard_counts():
    assert sg.tests_on("if (row.category === want) return true;", "category")


def test_jsx_key_is_not_a_guard():
    """`<tr key={s.key} ...>` was reported as a branch by the first cut."""
    assert not sg.tests_on("<tr key={s.key} className={extra ? 'a' : 'b'}>", "key")


# --- wiring into review.py -------------------------------------------------

def _review():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "review", ROOT / "review.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def test_a_clean_diff_adds_no_section(tmp_path, monkeypatch):
    """"0 guard run(s)" is not a finding. An early version passed that string through,
    which put an empty section in every reviewer's context pack."""
    complete = AFTER.replace("  return true;", "  if (f.c && !r.c) return false;\n  return true;")
    repo = _fixture_repo(tmp_path, after=complete)
    monkeypatch.chdir(repo)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert sg.collect("HEAD~1", "HEAD") == [], "the fixture must extend the whole checklist"
    assert _review().pull_stale_guards("HEAD~1", "HEAD") == ""


def test_unresolvable_refs_fail_open():
    """A context section must never be able to kill the reviewer."""
    assert _review().pull_stale_guards("nope-xyz", "also-nope") == ""


def test_guards_land_before_the_file_bodies():
    """compose_pack's injection slice is a PREFIX cut, so a section appended after the
    file bodies is dropped whenever they fill the cap."""
    pack = _review().compose_pack("FILES", "LEARN", "PRINCIPLES", "BLAST", "GUARDS")
    assert pack.index("GUARDS") < pack.index("FILES")


# --- the judge -------------------------------------------------------------

def test_evidence_carries_the_signature_and_the_declaration(monkeypatch):
    """Field names and guard lines alone lack the enclosing contract. The parsed
    declaration and signature provide that context without exposing whole bodies."""
    ev = _fixture_evidence(monkeypatch)
    assert set(ev) == {"new_fields", "type_declaration", "enclosing_signature", "guard_run"}
    assert ev["guard_run"], "the judge must never be handed an empty run"
    assert ev["new_fields"] == ["c"]
    assert ev["type_declaration"] == "Row { a: boolean, b: boolean, c: boolean }"


def test_judge_without_a_key_exits_three_rather_than_guessing():
    import os
    import pytest
    saved = os.environ.pop("TYPESAFE_API_KEY", None)
    try:
        with pytest.raises(SystemExit) as e:
            sg.judge([{"id": "a", "file": __file__, "new": "x", "type": "T",
                       "a": 1, "b": 1, "fields": ["y"], "lines": [(1, "z")]}])
        assert e.value.code == 3
    finally:
        if saved is not None:
            os.environ["TYPESAFE_API_KEY"] = saved


# --- the judge inside the reviewer path ------------------------------------

def _with_key(value):
    import os
    saved = os.environ.get("TYPESAFE_API_KEY")
    if value is None:
        os.environ.pop("TYPESAFE_API_KEY", None)
    else:
        os.environ["TYPESAFE_API_KEY"] = value
    return saved


def _restore(saved):
    import os
    if saved is None:
        os.environ.pop("TYPESAFE_API_KEY", None)
    else:
        os.environ["TYPESAFE_API_KEY"] = saved


# One field per line, because that is how the detector reads a type body and how
# real code is written. A first fixture declared them inline on one line and the
# detector correctly found nothing, which looked like a bug in the detector.
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

AFTER = BEFORE.replace("  b: boolean;\n", "  b: boolean;\n  c: boolean;\n")


def _fixture_evidence(monkeypatch):
    """Evidence from a complete local source fixture, independent of Git history."""
    blob = AFTER.replace("  return true;", "  const bodyOnly = 'do not transmit';\n  return true;")
    monkeypatch.setattr(sg, "sh", lambda *args: blob)
    finding = {"file": "row.ts", "new": ["c"], "type": "Row",
               "a": 8, "b": 9, "fields": ["a", "b"],
               "lines": [(8, "if (f.a && !r.a) return false;"),
                         (9, "if (f.b && !r.b) return false;")]}
    return sg.evidence_for(finding, "HEAD")


def _fixture_repo(tmp_path, after=AFTER):
    """A two-commit repo whose second commit adds a field to a guard run.

    History-dependent tests create their own commits. A checkout that lacks a
    pinned commit would otherwise resolve no refs and return "", falsely passing
    a clean-diff assertion without exercising the detector.
    """
    import subprocess
    d = tmp_path / "repo"
    d.mkdir()
    run = lambda *a: subprocess.run(a, cwd=d, capture_output=True, check=True)
    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "t@t.t")
    run("git", "config", "user.name", "t")
    (d / "row.ts").write_text(BEFORE)
    run("git", "add", "row.ts")
    run("git", "commit", "-qm", "base")
    (d / "row.ts").write_text(after)
    run("git", "commit", "-qam", "add c")
    return d


def _guards_in(repo, key):
    import os
    saved_key, cwd = _with_key(key), os.getcwd()
    try:
        os.chdir(repo)
        return _review().pull_stale_guards("HEAD~1", "HEAD")
    finally:
        os.chdir(cwd)
        _restore(saved_key)


def test_a_rejected_key_still_ships_the_section(tmp_path, monkeypatch):
    """The judge is an improvement, never a dependency. A key that rotates or an API
    that is down must cost the probabilities and nothing else, because `--judge`
    exits 3 and a naive caller would drop the whole section."""
    import subprocess
    repo = _fixture_repo(tmp_path)
    review = _review()
    actual_run = subprocess.run
    judged_calls = []

    def reject_judge(args, **kwargs):
        if "--judge" in args:
            judged_calls.append(args)
            return subprocess.CompletedProcess(args, 3, "", "judge key rejected")
        return actual_run(args, **kwargs)

    # No vendor request: exercise the reviewer's real fallback, then execute the
    # deterministic helper against the fixture checkout.
    monkeypatch.setattr(review.subprocess, "run", reject_judge)
    out = _guards_in(repo, "fixture-key-rejected-by-stub")
    assert judged_calls, "the judged path must be attempted before falling back"
    assert "guard run over" in out, "the deterministic finding must survive a dead judge"
    assert "p=" not in out


def test_no_key_at_all_still_ships_the_section(tmp_path):
    out = _guards_in(_fixture_repo(tmp_path), None)
    assert "guard run over" in out and "p=" not in out


# --- what leaves the machine ------------------------------------------------

def test_no_function_body_is_sent(monkeypatch):
    """Bodies can contain secrets or user data. The judge needs the guard run and
    the enclosing contract, not unrelated implementation details."""
    ev = _fixture_evidence(monkeypatch)
    assert "enclosing_code" not in ev
    assert set(ev) == {"new_fields", "type_declaration", "enclosing_signature", "guard_run"}
    assert "\n" not in ev["enclosing_signature"], "a signature is one line, never a body"
    assert ev["type_declaration"].startswith("Row"), "the type is parsed, never raw source"
    assert "bodyOnly" not in str(ev) and "do not transmit" not in str(ev)


def test_string_literals_never_leave_the_box():
    assert sg.scrub("if (r.email === 'buyer@example.com') return false;") == \
        "if (r.email === '<str>') return false;"
    assert sg.scrub('const k = "sk-live-abc123";') == "const k = '<str>';"
    assert sg.scrub("const t = `Bearer ${tok}`;") == "const t = '<str>';"


def test_comments_never_leave_the_box():
    """A comment is not a string literal. Redacting literals alone leaves contact
    comments above fields untouched. Comments are free text, a likely place for a real
    address, and carry nothing the judge needs."""
    assert "ops@example.com" not in sg.scrub("if (r.x) return false; // ping ops@example.com")
    assert "buyer@" not in sg.scrub("/** escalation buyer@example.com */\n  a: boolean;")


def test_the_type_arrives_as_parsed_members_never_as_source():
    """Sending 30 raw declaration lines shipped whatever sat between the fields."""
    blob = ["export interface Row {", "  /** contact buyer@example.com */",
            "  a: boolean;", "  // internal note", "  b?: string;", "}"]
    out = sg.type_summary(blob, "Row")
    assert out == "Row { a: boolean, b?: string }"
    assert "buyer@" not in out and "internal note" not in out


def test_redaction_keeps_the_structure_the_judge_reasons_about():
    out = sg.scrub("if (f.status === 'open' && !r.isActive) return false;")
    assert "f.status" in out and "r.isActive" in out and "open" not in out


def test_one_guard_run_is_one_finding_however_many_fields_arrived(tmp_path):
    """Several new fields can all point at the same guard run. That is one run and
    one question, not a separate finding per field. Single-field fixtures cannot
    expose this duplicate-reporting bug."""
    import os
    import subprocess
    before = """export interface Row {
  a: boolean;
  b: boolean;
}

export function keep(r: Row, f: Row): boolean {
  if (f.a && !r.a) return false;
  if (f.b && !r.b) return false;
  return true;
}
"""
    d = tmp_path / "multi"
    d.mkdir()
    run = lambda *a: subprocess.run(a, cwd=d, capture_output=True, check=True)
    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "t@t.t")
    run("git", "config", "user.name", "t")
    (d / "row.ts").write_text(before)
    run("git", "add", "row.ts")
    run("git", "commit", "-qm", "base")
    (d / "row.ts").write_text(before.replace(
        "  b: boolean;\n", "  b: boolean;\n  c: boolean;\n  d: boolean;\n  e: boolean;\n"))
    run("git", "commit", "-qam", "add three")

    cwd = os.getcwd()
    try:
        os.chdir(d)
        found = sg.collect("HEAD~1", "HEAD")
    finally:
        os.chdir(cwd)

    assert len(found) == 1, f"three new fields, one run, expected 1 finding, got {len(found)}"
    assert sorted(found[0]["new"]) == ["c", "d", "e"], "the finding must name every new field"


def test_two_types_sharing_one_guard_span_stay_two_findings(tmp_path):
    """Keying on (file, span) alone merged them: one type's new field was reported
    against the other type's declaration and the second question vanished silently.
    The type is part of the identity."""
    import os
    import subprocess
    before = """export interface Foo {
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
    d = tmp_path / "twotypes"
    d.mkdir()
    run = lambda *a: subprocess.run(a, cwd=d, capture_output=True, check=True)
    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "t@t.t")
    run("git", "config", "user.name", "t")
    (d / "row.ts").write_text(before)
    run("git", "add", "row.ts")
    run("git", "commit", "-qm", "base")
    after = before.replace("export interface Foo {\n  a: boolean;\n  b: boolean;\n}",
                           "export interface Foo {\n  a: boolean;\n  b: boolean;\n  c: boolean;\n}")
    after = after.replace("export interface Bar {\n  a: boolean;\n  b: boolean;\n}",
                          "export interface Bar {\n  a: boolean;\n  b: boolean;\n  d: boolean;\n}")
    (d / "row.ts").write_text(after)
    run("git", "commit", "-qam", "add c and d")

    cwd = os.getcwd()
    try:
        os.chdir(d)
        found = sg.collect("HEAD~1", "HEAD")
    finally:
        os.chdir(cwd)

    by_type = {x["type"]: sorted(x["new"]) for x in found}
    assert by_type.get("Foo") == ["c"], f"Foo's finding is wrong: {by_type}"
    assert by_type.get("Bar") == ["d"], f"Bar's finding is wrong or lost: {by_type}"


def test_finding_ids_are_unique(tmp_path):
    """judge() returns scores in a dict keyed by finding id. Two findings sharing an id
    means one score silently overwrites the other and both render the survivor's
    probability. Fixing the grouping key without fixing the id leaves this collision
    in place."""
    import os
    import subprocess
    before = """export interface Foo {
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
    d = tmp_path / "ids"
    d.mkdir()
    run = lambda *a: subprocess.run(a, cwd=d, capture_output=True, check=True)
    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "t@t.t")
    run("git", "config", "user.name", "t")
    (d / "row.ts").write_text(before)
    run("git", "add", "row.ts")
    run("git", "commit", "-qm", "base")
    (d / "row.ts").write_text(
        before.replace("export interface Foo {\n  a: boolean;\n  b: boolean;\n}",
                       "export interface Foo {\n  a: boolean;\n  b: boolean;\n  c: boolean;\n}")
              .replace("export interface Bar {\n  a: boolean;\n  b: boolean;\n}",
                       "export interface Bar {\n  a: boolean;\n  b: boolean;\n  d: boolean;\n}"))
    run("git", "commit", "-qam", "add c and d")

    cwd = os.getcwd()
    try:
        os.chdir(d)
        found = sg.collect("HEAD~1", "HEAD")
    finally:
        os.chdir(cwd)

    ids = [x["id"] for x in found]
    assert len(ids) == len(set(ids)), f"two findings share an id, scores would collide: {ids}"
