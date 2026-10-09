"""review.py's command line: exit codes, what is run, what is printed and what is kept.

main() is driven in-process with a fake graph, so no model, GitHub or database is touched.
STATE_DIR and the working directory are redirected into tmp_path, so nothing is written
beside the script or the developer's checkout.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import types
from pathlib import Path

import pytest

REVIEW_PATH = Path(__file__).resolve().parents[1] / "review.py"
spec = importlib.util.spec_from_file_location("review_cli_tests", REVIEW_PATH)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
REAL_PROVIDER_KEY = review.provider_key

STATS = {"raw": 0, "kept": 0, "killed": 0, "downgraded": 0, "meta": [], "gaps": 0,
         "cap_drops": 0, "resolved": None}
FINAL = {
    "pr": {"number": 12, "title": "Add retries", "headSha": "a" * 40},
    "provider": "azure", "model": "m", "mode": "discovery",
    "advisory": "## Product review (advisory)\n\nfine",
    "result": {"findings": [], "killed": [], "resolved": [], "stats": STATS},
}


class FakeGraph:
    def __init__(self, final=None, error=None, history=()):
        self.final, self.error, self.history = final, error, list(history)
        self.invocations = []

    def invoke(self, state, cfg):
        self.invocations.append((state, {**cfg, "configurable": dict(cfg["configurable"])}))
        if self.error:
            raise self.error
        return self.final

    def get_state_history(self, cfg):
        return iter(self.history)


@pytest.fixture
def cli(monkeypatch, tmp_path):
    for name in tuple(os.environ):
        if name.startswith(("OPENAI_", "AZURE_OPENAI_", "REVIEW_", "LANGFUSE_", "VERTEX_")):
            monkeypatch.delenv(name)
    state_dir = tmp_path / "state"
    graph = FakeGraph(final=FINAL)
    cmds = []
    monkeypatch.chdir(tmp_path)  # main() chdirs to the target root; monkeypatch undoes it
    monkeypatch.setattr(review, "STATE_DIR", state_dir)
    monkeypatch.setattr(review, "target_repo_root", lambda: tmp_path)
    monkeypatch.setattr(review, "run_cmd", lambda cmd, timeout=120: cmds.append(cmd) or "")
    monkeypatch.setattr(review, "provider_key", lambda provider: "test-key")
    monkeypatch.setattr(review, "open_saver", lambda: state_dir.mkdir(exist_ok=True) or object())
    monkeypatch.setattr(review, "build_graph", lambda saver: graph)
    monkeypatch.setattr(review, "make_trace_callbacks", lambda thread: [])
    return types.SimpleNamespace(graph=graph, state_dir=state_dir, cmds=cmds, tmp=tmp_path)


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["review.py", *argv])
    return review.main()


def usage_rows(cli):
    return [json.loads(line) for line in (cli.state_dir / "usage.jsonl").read_text().splitlines()]


# --------------------------------------------------------------- argument errors
def test_no_target_is_a_usage_error(cli, monkeypatch, capsys):
    assert run_main(monkeypatch) == 2
    assert "need --pr <number|branch>" in capsys.readouterr().err
    assert cli.graph.invocations == []


@pytest.mark.parametrize("flags", [["--history"], ["--replay-from", "cp1"]])
def test_history_and_replay_need_a_thread(cli, monkeypatch, capsys, flags):
    assert run_main(monkeypatch, "--pr", "12", *flags) == 2
    assert "need --thread" in capsys.readouterr().err


def test_an_unknown_review_mode_in_the_environment_is_a_usage_error(cli, monkeypatch, capsys):
    monkeypatch.setenv("REVIEW_MODE", "unbounded")
    assert run_main(monkeypatch, "--pr", "12", "--provider", "azure", "--model", "m") == 2
    assert "invalid review mode 'unbounded'" in capsys.readouterr().err
    assert cli.graph.invocations == []


def test_openai_without_a_model_exits_3_before_any_work(cli, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--pr", "12")
    assert exc.value.code == 3 and "OPENAI_MODEL" in capsys.readouterr().err
    assert cli.cmds == [] and cli.graph.invocations == []


def test_a_missing_provider_credential_exits_3_before_touching_github(cli, monkeypatch, capsys):
    monkeypatch.setattr(review, "provider_key", REAL_PROVIDER_KEY)
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--pr", "12", "--provider", "azure", "--model", "m")
    assert exc.value.code == 3 and "AZURE_OPENAI_ENDPOINT" in capsys.readouterr().err
    assert cli.cmds == [] and cli.graph.invocations == []


# --------------------------------------------------------------- environment errors
def test_running_outside_a_git_checkout_exits_3(cli, monkeypatch, capsys):
    def not_a_repo():
        raise RuntimeError("fatal: not a git repository")

    monkeypatch.setattr(review, "target_repo_root", not_a_repo)
    assert run_main(monkeypatch, "--pr", "12", "--provider", "azure", "--model", "m") == 3
    assert "run review.py from inside a checkout" in capsys.readouterr().err
    assert cli.graph.invocations == []


def test_an_unauthenticated_gh_exits_3_with_the_fix(cli, monkeypatch, capsys):
    def no_auth(cmd, timeout=120):
        raise RuntimeError("not logged in")

    monkeypatch.setattr(review, "run_cmd", no_auth)
    assert run_main(monkeypatch, "--pr", "12", "--provider", "azure", "--model", "m") == 3
    assert "Fix: gh auth login" in capsys.readouterr().err
    assert cli.graph.invocations == []


# --------------------------------------------------------------- a normal run
def test_a_run_prints_the_review_keeps_state_and_logs_usage(cli, monkeypatch, capsys):
    code = run_main(monkeypatch, "--pr", "12", "--provider", "azure", "--model", "m")
    assert code == 0
    out, err = capsys.readouterr()
    expected = review.render_markdown(FINAL["pr"], FINAL["result"], FINAL["advisory"])
    assert out == expected + "\n"
    (state, cfg), = cli.graph.invocations
    assert state == {"pr_ref": "12", "provider": "azure", "model": "m", "light_model": "",
                     "timeout": 300, "mode": "discovery"}
    thread = cfg["configurable"]["thread_id"]
    assert thread.startswith("pr12-") and "/" not in thread
    assert cfg["max_concurrency"] == review.PROVIDERS["azure"]["concurrency"]
    assert cfg["recursion_limit"] == 100 and "callbacks" not in cfg
    saved = cli.state_dir / f"review-{thread}.md"
    assert saved.read_text() == expected
    assert f"-- saved: {saved}" in err and f"--history --thread {thread}" in err
    (row,) = usage_rows(cli)
    assert (row["ok"], row["pr"], row["thread"], row["provider"], row["model"], row["error"]) == (
        True, "12", thread, "azure", "m", None)
    assert row["mode"] == "discovery" and isinstance(row["ms"], int)


def test_the_reviewer_never_posts_it_only_prints_the_comment_command(cli, monkeypatch, capsys):
    assert run_main(monkeypatch, "--pr", "12", "--provider", "azure", "--model", "m", "--json") == 0
    doc = json.loads(capsys.readouterr().out)
    thread = doc["thread_id"]
    saved = cli.state_dir / f"review-{thread}.md"
    assert doc["pr"] == 12 and doc["title"] == "Add retries" and doc["headSha"] == "a" * 40
    assert (doc["provider"], doc["model"], doc["mode"]) == ("azure", "m", "discovery")
    assert doc["stats"] == STATS and doc["findings"] == [] and doc["killed"] == []
    assert doc["markdown_file"] == str(saved)
    assert doc["gh_command"] == f"gh pr review 12 --comment --body-file {saved}"
    assert cli.cmds == [["gh", "auth", "status"]], "only a read-only auth check may run"


def test_flags_override_the_environment_for_model_mode_and_throttle(cli, monkeypatch):
    monkeypatch.setenv("REVIEW_MODE", "discovery")
    monkeypatch.setenv("REVIEW_CONCURRENCY", "5")
    run_main(monkeypatch, "--pr", "feat/x", "--provider", "azure", "--model", "deploy-1",
             "--light-model", "tiny", "--concurrency", "3", "--timeout", "60", "--mode", "verification")
    (state, cfg), = cli.graph.invocations
    assert state == {"pr_ref": "feat/x", "provider": "azure", "model": "deploy-1", "light_model": "tiny",
                     "timeout": 60, "mode": "verification"}
    assert cfg["max_concurrency"] == 3
    assert cfg["configurable"]["thread_id"].startswith("prfeat-x-"), "a branch name is made path-safe"


def test_environment_supplies_mode_and_throttle_when_flags_are_absent(cli, monkeypatch):
    monkeypatch.setenv("REVIEW_MODE", "verification")
    monkeypatch.setenv("REVIEW_CONCURRENCY", "5")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    run_main(monkeypatch, "--pr", "12")
    (state, cfg), = cli.graph.invocations
    assert (state["mode"], state["model"], state["provider"]) == ("verification", "env-model", "openai")
    assert cfg["max_concurrency"] == 5


def test_a_failed_run_exits_1_and_records_the_error(cli, monkeypatch, capsys):
    cli.graph.error = RuntimeError("provider outage " + "x" * 400)
    assert run_main(monkeypatch, "--pr", "12", "--provider", "azure", "--model", "m") == 1
    out, err = capsys.readouterr()
    assert out == "" and "error: provider outage" in err
    (row,) = usage_rows(cli)
    assert row["ok"] is False and row["error"].startswith("provider outage") and len(row["error"]) == 300


def test_a_result_that_cannot_be_rendered_is_a_failure_not_a_silent_success(cli, monkeypatch, capsys):
    cli.graph.final = {}
    assert run_main(monkeypatch, "--pr", "12", "--provider", "azure", "--model", "m") == 1
    assert usage_rows(cli)[0]["ok"] is False


def test_replay_resumes_from_the_checkpoint_without_a_new_pr_reference(cli, monkeypatch, capsys):
    code = run_main(monkeypatch, "--thread", "pr12/old", "--replay-from", "cp-9",
                    "--provider", "azure", "--model", "m")
    assert code == 0
    (state, cfg), = cli.graph.invocations
    assert state is None, "replay re-enters the saved graph state"
    assert cfg["configurable"] == {"thread_id": "pr12-old", "checkpoint_id": "cp-9"}
    assert "[replay] thread pr12-old from checkpoint cp-9" in capsys.readouterr().err
    assert usage_rows(cli)[0]["replay_from"] == "cp-9"


# --------------------------------------------------------------- history
CHECKPOINTS = [
    types.SimpleNamespace(config={"configurable": {"checkpoint_id": "cp-2"}}, next=("verify_one",),
                          created_at="2026-10-01T00:00:02"),
    types.SimpleNamespace(config={"configurable": {"checkpoint_id": "cp-1"}}, next=(),
                          created_at="2026-10-01T00:00:01"),
]


def test_history_lists_checkpoints_and_needs_no_model_or_key(cli, monkeypatch, capsys):
    monkeypatch.setattr(review, "provider_key", lambda p: pytest.fail("history must not need a key"))
    cli.graph.history = CHECKPOINTS
    assert run_main(monkeypatch, "--history", "--thread", "pr12-x") == 0
    assert capsys.readouterr().out.splitlines() == ["cp-2  next=verify_one", "cp-1  next=(end)"]
    cli.graph.history = CHECKPOINTS
    assert run_main(monkeypatch, "--history", "--thread", "pr12-x", "--json") == 0
    assert json.loads(capsys.readouterr().out) == [
        {"checkpoint_id": "cp-2", "next": ["verify_one"], "ts": "2026-10-01T00:00:02"},
        {"checkpoint_id": "cp-1", "next": [], "ts": "2026-10-01T00:00:01"}]
    assert cli.graph.invocations == []


def test_history_of_an_unknown_thread_exits_1_and_points_at_the_usage_log(cli, monkeypatch, capsys):
    assert run_main(monkeypatch, "--history", "--thread", "nope") == 1
    err = capsys.readouterr().err
    assert "no checkpoints for thread 'nope'" in err and "usage.jsonl" in err


# --------------------------------------------------------------- tracing hook-up
def test_tracing_adds_callbacks_session_and_tags_and_wraps_the_run(cli, monkeypatch):
    handler, traced = object(), []
    monkeypatch.setattr(review, "make_trace_callbacks", lambda thread: [handler])
    monkeypatch.setattr(review, "run_traced", lambda fn: traced.append("traced") or fn())
    assert run_main(monkeypatch, "--pr", "12", "--provider", "azure", "--model", "m") == 0
    (_, cfg), = cli.graph.invocations
    thread = cfg["configurable"]["thread_id"]
    assert traced == ["traced"]
    assert cfg["callbacks"] == [handler] and cfg["run_name"] == "pr-review:run"
    assert cfg["metadata"] == {"langfuse_session_id": thread, "langfuse_tags": ["pr-review", "pr-12"]}


# --------------------------------------------------------------- state helpers
def test_log_usage_appends_one_json_line_per_run(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "STATE_DIR", tmp_path / "new-state")
    review.log_usage({"ok": True, "n": 1})
    review.log_usage({"ok": False, "n": 2})
    lines = (tmp_path / "new-state" / "usage.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [{"ok": True, "n": 1}, {"ok": False, "n": 2}]


def test_log_usage_never_fails_the_review_when_state_is_unwritable(tmp_path, monkeypatch):
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("")
    monkeypatch.setattr(review, "STATE_DIR", blocker / "state")
    review.log_usage({"ok": True})  # must not raise


def test_checkpoints_live_in_the_state_dir_beside_the_script(tmp_path, monkeypatch):
    seen = {}

    class FakeSaver:
        def __init__(self, conn):
            seen["conn"] = conn

    sqlite_mod = types.ModuleType("langgraph.checkpoint.sqlite")
    sqlite_mod.SqliteSaver = FakeSaver
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", sqlite_mod)
    monkeypatch.setattr(review, "STATE_DIR", tmp_path / "state")
    saver = review.open_saver()
    assert isinstance(saver, FakeSaver) and isinstance(seen["conn"], sqlite3.Connection)
    assert (tmp_path / "state" / "checkpoints.sqlite").exists()
    seen["conn"].close()


def test_flush_traces_is_silent_without_keys_and_fail_open_with_them(monkeypatch):
    flushes = []
    client = types.SimpleNamespace(flush=lambda: flushes.append(1))
    fake = types.ModuleType("langfuse")
    fake.get_client = lambda: client
    monkeypatch.setitem(sys.modules, "langfuse", fake)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    review.flush_traces()
    assert flushes == [], "an untraced run must not touch the SDK"
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    review.flush_traces()
    assert flushes == [1]

    def broken():
        raise RuntimeError("collector unreachable")

    client.flush = broken
    review.flush_traces()  # must not raise


def test_the_trace_mask_hands_back_the_original_if_redaction_itself_fails():
    class Hostile(dict):
        def items(self):
            raise RuntimeError("cannot iterate")

    payload = Hostile(email="reader@example.test")
    assert review.langfuse_mask(payload) is payload
