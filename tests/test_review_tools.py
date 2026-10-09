"""Behaviour of the tools review.py hands to its agents, and the guards around them.

Reviewers and verifiers run model-chosen arguments through these tools, so the tools are
the trust boundary: file reads must stay inside the PR head, the database tool must stay
read-only, URL fetches must stay on public hosts, and a repeated call must be refused.
Everything runs offline: real Git in a throwaway checkout, faked subprocess/HTTP/DNS.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import types
import urllib.request
from pathlib import Path

import pytest

REVIEW_PATH = Path(__file__).resolve().parents[1] / "review.py"
spec = importlib.util.spec_from_file_location("review_tools_tests", REVIEW_PATH)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

DIFF = (
    "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"
    "diff --git a/lib/util.py b/lib/util.py\n--- a/lib/util.py\n+++ b/lib/util.py\n@@ -1 +1 @@\n-x\n+y\n"
)


def _git(repo, *args):
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A one-commit checkout with a secret file OUTSIDE it. cwd is the checkout root,
    as it is when main() runs. Returns (repo, head_sha)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "app.py").write_text("".join(f"line {i}\n" for i in range(1, 11)))
    (repo / "notes.md").write_text("needle in notes\n")
    (repo / "flags.txt").write_text("--verbose-flag\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    (tmp_path / "outside.txt").write_text("OUTSIDE-SECRET\n")
    monkeypatch.chdir(repo)
    return repo, _git(repo, "rev-parse", "HEAD")


def build(sha, diff=DIFF, **kw):
    return {t.name: t for t in review.make_tools(sha, 7, diff, **kw)}


# --------------------------------------------------------------- show_file
def test_show_file_reads_the_pr_head_not_the_working_tree(checkout):
    repo, sha = checkout
    (repo / "app.py").write_text("DIRTY working-tree edit\n")
    out = build(sha)["show_file"].invoke({"path": "app.py", "start": 1, "end": 2})
    assert out.splitlines()[:2] == ["1\tline 1", "2\tline 2"]
    assert "DIRTY" not in out


def test_show_file_cannot_read_outside_the_repository(checkout, tmp_path):
    """A model-chosen path must not reach files beside the checkout."""
    _, sha = checkout
    show = build(sha)["show_file"]
    for path in ("../outside.txt", str(tmp_path / "outside.txt")):
        out = show.invoke({"path": path})
        assert out.startswith("tool error:"), path
        assert "OUTSIDE-SECRET" not in out


def test_show_file_range_names_the_next_line_to_continue_from(checkout):
    _, sha = checkout
    out = build(sha)["show_file"].invoke({"path": "app.py", "start": 3, "end": 5})
    lines = out.splitlines()
    assert lines[:3] == ["3\tline 3", "4\tline 4", "5\tline 5"]
    assert lines[3] == "...(showing lines 3-5 of 10 -- continue with start=6)"
    whole = build(sha)["show_file"].invoke({"path": "app.py"})
    assert "continue with" not in whole and whole.splitlines()[-1] == "10\tline 10"


def test_show_file_past_the_end_reports_the_line_count(checkout):
    _, sha = checkout
    out = build(sha)["show_file"].invoke({"path": "app.py", "start": 50})
    assert out == "(empty range; file has 10 lines)"


def test_show_file_truncates_by_line_and_the_hint_skips_nothing(checkout, monkeypatch):
    """The continuation hint must name the last line actually shown, never the requested end."""
    _, sha = checkout
    monkeypatch.setattr(review, "TOOL_OUTPUT_CAP", 40)
    show = build(sha)["show_file"]
    out = show.invoke({"path": "app.py"})
    body, hint = out.rsplit("\n", 1)
    last_shown = int(body.splitlines()[-1].split("\t")[0])
    assert 1 < last_shown < 10
    assert f"continue with start={last_shown + 1}" in hint
    nxt = show.invoke({"path": "app.py", "start": last_shown + 1})
    assert nxt.splitlines()[0].startswith(f"{last_shown + 1}\t")


# --------------------------------------------------------------- grep_repo
def test_grep_repo_searches_the_head_and_strips_the_sha_prefix(checkout):
    repo, sha = checkout
    (repo / "app.py").write_text("dirtyword\n")
    grep = build(sha)["grep_repo"]
    assert grep.invoke({"pattern": "needle"}) == "notes.md:1:needle in notes"
    assert grep.invoke({"pattern": "dirtyword"}).startswith("no matches for /dirtyword/ at the PR head")


def test_grep_repo_pathspec_narrows_the_search(checkout):
    _, sha = checkout
    grep = build(sha)["grep_repo"]
    assert "needle" in grep.invoke({"pattern": "needle", "pathspec": "*.md"})
    miss = grep.invoke({"pattern": "needle", "pathspec": "*.py"})
    assert miss.startswith("no matches for /needle/") and "under *.py" in miss


def test_grep_repo_pattern_cannot_inject_a_git_option(checkout):
    """The pattern is passed after -e, so a leading dash is searched, not parsed as a flag."""
    _, sha = checkout
    out = build(sha)["grep_repo"].invoke({"pattern": "--verbose-flag"})
    assert out == "flags.txt:1:--verbose-flag"


def test_grep_repo_rejects_expensive_patterns_and_reports_regex_errors(checkout):
    _, sha = checkout
    grep = build(sha)["grep_repo"]
    assert grep.invoke({"pattern": "a" * 251}).startswith("rejected: pattern too long")
    assert grep.invoke({"pattern": "("}).startswith("error:")


def test_a_tool_that_times_out_returns_text_instead_of_raising(checkout, monkeypatch):
    """An uncaught TimeoutExpired once killed a whole reviewer."""
    _, sha = checkout

    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(review.subprocess, "run", hang)
    out = build(sha)["grep_repo"].invoke({"pattern": "needle"})
    assert out.startswith("tool error:") and "do not repeat the same call" in out


# --------------------------------------------------------------- pr_diff / changed_files
def test_pr_diff_returns_one_files_hunk_or_says_there_is_none(checkout):
    _, sha = checkout
    pr_diff = build(sha)["pr_diff"]
    assert pr_diff.invoke({}) == DIFF
    hunk = pr_diff.invoke({"path": "lib/util.py"})
    assert hunk.startswith("diff --git a/lib/util.py") and "app.py" not in hunk
    assert pr_diff.invoke({"path": "other.py"}) == "no diff hunk for other.py; see changed_files"


def test_pr_diff_output_is_capped_with_a_narrowing_hint(checkout, monkeypatch):
    _, sha = checkout
    monkeypatch.setattr(review, "TOOL_OUTPUT_CAP", 50)
    out = build(sha)["pr_diff"].invoke({})
    assert out.startswith(DIFF[:50]) and "narrow the request" in out and len(out) < len(DIFF) + 100


def test_changed_files_prefers_the_known_paths_and_falls_back_to_gh(checkout, monkeypatch):
    _, sha = checkout
    assert build(sha, changed_paths=["a.py", "b.py"])["changed_files"].invoke({}) == "a.py\nb.py"
    seen = []
    monkeypatch.setattr(review, "run_cmd", lambda cmd, timeout=120: seen.append(cmd) or "x.py\n")
    assert build(sha)["changed_files"].invoke({}) == "x.py\n"
    assert seen == [["gh", "pr", "diff", "7", "--name-only"]]


# --------------------------------------------------------------- tool exposure
def test_database_and_web_tools_are_opt_in(checkout):
    """Least privilege: only the advisory agent may query data, only the brief may use the web."""
    _, sha = checkout
    base = {"show_file", "grep_repo", "pr_diff", "changed_files"}
    assert set(build(sha)) == base
    assert set(build(sha, with_db=True)) == base | {"query_prod"}
    assert set(build(sha, with_web=True)) == base | {"web_search", "scrape_url"}


# --------------------------------------------------------------- repeat-call guard
def test_an_identical_repeat_call_is_refused_not_re_answered(checkout):
    _, sha = checkout
    show = build(sha)["show_file"]
    first = show.invoke({"path": "app.py", "start": 1, "end": 3})
    second = show.invoke({"path": "app.py", "start": 1, "end": 3})
    assert first.startswith("1\tline 1")
    assert second.startswith(review.CONVERGENCE_MARKER) and "line 1" not in second
    # The convergence wall keys off this exact wording, so the two must stay joined.
    msg = types.SimpleNamespace(type="tool", content=second)
    assert review._refused_repeat_count([msg]) == 1
    # A different range is a different call.
    assert show.invoke({"path": "app.py", "start": 4, "end": 5}).startswith("4\tline 4")


def test_a_failing_call_is_cached_so_repeating_it_is_refused(checkout):
    _, sha = checkout
    show = build(sha)["show_file"]
    assert show.invoke({"path": "missing.py"}).startswith("tool error:")
    assert show.invoke({"path": "missing.py"}).startswith(review.CONVERGENCE_MARKER)


def test_each_tool_set_has_its_own_repeat_cache(checkout):
    """A retried agent gets fresh tools; its first legitimate call must not be refused."""
    _, sha = checkout
    args = {"path": "app.py", "start": 1, "end": 2}
    assert build(sha)["show_file"].invoke(args).startswith("1\tline 1")
    assert build(sha)["show_file"].invoke(args).startswith("1\tline 1")


# --------------------------------------------------------------- query_prod
DSN = "postgresql://reader:p%40ss@db.example.com:6543/appdb?sslmode=verify-full"


class FakeRun:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr, self.calls = stdout, returncode, stderr, []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


@pytest.fixture
def prod(checkout, monkeypatch):
    _, sha = checkout
    monkeypatch.setenv("REVIEW_DATABASE_URL", DSN)
    monkeypatch.delenv("PGOPTIONS", raising=False)
    psql = FakeRun(stdout="42\n")
    monkeypatch.setattr(review.subprocess, "run", psql)
    return build(sha, with_db=True)["query_prod"], psql


def test_query_prod_without_a_database_url_is_unavailable(checkout, monkeypatch):
    _, sha = checkout
    monkeypatch.delenv("REVIEW_DATABASE_URL", raising=False)
    out = build(sha, with_db=True)["query_prod"].invoke({"sql": "select 1"})
    assert out.startswith("unavailable: REVIEW_DATABASE_URL not set")


@pytest.mark.parametrize("sql", [
    "DROP TABLE users",
    "delete from users",
    "UPDATE users SET admin = true",
    "INSERT INTO users VALUES (1)",
    "SELECT 1; DROP TABLE users",
    "select 1;select 2",
    "selection from users",
    "-- comment\nDROP TABLE users",
    "",
])
def test_query_prod_runs_only_a_single_select_or_with(prod, sql):
    query, psql = prod
    assert query.invoke({"sql": sql}).startswith("rejected:")
    assert psql.calls == [], "a rejected statement must never reach psql"


@pytest.mark.parametrize("sql,sent", [
    ("SELECT 1;", "SELECT 1"),
    ("  select count(*) from t  ", "select count(*) from t"),
    ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
])
def test_query_prod_passes_the_clean_statement_to_psql(prod, sql, sent):
    query, psql = prod
    assert query.invoke({"sql": sql}) == "42"
    (cmd, _), = psql.calls
    assert cmd == ["psql", "-t", "-A", "-c", sent]


def test_query_prod_keeps_credentials_out_of_argv_and_forces_read_only(prod, monkeypatch):
    monkeypatch.setenv("PGOPTIONS", "-c statement_timeout=5s")
    query, psql = prod
    query.invoke({"sql": "select 1"})
    (cmd, kw), = psql.calls
    assert not any("p@ss" in a or "p%40ss" in a or "db.example.com" in a for a in cmd)
    env = kw["env"]
    assert (env["PGHOST"], env["PGPORT"], env["PGUSER"], env["PGPASSWORD"], env["PGDATABASE"]) == (
        "db.example.com", "6543", "reader", "p@ss", "appdb")
    assert env["PGSSLMODE"] == "verify-full"
    # A WITH clause can hide a write, so the session itself must be read-only too.
    assert env["PGOPTIONS"] == "-c statement_timeout=5s -c default_transaction_read_only=on"


def test_query_prod_defaults_to_tls_and_the_standard_port(prod, monkeypatch):
    monkeypatch.setenv("REVIEW_DATABASE_URL", "postgresql://u:pw@db.example.com/appdb")
    query, psql = prod
    query.invoke({"sql": "select 1"})
    env = psql.calls[0][1]["env"]
    assert env["PGSSLMODE"] == "require" and env["PGPORT"] == "5432"
    assert env["PGOPTIONS"] == "-c default_transaction_read_only=on"


def test_query_prod_reports_psql_errors_and_empty_results(prod, monkeypatch):
    query, psql = prod
    psql.stdout = ""
    assert query.invoke({"sql": "select 1 where false"}) == "(0 rows)"
    psql.returncode, psql.stderr = 1, 'ERROR: relation "t" does not exist'
    assert query.invoke({"sql": "select * from t"}) == 'psql error: ERROR: relation "t" does not exist'


def test_query_prod_repeat_is_refused_without_a_second_query(prod):
    query, psql = prod
    query.invoke({"sql": "select 1"})
    assert query.invoke({"sql": "select 1;"}).startswith(review.CONVERGENCE_MARKER)
    assert len(psql.calls) == 1


# --------------------------------------------------------------- _public_host
def _resolver(monkeypatch, mapping):
    calls = []

    def getaddrinfo(host, port, *a, **k):
        calls.append(host)
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))
                for ip in mapping[host]]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return calls


@pytest.mark.parametrize("url", [
    "ftp://docs.example.com/x", "file:///etc/passwd", "javascript:alert(1)",
    "//docs.example.com/x", "http://", "docs.example.com", "",
])
def test_public_host_refuses_non_http_urls_without_resolving(monkeypatch, url):
    calls = _resolver(monkeypatch, {})
    assert review._public_host(url) is False
    assert calls == []


@pytest.mark.parametrize("url", ["http://docs.example.com/a", "https://docs.example.com:8443/a?b=1"])
def test_public_host_accepts_a_public_address(monkeypatch, url):
    _resolver(monkeypatch, {"docs.example.com": ["93.184.216.34"]})
    assert review._public_host(url) is True


@pytest.mark.parametrize("ip", [
    "10.0.0.5", "172.16.0.1", "192.168.1.1", "127.0.0.1", "169.254.169.254",
    "0.0.0.0", "224.0.0.1", "240.0.0.1", "::1", "fe80::1", "fc00::1",
])
def test_public_host_refuses_private_loopback_link_local_and_reserved(monkeypatch, ip):
    _resolver(monkeypatch, {"docs.example.com": [ip]})
    assert review._public_host("https://docs.example.com/") is False


def test_public_host_refuses_a_name_when_any_resolved_address_is_internal(monkeypatch):
    """DNS rebinding: one public record next to one internal record is still refused,
    whichever order the resolver returns them in."""
    for ips in (["93.184.216.34", "10.0.0.5"], ["10.0.0.5", "93.184.216.34"]):
        _resolver(monkeypatch, {"docs.example.com": ips})
        assert review._public_host("https://docs.example.com/") is False


def test_public_host_refuses_a_name_that_does_not_resolve(monkeypatch):
    def fail(*a, **k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    assert review._public_host("https://nope.example.invalid/") is False


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8080/admin", "http://[::1]/", "http://169.254.169.254/latest/meta-data/",
    "http://2130706433/",  # integer-encoded 127.0.0.1; a string host check would pass it
])
def test_public_host_refuses_literal_internal_addresses(url):
    """Numeric hosts resolve without any DNS lookup, so these stay offline."""
    assert review._public_host(url) is False


# --------------------------------------------------------------- web_search / scrape_url
@pytest.fixture
def web(checkout, tmp_path, monkeypatch):
    _, sha = checkout
    tools_dir = tmp_path / "webtools"
    tools_dir.mkdir()
    monkeypatch.setattr(review, "WEB_TOOLS_DIR", tools_dir)
    monkeypatch.setattr(review, "brave_search", lambda q: None)
    return build(sha, with_web=True), tools_dir


def test_web_search_uses_the_fast_snippet_lane_when_it_answers(web, monkeypatch):
    tools, _ = web
    monkeypatch.setattr(review, "brave_search", lambda q: f"snippet for {q}")
    monkeypatch.setattr(review.subprocess, "run", lambda *a, **k: pytest.fail("deep tool must not run"))
    assert tools["web_search"].invoke({"query": "pydantic v2"}) == "snippet for pydantic v2"


def test_web_search_says_so_when_no_search_tool_exists(web):
    tools, _ = web
    assert tools["web_search"].invoke({"query": "q"}).startswith("web search unavailable")


def test_web_search_falls_back_to_the_deep_tool(web, monkeypatch):
    tools, tools_dir = web
    (tools_dir / "web_search.sh").write_text("")
    pages = {"content": [{"url": "https://docs.example.com/a", "title": "A", "content": "Z" * 3000}]}
    shell = FakeRun(stdout=json.dumps(pages))
    monkeypatch.setattr(review.subprocess, "run", shell)
    out = tools["web_search"].invoke({"query": "how to"})
    cmd = shell.calls[0][0]
    assert cmd[:3] == ["bash", str(tools_dir / "web_search.sh"), "how to"] and "--no-stealth" in cmd
    assert out.startswith("[source: https://docs.example.com/a]\nA\n")
    assert out.count("Z") == 2500, "page text is clipped to keep the brief small"
    shell.stdout = "{}"
    assert tools["web_search"].invoke({"query": "nothing"}) == "no results for 'nothing'"


def test_scrape_url_refuses_non_public_hosts_before_running_anything(web, monkeypatch):
    tools, tools_dir = web
    (tools_dir / "scrape.sh").write_text("")
    _resolver(monkeypatch, {"internal.example.com": ["10.1.2.3"]})
    monkeypatch.setattr(review.subprocess, "run", lambda *a, **k: pytest.fail("must not fetch"))
    out = tools["scrape_url"].invoke({"url": "http://internal.example.com/secret"})
    assert out == "rejected: only public http(s) documentation URLs are allowed"


def test_scrape_url_fetches_a_public_page_and_caps_the_result(web, monkeypatch):
    tools, tools_dir = web
    _resolver(monkeypatch, {"docs.example.com": ["93.184.216.34"]})
    assert tools["scrape_url"].invoke({"url": "https://docs.example.com/a"}).startswith(
        "scrape unavailable")
    (tools_dir / "scrape.sh").write_text("")
    shell = FakeRun(stdout="page text\n")
    monkeypatch.setattr(review.subprocess, "run", shell)
    assert tools["scrape_url"].invoke({"url": "https://docs.example.com/b"}) == "page text"
    assert shell.calls[0][0] == ["bash", str(tools_dir / "scrape.sh"), "https://docs.example.com/b",
                                 "--no-stealth"]
    shell.stdout = ""
    assert tools["scrape_url"].invoke({"url": "https://docs.example.com/c"}) == \
        "empty fetch for https://docs.example.com/c"


# --------------------------------------------------------------- brave_search
class FakeHttp:
    def __init__(self, body=None, error=None):
        self.body, self.error, self.requests = body, error, []

    def __call__(self, req, timeout=None):
        self.requests.append((req, timeout))
        if self.error:
            raise self.error
        return self

    def read(self):
        return json.dumps(self.body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def brave(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAVE_API_KEY", "test-brave-key")
    monkeypatch.setattr(review.Path, "home", lambda: tmp_path)
    http = FakeHttp({"web": {"results": [
        {"title": "T1", "url": "https://a.example.com", "description": "d1"},
        {"title": "T2", "url": "https://b.example.com", "description": "d2"},
        {"title": "T3", "url": "https://c.example.com"},
    ]}})
    monkeypatch.setattr(urllib.request, "urlopen", http)
    return http


def test_brave_search_sends_the_key_and_a_quoted_query(brave):
    out = review.brave_search("python 3.12 & asyncio", count=2, timeout=7)
    (req, timeout), = brave.requests
    assert timeout == 7
    assert req.full_url == ("https://api.search.brave.com/res/v1/web/search"
                            "?q=python%203.12%20%26%20asyncio&count=2")
    assert {k.lower(): v for k, v in req.header_items()}["x-subscription-token"] == "test-brave-key"
    assert out == "T1 -- https://a.example.com\nd1\n\nT2 -- https://b.example.com\nd2"


def test_brave_search_clamps_the_result_count(brave):
    review.brave_search("q", count=500)
    assert brave.requests[0][0].full_url.endswith("&count=20")


def test_brave_search_reports_an_empty_result_and_swallows_http_faults(brave):
    brave.body = {"web": {"results": []}}
    assert review.brave_search("rare") == "no results for 'rare'"
    brave.error = OSError("connection reset")
    assert review.brave_search("rare") is None, "None lets the caller fall back to the deep tool"


def test_brave_search_reads_the_key_file_when_the_env_is_empty(brave, monkeypatch, tmp_path):
    monkeypatch.setenv("BRAVE_API_KEY", " ")
    keyfile = tmp_path / ".config" / "brave" / "api_key"
    keyfile.parent.mkdir(parents=True)
    keyfile.write_text("file-key\n")
    review.brave_search("q")
    assert {k.lower(): v for k, v in brave.requests[0][0].header_items()}[
        "x-subscription-token"] == "file-key"


def test_brave_search_with_an_unreadable_key_file_is_keyless(brave, monkeypatch, tmp_path):
    monkeypatch.setenv("BRAVE_API_KEY", "")
    (tmp_path / ".config" / "brave" / "api_key").mkdir(parents=True)  # exists, but not a file
    assert review.brave_search("q") is None
    assert brave.requests == []
