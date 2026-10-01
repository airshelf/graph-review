"""Pins for review.py's context pack (changed-file bodies + matched learnings).

The measured failure mode the pack kills: 7 reviewers independently re-fetching
the same changed files -- 45 LLM calls / 475k input tokens on a one-word diff.
These tests pin the pack's caps, the learnings matcher's precision floor, and the
provider-aware inline budgets so a refactor can't silently regress them.
"""
import importlib.util
import json
import pytest
import subprocess
import sys
import types
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "review", Path(__file__).resolve().parents[1] / "review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


@pytest.fixture(autouse=True)
def isolate_provider_tuning(monkeypatch):
    for name in ("AZURE_OPENAI_RESPONSES_API", "AZURE_OPENAI_REASONING_EFFORT",
                 "AZURE_OPENAI_SERVICE_TIER", "AZURE_OPENAI_NO_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)
    # Hold the real function: a test may monkeypatch make_model with a plain lambda,
    # and that patch is still in place when this fixture tears down.
    make_model = review.make_model
    make_model.cache_clear()
    yield
    make_model.cache_clear()


def test_provider_caps_claude_vertex_gets_big_budgets():
    assert review.provider_caps("claude-vertex") == (50_000, 120_000)
    assert review.provider_caps("gemini") == (15_000, 40_000)  # token-priced: stay lean
    assert review.provider_caps("nonexistent") == (review.DIFF_INLINE_CAP, 40_000)


def test_pack_block_empty_is_empty():
    assert review.pack_block("") == ""
    assert "Do NOT re-open" in review.pack_block("===== x =====\nbody")


def test_pr_context_tool_rule_flips_with_pack():
    pr = {"number": 1, "title": "t", "author": "a", "headRef": "h", "baseRef": "b",
          "additions": 1, "deletions": 0, "body": "", "files": ["x.py"]}
    without = review.pr_context(pr, "diff")
    with_pack = review.pr_context(pr, "diff", pack="===== x.py @ head =====\nbody")
    assert "zero tool calls is invalid" in without
    assert "do NOT re-read them" in with_pack and "CONSUMERS" in with_pack
    assert "===== x.py @ head =====" in with_pack


def test_pr_context_respects_diff_cap():
    pr = {"number": 1, "title": "t", "author": "a", "headRef": "h", "baseRef": "b",
          "additions": 1, "deletions": 0, "body": "", "files": []}
    out = review.pr_context(pr, "x" * 1000, diff_cap=100)
    assert "diff truncated" in out and "x" * 101 not in out


def test_verification_context_is_not_discovery():
    pr = {"number": 1, "title": "t", "author": "a", "headRef": "h", "baseRef": "b",
          "additions": 1, "deletions": 0, "body": "", "files": ["x.py"],
          "review_mode": "verification", "verification_scope": "- blocker A"}
    out = review.pr_context(pr, "fix delta")
    assert "VERIFICATION MODE — NOT DISCOVERY" in out
    assert "Do not report independent issues" in out
    assert "- blocker A" in out


def test_review_mode_resolves_cli_then_env_then_default():
    assert review.resolve_review_mode("verification", {"REVIEW_MODE": "discovery"}) == "verification"
    assert review.resolve_review_mode(None, {"REVIEW_MODE": "verification"}) == "verification"
    assert review.resolve_review_mode(None, {}) == "discovery"
    with pytest.raises(ValueError):
        review.resolve_review_mode(None, {"REVIEW_MODE": "unbounded"})


def test_fetch_thread_extracts_marked_blocker_scope(monkeypatch):
    body = """## 🕸️ Graph Review (model)

<!-- graph-review-head:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa mode:discovery -->

### Blockers

- **`x.py:7` -- concrete break**
  - Why: bad

### Majors

- **`y.py:9` -- real major**

### Minors

- **`z.py:3` -- nitpick**
"""
    monkeypatch.setattr(review, "gh_api_list", lambda path: [{
        "body": body, "created_at": "2026-07-24T00:00:00Z",
        "user": {"login": "github-actions[bot]", "type": "Bot"},
    }] if "/comments" in path else [])
    thread, findings, head, scope = review.fetch_thread(1)
    assert "Graph Review" in thread
    assert findings == [
        {"file": "x.py", "line": 7, "title": "concrete break"},
        {"file": "y.py", "line": 9, "title": "real major"},
        {"file": "z.py", "line": 3, "title": "nitpick"},
    ]
    assert head == "a" * 40
    # Majors joined the verification scope (2026-08-17) — they withhold
    # auto-approval, so the fix head must re-check them. Minors stay out.
    assert "concrete break" in scope and "real major" in scope
    assert "nitpick" not in scope


def test_node_fetch_verification_reviews_only_fix_delta(monkeypatch):
    previous, current = "a" * 40, "b" * 40
    meta = {"number": 1, "title": "t", "author": {"login": "a"},
            "baseRefName": "main", "headRefName": "fix", "body": "",
            "additions": 99, "deletions": 99, "files": [{"path": "whole-pr.py"}]}

    def fake_run(args, timeout=60, cwd=None):
        if args[:4] == ["gh", "pr", "view", "1"]:
            return json.dumps(meta)
        if args[:3] == ["git", "rev-parse", "FETCH_HEAD"]:
            return current
        if args[:3] == ["git", "diff", "--unified=80"]:
            return "focused fix delta"
        if args[:3] == ["git", "diff", "--name-only"]:
            return "fix.py\n"
        if args[:3] == ["git", "diff", "--numstat"]:
            return "2\t1\tfix.py\n"
        return ""

    monkeypatch.setattr(review, "run_cmd", fake_run)
    monkeypatch.setattr(review, "fetch_thread", lambda n: ("thread", [], previous, "blocker A"))
    monkeypatch.setattr(review, "pull_learnings", lambda files: "")
    monkeypatch.setattr(review, "pull_principles", lambda: "")
    monkeypatch.setattr(review, "pull_blast_radius", lambda *args: "")
    monkeypatch.setattr(review, "pull_stale_guards", lambda *args: "")
    monkeypatch.setattr(review, "build_file_pack", lambda sha, files: "pack")
    monkeypatch.setattr(review, "make_model", lambda *args, **kwargs: None)

    out = review.node_fetch({"pr_ref": "1", "provider": "azure", "model": "m",
                             "timeout": 1, "mode": "verification"})
    assert out["diff"] == "focused fix delta"
    assert out["pr"]["files"] == ["fix.py"]
    assert out["pr"]["additions"] == 2 and out["pr"]["deletions"] == 1
    assert out["pr"]["verification_scope"] == "blocker A"


def test_pull_learnings_matches_touched_paths(tmp_path):
    ldir = tmp_path / "learnings"
    ldir.mkdir()
    (ldir / "vertex.md").write_text(
        "# Vertex\n\n## ADC auth for review.py\nreview.py uses ADC. review.py needs no key.\n\n"
        "## Unrelated section\nnothing about the touched files here at all.\n")
    (ldir / "README.md").write_text("## Index\nreview.py review.py review.py\n")  # TOC: excluded
    out = review.pull_learnings(["review.py"], root=tmp_path)
    assert "ADC auth" in out               # score 3 (basename x2 + stem) >= 2 -> included
    assert "Unrelated section" not in out  # score 0 -> excluded
    assert "Index" not in out              # README.md is a TOC, never content


def test_pull_learnings_single_mention_is_noise(tmp_path):
    ldir = tmp_path / "learnings"
    ldir.mkdir()
    (ldir / "misc.md").write_text("## One stray mention\nsomething touches shop.ts once\n")
    assert review.pull_learnings(["src/shop.ts"], root=tmp_path) == ""


def test_pull_learnings_respects_cap(tmp_path):
    ldir = tmp_path / "learnings"
    ldir.mkdir()
    (ldir / "big.md").write_text("## pack pack pack section\n" + "pack " * 2000)
    out = review.pull_learnings(["pack.py"], root=tmp_path, cap=500)
    assert len(out) <= 500


def test_build_file_pack_real_repo(tmp_path, monkeypatch):
    # Exercise real Git objects in an isolated target, without requiring this
    # source checkout to have any commits. A tree is sufficient for git show.
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    blob = subprocess.run(["git", "hash-object", "-w", "--stdin"],
                          input="print('fixture')\n" * 100, text=True,
                          capture_output=True, check=True).stdout.strip()
    tree = subprocess.run(["git", "mktree"], input=f"100644 blob {blob}\treview.py\n",
                          text=True, capture_output=True, check=True).stdout.strip()
    pack = review.build_file_pack(tree, ["review.py", "does/not/exist.py"], per_cap=500)
    assert "===== review.py @ head =====" in pack
    assert "truncated at 500 chars" in pack
    assert "does/not/exist.py" not in pack  # deleted/missing files skipped silently


def test_cached_user_only_on_cache_providers():
    # claude-vertex: anthropic cache block (one breakpoint caches the whole prefix);
    # token-priced providers keep plain strings (their formatters don't know the block).
    out = review.cached_user("ctx", "claude-vertex")
    assert out == [{"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}}]
    assert review.cached_user("ctx", "gemini") == "ctx"
    assert review.cached_user("ctx", "glm") == "ctx"


def test_compose_pack_learnings_survive_prefix_slice():
    # Learnings appended LAST were sliced off by
    # pack[:pack_cap] whenever file bodies alone filled the cap. Learnings lead now.
    pack = review.compose_pack("F" * 50_000, "the gotcha")
    assert "the gotcha" in pack[:40_000]  # survives the gemini/glm 40k prefix cut
    assert pack.startswith("===== learnings/")
    assert review.compose_pack("files", "") == "files"      # no learnings -> pass-through
    assert review.compose_pack("", "g").endswith("g")        # learnings only


def test_pull_learnings_word_boundary(tmp_path):
    # "review" (loose stem signal) must not score inside "human-reviewable"
    # (substring counts pulled unrelated sections).
    ldir = tmp_path / "learnings"
    ldir.mkdir()
    (ldir / "noise.md").write_text(
        "## Unrelated feed section\nhuman-reviewable output, eyeball-reviewable too\n")
    (ldir / "real.md").write_text(
        "## The review lane\nthe review graph and the review approver\n")
    out = review.pull_learnings(["review.py"], root=tmp_path)
    assert "Unrelated feed section" not in out
    assert "The review lane" in out


def test_pull_learnings_root_defaults_to_repo(tmp_path, monkeypatch):
    # Resolve the target's Git root, not the script's checkout or a nested CWD.
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    ldir = tmp_path / "learnings"
    ldir.mkdir()
    (ldir / "context.md").write_text("## Pack\nreview.py needs context. review.py reads it.\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    expected = review.pull_learnings(["review.py"], root=tmp_path)
    assert "needs context" in expected
    assert review.target_repo_root() == tmp_path.resolve()
    assert review.pull_learnings(["review.py"]) == expected


def test_pull_learnings_missing_directory_is_silent(tmp_path):
    assert review.pull_learnings(["review.py"], root=tmp_path) == ""


def test_brave_search_keyless_returns_none(tmp_path, monkeypatch):
    # No key anywhere -> None, so web_search falls back to the deep shell tool
    # (fail-soft chain; the brief still works in keyless environments).
    monkeypatch.setenv("BRAVE_API_KEY", "")
    monkeypatch.setattr(review.Path, "home", lambda: tmp_path)  # no local key file
    assert review.brave_search("anything") is None


def test_pull_principles_reads_live_claude_md(tmp_path, monkeypatch):
    # Runtime loading includes every section rather than assuming a doc layout.
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    text = "# Instructions\nKeep it small.\n\n## Invariants\nNo writes.\n\n## Documentation\nKeep examples current.\n"
    (tmp_path / "CLAUDE.md").write_text(text)
    (tmp_path / "AGENTS.md").write_text("This lower-priority file must not be merged.\n")
    nested = tmp_path / "src"
    nested.mkdir()
    monkeypatch.chdir(nested)
    out = review.pull_principles()
    assert out == text
    assert "## Documentation" in out
    assert len(out) <= 18_000
    assert "[preamble truncated" not in out
    assert review.pull_principles(root=tmp_path / "missing") == ""


def test_pull_principles_falls_back_to_agents_md(tmp_path):
    text = "# Agent rules\n\n## Invariants\nReads must not mutate state.\n"
    (tmp_path / "AGENTS.md").write_text(text)
    assert review.pull_principles(root=tmp_path) == text


def test_pull_principles_first_existing_file_wins_even_if_empty(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("")
    (tmp_path / "AGENTS.md").write_text("Do not use this fallback when the first file exists.")
    assert review.pull_principles(root=tmp_path) == ""


@pytest.mark.parametrize("name", ["CLAUDE.md", "AGENTS.md"])
def test_pull_principles_marks_truncation_of_whole_file(tmp_path, name):
    text = "# Instructions\nKeep it small.\n" + "- Rule\n" * 2000
    (tmp_path / name).write_text(text)
    out = review.pull_principles(root=tmp_path, cap=3000)
    marker = "\n...[preamble truncated -- raise pull_principles(cap=)]"
    assert out == text[:3000 - len(marker)] + marker
    assert len(out) == 3000


def test_pull_principles_marks_the_cut_even_with_no_room_to_spare(tmp_path):
    # When no content fits beside the marker, visible loss still wins over silence.
    (tmp_path / "CLAUDE.md").write_text("# Instructions\n" + "- Rule\n" * 500)
    out = review.pull_principles(root=tmp_path, cap=20)
    marker = "\n...[preamble truncated -- raise pull_principles(cap=)]"
    assert out == marker[:20]
    assert len(out) == 20
    assert out.startswith("\n...[preamble")


def test_compose_pack_order_principles_learnings_files():
    pack = review.compose_pack("F" * 50_000, "the gotcha", "BE SIMPLE")
    head = pack[:40_000]  # gemini/glm prefix slice
    assert "BE SIMPLE" in head and "the gotcha" in head
    assert pack.index("BE SIMPLE") < pack.index("the gotcha") < pack.index("FFF")
    assert review.compose_pack("files", "", "") == "files"
    assert "repo agent instructions (CLAUDE.md / AGENTS.md, live)" in pack


def test_azure_review_requests_priority_processing(monkeypatch):
    """The CI review's env pin must reach Azure as a top-level request field."""
    seen = {}

    class FakeAzureChatOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        types.SimpleNamespace(AzureChatOpenAI=FakeAzureChatOpenAI),
    )
    monkeypatch.setattr(review, "provider_key", lambda _provider: "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_NO_TEMPERATURE", "1")
    monkeypatch.setenv("AZURE_OPENAI_SERVICE_TIER", "priority")
    review.make_model.cache_clear()

    review.make_model("azure", "custom-deployment", 120)

    assert seen["extra_body"] == {"service_tier": "priority"}
    review.make_model.cache_clear()


def test_azure_review_omits_service_tier_unless_requested(monkeypatch):
    """Account-specific processing tiers must not be sent by default."""
    seen = {}

    class FakeAzureChatOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        types.SimpleNamespace(AzureChatOpenAI=FakeAzureChatOpenAI),
    )
    monkeypatch.setattr(review, "provider_key", lambda _provider: "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_NO_TEMPERATURE", "1")
    monkeypatch.delenv("AZURE_OPENAI_SERVICE_TIER", raising=False)
    review.make_model.cache_clear()

    review.make_model("azure", "custom-deployment", 120)

    assert "extra_body" not in seen
    review.make_model.cache_clear()


def test_light_model_is_scoped_to_the_non_gating_aux_nodes():
    """--light-model must reach ONLY the two
    non-gating aux nodes -- brief + advisory. A gating reviewer or verifier that
    read light_model would be silently downgraded to the cheap tier, which is
    exactly the wiring bug this reviewer's own `wiring` dimension exists to catch.
    Source-level pin: inspect the payload boundary directly so a light model
    cannot silently enter a gating node's configuration."""
    import inspect
    assert "light_model" in inspect.getsource(review.node_brief)
    assert "light_model" in inspect.getsource(review.node_advisory)
    assert "light_model" not in inspect.getsource(review.node_review_dim)
    assert "light_model" not in inspect.getsource(review.node_verify_one)
    # fan_out attaches light_model to the advisory Send ONLY -- the gating
    # review_dim Send carries just {**base, "dimension"}, and `base` itself must
    # not smuggle it in (that spread reaches all 6 gating dimensions).
    src = inspect.getsource(review.fan_out_reviews)
    review_dim_line = next(l for l in src.splitlines() if 'Send("review_dim"' in l)
    assert "light_model" not in review_dim_line
    assert 'Send("advisory", {**base, "light_model"' in src
    base_literal = src[src.index("base = {"): src.index("sends =")]
    # strip comment lines so this pins the dict literal, not the prose around it
    base_code = "\n".join(l for l in base_literal.splitlines() if not l.lstrip().startswith("#"))
    assert "light_model" not in base_code


def test_resolve_tuning_reads_the_gate_throttle_env():
    """Pin that the reviewer consumes tuning env, not merely that CI sets it."""
    default_conc = review.PROVIDERS["azure"]["concurrency"]
    default_model = review.PROVIDERS["azure"]["model"]
    # env supplies the throttle when no CLI flag is given
    m, light, conc = review.resolve_tuning(
        "azure", None, None, None,
        {"REVIEW_CONCURRENCY": "4", "REVIEW_LIGHT_MODEL": "light-fixture-model"})
    assert (m, light, conc) == (default_model, "light-fixture-model", 4)
    # CLI flags win over env
    m, light, conc = review.resolve_tuning(
        "azure", "cli-model", "cli-light", 2,
        {"REVIEW_CONCURRENCY": "4", "REVIEW_LIGHT_MODEL": "light-fixture-model"})
    assert (m, light, conc) == ("cli-model", "cli-light", 2)
    # neither flag nor env -> provider default, no light split
    m, light, conc = review.resolve_tuning("azure", None, None, None, {})
    assert (m, light, conc) == (default_model, "", default_conc)


def _fake_azure(monkeypatch, seen, no_temperature=True):
    class FakeAzureChatOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        types.SimpleNamespace(AzureChatOpenAI=FakeAzureChatOpenAI),
    )
    monkeypatch.setattr(review, "provider_key", lambda _provider: "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.delenv("AZURE_OPENAI_RESPONSES_API", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_REASONING_EFFORT", raising=False)
    if no_temperature:
        monkeypatch.setenv("AZURE_OPENAI_NO_TEMPERATURE", "1")
    else:
        monkeypatch.delenv("AZURE_OPENAI_NO_TEMPERATURE", raising=False)
    review.make_model.cache_clear()


def test_gpt6_uses_responses_api_so_tools_survive(monkeypatch):
    """GPT-6 + function tools 400s on /v1/chat/completions ("Function tools with
    reasoning_effort are not supported"). Every reviewer binds tools, and the lane
    is continue-on-error, so without the Responses API each reviewer dies in ~2s
    and the run reports a clean, blocker-free review -- a green gate that reviewed
    nothing (measured 2026-09-23). This is the red path for that."""
    seen = {}
    _fake_azure(monkeypatch, seen)
    review.make_model("azure", "gpt-6-sol", 120)
    assert seen.get("use_responses_api") is True
    assert seen.get("reasoning") == {"effort": "medium"}
    review.make_model.cache_clear()


def test_gpt56_stays_on_chat_completions(monkeypatch):
    """5.6 serves tools on chat/completions and must NOT be moved: the Responses
    switch is scoped to the tier that needs it."""
    seen = {}
    _fake_azure(monkeypatch, seen)
    review.make_model("azure", "gpt-56-terra", 120)
    assert seen.get("use_responses_api") is False
    assert "reasoning" not in seen
    review.make_model.cache_clear()


def test_responses_api_env_overrides_name_derivation(monkeypatch):
    """A deployment whose name doesn't carry the tier can still be pinned."""
    seen = {}
    _fake_azure(monkeypatch, seen)
    monkeypatch.setenv("AZURE_OPENAI_RESPONSES_API", "1")
    review.make_model("azure", "house-style-deploy", 120)
    assert seen.get("use_responses_api") is True
    review.make_model.cache_clear()


def test_gpt6_never_sends_temperature_without_the_env(monkeypatch):
    """The GPT-6 tier REJECTS a non-default temperature. The earlier version of this
    suite set AZURE_OPENAI_NO_TEMPERATURE=1 in its helper, which hid that a local run
    (`--provider azure --model gpt-6-sol`, no env) would 400 every reviewer and the
    continue-on-error lane would then report no findings."""
    seen = {}
    _fake_azure(monkeypatch, seen, no_temperature=False)
    review.make_model("azure", "gpt-6-sol", 120)
    assert "temperature" not in seen
    review.make_model.cache_clear()


def test_responses_override_also_drops_temperature(monkeypatch):
    """Same rule via the env override, on a deployment whose name hides the tier --
    the case a `model.startswith("gpt-6")` condition would have missed."""
    seen = {}
    _fake_azure(monkeypatch, seen, no_temperature=False)
    monkeypatch.setenv("AZURE_OPENAI_RESPONSES_API", "1")
    review.make_model("azure", "house-style-deploy", 120)
    assert "temperature" not in seen
    review.make_model.cache_clear()


def test_chat_tier_keeps_its_review_temperature(monkeypatch):
    """gpt-54-mini is a chat tier that ACCEPTS temperature: the no-temperature rule
    must stay scoped to the reasoning tiers, not widen to every gpt-5/6 name."""
    seen = {}
    _fake_azure(monkeypatch, seen, no_temperature=False)
    review.make_model("azure", "gpt-54-mini", 120)
    assert seen.get("temperature") == 0.1
    review.make_model.cache_clear()


def test_old_api_version_pin_is_upgraded_for_responses(monkeypatch):
    """An AZURE_OPENAI_API_VERSION pin older than the Responses API would rebuild the
    silent failure."""
    seen = {}
    _fake_azure(monkeypatch, seen)
    monkeypatch.setitem(review.PROVIDERS["azure"], "api_version", "2024-05-01-preview")
    review.make_model("azure", "gpt-6-sol", 120)
    assert seen["api_version"] == review.RESPONSES_MIN_API_VERSION
    review.make_model.cache_clear()


def test_new_enough_api_version_pin_is_respected(monkeypatch):
    seen = {}
    _fake_azure(monkeypatch, seen)
    monkeypatch.setitem(review.PROVIDERS["azure"], "api_version", "2025-11-01-preview")
    review.make_model("azure", "gpt-6-sol", 120)
    assert seen["api_version"] == "2025-11-01-preview"
    review.make_model.cache_clear()


def test_api_version_at_least_orders_by_date_prefix():
    assert review.api_version_at_least("2025-04-01-preview", "2025-03-01-preview")
    assert review.api_version_at_least("2025-03-01-preview", "2025-03-01-preview")
    assert not review.api_version_at_least("2024-12-01-preview", "2025-03-01-preview")
    assert not review.api_version_at_least("", "2025-03-01-preview")
