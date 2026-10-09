"""Behaviour of review.py's graph nodes, report logic and parsing helpers, with no real model.

The model layer is replaced by scripted fake agents, so these tests pin what the pipeline
does WITH model output: how findings are parsed, scoped, retried, verified, downgraded,
deduplicated and rendered. A final test runs the whole compiled graph end to end.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REVIEW_PATH = Path(__file__).resolve().parents[1] / "review.py"
spec = importlib.util.spec_from_file_location("review_nodes_tests", REVIEW_PATH)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

PR = {"number": 7, "title": "Add retries", "author": "dev", "headRef": "feat", "baseRef": "main",
      "headSha": "a" * 40, "additions": 3, "deletions": 1, "body": "body", "files": ["x.py"]}


# --------------------------------------------------------------- scripted model layer
class FakeModel:
    """Stands in for a chat model; only the JSON-repair pass ever calls it directly."""

    def __init__(self):
        self.repair_reply, self.invocations = "{}", []

    def invoke(self, messages):
        self.invocations.append(messages)
        return types.SimpleNamespace(content=self.repair_reply)


class FakeAgent:
    def __init__(self, rig, prompt):
        self.rig, self.prompt, self.calls = rig, prompt, []

    def invoke(self, messages, config=None):
        user = messages["messages"][0][1]
        self.calls.append({"user": user, "config": config})
        reply = self.rig.reply(self.prompt, user)
        if isinstance(reply, Exception):
            raise reply
        return {"messages": [types.SimpleNamespace(content=reply)]}


class Rig:
    def __init__(self):
        self.models, self.agents, self.sleeps = [], [], []
        self.model = FakeModel()
        self.reply = lambda prompt, user: "{}"

    def last(self):
        return self.agents[-1]


@pytest.fixture
def rig(monkeypatch, tmp_path):
    rig = Rig()

    def make_model(provider, model, timeout, *a, **k):
        rig.models.append((provider, model, timeout))
        return rig.model

    def react_agent(model, tools, system_prompt, tool_budget=review.REVIEWER_TOOL_BUDGET):
        agent = FakeAgent(rig, system_prompt)
        rig.agents.append(types.SimpleNamespace(
            agent=agent, tools=[t.name for t in tools], prompt=system_prompt, budget=tool_budget))
        return agent

    monkeypatch.setattr(review, "make_model", make_model)
    monkeypatch.setattr(review, "react_agent", react_agent)
    monkeypatch.setattr(review.time, "sleep", rig.sleeps.append)
    monkeypatch.setattr(review, "WEB_TOOLS_DIR", tmp_path / "no-web-tools")
    return rig


def fenced(obj):
    return "```json\n" + json.dumps(obj) + "\n```"


def finding(title="real bug", severity="major", file="x.py", line=5, **extra):
    return {"file": file, "line": line, "severity": severity, "title": title,
            "why": "because", "failure_scenario": "input -> wrong output",
            "suggestion": "fix it", **extra}


def review_payload(dim="correctness", **kw):
    p = {"dimension": dim, "pr": PR, "diff": "diff text", "provider": "azure", "model": "main-model",
         "timeout": 9, "brief": "THE-BRIEF", "pack": "", "mode": "discovery"}
    return {**p, **kw}


# --------------------------------------------------------------- node_review_dim
def test_reviewer_findings_are_parsed_stamped_and_scoped_to_its_dimension(rig):
    rig.reply = lambda prompt, user: fenced({"findings": [finding(line="5"), {"title": "None"}]})
    out = review.node_review_dim(review_payload())
    (f,) = out["findings"]
    assert (f["key"], f["dimension"], f["line"], f["severity"]) == ("correctness:0", "correctness", 5, "major")
    agent = rig.last()
    assert "Your single dimension: correctness." in agent.prompt
    assert review.DIMENSIONS["correctness"] in agent.prompt
    assert review.VERIFICATION_PROMPT not in agent.prompt
    call = agent.agent.calls[0]["config"]
    assert call == {"recursion_limit": review.AGENT_RECURSION_LIMIT, "run_name": "review:correctness"}
    assert rig.models == [("azure", "main-model", 9)], "gating reviewers always use the main model"


def test_verification_mode_swaps_in_the_focused_prompt(rig):
    rig.reply = lambda prompt, user: json.dumps({"findings": []})
    assert review.node_review_dim(review_payload(mode="verification")) == {"findings": []}
    assert review.VERIFICATION_PROMPT in rig.last().prompt


@pytest.mark.parametrize("dim,sees_brief", [
    ("correctness", True), ("wiring", True), ("invariants", False), ("security-data", False),
])
def test_only_the_dimensions_that_check_external_facts_get_the_web_brief(rig, dim, sees_brief):
    rig.reply = lambda prompt, user: json.dumps({"findings": []})
    review.node_review_dim(review_payload(dim))
    assert ("THE-BRIEF" in rig.last().agent.calls[0]["user"]) is sees_brief


def test_the_pack_is_cut_to_the_providers_budget_before_it_reaches_the_agent(rig):
    rig.reply = lambda prompt, user: json.dumps({"findings": []})
    pack_cap = review.provider_caps("azure")[1]
    review.node_review_dim(review_payload(pack="§" * (pack_cap + 5000)))
    assert rig.last().agent.calls[0]["user"].count("§") == pack_cap


def test_cap_overflow_and_placeholder_findings_are_reported_not_hidden(rig):
    rig.reply = lambda prompt, user: json.dumps(
        {"findings": [finding("one"), {"title": "N/A"}, finding("two")], "droppedCount": 3})
    out = review.node_review_dim(review_payload("wiring"))["findings"]
    assert [f["title"] for f in out[:2]] == ["one", "two"]
    assert out[-1] == {"key": "wiring:dropped", "dimension": "wiring", "severity": "meta",
                       "title": "3 finding(s) dropped by the per-dimension cap"}
    assert len(out) == 3


def test_a_crashed_reviewer_becomes_a_visible_coverage_gap_not_a_clean_pass(rig):
    rig.reply = lambda prompt, user: RuntimeError("boom")
    out = review.node_review_dim(review_payload())
    assert out == {"findings": [{"key": "correctness:error", "dimension": "correctness",
                                 "severity": "meta", "title": "reviewer failed: boom"}]}


def test_unparseable_reviewer_output_is_a_coverage_gap(rig):
    rig.reply = lambda prompt, user: "I looked and it seems fine."
    rig.model.repair_reply = "still not json"
    out = review.node_review_dim(review_payload("invariants"))
    assert out["findings"][0]["key"] == "invariants:error"
    assert out["findings"][0]["title"] == "reviewer output unparseable"


def test_malformed_output_gets_one_repair_pass_before_being_given_up_on(rig):
    rig.reply = lambda prompt, user: 'findings: [{"title": "oops"'
    rig.model.repair_reply = json.dumps({"findings": [finding("repaired")]})
    out = review.node_review_dim(review_payload())
    assert [f["title"] for f in out["findings"]] == ["repaired"]
    (messages,) = rig.model.invocations
    assert messages[0][0] == "system" and "Output ONLY the JSON" in messages[0][1]
    assert messages[1] == ("user", 'findings: [{"title": "oops"')


def test_a_rate_limited_reviewer_is_retried_on_a_fresh_agent(rig):
    replies = [RuntimeError("429 Too Many Requests"), fenced({"findings": [finding("ok")]})]
    rig.reply = lambda prompt, user: replies.pop(0)
    out = review.node_review_dim(review_payload())
    assert [f["title"] for f in out["findings"]] == ["ok"]
    assert rig.sleeps == [review.RETRY_SLEEPS[0]]
    assert len(rig.agents) == 2 and rig.agents[0].agent is not rig.agents[1].agent


# --------------------------------------------------------------- node_advisory
def test_advisory_is_skipped_in_verification_mode_without_building_a_model(rig):
    out = review.node_advisory(review_payload(mode="verification"))
    assert out["advisory"].startswith("## Product review (advisory)") and "n/a" in out["advisory"]
    assert rig.models == [] and rig.agents == []


def test_advisory_uses_the_light_model_and_is_the_only_agent_with_database_access(rig):
    rig.reply = lambda prompt, user: "  ## Product review (advisory)\n\nLooks right.  \n"
    out = review.node_advisory(review_payload(light_model="light-model"))
    assert out == {"advisory": "## Product review (advisory)\n\nLooks right."}
    assert rig.models == [("azure", "light-model", 9)]
    assert "query_prod" in rig.last().tools and rig.last().prompt == review.ADVISORY_PROMPT
    review.node_advisory(review_payload())
    assert rig.models[-1] == ("azure", "main-model", 9), "no light model configured -> main model"


def test_a_failed_advisory_never_blocks_the_review(rig):
    rig.reply = lambda prompt, user: RuntimeError("quota exhausted")
    out = review.node_advisory(review_payload())
    assert out["advisory"] == "## Product review (advisory)\n\n_advisory agent failed: quota exhausted_"


# --------------------------------------------------------------- node_brief
@pytest.fixture
def web_tools(tmp_path, monkeypatch):
    d = tmp_path / "web-tools"
    d.mkdir()
    (d / "web_search.sh").write_text("")
    monkeypatch.setattr(review, "WEB_TOOLS_DIR", d)


def brief_state(**kw):
    state = {"pr": {**PR, "files": ["pyproject.toml"]}, "diff": "+dep", "provider": "azure",
             "model": "main-model", "timeout": 4, "pack": "PACK"}
    return {**state, **kw}


def test_brief_is_skipped_in_verification_mode(rig, web_tools):
    assert review.node_brief(brief_state(mode="verification")) == {"brief": ""}
    assert rig.models == []


def test_brief_is_skipped_when_the_web_tools_are_absent(rig):
    assert review.node_brief(brief_state()) == {"brief": ""}
    assert rig.models == []


def test_brief_is_skipped_for_a_repo_internal_diff(rig, web_tools):
    state = brief_state(pr={**PR, "files": ["x.py"]}, diff="+x = compute()\n")
    assert review.node_brief(state) == {"brief": ""}
    assert rig.agents == []


def test_brief_runs_a_web_enabled_agent_on_the_light_model(rig, web_tools):
    rig.reply = lambda prompt, user: "  - fact one  "
    out = review.node_brief(brief_state(light_model="light-model"))
    assert out == {"brief": "- fact one"}
    assert rig.models == [("azure", "light-model", 4)]
    agent = rig.last()
    assert {"web_search", "scrape_url"} <= set(agent.tools) and "query_prod" not in agent.tools
    assert agent.prompt == review.BRIEF_PROMPT and agent.budget == review.BRIEF_TOOL_BUDGET
    assert agent.agent.calls[0]["config"] == {"recursion_limit": review.BRIEF_RECURSION_LIMIT,
                                             "run_name": "brief"}


def test_a_failed_brief_leaves_reviewers_running_unbriefed(rig, web_tools):
    rig.reply = lambda prompt, user: RuntimeError("search provider down")
    assert review.node_brief(brief_state()) == {"brief": ""}


@pytest.mark.parametrize("files,diff", [
    (["package.json"], ""), (["svc/pyproject.toml"], ""), (["go.mod"], ""), (["Cargo.toml"], ""),
    (["a.py"], "# /// script\n+x"), (["docs/specs/auth.txt"], ""), (["docs/payment-spec.md"], ""),
    (["a.py"], "+import requests\n"), (["a.py"], "+from pkg.sub import thing\n"),
    (["a.js"], '+const x = require("y")\n'), (["a.js"], "+export * from './other'\n"),
    (["a.rs"], "+use std::fmt;\n"), (["package.json"], '+  "left-pad": "^1.3.0",\n'),
])
def test_diffs_that_lean_on_an_external_surface_trigger_the_brief(files, diff):
    assert review.touches_external_surface({"files": files}, diff) is True


@pytest.mark.parametrize("files,diff", [
    (["README.md"], "+We import the data from the old system and use it.\n"),
    (["a.py"], "+x = compute()\n"),
    (["docs/guide.md"], "+from the top, restart the service\n"),
    ([], ""),
])
def test_internal_and_prose_only_diffs_skip_the_brief(files, diff):
    assert review.touches_external_surface({"files": files}, diff) is False


# --------------------------------------------------------------- fan-out wiring
def test_reviewers_fan_out_per_dimension_and_only_the_advisory_gets_the_light_model():
    state = {"pr": PR, "diff": "d", "provider": "azure", "model": "m", "timeout": 5,
            "brief": "b", "pack": "p", "light_model": "light"}
    sends = review.fan_out_reviews(state)
    assert [s.node for s in sends] == ["review_dim"] * len(review.DIMENSIONS) + ["advisory"]
    assert {s.arg["dimension"] for s in sends[:-1]} == set(review.DIMENSIONS)
    assert all("light_model" not in s.arg for s in sends[:-1])
    assert sends[-1].arg["light_model"] == "light"
    assert all(s.arg["mode"] == "discovery" and s.arg["brief"] == "b" for s in sends)


def test_verification_runs_only_for_blockers_and_majors_one_per_lens():
    deduped = [{**finding("a", "blocker"), "key": "d:0"}, {**finding("b", "major"), "key": "d:1"},
               {**finding("c", "minor"), "key": "d:2"}]
    state = {"pr": PR, "diff": "d", "provider": "azure", "model": "m", "timeout": 5, "deduped": deduped}
    sends = review.fan_out_verify(state)
    assert len(sends) == 2 * len(review.LENSES)
    assert {s.node for s in sends} == {"verify_one"}
    assert {(s.arg["finding"]["key"], s.arg["lens"]) for s in sends} == {
        (k, lens) for k in ("d:0", "d:1") for lens in review.LENSES}


def test_nothing_to_verify_goes_straight_to_the_report():
    state = {"deduped": [{**finding("c", "minor"), "key": "d:0"}]}
    assert review.fan_out_verify(state) == "report"
    assert review.fan_out_verify({}) == "report"


def test_collect_dedupes_before_verification_spends_tokens():
    same_line = [{**finding("x", "major"), "key": "a:0", "dimension": "a"},
                 {**finding("y", "major"), "key": "b:0", "dimension": "b"}]
    (merged,) = review.node_collect({"findings": same_line})["deduped"]
    assert merged["dimensions"] == ["a", "b"]


# --------------------------------------------------------------- node_verify_one
def verify_payload(lens="repro", **kw):
    f = {**finding("claimed defect", "blocker"), "key": "correctness:0", "dimension": "correctness"}
    return {"finding": f, "lens": lens, "pr": PR, "diff": "d", "provider": "azure",
            "model": "main-model", "timeout": 6, **kw}


def test_a_verifier_gets_one_lens_and_the_claim_not_the_reviewers_context(rig):
    rig.reply = lambda prompt, user: json.dumps({"refuted": True, "inflated": False, "reasoning": "r" * 700})
    out = review.node_verify_one(verify_payload("repro"))
    (v,) = out["verdicts"]
    assert (v["key"], v["lens"], v["refuted"], v["inflated"]) == ("correctness:0", "repro", True, False)
    assert v["reasoning"] == "r" * 600
    agent = rig.last()
    assert review.LENSES["repro"] in agent.prompt and review.LENSES["severity"] not in agent.prompt
    assert review.VERDICT_FORMAT in agent.prompt and agent.budget == review.VERIFY_TOOL_BUDGET
    user = agent.agent.calls[0]["user"]
    assert "CLAIM: claimed defect" in user and "FILE: x.py:5" in user and "SEVERITY: blocker" in user
    assert agent.agent.calls[0]["config"] == {
        "recursion_limit": review.VERIFY_RECURSION_LIMIT, "run_name": "verify:repro:correctness:0"}


def test_a_verifier_that_dies_or_rambles_casts_no_vote(rig):
    rig.reply = lambda prompt, user: RuntimeError("provider exploded")
    assert review.node_verify_one(verify_payload()) == {"verdicts": []}
    rig.reply = lambda prompt, user: "no json here"
    rig.model.repair_reply = "none either"
    assert review.node_verify_one(verify_payload()) == {"verdicts": []}


# --------------------------------------------------------------- dedupe / similarity
def test_similar_needs_the_same_file_and_either_the_same_line_or_overlapping_titles():
    base = {"file": "a.py", "line": 10, "title": "retry loop never sleeps between attempts"}
    assert review.similar(base, {**base, "title": "unrelated words entirely"})          # same line
    assert review.similar(base, {**base, "line": 99, "title": "retry loop never sleeps"})  # >=0.5 overlap
    assert not review.similar(base, {**base, "line": 99, "title": "unrelated words entirely"})
    assert not review.similar(base, {**base, "file": "b.py"})
    assert not review.similar({"file": "a.py", "line": 1, "title": "is it"},
                              {"file": "a.py", "line": 2, "title": "is it"}), "2-letter words carry no signal"


def test_dedupe_keeps_the_highest_severity_and_every_raising_dimension():
    findings = [
        {**finding("shared bug", "minor"), "key": "wiring:0", "dimension": "wiring"},
        {**finding("shared bug", "blocker"), "key": "correctness:0", "dimension": "correctness"},
        {**finding("shared bug", "major"), "key": "invariants:0", "dimension": "invariants"},
        {**finding("elsewhere", "major", file="z.py"), "key": "wiring:1", "dimension": "wiring"},
        {"key": "x:error", "dimension": "x", "severity": "meta", "title": "reviewer failed"},
    ]
    out = review.dedupe_findings(findings)
    assert [(f["file"], f["severity"], f["key"]) for f in out] == [
        ("x.py", "blocker", "correctness:0"), ("z.py", "major", "wiring:1")]
    assert out[0]["dimensions"] == ["correctness", "invariants", "wiring"]


# --------------------------------------------------------------- node_report
def vote(key, refuted=False, inflated=False, why="r"):
    return {"key": key, "lens": "repro", "refuted": refuted, "inflated": inflated, "reasoning": why}


def report(findings=None, deduped=None, verdicts=None, prev=None):
    pr = {"prev_findings": prev} if prev is not None else {}
    state = {"findings": findings or [], "deduped": deduped or [], "verdicts": verdicts or [], "pr": pr}
    return review.node_report(state)["result"]


def cand(key, severity="blocker", **kw):
    return {**finding(f"t-{key}", severity, **kw), "key": key, "dimension": "d", "dimensions": ["d"]}


def test_a_majority_of_refuting_verifiers_kills_the_finding():
    f = cand("d:0")
    res = report([f], [f], [vote("d:0", True, why="guard upstream"), vote("d:0", True, why="never called"),
                            vote("d:0", False)])
    assert res["findings"] == []
    assert res["killed"][0]["refutations"] == ["guard upstream", "never called"]
    assert res["stats"]["killed"] == 1 and res["stats"]["kept"] == 0


def test_an_even_split_does_not_kill_the_finding():
    f = cand("d:0")
    res = report([f], [f], [vote("d:0", True), vote("d:0", False)])
    assert [x["verification"] for x in res["findings"]] == ["1/2 upheld"]


def test_a_blocker_with_no_surviving_verifier_vote_is_killed_not_trusted():
    f = cand("d:0")
    res = report([f], [f], [])
    assert res["killed"][0]["refutations"] == ["no verifier vote survived"]


def test_an_upheld_finding_keeps_its_severity_and_records_the_vote():
    f = cand("d:0")
    res = report([f], [f], [vote("d:0")] * 3)
    (kept,) = res["findings"]
    assert (kept["severity"], kept["verification"]) == ("blocker", "3/3 upheld")
    assert res["stats"]["downgraded"] == 0


def test_a_majority_calling_it_inflated_downgrades_one_step():
    blocker, major = cand("d:0", "blocker", line=1), cand("d:1", "major", line=2)
    votes = [vote("d:0", inflated=True), vote("d:0", inflated=True), vote("d:0"),
             vote("d:1", inflated=True), vote("d:1", inflated=True), vote("d:1")]
    res = report([blocker, major], [blocker, major], votes)
    got = {f["key"]: (f["severity"], f["verification"]) for f in res["findings"]}
    assert got == {"d:0": ("major", "3/3 upheld, severity downgraded"),
                   "d:1": ("minor", "3/3 upheld, severity downgraded")}
    assert res["stats"]["downgraded"] == 2


def test_a_minority_inflated_vote_does_not_downgrade():
    f = cand("d:0")
    res = report([f], [f], [vote("d:0", inflated=True), vote("d:0"), vote("d:0")])
    assert res["findings"][0]["severity"] == "blocker"


def test_minors_pass_through_unverified_and_findings_sort_by_severity_then_file():
    items = [cand("d:0", "minor", file="a.py"), cand("d:1", "major", file="z.py"),
             cand("d:2", "nit", file="a.py"), cand("d:3", "major", file="b.py")]
    res = report(items, items, [vote("d:1"), vote("d:3")])
    assert [(f["severity"], f["file"]) for f in res["findings"]] == [
        ("major", "b.py"), ("major", "z.py"), ("minor", "a.py"), ("nit", "a.py")]
    assert res["stats"]["raw"] == 4


def test_continuity_marks_still_open_new_and_resolved_findings():
    open_f, new_f = cand("d:0", "minor", file="a.py", line=10), cand("d:1", "minor", file="b.py", line=3)
    prev = [{"file": "a.py", "line": 10, "title": "old wording"},
            {"file": "gone.py", "line": 1, "title": "fixed since"}]
    res = report([open_f, new_f], [open_f, new_f], prev=prev)
    assert {f["key"]: f["continuity"] for f in res["findings"]} == {"d:0": "still open", "d:1": "new"}
    assert res["resolved"] == [prev[1]] and res["stats"]["resolved"] == 1
    assert report([new_f], [new_f])["stats"]["resolved"] is None, "no previous review -> no resolved count"


# --------------------------------------------------------------- render_markdown
def rendered_fixture():
    blocker = {**cand("d:0", "blocker"), "dimensions": ["correctness", "wiring"],
               "verification": "3/3 upheld", "continuity": "new"}
    major = {**cand("d:1", "major", file="m.py", line=2), "continuity": "still open"}
    minors = [cand("d:2", "minor", file="n.py", line=3), cand("d:3", "minor", file="o.py", line=4)]
    result = {
        "findings": [blocker, major, *minors],
        "killed": [{**cand("d:4"), "refutations": ["R" * 300]}],
        "resolved": [{"file": "old.py", "line": 9, "title": "fixed since"}],
        "stats": {"raw": 6, "kept": 4, "killed": 1, "downgraded": 2, "meta": ["reviewer failed: boom"],
                  "gaps": 1, "cap_drops": 0, "resolved": 1},
    }
    return review.render_markdown(PR, result, "## Product review (advisory)\n\nfine")


def test_report_header_summary_and_coverage_notes():
    md = rendered_fixture()
    assert md.startswith("## Multi-agent review of PR #7 -- Add retries\n")
    assert "Blocker-free." not in md
    assert ("**1 blocker, 1 major, 2 minor** -- 6 raw finding(s); 1 killed by 3-lens adversarial "
            "verification; 2 downgraded as severity-inflated.") in md
    assert "Vs the previous review: 3 new, 1 still open, 1 resolved." in md
    assert "_coverage note: reviewer failed: boom_" in md


def test_report_groups_by_severity_in_order_and_pluralises_only_repeats():
    md = rendered_fixture()
    assert md.index("### Blocker\n") < md.index("### Major\n") < md.index("### Minors\n")
    assert "### Blockers" not in md and "### Majors" not in md


def test_report_finding_lines_carry_location_dimensions_verification_and_continuity():
    md = rendered_fixture()
    assert "- **`x.py:5` -- t-d:0** _(correctness, wiring, verified 3/3 upheld; new)_" in md
    assert "- **`m.py:2` -- t-d:1** _(d; still open)_" in md
    assert "  - Why: because\n  - Failure: input -> wrong output\n  - Fix: fix it" in md


def test_report_lists_resolved_findings_killed_claims_and_the_advisory_last():
    md = rendered_fixture()
    assert "### Resolved since the previous review (1)\n\n- `old.py:9` fixed since" in md
    assert "<summary>Killed by verification (1)</summary>" in md
    assert "- `x.py:5` t-d:4 -- " + "R" * 200 + "\n" in md and "R" * 201 not in md
    assert md.rstrip().endswith("## Product review (advisory)\n\nfine")


def test_a_clean_report_says_so_plainly():
    stats = {"raw": 0, "kept": 0, "killed": 0, "downgraded": 0, "meta": [], "gaps": 0,
             "cap_drops": 0, "resolved": None}
    md = review.render_markdown(PR, {"findings": [], "killed": [], "resolved": [], "stats": stats}, "")
    assert "Blocker-free." in md and "**no findings** -- 0 raw finding(s)" in md
    assert "Vs the previous review" not in md and "<details>" not in md


# --------------------------------------------------------------- extract_json / repair
@pytest.mark.parametrize("text,expected", [
    ('```json\n{"a": {"b": 1}}\n```', {"a": {"b": 1}}),
    ('```\n{"a": 1}\n```', {"a": 1}),
    ('Here you go: {"a": [1, 2]} hope that helps', {"a": [1, 2]}),
    ('```json\n{"a": 1}\n``` and later ```json\n{"a": 2}\n```', {"a": 1}),
    ([{"type": "text", "text": '{"a": 1}'}], {"a": 1}),
])
def test_extract_json_finds_the_object_in_model_output(text, expected):
    assert review.extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "no braces at all", '{"a": 1', '{"a": }', "} {", None])
def test_extract_json_returns_none_for_unusable_output(text):
    assert review.extract_json(text) is None


def test_parse_json_or_repair_skips_the_model_when_the_output_already_parses():
    class Boom:
        def invoke(self, *a):
            raise AssertionError("no repair call expected")

    assert review.parse_json_or_repair('{"ok": true}', "hint", Boom(), 5) == {"ok": True}


def test_parse_json_or_repair_asks_once_with_the_schema_and_bounded_input():
    model = FakeModel()
    model.repair_reply = '{"fixed": true}'
    assert review.parse_json_or_repair("junk " * 10_000, "SCHEMA-HINT", model, 5) == {"fixed": True}
    (messages,) = model.invocations
    assert "SCHEMA-HINT" in messages[0][1] and len(messages[1][1]) == 20_000
    model.repair_reply = "still junk"
    assert review.parse_json_or_repair("junk", "hint", model, 5) is None


# --------------------------------------------------------------- invoke_with_backoff
class Flaky:
    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def invoke(self, messages, config=None):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"ok": messages, "config": config}


def test_backoff_retries_transient_faults_with_growing_pauses(monkeypatch):
    sleeps = []
    monkeypatch.setattr(review.time, "sleep", sleeps.append)
    made = []

    def factory():
        made.append(Flaky([]) if len(made) == 3 else Flaky([RuntimeError("503 Service Unavailable")]))
        return made[-1]

    out = review.invoke_with_backoff(factory, {"m": 1}, {"c": 2})
    assert out == {"ok": {"m": 1}, "config": {"c": 2}}
    assert sleeps == [20, 45, 90]
    assert len(made) == 4, "every attempt builds a fresh agent"


def test_backoff_gives_up_after_the_last_pause_and_raises_the_original_error(monkeypatch):
    sleeps = []
    monkeypatch.setattr(review.time, "sleep", sleeps.append)
    made = []
    error = RuntimeError("Server disconnected without sending a response")

    def factory():
        made.append(Flaky([error]))
        return made[-1]

    with pytest.raises(RuntimeError) as exc:
        review.invoke_with_backoff(factory, {}, {})
    assert exc.value is error
    assert len(made) == len(review.RETRY_SLEEPS) + 1 and sleeps == list(review.RETRY_SLEEPS)


def test_backoff_does_not_retry_a_deterministic_failure(monkeypatch):
    sleeps = []
    monkeypatch.setattr(review.time, "sleep", sleeps.append)
    agent = Flaky([ValueError("schema validation failed")])
    with pytest.raises(ValueError):
        review.invoke_with_backoff(lambda: agent, {}, {})
    assert agent.calls == 1 and sleeps == []


# --------------------------------------------------------------- gh_api_list / fetch_thread
def test_gh_api_list_reads_arrays_however_gh_paginates(monkeypatch):
    seen = []
    outputs = {
        "merged": '[{"id": 1}, {"id": 2}]',
        "concatenated": '[{"id": 1}]\n[{"id": 2}]\n\n[{"id": 3}]',
        "single object": '{"id": 9}',
        "empty": "  \n",
    }
    for name, text in outputs.items():
        monkeypatch.setattr(review, "run_cmd", lambda cmd, timeout=120, t=text: seen.append(cmd) or t)
        got = [x["id"] for x in review.gh_api_list("repos/o/r/issues/1/comments")]
        assert got == {"merged": [1, 2], "concatenated": [1, 2, 3], "single object": [9], "empty": []}[name]
    assert seen[0] == ["gh", "api", "--paginate", "repos/o/r/issues/1/comments"]


def _gh_items(monkeypatch, comments, reviews=()):
    monkeypatch.setattr(review, "gh_api_list",
                        lambda path: list(comments) if "/comments" in path else list(reviews))


def comment(login, body, ts="2026-07-24T00:00:00Z", kind="User"):
    return {"body": body, "created_at": ts, "user": {"login": login, "type": kind}}


def test_fetch_thread_fails_soft_when_gh_errors(monkeypatch):
    def fail(path):
        raise RuntimeError("HTTP 502")

    monkeypatch.setattr(review, "gh_api_list", fail)
    assert review.fetch_thread(7) == ("", [], "", "")


def test_fetch_thread_keeps_only_lane_human_and_claude_comments(monkeypatch):
    _gh_items(monkeypatch, [
        comment("alice", "human reply", "2026-07-24T01:00:00Z"),
        comment("github-actions[bot]", "## \U0001F9EA lane report", "2026-07-24T02:00:00Z", "Bot"),
        comment("github-actions[bot]", "deploy finished", "2026-07-24T03:00:00Z", "Bot"),
        comment("claude-review[bot]", "claude says", "2026-07-24T04:00:00Z", "Bot"),
        comment("dependabot[bot]", "bump", "2026-07-24T05:00:00Z", "Bot"),
        comment("alice", "   ", "2026-07-24T06:00:00Z"),
    ])
    thread, prev, head, scope = review.fetch_thread(7)
    assert [line for line in thread.splitlines() if line.startswith("[")] == [
        "[alice 2026-07-24T01:00]", "[github-actions[bot] 2026-07-24T02:00]",
        "[claude-review[bot] 2026-07-24T04:00]"]
    assert (prev, head, scope) == ([], "", "")


def test_fetch_thread_continuity_follows_the_latest_graph_review_only(monkeypatch):
    def graph_review(head, title):
        return (f"{review.GRAPH_COMMENT_PREFIX} (model)\n\n"
                f"<!-- graph-review-head:{head} mode:discovery -->\n\n"
                f"### Blockers\n\n- **`a.py:1` -- {title}**\n")

    _gh_items(monkeypatch, [
        comment("github-actions[bot]", graph_review("1" * 40, "old finding"), "2026-07-24T01:00:00Z", "Bot"),
        comment("github-actions[bot]", graph_review("2" * 40, "new finding"), "2026-07-24T02:00:00Z", "Bot"),
    ])
    _, prev, head, scope = review.fetch_thread(7)
    assert [f["title"] for f in prev] == ["new finding"]
    assert head == "2" * 40 and "new finding" in scope and "old finding" not in scope


def test_fetch_thread_bounds_what_it_hands_to_the_prompt(monkeypatch):
    _gh_items(monkeypatch, [comment("alice", f"msg {i} " + "x" * 3000, f"2026-07-24T{i:02d}:00:00Z")
                            for i in range(12)])
    thread = review.fetch_thread(7)[0]
    assert len(thread) <= 10_000
    assert "msg 11" in thread and "msg 0 " not in thread, "the recent tail wins"


# --------------------------------------------------------------- node_fetch (discovery path)
META = {"number": 5, "title": "Add retries", "author": {"login": "dev"}, "baseRefName": "main",
        "headRefName": "feat/x", "body": None, "additions": 4, "deletions": 2,
        "files": [{"path": "a.py"}, {"path": "b/c.py"}]}


class Cmds:
    """Scripted run_cmd: records every command and answers from a table by prefix."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, args, timeout=120, cwd=None):
        self.calls.append(list(args))
        for prefix, answer in self.answers:
            if list(args[:len(prefix)]) == prefix:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        return ""


@pytest.fixture
def fetch_env(monkeypatch):
    cmds = Cmds([
        (["gh", "pr", "list"], '[{"number": 5}]'),
        (["gh", "pr", "view"], json.dumps(META)),
        (["git", "rev-parse", "FETCH_HEAD"], "c" * 40 + "\n"),
        (["gh", "pr", "diff", "5"], "FULL PR DIFF"),
    ])
    monkeypatch.setattr(review, "run_cmd", cmds)
    monkeypatch.setattr(review, "fetch_thread", lambda n: ("THREAD", [{"file": "a.py", "line": 1,
                                                                        "title": "t"}], "", ""))
    monkeypatch.setattr(review, "pull_learnings", lambda files: "")
    monkeypatch.setattr(review, "pull_principles", lambda: "")
    monkeypatch.setattr(review, "pull_blast_radius", lambda base, head: "")
    monkeypatch.setattr(review, "pull_stale_guards", lambda base, head: "")
    monkeypatch.setattr(review, "build_file_pack", lambda sha, files: "FILE-PACK")
    monkeypatch.setattr(review, "make_model", lambda *a, **k: None)
    return cmds


def fetch_state(**kw):
    return {"pr_ref": "feat/x", "provider": "azure", "model": "m", "timeout": 1, **kw}


def test_fetch_resolves_a_branch_name_and_reviews_the_full_pr_diff(fetch_env):
    out = review.node_fetch(fetch_state())
    pr = out["pr"]
    assert fetch_env.calls[0] == ["gh", "pr", "list", "--head", "feat/x", "--json", "number", "--limit", "1"]
    assert ["git", "fetch", "origin", "pull/5/head"] in fetch_env.calls
    assert (pr["number"], pr["author"], pr["headSha"], pr["body"]) == (5, "dev", "c" * 40, "")
    assert (pr["files"], pr["baseRef"], pr["headRef"]) == (["a.py", "b/c.py"], "main", "feat/x")
    assert (pr["thread"], pr["review_mode"]) == ("THREAD", "discovery")
    assert out["diff"] == "FULL PR DIFF" and out["pack"] == "FILE-PACK"
    assert not any(c[:2] == ["git", "diff"] for c in fetch_env.calls), "discovery never reviews a delta"


def test_fetch_numeric_ref_skips_the_branch_lookup(fetch_env):
    review.node_fetch(fetch_state(pr_ref="5"))
    assert not any(c[:3] == ["gh", "pr", "list"] for c in fetch_env.calls)


def test_fetch_fails_loudly_when_a_branch_has_no_open_pr(fetch_env):
    fetch_env.answers.insert(0, (["gh", "pr", "list"], "[]"))
    with pytest.raises(RuntimeError, match="no open PR for branch 'feat/x'"):
        review.node_fetch(fetch_state())


def test_fetch_composes_reachability_and_guard_sections_into_the_pack(fetch_env, monkeypatch, capsys):
    monkeypatch.setattr(review, "pull_blast_radius",
                        lambda base, head: "ENTRY POINTS AFFECTED BUT NOT IN THE DIFF: 3\n  app/page.tsx")
    monkeypatch.setattr(review, "pull_stale_guards", lambda base, head: "2 guard run(s) predate a field")
    pack = review.node_fetch(fetch_state())["pack"]
    assert pack.index("reachability of this diff") < pack.index("guard runs this diff's new fields")
    assert pack.index("guard runs") < pack.index("FILE-PACK")
    err = capsys.readouterr().err
    assert "[fetch] reach: 3 entry point(s) affected but not in the diff" in err
    assert "[fetch] stale guards: 2 guard run(s)" in err


def test_verification_without_a_prior_marked_review_is_refused(fetch_env):
    with pytest.raises(RuntimeError, match="verification mode requires a prior marked graph review"):
        review.node_fetch(fetch_state(mode="verification"))


def test_verification_fetches_the_previous_head_when_it_is_not_local(fetch_env, monkeypatch):
    previous = "d" * 40
    monkeypatch.setattr(review, "fetch_thread", lambda n: ("T", [], previous, "### Blockers\n- x"))
    fetch_env.answers[:0] = [
        (["git", "cat-file"], RuntimeError("not a valid object")),
        (["git", "diff", "--unified=80"], "DELTA"),
        (["git", "diff", "--name-only"], "fix.py\n"),
        (["git", "diff", "--numstat"], "5\t2\tfix.py\n-\t-\tlogo.png\n"),
    ]
    out = review.node_fetch(fetch_state(mode="verification"))
    assert ["git", "fetch", "origin", previous] in fetch_env.calls
    assert out["diff"] == "DELTA" and out["pr"]["files"] == ["fix.py"]
    assert (out["pr"]["additions"], out["pr"]["deletions"]) == (5, 2), "binary '-' rows are not counted"
    assert out["pr"]["previousReviewHead"] == previous


# --------------------------------------------------------------- reach / guard sections
class Runner:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    def __call__(self, args, **kw):
        self.calls.append(list(args))
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        code, out = r
        return subprocess.CompletedProcess(args, code, out, "")


def test_blast_radius_prefers_the_remote_ref_then_the_bare_branch(monkeypatch):
    run = Runner([(2, ""), (0, "ENTRY POINTS AFFECTED BUT NOT IN THE DIFF: 1\n app/page.tsx\n")])
    monkeypatch.setattr(review.subprocess, "run", run)
    out = review.pull_blast_radius("main", "abc")
    assert out.endswith("app/page.tsx") and out.startswith("ENTRY POINTS")
    assert [c[c.index("--base") + 1] for c in run.calls] == ["origin/main", "main"]
    assert all(c[c.index("--head") + 1] == "abc" for c in run.calls)


def test_blast_radius_stays_quiet_on_failure_and_is_capped(monkeypatch):
    monkeypatch.setattr(review.subprocess, "run", Runner([(0, "reach: nothing"), (1, "ENTRY POINTS AFFECTED")]))
    assert review.pull_blast_radius("main", "abc") == ""
    monkeypatch.setattr(review.subprocess, "run", Runner([OSError("no interpreter")]))
    assert review.pull_blast_radius("main", "abc") == "", "optional context must never kill the reviewer"
    big = "ENTRY POINTS AFFECTED " + "x" * 20_000
    monkeypatch.setattr(review.subprocess, "run", Runner([(0, big)]))
    assert len(review.pull_blast_radius("main", "abc")) == review.BLAST_PACK_CAP


def test_stale_guards_say_nothing_for_a_clean_run_and_survive_a_missing_interpreter(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(review.subprocess, "run", Runner([(0, "\n  0 guard run(s) predate a new field\n")] * 2))
    assert review.pull_stale_guards("main", "abc") == ""
    monkeypatch.setattr(review.subprocess, "run", Runner([OSError("gone")]))
    assert review.pull_stale_guards("main", "abc") == ""
    monkeypatch.setattr(review.subprocess, "run", Runner([(0, "2 guard run(s) predate a new field\n" + "y" * 9000)]))
    assert len(review.pull_stale_guards("main", "abc")) == review.GUARDS_PACK_CAP


# --------------------------------------------------------------- make_vertex_cache
def _fake_vertex(monkeypatch, fail=None):
    seen = {}

    class CachedContent:
        @staticmethod
        def create(**kw):
            if fail:
                raise RuntimeError(fail)
            seen["create"] = kw
            return types.SimpleNamespace(name="cachedContents/42")

    root, preview = types.ModuleType("vertexai"), types.ModuleType("vertexai.preview")
    caching, models = types.ModuleType("vertexai.preview.caching"), types.ModuleType("vertexai.generative_models")
    root.__path__ = preview.__path__ = []
    root.init = lambda **kw: seen.setdefault("init", kw)
    caching.CachedContent = CachedContent
    preview.caching = caching
    models.Content = lambda role, parts: (role, parts)
    models.Part = types.SimpleNamespace(from_text=lambda text: ("part", text))
    for name, mod in {"vertexai": root, "vertexai.preview": preview,
                      "vertexai.preview.caching": caching, "vertexai.generative_models": models}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return seen


def test_vertex_cache_is_opt_in_and_ignores_an_empty_pack(monkeypatch):
    seen = _fake_vertex(monkeypatch)
    monkeypatch.delenv("GEMINI_VERTEX_CACHE", raising=False)
    assert review.make_vertex_cache("pack text", "gemini-x") is None
    monkeypatch.setenv("GEMINI_VERTEX_CACHE", "1")
    assert review.make_vertex_cache("   ", "gemini-x") is None
    assert seen == {}


def test_vertex_cache_creates_a_cached_content_with_the_requested_ttl(monkeypatch, capsys):
    seen = _fake_vertex(monkeypatch)
    monkeypatch.setenv("GEMINI_VERTEX_CACHE", "1")
    assert review.make_vertex_cache("pack text", "gemini-x", ttl_minutes=5) == "cachedContents/42"
    kw = seen["create"]
    assert kw["model_name"] == "gemini-x" and kw["ttl"] == datetime.timedelta(minutes=5)
    assert kw["contents"] == [("user", [("part", "pack text")])]
    assert "[cache] vertex context cache created (9 chars)" in capsys.readouterr().err


def test_vertex_cache_failure_runs_the_lane_uncached(monkeypatch, capsys):
    _fake_vertex(monkeypatch, fail="quota exceeded")
    monkeypatch.setenv("GEMINI_VERTEX_CACHE", "1")
    assert review.make_vertex_cache("pack text", "gemini-x") is None
    assert "vertex caching unavailable (quota exceeded) -- running uncached" in capsys.readouterr().err


# --------------------------------------------------------------- the whole graph
def test_the_compiled_graph_runs_end_to_end_with_scripted_agents(rig, monkeypatch):
    """fetch -> brief -> six reviewers + advisory -> dedupe -> 3-lens verify -> report."""
    from langgraph.checkpoint.memory import InMemorySaver

    # LangGraph resolves State's string annotations through sys.modules[State.__module__];
    # a script run as __main__ is registered there, a spec-loaded test module is not.
    monkeypatch.setitem(sys.modules, review.__name__, review)
    pr = {**PR, "prev_findings": [{"file": "a.py", "line": 5, "title": "real blocker"},
                                  {"file": "old.py", "line": 1, "title": "fixed since"}]}
    monkeypatch.setattr(review, "node_fetch", lambda state: {"pr": pr, "diff": "d", "pack": ""})

    by_dim = {
        "correctness": [finding("real blocker", "blocker", "a.py", 5),
                        finding("phantom blocker", "blocker", "b.py", 9)],
        "wiring": [finding("real blocker, restated", "blocker", "a.py", 5)],
        "invariants": [finding("inflated major", "major", "c.py", 2)],
        "security-data": [finding("small nit", "minor", "d.py", 1)],
        "simplicity-reuse": RuntimeError("reviewer crashed"),
    }
    # (repro, context, severity) votes as (refuted, inflated) per claim title
    verdicts = {"real blocker": [(False, False)] * 3,
                "phantom blocker": [(True, False), (True, False), (False, False)],
                "inflated major": [(False, False), (False, True), (False, True)]}
    lens_order = list(review.LENSES)

    def reply(prompt, user):
        if prompt == review.ADVISORY_PROMPT:
            return "## Product review (advisory)\n\nship it"
        if prompt.startswith("You are an adversarial verifier"):
            lens = next(name for name, text in review.LENSES.items() if text in prompt)
            title = user.split("CLAIM: ", 1)[1].split("\n", 1)[0]
            refuted, inflated = verdicts[title][lens_order.index(lens)]
            return json.dumps({"refuted": refuted, "inflated": inflated, "reasoning": f"{lens} on {title}"})
        dim = prompt.split("Your single dimension: ", 1)[1].split(".", 1)[0]
        out = by_dim.get(dim, [])
        return out if isinstance(out, Exception) else json.dumps({"findings": out})

    rig.reply = reply
    graph = review.build_graph(InMemorySaver())
    final = graph.invoke(
        {"pr_ref": "7", "provider": "azure", "model": "m", "light_model": "", "timeout": 5, "mode": "discovery"},
        {"configurable": {"thread_id": "t1"}, "max_concurrency": 4, "recursion_limit": 100})

    result = final["result"]
    got = {f["title"]: (f["severity"], f.get("verification"), f["continuity"]) for f in result["findings"]}
    assert got == {
        "real blocker": ("blocker", "3/3 upheld", "still open"),
        "inflated major": ("minor", "3/3 upheld, severity downgraded", "new"),
        "small nit": ("minor", None, "new"),
    }
    real = next(f for f in result["findings"] if f["title"] == "real blocker")
    assert real["dimensions"] == ["correctness", "wiring"], "the restated claim merged into one verified finding"
    assert [k["title"] for k in result["killed"]] == ["phantom blocker"]
    assert result["resolved"] == [pr["prev_findings"][1]]
    stats = result["stats"]
    assert (stats["raw"], stats["kept"], stats["killed"], stats["downgraded"], stats["gaps"]) == (5, 3, 1, 1, 1)
    assert stats["meta"] == ["reviewer failed: reviewer crashed"]
    assert final["advisory"].endswith("ship it")
    # 3 deduped blocker/major findings x 3 lenses, and nothing was verified twice.
    verifiers = [a for a in rig.agents if a.prompt.startswith("You are an adversarial verifier")]
    assert len(verifiers) == 9
