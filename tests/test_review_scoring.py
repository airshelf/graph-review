"""Pin for review.py's coverage-gap scoring split.

The regression this walls: a reviewer that hits the per-dimension finding CAP emits a
`<dim>:dropped` meta note (overflow -- it found MORE than the cap), which is the
OPPOSITE of a coverage gap. The old scorer counted every meta note as a gap, so
a thorough reviewer that capped 3 dimensions scored review_coverage_gaps=3 and
triggered a false alarm about dead dimensions. node_report must score ONLY
genuine deaths (`<dim>:error`) as gaps.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "review", Path(__file__).resolve().parents[1] / "review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def _meta(key, title="note"):
    return {"key": key, "dimension": key.split(":")[0], "severity": "meta", "title": title}


def _finding(dim, sev="minor"):
    return {"key": f"{dim}:0", "dimension": dim, "severity": sev, "title": f"{dim} issue",
            "file": f"{dim}.py", "line": 1}


def _stats(findings):
    return review.node_report({"findings": findings, "verdicts": [], "pr": {}})["result"]["stats"]


def test_cap_overflow_is_not_a_gap():
    # 3 dimensions capped (found too much), 0 genuine deaths.
    s = _stats([_meta("wiring:dropped"), _meta("spec:dropped"), _meta("simplicity:dropped"),
                _finding("correctness")])
    assert s["gaps"] == 0, "cap overflow must NOT count as a coverage gap"
    assert s["cap_drops"] == 3
    assert len(s["meta"]) == 3  # both kinds still render as coverage notes


def test_genuine_death_scores_a_gap():
    # 2 reviewers errored/unparseable -> 2 real gaps; 1 cap overflow -> 0.
    s = _stats([_meta("security:error", "reviewer failed: boom"),
                _meta("invariants:error", "reviewer output unparseable"),
                _meta("wiring:dropped")])
    assert s["gaps"] == 2
    assert s["cap_drops"] == 1
    assert len(s["meta"]) == 3


def test_clean_run_has_no_gaps():
    s = _stats([_finding("correctness"), _finding("wiring")])
    assert s["gaps"] == 0 and s["cap_drops"] == 0 and s["meta"] == []


class _Msg:
    """Minimal stand-in for a langchain message (tool_calls / type / content)."""
    def __init__(self, tool_calls=None, type="", content=""):
        if tool_calls is not None:
            self.tool_calls = tool_calls
        self.type = type
        self.content = content


def _refusal():
    return _Msg(type="tool", content="(repeat call refused -- you already made this exact call)")


def _ok_tool():
    return _Msg(type="tool", content="some real file content")


def test_refusal_count_catches_ignored_wall():
    # Observed loop shape: the model ignores once()'s refusal and
    # re-issues the identical grep. 3 refusals in history = force-conclude signal.
    msgs = [_ok_tool(), _refusal(), _ok_tool(), _refusal(), _refusal()]
    assert review._refused_repeat_count(msgs) == 3


def test_refusal_count_robust_to_dict_messages():
    # ChatVertexAI history can surface as dicts, not objects -- the count must not
    # silently zero out and let a reviewer wander past its budget.
    msgs = [{"type": "tool", "content": "(repeat call refused -- ...)"},
            {"type": "tool", "content": "real content"},
            {"type": "tool", "content": "(repeat call refused -- ...)"}]
    assert review._refused_repeat_count(msgs) == 2


def test_refusal_count_ignores_marker_inside_legitimate_output():
    # A real tool result can CONTAIN the marker -- grepping review.py finds the
    # literal refusal string. once() returns the refusal as the ENTIRE result, so
    # only a leading match counts; `in` would fake refusals and trip the wall.
    quoting = _Msg(type="tool", content=(
        'review.py:641:  return ("(repeat call refused -- you already made this")'))
    assert review._refused_repeat_count([quoting, quoting, quoting]) == 0


def test_refusal_count_ignores_non_tool_and_clean_results():
    # An assistant message that merely quotes the marker, and healthy tool
    # results, must not count as refusals.
    msgs = [_Msg(type="ai", content="I will avoid a (repeat call refused"),
            _ok_tool(), _ok_tool()]
    assert review._refused_repeat_count(msgs) == 0


def test_convergence_middleware_import_path_is_real():
    """Mechanical proof of the middleware import path.

    The graph reviewer has twice raised a BLOCKER claiming
    `langchain.agents.middleware` does not exist (stale model knowledge about
    langchain 1.x). It does — it is the documented home of `wrap_model_call` —
    and this test makes that a CI fact rather than an argument: if the path ever
    really moves, pytest reds here instead of every reviewer dying at runtime
    with an ImportError.
    """
    from langchain.agents.middleware import ModelRequest, wrap_model_call

    assert callable(wrap_model_call)
    assert ModelRequest is not None
    # And the production factory builds against it without raising.
    assert review.make_convergence_middleware(5) is not None
