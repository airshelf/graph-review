"""Behaviour tests for the agent-loop convergence wall.

WHY THESE LIVE HERE. A construction-only smoke test passes even when the wall is
completely disabled. One definition deserves one real behavioural test suite,
so every agent build shares the same termination guarantees.

These are the invariants that matter, i.e. the ones whose failure is silent:
a wall that never fires produces a dead dimension (code unreviewed, output
indistinguishable from a clean review), and a wall
that fires too eagerly truncates every review into shallowness.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convergence import (  # noqa: E402
    CONVERGENCE_MARKER,
    CONVERGENCE_REFUSAL_LIMIT,
    content_to_text,
    make_convergence_middleware,
    refused_repeat_count,
)


class FakeRequest:
    """Stands in for langchain's ModelRequest: only .tools/.state/.system_message
    and .override() are touched by the wall."""

    def __init__(self, messages=None, tools=("show_file",)):
        self.tools = list(tools)
        self.state = {"messages": list(messages or [])}
        self.system_message = None
        self.overridden = None

    def override(self, **kw):
        self.overridden = kw
        for k, v in kw.items():
            setattr(self, k, v)
        return self


def _run(mw, req, times=1):
    """Drive the middleware `times` turns; return what the handler observed."""
    seen = []

    def handler(r):
        seen.append(r.tools)
        return "done"

    fn = mw.wrap_model_call        # @wrap_model_call yields an object, not a function
    for _ in range(times):
        fn(req, handler)
    return seen


def _tool_msg(text):
    class M:
        type = "tool"
        content = text
    return M()


# --- the wall fires -----------------------------------------------------------

def test_strips_tools_once_over_budget():
    """Turn N+1 past the budget must reach the model with NO tools, so it cannot
    loop another round. Without this the agent runs to the recursion ceiling and
    returns nothing at all."""
    mw = make_convergence_middleware(model_call_budget=1)
    req = FakeRequest()
    seen = _run(mw, req, times=2)
    assert seen[0] == ["show_file"], "turn 1 is within budget -- tools must survive"
    assert seen[1] == [], "turn 2 is over budget -- tools must be stripped"
    assert req.overridden is not None and req.overridden["tools"] == []


def test_strips_tools_when_refusal_limit_reached():
    """Ignoring the tool-layer refusal is the OTHER stuck signal: the model is
    burning turns re-issuing calls that already returned nothing new."""
    msgs = [_tool_msg(f"{CONVERGENCE_MARKER} -- you already made this call")
            for _ in range(CONVERGENCE_REFUSAL_LIMIT)]
    mw = make_convergence_middleware(model_call_budget=99)   # budget NOT the trigger
    seen = _run(mw, FakeRequest(messages=msgs))
    assert seen[0] == [], "refusal limit reached -- tools must be stripped"


def test_final_answer_is_ordered_not_just_tools_removed():
    """Stripping tools alone can leave a model narrating; the wall must also tell
    it to emit the final answer in the required format."""
    mw = make_convergence_middleware(model_call_budget=0)
    req = FakeRequest()
    _run(mw, req)
    sys_text = content_to_text(getattr(req.overridden["system_message"], "content", ""))
    assert "Do NOT call any more tools" in sys_text
    assert "final" in sys_text.lower()


# --- the wall stays out of the way --------------------------------------------

def test_passes_through_while_within_budget():
    """The common case: a healthy agent must keep every tool. A wall that fires
    early would silently shallow-out every review."""
    mw = make_convergence_middleware(model_call_budget=10)
    req = FakeRequest()
    assert _run(mw, req, times=3) == [["show_file"]] * 3
    assert req.overridden is None


def test_no_tools_means_no_override():
    """Already tool-less (e.g. a second wall pass) -- nothing to strip, and the
    system message must not accumulate the note twice."""
    mw = make_convergence_middleware(model_call_budget=0)
    req = FakeRequest(tools=[])
    _run(mw, req)
    assert req.overridden is None


def test_counter_is_per_build_not_global():
    """Budgets are per agent build, so a rebuild (invoke_with_backoff's retry)
    starts fresh. A shared counter would make retry 2 conclude immediately."""
    req = FakeRequest()
    _run(make_convergence_middleware(model_call_budget=1), req, times=2)
    fresh = FakeRequest()
    assert _run(make_convergence_middleware(model_call_budget=1), fresh)[0] == ["show_file"]


# --- refusal detection --------------------------------------------------------

def test_refused_repeat_count_handles_object_and_dict_messages():
    """ChatVertexAI's history shape must never silently zero this out -- a zeroed
    count unhooks wall 2 from wall 1."""
    msgs = [
        _tool_msg(f"{CONVERGENCE_MARKER} -- repeat"),
        {"type": "tool", "content": f"  {CONVERGENCE_MARKER} -- repeat"},   # leading ws
        {"type": "tool", "content": [{"text": f"{CONVERGENCE_MARKER} -- blocks"}]},
        {"type": "tool", "content": "a normal file listing"},
        {"type": "ai", "content": f"{CONVERGENCE_MARKER}"},                 # not a tool result
    ]
    assert refused_repeat_count(msgs) == 3


def test_refusal_must_start_the_result_not_merely_appear_in_it():
    """startswith, NOT `in`: a legitimate grep result can CONTAIN the marker
    (grepping the convergence module finds it). `in` would fake refusals and trip
    the wall on a perfectly healthy agent."""
    quoting = _tool_msg(f"convergence.py:31:CONVERGENCE_MARKER = \"{CONVERGENCE_MARKER}\"")
    assert refused_repeat_count([quoting] * 3) == 0


def test_content_to_text_flattens_block_lists():
    assert content_to_text([{"text": "a"}, "b", {"noise": 1}]) == "ab"
    assert content_to_text(None) == ""
    assert content_to_text("plain") == "plain"


# --- termination is STRUCTURAL, not advisory ----------------------------------
# The tests above drive a FakeRequest whose handler returns a bare string, so they
# can only prove the wall stripped tools from the REQUEST. That is exactly the
# blind spot that let real reviews lose dimensions with the wall installed and
# firing: unbinding tools does not unwire the ToolNode, so a model that emits
# tool calls anyway keeps looping to the recursion ceiling. These cover the
# RESPONSE half.

import dataclasses  # noqa: E402

import pytest  # noqa: E402

from convergence import drop_tool_calls  # noqa: E402


@dataclasses.dataclass
class FakeResponse:
    """Stands in for langchain's ModelResponse (fields: result, structured_response)."""
    result: list
    structured_response: object = None


class FakeAI:
    """Minimal pydantic-ish AIMessage: only what drop_tool_calls touches."""

    def __init__(self, content="", tool_calls=None, additional_kwargs=None):
        self.content = content
        self.tool_calls = list(tool_calls or [])
        self.invalid_tool_calls = []
        self.additional_kwargs = dict(additional_kwargs or {})

    def model_copy(self, update):
        new = FakeAI(self.content, self.tool_calls, self.additional_kwargs)
        for k, v in update.items():
            setattr(new, k, v)
        return new


def test_drop_tool_calls_clears_calls_but_keeps_the_text():
    """The salvaged answer is the POINT of the wall -- clearing must not eat it."""
    call = {"name": "show_file", "args": {}, "id": "t1"}
    resp = FakeResponse(result=[FakeAI(content="partial findings", tool_calls=[call],
                                       additional_kwargs={"function_call": call, "keep": 1})])
    out = drop_tool_calls(resp)
    msg = out.result[0]
    assert msg.tool_calls == [], "a surviving tool call routes the graph back to ToolNode"
    assert msg.content == "partial findings"
    # the provider-native copy must go too, or it can be re-inflated downstream
    assert "function_call" not in msg.additional_kwargs
    assert msg.additional_kwargs["keep"] == 1


def test_drop_tool_calls_works_on_a_NON_dataclass_response():
    """The response container's type must not matter. An earlier cut rebuilt it
    with `dataclasses.replace` and, on any non-dataclass shape, fell back to
    returning the ORIGINAL — clearing the calls only on discarded copies and
    failing OPEN into the recursion loop this exists to stop. ModelResponse is a
    dataclass today; that is not a contract."""

    class PlainResponse:                       # deliberately not a dataclass
        def __init__(self, result):
            self.result = result

    resp = PlainResponse([FakeAI(content="text",
                                 tool_calls=[{"name": "x", "args": {}, "id": "t1"}])])
    out = drop_tool_calls(resp)
    assert out.result[0].tool_calls == [], "tool call survived on a non-dataclass response"
    assert out.result[0].content == "text"


def test_drop_tool_calls_leaves_a_clean_response_untouched():
    resp = FakeResponse(result=[FakeAI(content="done")])
    assert drop_tool_calls(resp) is resp


def test_drop_tool_calls_fails_soft_on_an_unknown_shape():
    """Tool-layer discipline: never raise into the agent loop."""
    assert drop_tool_calls("not a response") == "not a response"
    assert drop_tool_calls(FakeResponse(result=[])).result == []


def test_wall_clears_tool_calls_the_model_emitted_anyway():
    """Through the real middleware: a stuck turn must come
    back with no tool calls even when the model insists on making them."""
    mw = make_convergence_middleware(model_call_budget=0)     # stuck immediately
    call = {"name": "show_file", "args": {}, "id": "t1"}
    resp = FakeResponse(result=[FakeAI(content="text", tool_calls=[call])])
    out = mw.wrap_model_call(FakeRequest(), lambda r: resp)
    assert out.result[0].tool_calls == [], (
        "wall let a tool call through -- the agent gets another loop iteration "
        "and dies at the recursion ceiling (dead dimension)")


# --- the salvaged answer can be EMPTY, and callers must know it ---------------

def test_a_textless_tool_call_salvages_NOTHING():
    """A textless forced turn must terminate without inventing an answer.

    Every other test here hands the wall a response whose content is
    "partial"/"text" -- they all quietly assume the model narrates alongside its
    tool call. Gemini does not: a ChatVertexAI turn that decides to call a
    function carries EMPTY content. So on the wall's forced turn Gemini emits a
    bare grep_repo call, drop_tool_calls clears it, and the "shallow but real
    answer" is the empty string.

    This is NOT a bug in the wall -- terminating the loop is its job, and it does
    that. It is a fact its CALLERS have to handle. Rebuilding an agent WITH tools
    can reproduce the same silence. A caller needing recovery can use a bare,
    tool-free model call outside the agent loop with flattened research.

    If someone ever makes the wall itself re-ask, this test should fail and be
    deleted deliberately. Re-asking the identical stripped request was tried on
    real reviews and rarely recovered text: the model can stay silent given the
    same conversation, so retries mostly just spend more calls.
    """
    call = {"name": "grep_repo", "args": {}, "id": "t1"}
    resp = FakeResponse(result=[FakeAI(content="", tool_calls=[call])])
    out = make_convergence_middleware(model_call_budget=0).wrap_model_call(
        FakeRequest(), lambda _r: resp)
    assert out.result[0].tool_calls == [], "termination must still be structural"
    assert content_to_text(out.result[0].content) == "", (
        "if this now returns text the wall gained a recovery path -- see the "
        "docstring before changing the caller's recovery policy")


def test_uncooperative_model_cannot_outlive_the_wall_in_a_real_agent_loop():
    """End-to-end against langchain's real create_agent, because the FakeRequest
    tests structurally cannot see this: a model can keep emitting calls past the
    stripped-tools wall and die at the recursion ceiling.
    The agent must now CONCLUDE instead of raising."""
    pytest.importorskip("langchain.agents")
    pytest.importorskip("langgraph")
    from langchain.agents import create_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langgraph.errors import GraphRecursionError

    turns = {"n": 0}

    class DefiantModel(BaseChatModel):
        """Emits a tool call on EVERY turn, bound or not."""

        @property
        def _llm_type(self):
            return "defiant"

        def bind_tools(self, tools, **kw):
            return self.bind(tools=tools, **kw)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            turns["n"] += 1
            msg = AIMessage(content="partial",
                            tool_calls=[{"name": "peek", "args": {"q": "x"},
                                         "id": f"t{turns['n']}"}])
            return ChatResult(generations=[ChatGeneration(message=msg)])

    @tool
    def peek(q: str) -> str:
        """Peek at something."""
        return "data"

    agent = create_agent(DefiantModel(), [peek],
                         middleware=[make_convergence_middleware(3)])
    try:
        out = agent.invoke({"messages": [("user", "go")]}, config={"recursion_limit": 12})
    except GraphRecursionError:                                  # pragma: no cover
        pytest.fail(f"agent died at the recursion ceiling after {turns['n']} turns "
                    "— the wall is advisory again")
    assert turns["n"] == 4, "must conclude on the first over-budget turn"
    # Assert the MECHANISM, not just that the loop stopped: the run has to end on
    # a message whose tool calls were cleared. Without this the test would still
    # pass if the agent halted for some unrelated reason (a changed langgraph
    # loop, a bypassed middleware) while the wall was quietly advisory again.
    last = out["messages"][-1]
    assert getattr(last, "tool_calls", []) == [], (
        "final message still carries a tool call — the router would route back "
        "to the ToolNode and the agent would loop to the ceiling")
    assert "function_call" not in (getattr(last, "additional_kwargs", None) or {})
    assert last.content == "partial", "the salvaged answer must survive the clearing"
