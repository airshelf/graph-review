"""The agent-loop convergence wall, imported by the sibling PR reviewer.

WHAT IT DOES. A langchain `wrap_model_call` middleware. Once an agent is stuck —
it has burned its tool-round budget, or it keeps re-issuing calls the tool layer
already refused — the wall STRIPS every tool, orders a final answer, and then
CLEARS any tool call the model emits regardless. That converts a recursion-limit
death (a DEAD DIMENSION: code silently unreviewed, output indistinguishable from
a clean review) into graceful degradation: a real, if shallower, answer.

BOTH HALVES ARE LOAD-BEARING. Unbinding tools is advisory — the ToolNode stays
wired into the graph, so the router still follows a tool call the model emits
anyway. Some models conclude when tools disappear; others keep calling them
until the recursion ceiling. Measured on real reviews: the wall fired within
its budget, but repeated tool calls still killed dimensions. Only clearing the
calls (drop_tool_calls) makes termination independent of provider cooperation.

WHY IT IS SEPARATE. One definition and one behavioural test suite keep every
agent build on the same termination contract. Copying the middleware into each
caller would let a fix reach one loop while another kept returning no review.

Pure and dependency-light on purpose: langchain is imported INSIDE the factory so
that importing this module costs nothing in a context that never builds an agent.
"""
from __future__ import annotations

# Must match the tool layer's repeat-refusal string in review.py's once() in
# make_tools. Wall 2 keys off wall 1's wording, so a reworded refusal would
# silently unhook this; the reviewer tests pin the tool-layer marker too.
CONVERGENCE_MARKER = "(repeat call refused"

# Refused repeats tolerated before force-concluding. One refusal is already
# evidence that the tool call produced no new information; waiting only spends
# more of the remaining budget on the same loop.
CONVERGENCE_REFUSAL_LIMIT = 1

# Tool-round budget as a fraction of a node's recursion budget. review.py's own
# ratios are 40/100, 10/25, 9/24 — i.e. ~40%, always BELOW the agent's recursion
# cycles so the wall fires BEFORE the ceiling does. Raising the ceiling instead
# just lengthens a doomed loop.
CONVERGENCE_BUDGET_RATIO = 0.4


def content_to_text(content) -> str:
    """LLM message content -> plain text (native Gemini returns block lists)."""
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "") if isinstance(b, dict) else ""
            for b in content
        )
    return content or ""


def refused_repeat_count(messages: list) -> int:
    """How many tool results in history are the tool layer's repeat-refusal (i.e.
    the model ignoring wall 1). Robust to object- OR dict-form messages —
    ChatVertexAI's history shape must never silently zero this out. Pure, so it
    can be tested independently of a model or agent build."""
    n = 0
    for m in messages:
        typ = m.get("type") if isinstance(m, dict) else getattr(m, "type", "")
        if typ != "tool":
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        # startswith, NOT `in`: the refusal is the ENTIRE tool result, whereas a
        # legitimate result can CONTAIN the marker (grepping this very file finds
        # it) -- `in` would fake refusals and trip the wall spuriously.
        if content_to_text(content or "").lstrip().startswith(CONVERGENCE_MARKER):
            n += 1
    return n


def drop_tool_calls(response):
    """Clear tool calls off a model response so the agent loop TERMINATES.

    Unbinding tools (`override(tools=[])`) is only ADVISORY: it stops langchain
    from offering the tools, but the ToolNode stays wired into the graph, so the
    router still follows any tool call the model emits anyway. A cooperative
    model concludes; an uncooperative provider can keep emitting calls PAST the
    stripped-tools wall and die at the recursion ceiling, a DEAD DIMENSION.
    Clearing the calls makes termination structural: the router sees no tool
    call and goes to END no matter how uncooperative the provider is.

    Whatever TEXT the model produced is preserved -- that is the shallow-but-real
    answer the wall exists to salvage. If it produced none, the dimension fails
    visibly (unparseable) instead of silently burning to the ceiling.

    THE SALVAGE IS OFTEN EMPTY ON GEMINI, and callers must handle that. A
    ChatVertexAI turn that decides to call a function carries NO text, so on the
    forced turn Gemini emits a bare tool call, this clears it, and the salvage is
    "". Measured on an adversarial-verification judge: the wall fired on one
    refused repeat and the node saw an empty final reply on most runs.
    Re-asking the identical stripped request rarely recovered text; rebuilding
    an agent with tools reproduced the silence. A caller needing recovery can
    make a bare, tool-free model call with the research flattened into one
    plain-text turn. Recovery belongs to the caller, not this termination wall.

    Fails soft, like the tool layer: an unrecognised response shape is returned
    untouched rather than raising into the agent loop.
    """
    result = getattr(response, "result", None)
    if not result:
        return response
    for m in result:
        if not (getattr(m, "tool_calls", None) or getattr(m, "invalid_tool_calls", None)):
            continue
        # Mutate the message IN PLACE and hand back the original container.
        # Rebuilding the container instead (dataclasses.replace) would clear the
        # calls only on COPIES, so any response type that is not a dataclass
        # would fall back to returning the untouched original -- failing open
        # into exactly the recursion loop this exists to stop. These are the same
        # objects the router reads, so in-place is both simpler and the only
        # shape-independent option.
        for attr in ("tool_calls", "invalid_tool_calls"):
            try:
                setattr(m, attr, [])
            except Exception:                       # frozen/exotic message type
                pass
        # additional_kwargs carries the provider-native copy (OpenAI
        # `tool_calls`, Gemini `function_call`). The ROUTER only reads
        # `.tool_calls`, so this is belt-and-braces against a serializer that
        # rehydrates from the provider payload.
        ak = getattr(m, "additional_kwargs", None)
        if isinstance(ak, dict):
            ak.pop("tool_calls", None)
            ak.pop("function_call", None)
    return response


def make_convergence_middleware(model_call_budget: int):
    """Build the wall for one agent. See the module docstring for what it does.

    The round count is the middleware's OWN invocation counter (one call per model
    turn), NOT a scan of the message history: counting AIMessages-with-tool_calls
    undercounted against real ChatVertexAI history, letting a reviewer exceed its
    tool budget. A closure counter is immune to how a provider structures its
    messages. Fresh per agent build, so a per-retry rebuild resets it (like the
    tool layer's once() cache).
    """
    from langchain.agents.middleware import wrap_model_call
    from langchain_core.messages import SystemMessage

    turns = {"n": 0}

    @wrap_model_call
    def force_convergence(request, handler):
        turns["n"] += 1
        stuck = (turns["n"] > model_call_budget
                 or refused_repeat_count(request.state["messages"]) >= CONVERGENCE_REFUSAL_LIMIT)
        if request.tools and stuck:
            note = ("\n\nTOOL BUDGET EXHAUSTED -- you have gathered enough evidence (or are "
                    "repeating refused calls). Do NOT call any more tools. Emit your final "
                    "answer in the required output format NOW, from what you already have.")
            # Plain STRING content, not a block list: the system message carries no
            # cache_control breakpoint (that lives on the user message), so blocks buy
            # nothing here and a list-content SystemMessage is the shape non-anthropic
            # providers are least happy with. content_to_text flattens either form.
            sys_msg = getattr(request, "system_message", None)
            base = content_to_text(getattr(sys_msg, "content", "") or "")
            request = request.override(tools=[], system_message=SystemMessage(content=base + note))
            # Structural, not advisory -- see drop_tool_calls. The stripped-tools
            # request is the model's one clean chance to answer; whatever it says
            # is kept, but it does NOT get to spend another loop iteration.
            return drop_tool_calls(handler(request))
        return handler(request)

    return force_convergence
