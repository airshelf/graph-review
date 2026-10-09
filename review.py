#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "langgraph>=0.6",
#   "langgraph-checkpoint-sqlite>=2.0",
#   "langchain-openai>=0.3",
#   "langchain-google-genai>=2.1",
#   "langchain-google-vertexai>=2.0",
#   "anthropic[vertex]>=0.40",  # ChatAnthropicVertex imports the anthropic SDK lazily; vertex extra = google-auth
#   "langfuse>=4,<5",
#   "langchain>=1.3",  # convergence middleware needs create_agent's wrap_model_call
#                      # + ModelRequest.override(tools=[]) -- verified on 1.3.14; the
#                      # an older resolved wheel let the middleware no-op in CI,
#                      # leaving reviewers to hit the recursion ceiling.
# ]
# ///
"""Local multi-agent PR review with LangGraph and selectable model providers.

A local, git-versioned, cheaply re-runnable graph:

    fetch -> brief -> review (6 dimensions + product advisory, parallel)
          -> verify (3 adversarial lenses per blocker/major, majority-refute)
          -> report (pure Python: dedup, rank, markdown)

GitHub is a read-only data source (gh pr view/diff). NOTHING is ever posted -- the
tool prints a ready-to-post markdown review and the gh command to post it.

Time travel (cheap re-run) via SqliteSaver checkpoints:
    review.py --pr 123                          # fresh run, prints thread id
    review.py --history --thread pr123-...      # list checkpoints
    review.py --replay-from <checkpoint_id> --thread pr123-...
Nodes before the checkpoint replay from cache; later nodes re-execute (edit a
prompt below, replay from the pre-verify checkpoint, pay only for what changed).

"""

from __future__ import annotations

import argparse
import functools
import json
import operator
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

# Keep the convergence wall in one sibling module so fixes cannot leave a copied
# agent loop unprotected. Resolve imports from this script, not the target checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convergence import (  # noqa: E402
    CONVERGENCE_BUDGET_RATIO,  # noqa: F401
    CONVERGENCE_MARKER,
    CONVERGENCE_REFUSAL_LIMIT,
    content_to_text,
    make_convergence_middleware,
    refused_repeat_count as _refused_repeat_count,
)

STATE_DIR = Path(__file__).resolve().parent / ".state"
ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
ZAI_KEY_FILE = Path.home() / ".config" / "zai-key"
# Both Vertex providers use the user's project and Application Default Credentials.
# The location must support the selected model; there is no project fallback.
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
# Azure model names are deployment names, which may differ from upstream model IDs.
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
# Azure serves the Responses API only from this api-version onward; an older
# AZURE_OPENAI_API_VERSION pin plus a GPT-6 deployment would otherwise rebuild the
# silent loss of reviewer coverage that this path exists to prevent.
RESPONSES_MIN_API_VERSION = "2025-03-01-preview"


def api_version_at_least(have: str, need: str) -> bool:
    """Azure api-versions order by their YYYY-MM-DD prefix; the -preview/-GA suffix
    never reorders two different dates, so a prefix compare is the whole rule."""
    return (have or "")[:10] >= need[:10]
# Foundry's /models endpoint speaks an OpenAI-compatible API.
FOUNDRY_ENDPOINT = os.environ.get("AZURE_FOUNDRY_ENDPOINT")
FOUNDRY_API_VERSION = os.environ.get("AZURE_FOUNDRY_API_VERSION", "2024-05-01-preview")
# Gemini uses native clients so tool calls retain their thought_signature.
PROVIDERS = {
    # diff_cap / pack_cap: chars of diff / context-pack inlined per agent prompt.
    # Larger packs trade input tokens for fewer repeated file reads. Measured on
    # a real one-word diff: 45 LLM calls / 475k input tokens re-fetching context.
    # Gemini's API-key lane and GLM retain leaner budgets for token control.
    "openai": {"model": os.environ.get("OPENAI_MODEL"), "concurrency": 8,
               "diff_cap": 50_000, "pack_cap": 120_000},
    "gemini": {"model": "gemini-3.8-flash", "concurrency": 8,
               "diff_cap": 15_000, "pack_cap": 40_000},
    # Vertex uses ADC instead of an API key. No prompt_cache here: the message
    # cache_control breakpoint is specific to the Anthropic client.
    "gemini-vertex": {"model": "gemini-3.8-flash", "concurrency": 8,
                      "project": VERTEX_PROJECT, "location": VERTEX_LOCATION,
                      "diff_cap": 50_000, "pack_cap": 120_000},
    "glm": {"base_url": ZAI_BASE_URL, "model": "glm-5.2", "concurrency": 2,
            "diff_cap": 15_000, "pack_cap": 40_000},
    # Cloud deployment names are user-overridable; they need not match model IDs.
    "foundry": {"model": os.environ.get("AZURE_FOUNDRY_DEPLOYMENT", "grok-43"), "concurrency": 8,
                "endpoint": FOUNDRY_ENDPOINT, "api_version": FOUNDRY_API_VERSION,
                "diff_cap": 50_000, "pack_cap": 120_000},
    "azure": {"model": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-54-mini"), "concurrency": 8,
              "endpoint": AZURE_ENDPOINT, "api_version": AZURE_API_VERSION,
              "diff_cap": 50_000, "pack_cap": 120_000},
    # prompt_cache: mark the pr_context user block with anthropic cache_control --
    # a breakpoint caches EVERYTHING before it (tools + system + context + pack),
    # so rounds 2..N of each agent's ReAct loop prefill the heavy prefix at ~0.1x.
    # Re-prefilling a large pack every round can dominate latency. Other clients
    # keep plain strings because they do not accept Anthropic cache blocks.
    "claude-vertex": {"model": "claude-sonnet-5", "concurrency": 8,
                      "project": VERTEX_PROJECT, "location": VERTEX_LOCATION,
                      "diff_cap": 50_000, "pack_cap": 120_000, "prompt_cache": True},
}
DEFAULT_PROVIDER = "openai"
REVIEW_MODES = ("discovery", "verification")


def resolve_tuning(provider: str, model_arg, light_arg, conc_arg, env) -> tuple[str, str, int]:
    """Resolve (model, light_model, concurrency): CLI arg > env > provider default.

    Environment settings let CI tune throughput without changing its invocation.
    Resolve them here so tests cover consumption, not just workflow declarations.
    """
    model = model_arg or (env.get("OPENAI_MODEL") if provider == "openai"
                          else PROVIDERS[provider]["model"])
    if not model or not model.strip():
        log("error: set OPENAI_MODEL or pass --model <model-id> for --provider openai.")
        sys.exit(3)
    model = model.strip()
    light_model = light_arg or env.get("REVIEW_LIGHT_MODEL") or ""
    env_conc = env.get("REVIEW_CONCURRENCY")
    concurrency = (conc_arg or (int(env_conc) if env_conc else 0)
                   or PROVIDERS[provider]["concurrency"])
    return model, light_model, concurrency


def resolve_review_mode(mode_arg: str | None, env) -> str:
    mode = mode_arg or env.get("REVIEW_MODE") or "discovery"
    if mode not in REVIEW_MODES:
        raise ValueError(f"invalid review mode {mode!r}; expected one of {REVIEW_MODES}")
    return mode


MAX_FINDINGS_PER_DIMENSION = 8
DIFF_INLINE_CAP = 15_000     # fallback diff inline cap for providers without a diff_cap
TOOL_OUTPUT_CAP = 15_000     # chars per tool result
# Context-pack BUILD caps (built once in node_fetch, sliced per provider pack_cap
# at injection). Full changed-file bodies + path-matched learnings sections, so
# reviewers spend tool rounds on CONSUMERS of the change, not re-fetching it.
# A small per-file cap forced reviewers to read large changed files in dozens of
# manual chunks until they exhausted the recursion ceiling. The full changed
# file in-pack removes that need. Providers with a smaller pack_cap still trade
# some of that coverage for a leaner input budget at injection time.
BLAST_PACK_CAP = 6_000       # chars of reachability facts; small on purpose, goes first
GUARDS_PACK_CAP = 4_000      # chars of stale-guard findings; it speaks on ~3% of commits
PACK_PER_FILE_CAP = 80_000   # chars per changed file
# >= 2x the per-file cap so a SECOND large changed file isn't dropped whole at
# build time (two 70k files used to blow a 120k build cap). The per-provider
# pack_cap still does the final slicing at injection.
PACK_BUILD_CAP = 170_000     # chars across all changed files (2x per-file + header slack)
LEARNINGS_PACK_CAP = 12_000  # chars of matched learnings sections
# Steps per reviewer agent. langchain 1.0's create_agent loop counts model +
# tool executions; 100 =~ 45 tool rounds. Exhaustion is NOT retried (a agent
# that loops deterministically would just loop again) -- it degrades to a
# coverage note in the report. RECURSION EXHAUSTION IS THE HARD BACKSTOP, not
# the primary loop guard: the convergence middleware (make_convergence_middleware)
# force-concludes a stuck agent BEFORE it reaches this ceiling, so a looping
# reviewer returns a real (if shallower) answer instead of dying to a coverage
# note. See CONVERGENCE_* below. Some models ignore both refusal text and tool
# unbinding, so the middleware also clears returned tool calls. This ceiling
# remains the backstop if a provider takes an unexpected path through the loop;
# raising it alone only lets a stuck agent wander longer.
AGENT_RECURSION_LIMIT = 100
# Convergence wall (agent-loop level, above the tool-level anti-loop wall).
# The tool wall's once() refusal caps the TOKEN cost of a repeat but cannot make
# the model STOP. Measured on a real PR: three reviewers re-issued the same grep
# or wandered through distinct searches until they reached the recursion limit. This
# middleware counts its OWN model-call turns and, once the agent is clearly stuck,
# STRIPS the tools + orders a final answer, so the next model turn must conclude.
# Two triggers map to the two observed loop shapes:
#   * REFUSAL trigger -> identical-repeat loopers (the once() wall already refused
#     them; N refusals in the history = the model is ignoring the wall)
#   * ROUND-BUDGET trigger -> distinct-but-wandering loopers (each call differs so
#     none are refused; only a hard model-call ceiling catches them). Budgets sit
#     below each agent's recursion cycles (~recursion_limit/2) so the wall trips
#     first. Counted via the middleware's own invocation counter, NOT a message
#     scan -- an AIMessage-with-tool_calls scan undercounted vs real ChatVertexAI
#     history and let a reviewer wander past its budget.
# 1, NOT higher: a refusal already means the model issued the EXACT same call
# twice, so one is an unambiguous loop signal. Waiting for more refusals only
# adds duplicate calls before the same forced conclusion.
REVIEWER_TOOL_BUDGET = 40       # model-call turns for reviewer/advisory/fix agents (recursion 100 ~= 49 cycles)
VERIFY_TOOL_BUDGET = 10         # verifiers (recursion 25 ~= 12 cycles)
BRIEF_TOOL_BUDGET = 9           # brief (recursion 24 ~= 11 cycles)
# Verifiers check ONE narrow claim -- they never need a reviewer-sized budget,
# and a wandering verifier is the tail that stalls the whole run.
VERIFY_RECURSION_LIMIT = 25  # cap the wandering/throttle tail for one narrow claim
# The brief sits ALONE on the critical path (fetch -> brief -> all reviewers; the
# parallel-brief topology was tried and reverted -- see fan_out_reviews). Measured
# 116-123s/run at the reviewer-sized budget: pure serial cost. With changed files
# now inlined via the context pack it needs no repo rounds -- a web-only budget.
BRIEF_RECURSION_LIMIT = 24   # ~10 tool rounds: a few web searches + scrapes, then write
RETRY_SLEEPS = (20, 45, 90)  # node-level backoff on transient provider faults
# Retryable provider faults: rate limits + connection drops/5xx. Gemini drops the
# connection on some long ReAct turns ("Server disconnected without sending a
# response") and the SDK's own max_retries does not always cover it -- the whole
# agent run is retried from scratch (its loop state is lost either way).
RETRYABLE_MARKERS = ("429", "Server disconnected", "Connection reset", "RemoteProtocolError",
                     "502", "503", "504", "Deadline Exceeded", "ReadTimeout")
# Dimensions that consume the web-grounded brief: only these check external-API
# facts. simplicity-reuse / invariants / spec-conformance are repo-internal --
# a web brief is dead weight (and an anchoring risk) for them.
BRIEF_DIMENSIONS = {"correctness", "wiring"}
# Optional web-tool adapters can live in the user's local tools directory.
# CI can set PR_REVIEW_WEB_TOOLS to a separately provisioned copy; without them
# the reviewers still work and ground their findings from the repository.
WEB_TOOLS_DIR = Path(os.environ.get("PR_REVIEW_WEB_TOOLS") or (Path.home() / ".claude" / "tools"))
SEVERITIES = ("blocker", "major", "minor", "nit")
SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}
SEV_DOWN = {"blocker": "major", "major": "minor", "minor": "nit", "nit": "nit"}
# A reviewer that has nothing to say sometimes fills the schema in rather than
# returning []. A model shipped `{file: "None", line: 0, title: "None", why:
# "None"}` to a real PR comment: it rendered as "**`None:0` -- None**",
# inflated the finding count, and burned a 3-lens verify fan-out on nothing.
# JSON null decodes to Python None, and models also emit the STRING "None"/"N/A",
# so both shapes have to die here.
EMPTY_TITLES = {"", "none", "n/a", "na", "null", "nil", "-", "--", "no findings", "no issues"}


def is_empty_finding(f) -> bool:
    """True for a finding with no substantive claim -- see EMPTY_TITLES.

    Keyed on the TITLE alone, deliberately. The title is the one field every
    renderer and the verify prompt depend on, so a placeholder there makes the
    finding unusable no matter what the other fields hold; conversely a real
    title with a vague `why` is a weak finding, not an empty one, and killing it
    here would silently narrow the review.
    """
    if not isinstance(f, dict):
        return True
    title = f.get("title")
    return str("" if title is None else title).strip().lower() in EMPTY_TITLES


def normalize_findings(raw, dim: str):
    """Reviewer output -> (findings, empty_dropped). The ONE ingestion path.

    Applies the per-dimension cap, drops empty findings, defaults an unknown
    severity to `minor`, coerces `line` to an int, and stamps key/dimension.
    Extracted from node_review so the filter is testable against the real path
    rather than only in isolation -- an unreachable guard is not a guard.
    """
    findings, empty = [], 0
    if not isinstance(raw, list):
        return findings, empty
    for i, f in enumerate(raw[:MAX_FINDINGS_PER_DIMENSION]):
        if is_empty_finding(f):
            empty += 1
            continue
        if f.get("severity") not in SEVERITIES:
            f["severity"] = "minor"
        try:
            f["line"] = int(f.get("line") or 0)
        except (TypeError, ValueError):
            f["line"] = 0
        f.update(key=f"{dim}:{i}", dimension=dim)
        findings.append(f)
    return findings, empty


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_cmd(cmd: list[str], timeout: int = 120) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}... failed: {p.stderr.strip()[:500]}")
    return p.stdout


def target_repo_root() -> Path:
    """Resolve the checkout being reviewed, independently of this script's location."""
    return Path(run_cmd(["git", "rev-parse", "--show-toplevel"]).strip()).resolve()


# ---------------------------------------------------------------------------
# Review dimensions. Keep prompts as constants so behavior is easy to inspect
# and checkpoint replay can isolate the effect of a prompt change.
# ---------------------------------------------------------------------------
DIMENSIONS: dict[str, str] = {
    "spec-conformance": """Spec conformance: does the code do what it was supposed to? The diff can be
flawless yet implement the wrong thing or silently drop a requirement.
- Read the PR body; find the linked issue / spec (docs/**/specs/, docs/plans/, or named in
  the body) via grep_repo. Check each stated requirement against the implementation; flag
  any requirement unmet, partially met, or contradicted.
- If the PR clearly implements a feature but you find NO spec, note it in one line (not a finding).""",
    "correctness": """Obvious bugs and stale siblings.
- Off-by-one, inverted conditions, null/undefined deref, swapped args, async races.
- Breaking changes: grep_repo callers of EVERY modified function/type/removed export -- did
  every caller, test fixture, and .toEqual assertion get updated?
- Silent-failure honesty: an infra failure (timeout, key unset, malformed JSON, swallowed
  exception) that degrades into status:"ok" with empty results is a bug -- if one failure
  branch returns an error status, its symmetric siblings must too.
- Shared-type changes: check the repo's actual type-checking gate and untouched consumers;
  a build that does not type-check them can stay green despite a broken shared type.
- Python: psql camelCase identifiers need double quotes; Path.name strips
  directories (startswith('dir/') on a basename is dead code).""",
    "wiring": """Cross-file integration bugs.
The bug is usually in a file the diff does NOT touch. For every artifact the diff produces,
verify its consumer gets what it expects:
- Config<->code: every env var / secret / path in new code -- does the workflow env: name
  EXACTLY match what the script reads? Is a required path gitignored (empty on a fresh CI
  runner)? Does the referenced secret exist?
- Caller<->callee: for every subprocess/CLI invocation, OPEN the callee (show_file) and read
  its argparse/flag parsing. Match every flag and choices=[...] value.
- End-to-end: trace entry point -> each script -> each output. Does an invoked script rebuild
  state from scratch (clobbering) or merge? Where does it actually read/write?
- Git semantics: push origin HEAD from workflow_dispatch on a feature branch pushes to that
  branch; git add <dir> won't stage deletions; failed pull --rebase poisons later steps.
- Day-2 behavior of anything scheduled: does the dedup key match what the producer emits?
- Subprojects: check the repo's actual workspace, package-manager and CI dependency setup.
  Dependencies must be available in the project and runner that import them.""",
    "security-data": """Security vulnerabilities and data-loss risks.
- Auth bypass, secrets in code/logs/committed JSON, SQL injection, SSRF, XSS.
- Cron handlers must fail closed: guard if(!secret) so an unset env never matches
  'Bearer undefined'.
- Destructive migrations: dropped/renamed columns, UNIQUE adds, missing transactions.
  New columns: audit ALL insert paths, including writers that supply their own identifiers.
- Update-wipe: including enriched JSON fields in an update can clobber fields other
  pipelines wrote; check create-only and merge semantics.
- PII: logged tool args must be allowlist-only summaries (never raw bodies with
  contact/notes); restricted data must stay within its authorized readers; test paths must
  not create real records or send email to real inboxes.""",
    "simplicity-reuse": """Simplicity and scope control: did this diff add code that can be
deleted while preserving the requested behavior?
- Prefer deletion or inlining. A wrapper, helper, config shape, or single implementation is
  not a defect by itself; report it only when it creates a concrete wrong outcome or an
  avoidable operating surface in this diff.
- Never ask for a new abstraction, module, gate, fallback system, or generalized framework
  merely to make the review cleaner. Review must shrink scope, not manufacture follow-up work.
- Speculative handling for inputs that cannot occur is advisory at most.
- Reinvention is a finding only when an existing shared mechanism prevents a concrete bug
  (for example shared auth); name the existing mechanism and the actual failure path.
- Pure maintainability preferences are minor/nit advisory. Essential complexity is fine.""",
    "invariants": """Named invariants stated in the repo's agent instructions
(CLAUDE.md / AGENTS.md) in the context pack. Flag concrete violations or missed extensions,
not stylistic proximity. If the repo states no invariants, return no findings.
- Allowlist twins: verifiers with hardcoded allowlists must be updated in the SAME PR as
  the thing they allowlist. If the repo's agent instructions maintain a corresponding
  table, update its matching row when the allowed surface changes.
- Gate over prose applies only when the diff fixes a recurring, high-impact bug class that
  is cheaply and deterministically checkable. Do not demand a new verifier for a one-off,
  speculative edge case, reviewer preference, or bug class with no real recurrence signal.
  The smallest code fix is often sufficient.""",
}

ADVISORY_PROMPT = """Write the "## Product review (advisory)" section for this PR -- it never
changes the verdict. Answer briefly, numbered:
1. Product translation -- 2-3 plain sentences: what does this change do, for which user/surface?
2. Linked issue -- does the PR do what the linked issue asked; does the issue target the right problem?
3. Right lever -- root cause or shim? If it tolerates drift at read time, would write-time
   normalization plus a mechanical gate be the smaller honest fix?
4. Data grounding -- if the PR's justification rests on claims about prod data (missing
   fields, needed fallbacks, coverage gaps), verify them with query_prod and report the
   counts. A fallback defending data that has never existed is YAGNI noise -- say so with
   numbers. If the tool reports unavailable, skip this question with one line saying so.
5. Principle alignment -- judge the diff against the live repo agent instructions
   (CLAUDE.md / AGENTS.md) in your context pack. Use the principles stated there, such as
   simplicity, avoiding speculative work, entity modeling, mechanical checks, scope
   discipline and honest numbers. Report ONLY principles this diff MATERIALLY engages, one line each,
   verdict-first: "upheld" (notably), "tension" (defensible -- name the cost), or
   "violated" (name the smaller/honest version). Nothing materially engaged -> the single
   line "principles: none engaged". Do not grade principle-neutral diffs against the full
   list. Also: does this serve a locked spec/positioning doc, and is there a simpler lever?
6. Churn -- does the diff smell like the Nth consecutive fix to the same area? (grep_repo for
   sibling fix markers / TODOs if unsure)
For trivial PRs the whole section is one line: "Product review: n/a -- <reason>".
Return ONLY the markdown section text, starting with the heading."""

LENSES: dict[str, str] = {
    "repro": """REPRODUCTION lens: does this defect actually occur on the real code at the PR
head? Trace the exact inputs/state from the failure_scenario through the actual code
(show_file / grep_repo). If you cannot make it fire concretely, refute it.""",
    "context": """FALSE-POSITIVE lens: is the finding invalidated by surrounding code the
reviewer missed -- a guard upstream, a caller that never passes the bad input, an allowlist
entry, a pinning test, or a deliberate documented exception (code comment or a rule in
CLAUDE.md / AGENTS.md)? Hunt for the exonerating context; if you find it, refute.""",
    "severity": """SEVERITY lens: assume the defect is real; is the severity honest?
blocker = concrete release stopper: production break, data corruption/loss, secrets/PII
exposure, or CI-wide breakage on the current diff. It needs an exact triggering path.
major = real bug, but not a release stopper; it keeps the check green but withholds
auto-approval until the author fixes it or a human accepts it.
If real but overrated, set inflated=true (refuted=false). If not even real, refute.""",
}

FINDINGS_FORMAT = f"""Respond with ONLY a JSON object (no prose before/after):
{{"findings": [{{"file": "repo/relative/path", "line": 123, "severity": "blocker|major|minor|nit",
  "title": "one-line defect statement", "why": "why this is wrong, citing actual code",
  "failure_scenario": "concrete inputs/state -> wrong outcome",
  "suggestion": "concrete fix"}}], "droppedCount": 0}}
At most {MAX_FINDINGS_PER_DIMENSION} findings, most severe first; count anything cut in
droppedCount. Work in SILENCE between tool calls -- no narration, no interim summaries, no
"Now I'll check..." prose; your ONLY visible output is the final JSON. (Reasoning is fine;
narrating it is wasted latency on a gating lane.) Zero findings is a perfectly good answer -- do not manufacture. Every finding
needs a CONCRETE failure_scenario; "could be confusing" is not a finding. Findings must be
caused-or-obligated by this diff.

PRECISION over exhaustiveness (this reviewer GATES merges -- noise erodes trust and burns
review cycles). Report a finding ONLY if you can name specific inputs/state that produce a
wrong outcome and you are confident it is real. Do NOT report:
- theoretical fragilities that cannot occur in practice, or that the pinned version / a guard
  upstream / an existing test already prevents (check before flagging);
- style, naming, or efficiency PREFERENCES (fetching more than needed, a "complex" regex, a
  bespoke-but-working helper) -- those are not defects;
- the SAME root cause as more than one finding -- report it once, at its highest severity.
When unsure whether something is real, LEAVE IT OUT. Five solid findings beat ten padded ones;
an empty list on clean code is the correct answer, not a failure to try.

This tool performs one stochastic review of the current PR head. Only a blocker reds the check;
verified majors withhold auto-approval (the author fixes the batch or a human approves — never a
new review); minor/nit findings are advisory telemetry. The surrounding process may run at most two focused
verification heads for the open blocker/major batch; it must not start a fourth stochastic review or expand
scope with unrelated findings. Never recommend review-driven architecture, a new gate, or generalized machinery.
The suggestion must be the smallest root-cause change, preferably deletion, and may not expand the
requested product scope."""

VERDICT_FORMAT = """Respond with ONLY a JSON object:
{"refuted": true|false, "inflated": true|false, "reasoning": "2-4 sentences citing the code you checked"}
If uncertain after checking the code, refuted=true (unconfirmed findings must not reach the human)."""


BRIEF_PROMPT = """You are the RESEARCH agent for a multi-agent PR review.
Produce a short FACTUAL brief that the correctness + wiring reviewers will read as background.

Do:
1. Read the diff and identify the external libraries / frameworks / APIs it leans on where
   CURRENT behavior matters (versions, signatures, deprecations). Check the pinned version in
   the diff or the manifest (package.json / pyproject / PEP 723 header) FIRST.
2. For the uncertain ones, use web_search / scrape_url to pull the CURRENT docs / changelog and
   extract the concrete facts: current API signatures, what changed across versions, what is
   deprecated. Cite the source URL for each fact.
3. Note the spec/issue the PR implements and its concrete requirements (from the inlined context).
4. Note which named repo invariants the diff touches (from CLAUDE.md / AGENTS.md), one line each.

Rules:
- FACTS ONLY. API signatures, version behavior, spec requirements, invariant definitions.
  NEVER judge whether the code is correct -- that is the reviewers' job; you only arm them.
- Web results are an UNVERIFIED, possibly-stale source. Say the version each fact applies to,
  and flag where the reviewer must re-verify against the pinned version + the actual code.
- Be concise: a tight bulleted brief, not an essay. If there is no meaningful external surface,
  say so in one line.
- HARD BUDGET -- you sit alone on the run's critical path. web_search returns SNIPPETS in
  ~2s (cheap, up to 3 calls); scrape_url fetches a full page in 20-40s (expensive, at most
  2 calls, only when a snippet can't answer). Target <= 350 words. The changed files are
  already inlined in your context; do not re-read them with repo tools. Research the 1-3
  MOST uncertain external facts and stop -- a fast partial brief beats a slow complete one.
Output ONLY the brief markdown."""

REVIEWER_PROMPT = """You are one reviewer in a multi-agent PR review.
Your single dimension: {dimension}.

{dimension_prompt}

{findings_format}{mode_prompt}"""

VERIFICATION_PROMPT = """

VERIFICATION MODE: do not discover new issues. Check only the previous blocker
batch and regressions directly caused by its fix delta. Return zero findings
when those blockers are resolved."""


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------
def required_env(name: str) -> str:
    """Fail before graph construction, with a single actionable prerequisite error."""
    value = os.environ.get(name, "").strip()
    if not value:
        log(f"error: missing {name}. Fix: export {name} for the selected provider.")
        sys.exit(3)
    return value


def provider_key(provider: str) -> str:
    if provider in ("claude-vertex", "gemini-vertex"):
        required_env("VERTEX_PROJECT")
        return ""  # Vertex authenticates via ADC, not an API key
    if provider == "foundry":
        required_env("AZURE_FOUNDRY_ENDPOINT")
        return required_env("AZURE_FOUNDRY_API_KEY")
    if provider == "azure":
        required_env("AZURE_OPENAI_ENDPOINT")
        return required_env("AZURE_OPENAI_API_KEY")
    if provider == "openai":
        return required_env("OPENAI_API_KEY")
    if provider == "glm":
        key = os.environ.get("GLM_API_KEY", "").strip()
        if not key:
            try:
                key = ZAI_KEY_FILE.read_text().strip()
            except OSError:
                pass
        if not key:
            log(f"error: no GLM key. Put it in {ZAI_KEY_FILE} or export GLM_API_KEY.")
            sys.exit(3)
        return key
    key = (os.environ.get("GOOGLE_GEMINI_API_KEY", "").strip()
           or os.environ.get("GEMINI_API_KEY", "").strip())
    if not key:
        log("error: no Gemini key. Fix: export GOOGLE_GEMINI_API_KEY or GEMINI_API_KEY.")
        sys.exit(3)
    return key


# --- Vertex context caching for the gemini lane (opt-in: GEMINI_VERTEX_CACHE=1) ---
#
# WHY. Re-prefilling a large pack on every tool round can dominate latency.
# Shrinking it can instead force reviewers to re-read files until they hit the
# recursion ceiling. Explicit caching could keep the pack without repeated prefill.
#
# Feasibility verified 2026-07-20: CachedContent.create succeeds for
# gemini-3-flash-preview at location=global with a ~29k-token pack.
#
# Fail-soft by construction: any error returns None and the lane runs exactly as
# before (uncached), because a caching optimisation must never be able to break
# the merge gate.
#
# STATUS: helper + make_model plumbing only -- NOT YET WIRED, and inert until it
# is (GEMINI_VERTEX_CACHE is also off by default). Remaining work, deliberately
# left as one reviewable change rather than half-done here:
#   1. node_fetch: call make_vertex_cache(pack, model) once per run, thread the
#      handle through state to every make_model call.
#   2. pr_context: OMIT the pack from the user message when a cache handle is
#      live -- otherwise the payload is sent BOTH cached and inline, which is
#      strictly worse than today.
#   3. Validate on a real PR that wall-clock drops and `gaps` stays 0; only then
#      flip the env default.
# (2) is the load-bearing half: without it this buys nothing.
def make_vertex_cache(pack: str, model: str, ttl_minutes: int = 30) -> str | None:
    if os.environ.get("GEMINI_VERTEX_CACHE", "").strip() != "1" or not pack.strip():
        return None
    try:
        import datetime as _dt
        import vertexai
        from vertexai.preview import caching
        from vertexai.generative_models import Content, Part

        vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        cc = caching.CachedContent.create(
            model_name=model,
            contents=[Content(role="user", parts=[Part.from_text(pack)])],
            ttl=_dt.timedelta(minutes=ttl_minutes),
        )
        log(f"[cache] vertex context cache created ({len(pack):,} chars) {cc.name}")
        return cc.name
    except Exception as e:
        # Below the minimum cacheable size, model/location without cache support,
        # quota -- all just mean "run uncached".
        log(f"[cache] vertex caching unavailable ({str(e)[:120]}) -- running uncached")
        return None


def openai_model_kwargs(provider: str, model: str, timeout: int) -> dict:
    """Shared OpenAI-compatible request policy, with provider-scoped env overrides.

    No eager capability probe: a probe would consume a completion and turn a
    transient outage into a construction failure. GPT-6 tool use needs Responses
    with reasoning on some endpoints, so derive that route from the model name.
    Explicit overrides also support deployments whose names hide their model.
    """
    prefix = "AZURE_OPENAI" if provider == "azure" else "OPENAI"
    cfg = {"max_retries": 3, "timeout": timeout}
    responses_env = os.environ.get(f"{prefix}_RESPONSES_API", "").strip()
    use_responses = (responses_env == "1") if responses_env else model.startswith("gpt-6")
    # Always explicit: left unset, LangChain picks the route itself and can send a
    # chat-only endpoint to /responses even when the user turned Responses off.
    cfg["use_responses_api"] = use_responses
    if use_responses:
        cfg["reasoning"] = {"effort": os.environ.get(f"{prefix}_REASONING_EFFORT") or "medium"}
    # Responses also suppresses temperature when an opaque deployment name was
    # forced onto that path. Checking only the model prefix misses that case.
    if not use_responses and os.environ.get(f"{prefix}_NO_TEMPERATURE", "").strip() != "1":
        cfg["temperature"] = 0.1
    service_tier = os.environ.get(f"{prefix}_SERVICE_TIER", "").strip()
    if service_tier:
        # Availability belongs to the user's endpoint, not a built-in deployment list.
        cfg["extra_body"] = {"service_tier": service_tier}
    return cfg


@functools.lru_cache(maxsize=8)
def make_model(provider: str, model: str, timeout: int, cached_content: str | None = None):
    key = provider_key(provider)
    if provider == "claude-vertex":
        # Model access must be enabled in Vertex. Omit temperature for reasoning
        # models; the cache breakpoint is attached separately by cached_user().
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        return ChatAnthropicVertex(
            model_name=model, project=os.environ["VERTEX_PROJECT"].strip(),
            location=PROVIDERS[provider]["location"],
            max_retries=3, timeout=timeout)
    if provider == "foundry":
        # Foundry's /models endpoint is OpenAI-shaped, so plain ChatOpenAI works
        # (unlike Azure OpenAI, which needs the deployment-scoped AzureChatOpenAI).
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=key,
                          base_url=os.environ["AZURE_FOUNDRY_ENDPOINT"].strip(),
                          default_query={"api-version": PROVIDERS[provider]["api_version"]},
                          temperature=0.1, max_retries=3, timeout=timeout)
    if provider in ("azure", "openai"):
        cfg = openai_model_kwargs(provider, model, timeout)
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
            if base_url:
                cfg["base_url"] = base_url
            else:
                # The OpenAI SDK reads this env itself and treats "" as a real base
                # URL, so requests go to "/chat/completions". CI exports unset
                # variables as empty strings.
                os.environ.pop("OPENAI_BASE_URL", None)
            return ChatOpenAI(model=model, api_key=key, **cfg)

        from langchain_openai import AzureChatOpenAI

        cfg.update(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
                   azure_deployment=model,
                   api_version=PROVIDERS[provider]["api_version"],
                   api_key=key)
        if cfg.get("use_responses_api"):
            if not api_version_at_least(str(cfg["api_version"]), RESPONSES_MIN_API_VERSION):
                # Upgrade rather than raise: a raise here becomes a dead reviewer,
                # which can leave a green check with no reviewer coverage.
                log(f"[model] {model}: api-version {cfg['api_version']} predates the "
                    f"Responses API; using {RESPONSES_MIN_API_VERSION} for this build "
                    f"(pin AZURE_OPENAI_API_VERSION >= {RESPONSES_MIN_API_VERSION} to silence)")
                cfg["api_version"] = RESPONSES_MIN_API_VERSION
        return AzureChatOpenAI(**cfg)
    if provider == "gemini-vertex":
        # The native Vertex client round-trips Gemini tool thought signatures
        # while keeping this lane on ADC authentication.
        from langchain_google_vertexai import ChatVertexAI

        kw = dict(model=model,
                  project=os.environ["VERTEX_PROJECT"].strip(),
                  location=PROVIDERS[provider]["location"],
                  temperature=0.1, max_retries=3, timeout=timeout)
        if cached_content:
            kw["cached_content"] = cached_content
        return ChatVertexAI(**kw)
    if provider == "gemini":
        # Native client, NOT the OpenAI-compat endpoint: Gemini 3 function calls
        # require thought_signature round-tripping, which the compat layer drops
        # (400 "missing thought_signature" on every 2nd tool turn).
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            timeout=timeout,
            max_retries=3,
            temperature=0.1,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=PROVIDERS[provider]["base_url"],
        timeout=timeout,
        max_retries=3,
        temperature=0.1,
    )


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONGNUM_RE = re.compile(r"\b\d{7,}\b")  # phones, card/account numbers, long ids


def _redact(x):
    if isinstance(x, str):
        return _LONGNUM_RE.sub("[num]", _EMAIL_RE.sub("[email]", x))
    if isinstance(x, dict):
        return {k: _redact(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_redact(v) for v in x]
    return x


def langfuse_mask(data=None, **_):
    """Redact emails and long digit runs from traced inputs and outputs.

    Diffs, files and database rows can contain real contact details, including
    in fixtures. Mask before export; tracing itself remains fail-open.
    """
    try:
        return _redact(data)
    except Exception:
        return data


def tracing_environment() -> str:
    """Explicit trace environment wins; otherwise distinguish CI from local runs."""
    explicit = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", "").strip()
    if explicit:
        return explicit
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        return "github-actions"
    return "local"


def make_trace_callbacks(thread_id: str) -> list:
    """Env-gated Langfuse tracing:
    no-op unless LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set; fail-open --
    observability must never break a review run. One LangChain CallbackHandler on
    the top-level graph.invoke propagates to every node, LLM call, and tool call.
    Set LANGFUSE_HOST for a non-default host. All credentials come from env."""
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", tracing_environment())
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return []
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Construct the singleton WITH the PII mask FIRST -- langfuse v4 registers
        # the first-constructed client as the process singleton, so the later
        # CallbackHandler() and score_current_trace() both resolve get_client() to
        # THIS masked instance. Verified: after this line get_client()._mask is
        # langfuse_mask (True). This is the only Langfuse touch before the graph runs.
        Langfuse(mask=langfuse_mask)
        return [CallbackHandler()]
    except Exception as exc:
        log(f"[trace] init skipped: {exc}")
        return []


def score_trace(lf, stats: dict) -> None:
    """Attach the report's stats as Langfuse scores on the run's trace (v4:
    score_current_trace inside the active observation context) so runs are
    comparable in the UI and mechanically checkable by downstream telemetry
    gates. Fail-open like all tracing."""
    try:
        for name in ("raw", "kept", "killed", "downgraded"):
            lf.score_current_trace(name=f"review_{name}", value=float(stats[name]))
        # gaps = genuine dimension deaths only (key `<dim>:error`), NOT per-dimension
        # cap overflow (`<dim>:dropped`) -- see node_report. Fall back to len(meta)
        # for pre-split cached stats.
        lf.score_current_trace(name="review_coverage_gaps",
                               value=float(stats.get("gaps", len(stats.get("meta", [])))))
        lf.score_current_trace(name="review_cap_drops", value=float(stats.get("cap_drops", 0)))
    except Exception as exc:
        log(f"[trace] scores skipped: {exc}")


def run_traced(run_graph):
    """Fail-open trace setup/teardown without ever retrying the review itself.

    Keep graph execution outside the tracing-error catch. Retrying the whole
    block after a tracing error could accidentally run a second paid review.
    """
    observation, lf = None, None
    try:
        from langfuse import get_client

        lf = get_client()
        observation = lf.start_as_current_observation(name="pr-review:run", as_type="span")
        observation.__enter__()
    except Exception as exc:
        observation, lf = None, None
        log(f"[trace] observation skipped: {exc}")
    try:
        final = run_graph()
        if lf is not None:
            score_trace(lf, final["result"]["stats"])
        return final
    finally:
        if observation is not None:
            try:
                observation.__exit__(*sys.exc_info())
            except Exception as exc:
                log(f"[trace] observation close skipped: {exc}")


def flush_traces() -> None:
    # get_client() without keys prints an authentication error on every untraced run.
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass


def invoke_with_backoff(agent_factory, messages: dict, config: dict):
    """Build-and-invoke with patient node-level retries on transient provider
    faults (rate limits, connection drops, 5xx). Takes a FACTORY, not an agent:
    each attempt gets fresh tools, because the anti-loop `seen` cache must not
    leak between attempts (a retried agent starts a fresh conversation -- its
    first legitimate calls would otherwise be refused as repeats)."""
    for i, pause in enumerate((*RETRY_SLEEPS, None)):
        try:
            return agent_factory().invoke(messages, config=config)
        except Exception as e:
            msg = str(e)
            if pause is None or not any(m in msg for m in RETRYABLE_MARKERS):
                raise
            log(f"[backoff] transient provider fault ({msg[:80]}), sleeping {pause}s (attempt {i + 1}/{len(RETRY_SLEEPS)})")
            time.sleep(pause)




def extract_json(text: str) -> dict | None:
    text = content_to_text(text)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [m.group(1)] if m else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def parse_json_or_repair(text: str, schema_hint: str, model, timeout: int) -> dict | None:
    data = extract_json(text)
    if data is not None:
        return data
    # One repair pass: ask the model to reformat its own output.
    resp = model.invoke(
        [
            ("system", "Reformat the user's content into valid JSON per this contract. Output ONLY the JSON.\n" + schema_hint),
            ("user", text[:20_000]),
        ]
    )
    return extract_json(content_to_text(resp.content))


def _public_host(url: str) -> bool:
    """True only if the URL is http(s) and EVERY resolved IP is public -- defeats
    DNS-rebinding and integer/hex-encoded IPs that a string host-check would miss."""
    import ipaddress
    import socket
    from urllib.parse import urlsplit

    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    try:
        infos = socket.getaddrinfo(u.hostname, None)
    except Exception:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        # is_global alone is not enough: it is True for multicast.
        if (not ip.is_global or ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def brave_search(query: str, count: int = 6, timeout: int = 10) -> str | None:
    """Snippet search via the Brave API (~1-2s) -- the brief's fast lane.
    Key: BRAVE_API_KEY env (the CI secret graph-review.yml already passes) or
    ~/.config/brave/api_key (the web_search.sh convention). Returns None when
    keyless or on any HTTP fault so the caller falls back to the deep shell tool
    -- fail-soft, never an exception into the agent loop."""
    key = os.environ.get("BRAVE_API_KEY", "").strip()
    if not key:
        kf = Path.home() / ".config" / "brave" / "api_key"
        try:
            key = kf.read_text().strip() if kf.exists() else ""
        except OSError:
            key = ""
    if not key:
        return None
    import urllib.parse
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://api.search.brave.com/res/v1/web/search?q="
            + urllib.parse.quote(query) + f"&count={min(count, 20)}",
            headers={"X-Subscription-Token": key, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rows = (json.loads(r.read()).get("web") or {}).get("results") or []
    except Exception:
        return None  # fall back to the deep tool
    if not rows:
        return f"no results for {query!r}"
    return "\n\n".join(f"{x.get('title', '?')} -- {x.get('url')}\n{x.get('description', '')}"
                       for x in rows[:count])


def make_tools(sha: str, pr_number: int, diff: str, with_db: bool = False,
               with_web: bool = False, changed_paths: list[str] | None = None):
    from langchain_core.tools import tool

    def cap(s: str) -> str:
        return s if len(s) <= TOOL_OUTPUT_CAP else s[:TOOL_OUTPUT_CAP] + "\n...[truncated -- narrow the request instead of repeating it]"

    # Mechanical anti-loop wall: Langfuse traces showed reviewers re-issuing the
    # IDENTICAL call up to 11x (same grep, same show_file with the default range).
    # A repeat returns a short refusal instead of the same blob, which breaks the
    # loop and saves the context window.
    seen: dict[tuple, str] = {}

    def once(key: tuple, produce):
        if key in seen:
            # Built FROM the constant the convergence wall matches on: a reworded
            # refusal must not silently stop _refused_repeat_count from tripping.
            return (f"{CONVERGENCE_MARKER} -- you already made this exact call and the result "
                    "has not changed. Vary the arguments: a line range past what you saw, a "
                    "narrower pattern, or a pathspec.)")
        # Fail-soft chokepoint: a tool must NEVER raise into the agent loop --
        # an uncaught TimeoutExpired from one pathological regex once killed a
        # whole reviewer.
        # Errors are data for the model.
        try:
            out = produce()
        except Exception as e:
            out = f"tool error: {str(e)[:300]} -- adjust the arguments; do not repeat the same call"
        seen[key] = out
        return out

    @tool
    def show_file(path: str, start: int = 1, end: int | None = None) -> str:
        """Read a repo file at the PR head. path is repo-relative; start/end are 1-indexed line
        numbers (end defaults to start+399, so continuation calls only need start)."""
        def produce():
            # Reviewers read the immutable PR head sha. A bad path raises from
            # run_cmd; once()'s fail-soft chokepoint formats it.
            out = run_cmd(["git", "show", f"{sha}:{path}"])
            lines = out.splitlines()
            stop = end if end is not None else start + 399
            sel = lines[max(0, start - 1) : stop]
            if not sel:
                return f"(empty range; file has {len(lines)} lines)"
            # Truncate at LINE granularity so the continuation hint is truthful:
            # the hint must name the last line actually shown, never the requested
            # end (a char-cap mid-range would silently skip lines otherwise).
            body_lines: list[str] = []
            used = 0
            for i, l in enumerate(sel):
                ln = f"{i + start}\t{l}"
                if used + len(ln) + 1 > TOOL_OUTPUT_CAP and body_lines:
                    break
                body_lines.append(ln[: TOOL_OUTPUT_CAP])
                used += len(ln) + 1
            shown_end = start + len(body_lines) - 1
            body = "\n".join(body_lines)
            if shown_end < len(lines):
                body += f"\n...(showing lines {start}-{shown_end} of {len(lines)} -- continue with start={shown_end + 1})"
            return body
        return once(("show_file", path, start, end), produce)

    @tool
    def grep_repo(pattern: str, pathspec: str = "") -> str:
        """git grep -nE <pattern> across the whole repo at the PR head. Optional pathspec glob like 'scripts/**' or '*.ts'."""
        def produce():
            if len(pattern) > 250:
                return "rejected: pattern too long/expensive -- use a short, specific regex"
            cmd = ["git", "grep", "-n", "-E", "-e", pattern, sha]
            if pathspec:
                cmd += ["--", pathspec]
            p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
            if p.returncode == 1:
                return f"no matches for /{pattern}/ at the PR head" + (f" under {pathspec}" if pathspec else "") + ". Try a broader pattern or drop the pathspec."
            if p.returncode != 0:
                return f"error: {p.stderr.strip()[:300]}"
            # strip the "<sha>:" prefix for readability
            return cap("\n".join(l.split(":", 1)[1] if l.startswith(sha) else l for l in p.stdout.splitlines()))
        return once(("grep_repo", pattern, pathspec), produce)

    @tool
    def pr_diff(path: str = "") -> str:
        """The PR diff. Empty path = full diff; otherwise the diff hunk for that one file."""
        def produce():
            if not path:
                return cap(diff)
            chunks = re.split(r"(?m)^diff --git ", diff)
            for ch in chunks:
                # the header is `a/<old> b/<new>`: match the whole new path, never a substring
                if ch and ch.splitlines()[0].endswith(f" b/{path}"):
                    return cap("diff --git " + ch)
            return f"no diff hunk for {path}; see changed_files"
        return once(("pr_diff", path), produce)

    @tool
    def changed_files() -> str:
        """List files changed by this PR."""
        return once(("changed_files",), lambda: (
            "\n".join(changed_paths) if changed_paths is not None
            else run_cmd(["gh", "pr", "diff", str(pr_number), "--name-only"])))

    @tool
    def query_prod(sql: str) -> str:
        """One SELECT/WITH for data grounding, using a SELECT-only database role.
        PostgreSQL: double-quote case-sensitive identifiers."""
        dsn = os.environ.get("REVIEW_DATABASE_URL", "")
        if not dsn:
            return "unavailable: REVIEW_DATABASE_URL not set -- skip data grounding and say so in one line"
        stmt = sql.strip().rstrip(";")
        if ";" in stmt or not re.match(r"(?is)^(select|with)\b", stmt):
            return "rejected: a single read-only SELECT/WITH statement only"
        # DSN via libpq env vars, not argv -- a URI in argv is ps-visible, and
        # PGDATABASE does NOT accept a URI (verified: it tries a local socket).
        from urllib.parse import parse_qs, unquote, urlsplit

        u = urlsplit(dsn)
        q = parse_qs(u.query)
        pg_env = {**os.environ,
                  "PGHOST": u.hostname or "", "PGPORT": str(u.port or 5432),
                  "PGUSER": unquote(u.username or ""), "PGPASSWORD": unquote(u.password or ""),
                  "PGDATABASE": u.path.lstrip("/"),
                  "PGSSLMODE": (q.get("sslmode") or ["require"])[0],
                  # A WITH clause can contain a write despite its leading token.
                  # This is defense in depth; the supplied role must still be
                  # SELECT-only and restricted to data safe to share with a model.
                  "PGOPTIONS": (os.environ.get("PGOPTIONS", "")
                                + " -c default_transaction_read_only=on").strip()}
        def produce():
            p = subprocess.run(["psql", "-t", "-A", "-c", stmt], env=pg_env,
                               capture_output=True, text=True, errors="replace", timeout=30)
            if p.returncode != 0:
                return f"psql error: {p.stderr.strip()[:400]}"
            return cap(p.stdout.strip() or "(0 rows)")
        return once(("query_prod", stmt), produce)

    @tool
    def web_search(query: str) -> str:
        """Search the web for CURRENT library/API/spec docs. Returns titles, URLs and
        snippets in ~2s -- cheap to call. Snippets often answer version/deprecation
        questions outright; use scrape_url only when a snippet is not enough and you
        need the full page."""
        def produce():
            # Fast lane: Brave API directly (~1-2s). The deep shell tool
            # (web_search.sh) search+fetches PAGES with a ~13s uv-startup floor
            # per call -- measured as the brief's dominant wall cost (35-136s of
            # a run whose reviewers finish in 13-74s). Snippet triage + selective
            # scrape_url replaces it; the shell tool stays as the no-key fallback.
            fast = brave_search(query)
            if fast is not None:
                return cap(fast)
            wsh = WEB_TOOLS_DIR / "web_search.sh"
            if not wsh.exists():
                return "web search unavailable in this environment -- ground the brief from the repo/spec only"
            # --no-stealth: skip the camoufox headless browser (direct fetch +
            # BrightData fallback is enough for docs, and keeps CI light).
            out = subprocess.run(["bash", str(wsh), query, "-o", "json", "--fetch", "3", "--no-stealth"],
                                 capture_output=True, text=True, timeout=150)
            pages = (json.loads(out.stdout or "{}")).get("content", [])
            if not pages:
                return f"no results for {query!r}"
            return cap("\n\n".join(f"[source: {p.get('url')}]\n{p.get('title', '')}\n{(p.get('content') or '')[:2500]}"
                                   for p in pages))
        return once(("web_search", query), produce)

    @tool
    def scrape_url(url: str) -> str:
        """Fetch ONE specific doc / changelog / README URL to text. Use when you already know
        the exact page (e.g. a library's docs or GitHub release notes)."""
        def produce():
            # SSRF guard: only public http(s) doc pages (resolved-IP check defeats
            # DNS rebinding + integer-encoded IPs). A diff-derived "library name"
            # that is really an internal URL can't exfiltrate metadata into the brief.
            if not _public_host(url):
                return "rejected: only public http(s) documentation URLs are allowed"
            ssh = WEB_TOOLS_DIR / "scrape.sh"
            if not ssh.exists():
                return "scrape unavailable in this environment -- ground the brief from the repo/spec only"
            out = subprocess.run(["bash", str(ssh), url, "--no-stealth"], capture_output=True, text=True, timeout=120)
            return cap(out.stdout.strip() or f"empty fetch for {url}")
        return once(("scrape_url", url), produce)

    tools = [show_file, grep_repo, pr_diff, changed_files]
    # Least privilege: only the advisory agent gets database grounding.
    if with_db:
        tools.append(query_prod)
    if with_web:          # brief agent only
        tools += [web_search, scrape_url]
    return tools






def react_agent(model, tools, system_prompt: str, tool_budget: int = REVIEWER_TOOL_BUDGET):
    # langchain v1 home of the prebuilt ReAct agent (create_react_agent is
    # deprecated in langgraph 1.0 and its warning spams once per agent spawn).
    from langchain.agents import create_agent

    return create_agent(model, tools, system_prompt=system_prompt,
                        middleware=[make_convergence_middleware(tool_budget)])


def agent_text(result: dict) -> str:
    return content_to_text(result["messages"][-1].content)


# ---------------------------------------------------------------------------
# Graph state + nodes
# ---------------------------------------------------------------------------
class State(TypedDict, total=False):
    pr_ref: str
    provider: str
    model: str
    light_model: str       # optional cheaper model for the non-gating aux nodes (brief, advisory); falls back to `model`
    timeout: int
    mode: str
    pr: dict
    diff: str
    pack: str              # context pack: changed-file bodies @ head + matched learnings
    brief: str
    findings: Annotated[list, operator.add]
    deduped: list          # findings deduped at the collect barrier, BEFORE verify (fewer verify calls)
    advisory: str
    verdicts: Annotated[list, operator.add]
    result: dict


def brief_block(brief: str) -> str:
    if not brief:
        return ""
    return (f"\n\n--- BACKGROUND BRIEF (unverified, possibly-stale external source; "
            f"re-verify every fact against the pinned version + the actual code before you "
            f"flag anything) ---\n{brief}\n--- end brief ---")


def provider_caps(provider: str) -> tuple[int, int]:
    """(diff_cap, pack_cap) chars for inlined context -- see the PROVIDERS comment."""
    p = PROVIDERS.get(provider, {})
    return p.get("diff_cap", DIFF_INLINE_CAP), p.get("pack_cap", 40_000)


def cached_user(text: str, provider: str):
    """The agent's first user message, with an anthropic cache breakpoint on
    providers that support it (see the claude-vertex PROVIDERS comment). A
    breakpoint caches the ENTIRE prefix before it, so this one marker covers
    tools + system + the full context/pack for every later round of the loop."""
    if PROVIDERS.get(provider, {}).get("prompt_cache"):
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return text


_PACK_SKIP_SUFFIXES = (".lock", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf",
                       ".woff", ".woff2", "bun.lock", "package-lock.json")


def build_file_pack(sha: str, files: list[str], per_cap: int = PACK_PER_FILE_CAP,
                    total_cap: int = PACK_BUILD_CAP) -> str:
    """Full content of every changed file at the PR head, concatenated with caps.
    The measured failure mode this kills: each of 7 reviewers independently
    show_file'ing the same changed files -- 45 LLM calls / 475k input tokens on a
    one-word diff, all latency. Deleted/binary files are skipped silently."""
    parts, total = [], 0
    for f in files:
        if f.endswith(_PACK_SKIP_SUFFIXES):
            continue
        try:
            body = run_cmd(["git", "show", f"{sha}:{f}"], timeout=30)
        except Exception:
            continue  # deleted in this PR, or binary -- reviewers see it in the diff
        if "\x00" in body[:1000]:
            continue
        if len(body) > per_cap:
            body = body[:per_cap] + f"\n...[truncated at {per_cap} chars -- show_file for the rest]"
        block = f"===== {f} @ head =====\n{body}"
        if total + len(block) > total_cap:
            parts.append(f"===== {f} @ head ===== [omitted -- pack budget; use show_file]")
            continue
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _path_signals(files: list[str]) -> set[str]:
    """Terms worth matching learnings against: exact basenames (high precision),
    stems and directory names >=4 chars (topic-ish). Generic roots are dropped."""
    generic = {"scripts", "src", "app", "docs", "test", "tests", "index", "readme", "main"}
    sig = set()
    for f in files:
        p = Path(f)
        sig.add(p.name.lower())
        if len(p.stem) >= 4 and p.stem.lower() not in generic:
            sig.add(p.stem.lower())
        for part in p.parts[:-1]:
            part = part.lower().lstrip(".")
            if len(part) >= 4 and part not in generic:
                sig.add(part)
    return sig


def pull_learnings(files: list[str], root: Path | None = None, cap: int = LEARNINGS_PACK_CAP) -> str:
    """Sections of learnings/*.md matched to the touched paths, best-first.
    Mechanical stand-in for the human reflex 'grep learnings/ before touching X'
    -- reviewers otherwise never see the curated gotchas. No external index is
    required, so this also works on a fresh runner. Score = occurrences of path
    signals in a section; >=2 required so a stray mention doesn't qualify."""
    sig = _path_signals(files)
    # Resolve the target checkout even when called from one of its subdirectories.
    ldir = (root if root is not None else target_repo_root()) / "learnings"
    if not sig or not ldir.is_dir():
        return ""
    # Basenames vs looser stems/dirs are scored on disjoint text: a stem is a
    # substring of its own basename ("shop" in "shop.ts"), so counting both on the
    # same text double-counts every basename mention and one stray reference
    # would clear the >=2 noise floor.
    names = {s for s in sig if "." in s}
    loose = sig - names
    scored = []
    for lf in sorted(ldir.glob("*.md")):
        if lf.name == "README.md":  # table of contents, no content
            continue
        try:
            text = lf.read_text(errors="ignore")
        except OSError:
            continue
        for sec in re.split(r"(?m)^(?=## )", text):
            low = sec.lower()
            score = sum(low.count(n) for n in names)
            for n in names:
                low = low.replace(n, " ")
            # Loose signals match on WORD BOUNDARIES: bare `count` let "review"
            # score inside "human-reviewable"/"eyeball-reviewable" and pulled
            # unrelated sections into the pack.
            score += sum(len(re.findall(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", low))
                         for s in loose)
            if score >= 2:
                scored.append((score, f"[{lf.name}] {sec.strip()}"))
    scored.sort(key=lambda t: -t[0])
    parts, total = [], 0
    for _, sec in scored:
        if total + len(sec) > cap:
            break
        parts.append(sec)
        total += len(sec)
    return "\n\n".join(parts)


def pull_principles(root: Path | None = None, cap: int = 18_000) -> str:
    """Read the first existing agent-instructions file from the target checkout.

    Runtime loading keeps the prompt aligned with the repo. Read the whole file,
    not a section layout specific to one project. Any cap loss stays visible.
    """
    root = root if root is not None else target_repo_root()
    for name in ("CLAUDE.md", "AGENTS.md"):
        path = root / name
        if not path.exists():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            return ""
        if len(text) <= cap:
            return text
        # Retain the established marker so a clipped instruction pack is obvious
        # to both reviewers and tooling, even under a very small test cap.
        marker = "\n...[preamble truncated -- raise pull_principles(cap=)]"
        return text[:max(0, cap - len(marker))] + marker[:max(0, cap)]
    return ""


def pull_blast_radius(base: str, head: str) -> str:
    """Which entry points this diff reaches that the diff does NOT contain, or "".

    Reviewers read the PATCH but can miss a page several import hops away.
    The sibling reach.py reads the checkout's imports and walks them backwards.
    No index or extra installation is needed, so this also works in CI.

    FAILS OPEN. A nice-to-have context section must never kill the reviewer,
    including when a helper is absent or a subprocess cannot run.
    """
    script = Path(__file__).resolve().parent / "reach.py"
    if not script.exists():
        return ""
    # The PR base arrives as a branch NAME ("main"); in a CI checkout only the remote-
    # tracking ref is guaranteed to exist, so try that first and fall back to the bare
    # name for a local run.
    for ref in (f"origin/{base}", base):
        try:
            out = subprocess.run([sys.executable, str(script), "--base", ref, "--head", head],
                                 capture_output=True, text=True, timeout=180)
        except (subprocess.SubprocessError, OSError):
            return ""
        if out.returncode == 0 and "ENTRY POINTS AFFECTED" in out.stdout:
            return out.stdout.strip()[:BLAST_PACK_CAP]
    return ""


def pull_stale_guards(base: str, head: str) -> str:
    """Guard runs in this diff that a newly added field never reached, or "".

    The sibling of pull_blast_radius, aimed the other way. Reachability points
    beyond the diff; this looks inside it for old conditionals that never gained
    a newly added input. The two signals answer different questions.

    Quiet by construction and fail-open, like its sibling: optional context
    must never prevent the review from running.
    """
    script = Path(__file__).resolve().parent / "stale_guards.py"
    if not script.exists():
        return ""
    # Ask the judge only when a key exists. --judge exits 3 without one, and a section
    # that vanishes because a key rotated is worse than a section with no probabilities.
    judged = bool(os.environ.get("TYPESAFE_API_KEY"))
    for ref in (f"origin/{base}", base):
        for args in ([("--judge",)] if judged else []) + [()]:
            try:
                out = subprocess.run([sys.executable, str(script), "--base", ref,
                                      "--head", head, *args],
                                     capture_output=True, text=True, timeout=180)
            except (subprocess.SubprocessError, OSError):
                return ""
            if out.returncode != 0 or "guard run(s)" not in out.stdout:
                continue          # judge unreachable or key rejected -> try unjudged
            # A clean run says "0 guard run(s)". Nothing to tell a reviewer, so say nothing.
            if re.search(r"\b0 guard run\(s\)", out.stdout):
                return ""
            return out.stdout.strip()[:GUARDS_PACK_CAP]
    return ""


def compose_pack(file_pack: str, learn: str, principles: str = "",
                 blast: str = "", guards: str = "") -> str:
    """Blast radius, then principles, then learnings, then file bodies. The injection
    slice is a prefix cut (`pack[:pack_cap]`), so anything appended after the file bodies
    was silently dropped whenever they filled pack_cap -- two 20k files already
    exceed gemini/glm's 40k. Instructions and learnings have their own caps;
    leading with them keeps them visible when changed files fill the budget.
    Blast radius goes FIRST because it is the smallest section (<= BLAST_PACK_CAP) and
    the only one naming files the reviewer would otherwise never open."""
    parts = []
    if blast:
        parts.append("===== reachability of this diff (deterministic, not a model) =====\n"
                     "Entry points listed as NOT IN THE DIFF are reached by a changed symbol\n"
                     "through the call graph. Their behaviour can change even though this PR\n"
                     "does not touch them. Treat as facts to CHECK, not findings.\n" + blast)
    if guards:
        parts.append("===== guard runs this diff's new fields never reached (deterministic) =====\n"
                     "Each block is consecutive conditionals testing sibling fields of one\n"
                     "object, where this diff added another sibling that no line tests. Ask\n"
                     "whether the new field belongs there. Treat as a QUESTION, not a finding.\n"
                     "A `p=` score is a typed judge's estimate that the field really is missing;\n"
                     "it is calibrated on 13 labelled cases, so weigh it, never quote it.\n"
                     + guards)
    if principles:
        parts.append("===== repo agent instructions (CLAUDE.md / AGENTS.md, live) =====\n" + principles)
    if learn:
        parts.append("===== learnings/ sections matched to the touched paths =====\n" + learn)
    if file_pack:
        parts.append(file_pack)
    return "\n\n".join(parts)


def pack_block(pack: str) -> str:
    if not pack:
        return ""
    return f"""

Pre-fetched context pack -- the repo's live agent instructions (CLAUDE.md / AGENTS.md), learnings
sections matched to the touched paths, and the FULL content of the changed files at the
PR head. Judge the diff against those principles where relevant (they are context, not a
checklist -- do not manufacture findings from them). Do NOT re-open these files with
show_file; spend your tool budget on what is NOT here (callers/consumers of changed
symbols, unchanged files, tests).
{pack}"""


def pr_context(pr: dict, diff: str, include_thread: bool = False, brief: str = "",
               pack: str = "", diff_cap: int = DIFF_INLINE_CAP) -> str:
    inline = diff if len(diff) <= diff_cap else diff[:diff_cap] + "\n...[diff truncated -- use the pr_diff tool per file]"
    # With a pack, the changed files are already in-context: the mandatory-tool rule
    # shifts from "read the files" to "read their CONSUMERS" (the pack can't contain
    # callers -- they're outside the changed set by definition).
    must = ("The changed files are inlined IN FULL in the context pack below -- do NOT re-read them\n"
            "with show_file. You MUST still use tools to inspect CONSUMERS of changed symbols (grep\n"
            "callers, open callees of subprocess/CLI invocations) unless the change is provably\n"
            "self-contained (docs/tests only)." if pack else
            "Do NOT conclude from the inline diff\n"
            "alone -- before returning JSON you MUST inspect the changed files and their consumers with\n"
            "tools; a review with zero tool calls is invalid.")
    verification = ""
    if pr.get("review_mode") == "verification":
        verification = f"""

VERIFICATION MODE — NOT DISCOVERY. Review only the previous graph review's finding batch below
and regressions directly caused by the fix delta. Do not report independent issues or unrelated
defects even if real. If every listed finding is resolved, return zero findings. This is the
one allowed focused verification pass.

Previous blocker batch (verified blockers and majors — majors withhold auto-approval, so
re-check them exactly like blockers):
{pr.get('verification_scope') or '(missing — return zero findings)'}
"""
    return f"""PR #{pr['number']}: "{pr['title']}" by {pr['author']} ({pr['headRef']} -> {pr['baseRef']}, +{pr['additions']}/-{pr['deletions']}).
PR body:
{pr['body'][:4000]}

Changed files: {', '.join(pr['files'])}

Tools: show_file/grep_repo read the repo AT THE PR HEAD (already fetched -- no checkout
needed); pr_diff gives per-file hunks; changed_files lists paths. Walk the code, not just
the diff: grep callers of modified symbols, open every callee
of any subprocess/CLI invocation. You are READ-ONLY. {must} Investigate EFFICIENTLY: scale effort to
the diff (a small diff usually needs 5-15 tool calls), never repeat a search that returned
nothing, and when investigation stops yielding new information, STOP and return your JSON.

Full diff (may be truncated):
{inline}{verification}{pack_block(pack)}{brief_block(brief)}{thread_block(pr) if include_thread else ""}"""


def thread_block(pr: dict) -> str:
    t = pr.get("thread") or ""
    if not t:
        return ""
    return f"""

Prior review discussion (freshly fetched from the PR thread -- earlier lane reviews and
author replies):
{t}

Discussion rules: (1) do NOT re-raise a finding that was explicitly declined or refuted
with a stated reason in the discussion, unless the current diff changes the picture -- if
you considered one, note it in one line instead of a finding; (2) if a prior review
reported a blocker or major, explicitly check whether the current head fixes it."""


GRAPH_COMMENT_PREFIX = "## \U0001F578\uFE0F Graph Review"  # spider web emoji header
PREV_FINDING_RE = re.compile(r"- \*\*`(?P<file>[^:`]+):(?P<line>\d+)` -- (?P<title>.+?)\*\*")
GRAPH_MARKER_RE = re.compile(
    r"<!-- graph-review-head:(?P<head>[0-9a-f]{40}) mode:(?P<mode>discovery|verification) -->")
BLOCKER_SECTION_RE = re.compile(r"(?ms)^### Blockers?\s*\n(?P<body>.*?)(?=^### |\Z)")
# Majors joined the verification scope on 2026-08-17: they withhold auto-approval,
# so the fix head's focused pass must re-check them exactly like blockers.
MAJOR_SECTION_RE = re.compile(r"(?ms)^### Majors?\s*\n(?P<body>.*?)(?=^### |\Z)")


def gh_api_list(path: str) -> list:
    """gh api --paginate on an array endpoint EITHER merges pages into one array
    OR concatenates them (`[...][...]`, newline-separated) depending on gh version
    -- `json.loads` on the raw concatenated stream throws 'Extra data'. Decode array-by-array to be
    version-independent. --paginate keeps the RECENT tail (items[-8:]) + the latest
    graph review on a >100-comment PR."""
    text = run_cmd(["gh", "api", "--paginate", path]).strip()
    if not text:
        return []
    dec, items, pos, n = json.JSONDecoder(), [], 0, len(text)
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        obj, pos = dec.raw_decode(text, pos)
        items.extend(obj if isinstance(obj, list) else [obj])
    return items


def fetch_thread(pr_number: int) -> tuple[str, list[dict], str, str]:
    """Prior review-lane discussion, FRESHLY fetched from the PR (not stored
    state -- replay-safe): lane comments + formal review bodies + human
    replies, plus the previous graph review's findings parsed for the
    mechanical NEW / STILL-OPEN / RESOLVED diff in the report. Fail-soft."""
    try:
        comments = gh_api_list(f"repos/{{owner}}/{{repo}}/issues/{pr_number}/comments?per_page=100")
        reviews = gh_api_list(f"repos/{{owner}}/{{repo}}/pulls/{pr_number}/reviews?per_page=100")
    except Exception as e:
        log(f"[fetch] thread skipped: {str(e)[:120]}")
        return "", [], "", ""
    items = []
    for c in comments + reviews:
        body = (c.get("body") or "").strip()
        if not body:
            continue
        login = (c.get("user") or {}).get("login", "?")
        is_lane = (login == "github-actions[bot]"
                   and body.startswith(("## \U0001F578", "## \U0001F9EA")))
        is_human = (c.get("user") or {}).get("type") == "User"
        is_claude = login.startswith("claude") and login.endswith("[bot]")
        if is_lane or is_human or is_claude:
            items.append((c.get("created_at") or c.get("submitted_at") or "", login, body))
    items.sort(key=lambda x: x[0])
    prev_findings: list[dict] = []
    previous_head = ""
    blocker_scope = ""
    for _, _, body in items:
        if body.startswith(GRAPH_COMMENT_PREFIX):  # keep the LAST graph review's findings
            prev_findings = [
                {"file": m.group("file"), "line": int(m.group("line")), "title": m.group("title")}
                for m in PREV_FINDING_RE.finditer(body)]
            marker = GRAPH_MARKER_RE.search(body)
            blockers = BLOCKER_SECTION_RE.search(body)
            majors = MAJOR_SECTION_RE.search(body)
            if marker:
                previous_head = marker.group("head")
                parts = []
                if blockers:
                    parts.append("### Blockers\n" + blockers.group("body").strip())
                if majors:
                    parts.append("### Majors\n" + majors.group("body").strip())
                blocker_scope = "\n\n".join(parts)[:8_000]
    thread = "\n\n".join(f"[{login} {ts[:16]}]\n{body[:1500]}" for ts, login, body in items[-8:])
    return thread[-10_000:], prev_findings, previous_head, blocker_scope


def node_fetch(state: State) -> dict:
    ref = state["pr_ref"]
    if not ref.isdigit():
        out = json.loads(run_cmd(["gh", "pr", "list", "--head", ref, "--json", "number", "--limit", "1"]))
        if not out:
            raise RuntimeError(f"no open PR for branch {ref!r}")
        ref = str(out[0]["number"])
    n = int(ref)
    mode = state.get("mode", "discovery")
    log(f"[fetch] PR #{n}: metadata + {mode} diff + head objects")
    meta = json.loads(run_cmd([
        "gh", "pr", "view", str(n), "--json",
        "number,title,author,baseRefName,headRefName,body,additions,deletions,files",
    ]))
    run_cmd(["git", "fetch", "origin", f"pull/{n}/head"], timeout=300)
    sha = run_cmd(["git", "rev-parse", "FETCH_HEAD"]).strip()
    pr = {
        "number": meta["number"],
        "title": meta["title"],
        "author": meta["author"]["login"],
        "baseRef": meta["baseRefName"],
        "headRef": meta["headRefName"],
        "headSha": sha,
        "body": meta.get("body") or "",
        "additions": meta["additions"],
        "deletions": meta["deletions"],
        "files": [f["path"] for f in meta["files"]],
    }
    thread, prev_findings, previous_head, blocker_scope = fetch_thread(n)
    pr["thread"], pr["prev_findings"] = thread, prev_findings
    pr["review_mode"] = mode
    if mode == "verification":
        if not previous_head or not blocker_scope:
            raise RuntimeError(
                "verification mode requires a prior marked graph review with blockers or majors")
        try:
            run_cmd(["git", "cat-file", "-e", f"{previous_head}^{{commit}}"])
        except Exception:
            run_cmd(["git", "fetch", "origin", previous_head], timeout=300)
        diff = run_cmd(["git", "diff", "--unified=80", f"{previous_head}..{sha}"], timeout=180)
        files = run_cmd(["git", "diff", "--name-only", f"{previous_head}..{sha}"]).splitlines()
        numstat = run_cmd(["git", "diff", "--numstat", f"{previous_head}..{sha}"]).splitlines()
        pr["files"] = files
        pr["additions"] = sum(int(row.split("\t", 1)[0]) for row in numstat
                              if row and row.split("\t", 1)[0].isdigit())
        pr["deletions"] = sum(int(row.split("\t", 2)[1]) for row in numstat
                              if len(row.split("\t", 2)) > 1 and row.split("\t", 2)[1].isdigit())
        pr["verification_scope"] = blocker_scope
        pr["previousReviewHead"] = previous_head
    else:
        diff = run_cmd(["gh", "pr", "diff", str(n)], timeout=180)
    # Context pack: built ONCE here (checkpointed with fetch -> replay-safe),
    # sliced to each provider's pack_cap at injection time. Learnings go FIRST:
    # they are small (<= LEARNINGS_PACK_CAP) and the injection slice is a prefix
    # cut, so appending them last silently dropped them whenever the file bodies
    # alone filled pack_cap (two 20k files already exceed the smaller pack caps).
    learn = pull_learnings(pr["files"])
    principles = pull_principles()
    blast = pull_blast_radius(pr["baseRef"], sha)
    if blast:
        # Log the COUNT, not merely that it ran. The forward test is "over the next N PRs,
        # how often is this column non-empty and how often did it matter", and a boolean
        # cannot answer that -- it only says the section existed. Greppable on purpose.
        hit = re.search(r"NOT IN THE DIFF: (\d+)", blast)
        log(f"[fetch] reach: {hit.group(1) if hit else '?'} entry point(s) "
            "affected but not in the diff")
    guards = pull_stale_guards(pr["baseRef"], sha)
    if guards:
        n = re.search(r"(\d+) guard run\(s\)", guards)
        log(f"[fetch] stale guards: {n.group(1) if n else '?'} guard run(s) "
            "a new field never reached")
    pack = compose_pack(build_file_pack(sha, pr["files"]), learn, principles, blast, guards)
    # Warm the shared model here, single-threaded: the FIRST make_model pays
    # heavy lazy imports + (in CI) the WIF token mint.
    # lru_cache doesn't dedupe CONCURRENT first calls, so on brief-skipped PRs
    # the parallel reviewers would all pay it. One call before fan-out avoids that.
    make_model(state["provider"], state["model"], state["timeout"])
    log(f"[fetch] {mode} head {sha[:12]}, {len(pr['files'])} files, +{pr['additions']}/-{pr['deletions']}, "
        f"thread {len(pr['thread'])} chars, prev findings {len(pr['prev_findings'])}, "
        f"pack {len(pack)} chars ({len(learn)} learnings)")
    return {"pr": pr, "diff": diff, "pack": pack}


# Match real import/dependency STATEMENTS in added lines, not the English words
# "use/from/import" in prose (which over-triggered the brief on doc-only diffs).
# Keyword must be at statement position (after +/indent), or a dep-manifest pattern.
_IMPORT_RE = re.compile(
    r'(?m)^\+\s*('
    r'import\s|from\s+[\w.\'"@/]+\s+import\b|export\s.*\sfrom\s|'   # py / es import
    r'(?:const|let|var)\s+\S.*=\s*require\(|require\(|'             # cjs require
    r'use\s+[\w:]+\s*;|'                                           # rust use path::;
    r'"[\w@./-]+"\s*:\s*"[\^~]?\d|'                                # package.json dep line
    r'#\s*dependencies|#\s*///\s*script'                           # PEP 723 inline deps
    r')')
_MANIFEST_SUFFIXES = ("package.json", "pyproject.toml", "requirements.txt", "go.mod", "Cargo.toml")


def touches_external_surface(pr: dict, diff: str) -> bool:
    """Cheap gate: is there an external library / dependency / spec surface worth
    researching? Docs-only and pure-internal-refactor PRs skip the brief (its web
    cost only earns its keep when the diff actually leans on an external API)."""
    files = pr.get("files", [])
    if any(f.endswith(_MANIFEST_SUFFIXES) for f in files):
        return True
    if "PEP 723" in diff or "# /// script" in diff:  # inline uv deps
        return True
    if any("/specs/" in f or f.endswith(".md") and "spec" in f.lower() for f in files):
        return True
    return bool(_IMPORT_RE.search(diff))


def node_brief(state: State) -> dict:
    pr, diff = state["pr"], state["diff"]
    if state.get("mode") == "verification":
        log("[brief] verification mode -- skipping discovery research")
        return {"brief": ""}
    # Skip when the web tools aren't installed (the CI runner): without them the
    # brief is a repo-only grep the reviewers already do -- a full ReAct budget of
    # pure overhead on the critical path. The brief is a local affordance that
    # auto-activates in CI the moment the tools ship there.
    if not (WEB_TOOLS_DIR / "web_search.sh").exists():
        log("[brief] web tools absent (e.g. CI) -- skipping (reviewers ground from the repo themselves)")
        return {"brief": ""}
    if not touches_external_surface(pr, diff):
        log("[brief] no external library/spec surface -- skipping (repo-internal PR)")
        return {"brief": ""}
    # A light model can reduce auxiliary-node latency and avoid competing with
    # gating reviewers for the same model's quota.
    model = make_model(state["provider"], state.get("light_model") or state["model"], state["timeout"])
    # tools built INSIDE the factory so each retry gets a fresh anti-loop cache
    # (a stale `seen` would refuse a retried agent's first legitimate call).
    agent = lambda: react_agent(model, make_tools(
        pr["headSha"], pr["number"], diff, with_web=True, changed_paths=pr["files"]), BRIEF_PROMPT,
                                tool_budget=BRIEF_TOOL_BUDGET)
    t0 = time.time()
    try:
        dcap, pcap = provider_caps(state["provider"])
        ctx = pr_context(pr, diff, pack=state.get("pack", "")[:pcap], diff_cap=dcap)
        res = invoke_with_backoff(agent, {"messages": [("user", cached_user(ctx, state["provider"]))]},
                                  config={"recursion_limit": BRIEF_RECURSION_LIMIT, "run_name": "brief"})
        brief = agent_text(res).strip()
    except Exception as e:  # fail-open: reviewers run without a brief
        log(f"[brief] FAILED after {time.time() - t0:.0f}s: {str(e)[:160]} -- reviewers run unbriefed")
        return {"brief": ""}
    log(f"[brief] {len(brief)} chars in {time.time() - t0:.0f}s")
    return {"brief": brief}


def fan_out_reviews(state: State):
    from langgraph.types import Send

    # NOTE: brief stays BEFORE the fan-out (fetch -> brief -> here). A parallel-
    # brief branch (non-brief reviewers from fetch, brief-dims from brief) was
    # tried and REVERTED: LangGraph's `collect` barrier double-fires when
    # review_dim is fanned out across two supersteps (verified -- it ran once on
    # the incomplete first wave, missing correctness+wiring). The brief is ~40s on
    # a critical path dominated by the slowest reviewer (~400s), so the ~9% trim
    # isn't worth risking a premature verify/report on a gating reviewer.
    base = {"pr": state["pr"], "diff": state["diff"], "provider": state["provider"],
            "model": state["model"], "timeout": state["timeout"],
            "brief": state.get("brief", ""), "pack": state.get("pack", ""),
            "mode": state.get("mode", "discovery")}
    # light_model is scoped to advisory ONLY -- the 6 gating review_dim payloads
    # never carry it, so no future edit can silently downgrade a gating reviewer
    # to the cheap tier by copying node_advisory's `payload.get("light_model")`.
    sends = [Send("review_dim", {**base, "dimension": d}) for d in DIMENSIONS]
    sends.append(Send("advisory", {**base, "light_model": state.get("light_model", "")}))
    return sends


def node_review_dim(payload: dict) -> dict:
    dim = payload["dimension"]
    pr, diff = payload["pr"], payload["diff"]
    model = make_model(payload["provider"], payload["model"], payload["timeout"])
    mode_prompt = VERIFICATION_PROMPT if payload.get("mode") == "verification" else ""
    agent = lambda: react_agent(model, make_tools(
        pr["headSha"], pr["number"], diff, changed_paths=pr["files"]),
                        REVIEWER_PROMPT.format(dimension=dim, dimension_prompt=DIMENSIONS[dim],
                                               findings_format=FINDINGS_FORMAT, mode_prompt=mode_prompt))
    t0 = time.time()
    try:
        brief = payload.get("brief", "") if dim in BRIEF_DIMENSIONS else ""
        dcap, pcap = provider_caps(payload["provider"])
        ctx = pr_context(pr, diff, include_thread=True, brief=brief,
                         pack=payload.get("pack", "")[:pcap], diff_cap=dcap)
        res = invoke_with_backoff(agent, {"messages": [("user", cached_user(ctx, payload["provider"]))]},
                                  config={"recursion_limit": AGENT_RECURSION_LIMIT,
                                          "run_name": f"review:{dim}"})
        data = parse_json_or_repair(agent_text(res), FINDINGS_FORMAT, model, payload["timeout"])
    except Exception as e:  # one dimension failing must not kill the run
        log(f"[review:{dim}] FAILED after {time.time() - t0:.0f}s: {str(e)[:200]}")
        return {"findings": [{"key": f"{dim}:error", "dimension": dim, "severity": "meta",
                              "title": f"reviewer failed: {str(e)[:160]}"}]}
    if data is None:
        log(f"[review:{dim}] unparseable output -- dimension not covered")
        return {"findings": [{"key": f"{dim}:error", "dimension": dim, "severity": "meta",
                              "title": "reviewer output unparseable"}]}
    findings, empty = normalize_findings(data.get("findings", []), dim)
    if empty:
        # Logged, never silent: an empty finding is a REVIEWER defect, and the
        # raw/kept counts in the report header would otherwise disagree with the
        # finding list for no visible reason.
        log(f"[review:{dim}] dropped {empty} empty finding(s) (placeholder title)")
    dropped = int(data.get("droppedCount") or 0)
    if dropped:
        log(f"[review:{dim}] hit the {MAX_FINDINGS_PER_DIMENSION}-finding cap, dropped {dropped}")
        findings.append({"key": f"{dim}:dropped", "dimension": dim, "severity": "meta",
                         "title": f"{dropped} finding(s) dropped by the per-dimension cap"})
    log(f"[review:{dim}] {sum(1 for f in findings if f['severity'] != 'meta')} finding(s) in {time.time() - t0:.0f}s")
    return {"findings": findings}


def node_advisory(payload: dict) -> dict:
    if payload.get("mode") == "verification":
        return {"advisory": "## Product review (advisory)\n\n_n/a — focused blocker verification only._"}
    pr, diff = payload["pr"], payload["diff"]
    # Aux node -> light model when set (see node_brief). Advisory is non-gating
    # product/principle commentary; it need not share the reviewers' strong tier.
    model = make_model(payload["provider"], payload.get("light_model") or payload["model"], payload["timeout"])
    agent = lambda: react_agent(model, make_tools(
        pr["headSha"], pr["number"], diff, with_db=True, changed_paths=pr["files"]), ADVISORY_PROMPT)
    try:
        # No brief for advisory -- the brief is SELECTIVE to correctness+wiring
        # (external-API facts), and product advisory must not be anchored by it.
        dcap, pcap = provider_caps(payload["provider"])
        ctx = pr_context(pr, diff, include_thread=True,
                         pack=payload.get("pack", "")[:pcap], diff_cap=dcap)
        res = invoke_with_backoff(agent, {"messages": [("user", cached_user(ctx, payload["provider"]))]},
                                  config={"recursion_limit": AGENT_RECURSION_LIMIT,
                                          "run_name": "advisory"})
        text = agent_text(res).strip()
    except Exception as e:
        text = f"## Product review (advisory)\n\n_advisory agent failed: {str(e)[:160]}_"
    log("[advisory] done")
    return {"advisory": text}


def node_collect(state: State) -> dict:
    """Barrier: all reviewers are in. Dedupe here so verify runs once per distinct
    finding (a cross-dimension duplicate would otherwise be verified N x 3 lenses)."""
    return {"deduped": dedupe_findings(state.get("findings", []))}


def fan_out_verify(state: State):
    from langgraph.types import Send

    to_verify = [f for f in state.get("deduped", []) if f["severity"] in ("blocker", "major")]
    if not to_verify:
        log("[verify] nothing blocker/major -- skipping straight to report")
        return "report"
    log(f"[verify] {len(to_verify)} deduped finding(s) x {len(LENSES)} lenses")
    base = {"pr": state["pr"], "diff": state["diff"], "provider": state["provider"],
            "model": state["model"], "timeout": state["timeout"]}
    return [Send("verify_one", {**base, "finding": f, "lens": lens})
            for f in to_verify for lens in LENSES]


def node_verify_one(payload: dict) -> dict:
    f, lens = payload["finding"], payload["lens"]
    pr, diff = payload["pr"], payload["diff"]
    model = make_model(payload["provider"], payload["model"], payload["timeout"])
    agent = lambda: react_agent(model, make_tools(
        pr["headSha"], pr["number"], diff, changed_paths=pr["files"]),
                        f"You are an adversarial verifier in a PR review. Apply ONLY this lens:\n"
                        f"{LENSES[lens]}\n\n{VERDICT_FORMAT}", tool_budget=VERIFY_TOOL_BUDGET)
    claim = (f"A reviewer (dimension: {f['dimension']}) claims:\n"
             f"FILE: {f.get('file')}:{f.get('line')}\nSEVERITY: {f['severity']}\n"
             f"CLAIM: {f.get('title')}\nWHY: {f.get('why')}\n"
             f"FAILURE SCENARIO: {f.get('failure_scenario')}\n\n"
             f"Check the ACTUAL code, not the claim's plausibility.\n\n{pr_context(pr, diff)}")
    try:
        res = invoke_with_backoff(agent, {"messages": [("user", cached_user(claim, payload["provider"]))]},
                                  config={"recursion_limit": VERIFY_RECURSION_LIMIT,
                                          "run_name": f"verify:{lens}:{f['key']}"})
        data = parse_json_or_repair(agent_text(res), VERDICT_FORMAT, model, payload["timeout"])
    except Exception as e:
        log(f"[verify:{lens}:{f['key']}] FAILED: {str(e)[:160]}")
        data = None
    if data is None:
        return {"verdicts": []}  # a dead verifier casts no vote
    return {"verdicts": [{"key": f["key"], "lens": lens,
                          "refuted": bool(data.get("refuted")),
                          "inflated": bool(data.get("inflated")),
                          "reasoning": str(data.get("reasoning", ""))[:600]}]}


# ---------------------------------------------------------------------------
# Report (pure Python -- no LLM)
# ---------------------------------------------------------------------------
def title_tokens(t: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", str(t).lower()) if len(w) > 2}


def similar(a: dict, b: dict) -> bool:
    if a.get("file") != b.get("file"):
        return False
    if a.get("line") == b.get("line"):
        return True
    ta, tb = title_tokens(a.get("title", "")), title_tokens(b.get("title", ""))
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.5


def dedupe_findings(findings: list) -> list:
    """Merge cross-dimension duplicates (same file+line or >=0.5 title-token overlap),
    keeping the HIGHEST-severity representative (its `key` drives verification) and
    tracking every dimension that raised it. Excludes meta notes. Done at the collect
    barrier BEFORE verify so a finding raised by N dimensions is verified once, not N times."""
    real = [f for f in findings if f.get("severity") != "meta"]
    deduped: list = []
    for f in sorted(real, key=lambda x: (SEV_RANK[x["severity"]], x.get("file", ""))):
        twin = next((m for m in deduped if similar(m, f)), None)
        if twin is None:
            deduped.append({**f, "dimensions": [f["dimension"]]})
        elif f["dimension"] not in twin["dimensions"]:
            twin["dimensions"].append(f["dimension"])
    return deduped


def node_report(state: State) -> dict:
    raw_findings = state.get("findings", [])
    deduped_in = state.get("deduped", [])  # already merged at the collect barrier
    verdicts = state.get("verdicts", [])
    meta = [f for f in raw_findings if f["severity"] == "meta"]
    meta_notes = [f["title"] for f in meta]
    # review_coverage_gaps must count only GENUINE degradation -- a dimension that
    # errored / was unparseable (key `<dim>:error`) -- NOT the per-dimension finding
    # cap (`<dim>:dropped`), which is overflow (found MORE than the cap), the opposite
    # of a gap. Both still render as coverage notes below; only real deaths score.
    gaps = sum(1 for f in meta if f["key"].endswith(":error"))
    cap_drops = sum(1 for f in meta if f["key"].endswith(":dropped"))
    by_key: dict[str, list[dict]] = {}
    for v in verdicts:
        by_key.setdefault(v["key"], []).append(v)

    kept, killed, downgraded = [], [], 0
    for f in deduped_in:
        if f["severity"] not in ("blocker", "major"):
            kept.append(f)
            continue
        votes = by_key.get(f["key"], [])
        refutes = sum(v["refuted"] for v in votes)
        if not votes or refutes * 2 > len(votes):
            killed.append({**f, "refutations": [v["reasoning"] for v in votes if v["refuted"]] or ["no verifier vote survived"]})
            continue
        f = dict(f)
        f["verification"] = f"{len(votes) - refutes}/{len(votes)} upheld"
        if sum(1 for v in votes if not v["refuted"] and v["inflated"]) * 2 > len(votes):
            f["severity"] = SEV_DOWN[f["severity"]]
            f["verification"] += ", severity downgraded"
            downgraded += 1
        kept.append(f)

    # kept is already deduped; only re-sort (a downgrade can change severity order).
    deduped = sorted(kept, key=lambda x: (SEV_RANK[x["severity"]], x.get("file", "")))

    # Mechanical continuity vs the PREVIOUS graph review on this PR (parsed
    # from its posted comment at fetch time). Pure Python -- the LLM layers
    # stay clean-slate, so continuity can never anchor a judgment.
    prev = state.get("pr", {}).get("prev_findings") or []
    for f in deduped:
        f["continuity"] = "still open" if any(similar(f, p) for p in prev) else "new"
    resolved = [p for p in prev if not any(similar(p, f) for f in deduped)]

    raw_count = sum(1 for f in raw_findings if f["severity"] != "meta")
    stats = {"raw": raw_count, "kept": len(deduped), "killed": len(killed),
             "downgraded": downgraded, "meta": meta_notes, "gaps": gaps,
             "cap_drops": cap_drops, "resolved": len(resolved) if prev else None}
    return {"result": {"findings": deduped, "killed": killed, "resolved": resolved,
                       "stats": stats}}


def render_markdown(pr: dict, result: dict, advisory: str) -> str:
    deduped, stats = result["findings"], result["stats"]
    out = [f"## Multi-agent review of PR #{pr['number']} -- {pr['title']}", ""]
    blockers = [f for f in deduped if f["severity"] == "blocker"]
    if not blockers:
        out += ["Blocker-free.", ""]
    counts = {}
    for f in deduped:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: SEV_RANK[kv[0]])) or "no findings"
    out.append(f"**{summary}** -- {stats['raw']} raw finding(s); {stats['killed']} killed by 3-lens "
               f"adversarial verification; {stats['downgraded']} downgraded as severity-inflated.")
    if stats.get("resolved") is not None:
        still = sum(1 for f in deduped if f.get("continuity") == "still open")
        out.append(f"\nVs the previous review: {len(deduped) - still} new, {still} still open, "
                   f"{stats['resolved']} resolved.")
    for note in stats["meta"]:
        out.append(f"\n_coverage note: {note}_")
    out.append("")
    for sev in SEVERITIES:
        group = [f for f in deduped if f["severity"] == sev]
        if not group:
            continue
        out += [f"### {sev.capitalize()}{'s' if len(group) > 1 else ''}", ""]
        for f in group:
            ver = f", verified {f['verification']}" if f.get("verification") else ""
            cont = f"; {f['continuity']}" if f.get("continuity") else ""
            out.append(f"- **`{f.get('file')}:{f.get('line')}` -- {f.get('title')}** _({', '.join(f['dimensions'])}{ver}{cont})_")
            out.append(f"  - Why: {f.get('why')}")
            out.append(f"  - Failure: {f.get('failure_scenario')}")
            out.append(f"  - Fix: {f.get('suggestion')}")
        out.append("")
    if result.get("resolved"):
        out += [f"### Resolved since the previous review ({len(result['resolved'])})", ""]
        for p in result["resolved"]:
            out.append(f"- `{p['file']}:{p['line']}` {p['title']}")
        out.append("")
    if result["killed"]:
        out += ["<details><summary>Killed by verification (" + str(len(result["killed"])) + ")</summary>", ""]
        for f in result["killed"]:
            out.append(f"- `{f.get('file')}:{f.get('line')}` {f.get('title')} -- {f['refutations'][0][:200]}")
        out += ["", "</details>", ""]
    if advisory:
        out += [advisory, ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Graph assembly + CLI
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(State)
    g.add_node("fetch", node_fetch)
    g.add_node("brief", node_brief)
    g.add_node("review_dim", node_review_dim)
    g.add_node("advisory", node_advisory)
    g.add_node("collect", node_collect)
    g.add_node("verify_one", node_verify_one)
    g.add_node("report", node_report)
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "brief")
    g.add_conditional_edges("brief", fan_out_reviews, ["review_dim", "advisory"])
    g.add_edge("review_dim", "collect")
    g.add_edge("advisory", "collect")
    g.add_conditional_edges("collect", fan_out_verify, ["verify_one", "report"])
    g.add_edge("verify_one", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


def open_saver():
    from langgraph.checkpoint.sqlite import SqliteSaver

    STATE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(STATE_DIR / "checkpoints.sqlite", check_same_thread=False)
    return SqliteSaver(conn)


def log_usage(entry: dict) -> None:
    try:
        STATE_DIR.mkdir(exist_ok=True)
        with open(STATE_DIR / "usage.jsonl", "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def emit(final: State, thread_id: str, as_json: bool) -> None:
    pr, result, advisory = final["pr"], final["result"], final.get("advisory", "")
    md = render_markdown(pr, result, advisory)
    md_path = STATE_DIR / f"review-{thread_id}.md"
    md_path.write_text(md)
    gh_cmd = f"gh pr review {pr['number']} --comment --body-file {md_path}"
    if as_json:
        out = {"pr": pr["number"], "title": pr["title"], "headSha": pr["headSha"],
               "thread_id": thread_id, "provider": final["provider"], "model": final["model"],
               "mode": final.get("mode", "discovery"),
               "stats": result["stats"],
               "findings": result["findings"], "killed": result["killed"],
               "markdown_file": str(md_path), "gh_command": gh_cmd}
        print(json.dumps(out, indent=1))
    else:
        print(md)
    log(f"\n-- thread: {thread_id}")
    log(f"-- saved: {md_path}")
    log(f"-- optional comment command: {gh_cmd}")
    log(f"-- to iterate: edit prompts in {__file__}, then:")
    log(f"     {sys.argv[0]} --history --thread {thread_id}")
    log(f"     {sys.argv[0]} --replay-from <checkpoint_id> --thread {thread_id}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Local multi-agent PR review with LangGraph. Read-only vs GitHub; never posts.",
        epilog="examples: review.py --pr <number> | review.py --history --thread <thread> | "
               "review.py --replay-from <checkpoint_id> --thread <thread>")
    ap.add_argument("--pr", help="PR number or head branch name")
    ap.add_argument("--provider", choices=list(PROVIDERS), default=DEFAULT_PROVIDER,
                    help=f"LLM provider (default {DEFAULT_PROVIDER})")
    ap.add_argument("--mode", choices=REVIEW_MODES,
                    help="discovery or focused blocker verification (default: REVIEW_MODE or discovery)")
    ap.add_argument("--model", help="model id or deployment (openai: required here or in OPENAI_MODEL)")
    ap.add_argument("--light-model", dest="light_model",
                    help="cheaper model for the non-gating aux nodes (brief, advisory) so they draw "
                         "from a separate quota pool; same --provider (default: use --model everywhere)")
    ap.add_argument("--timeout", type=int, default=300, help="per-LLM-call timeout seconds (default 300)")
    ap.add_argument("--concurrency", type=int,
                    help="parallel graph tasks (default: provider's; z.ai FUP-429s higher values)")
    ap.add_argument("--json", action="store_true", help="machine output (bare object) on stdout")
    ap.add_argument("--history", action="store_true", help="list checkpoints for --thread")
    ap.add_argument("--replay-from", metavar="CHECKPOINT_ID", help="re-run from this checkpoint (earlier nodes replay from cache)")
    ap.add_argument("--thread", help="thread id from a previous run")
    args = ap.parse_args()

    if not (args.pr or args.thread):
        ap.print_usage(sys.stderr)
        log("error: need --pr <number|branch> (fresh run) or --thread (history/replay)")
        return 2
    if (args.history or args.replay_from) and not args.thread:
        log("error: --history/--replay-from need --thread <id> (printed at the end of the original run)")
        return 2
    if not args.history:  # history reads local checkpoints; it needs no model or key
        try:
            model, light_model, concurrency = resolve_tuning(
                args.provider, args.model, args.light_model, args.concurrency, os.environ)
            mode = resolve_review_mode(args.mode, os.environ)
        except ValueError as exc:
            log(f"error: {exc}")
            return 2
        provider_key(args.provider)  # exits 3 with guidance if required env is absent
    try:
        os.chdir(target_repo_root())
    except Exception:
        log("error: run review.py from inside a checkout of the target Git repository.")
        return 3
    try:
        run_cmd(["gh", "auth", "status"])
    except Exception:
        log("error: gh is not authenticated. Fix: gh auth login")
        return 3

    saver = open_saver()
    graph = build_graph(saver)
    t0 = time.time()

    if args.history:
        cfg = {"configurable": {"thread_id": args.thread}}
        rows = [{"checkpoint_id": s.config["configurable"]["checkpoint_id"],
                 "next": list(s.next), "ts": s.created_at}
                for s in graph.get_state_history(cfg)]
        if not rows:
            log(f"no checkpoints for thread {args.thread!r}. List past runs: tail {STATE_DIR / 'usage.jsonl'}")
            return 1
        print(json.dumps(rows, indent=1) if args.json else
              "\n".join(f"{r['checkpoint_id']}  next={','.join(r['next']) or '(end)'}" for r in rows))
        return 0

    thread_id = (args.thread or f"pr{args.pr}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}").replace('/', '-')
    cfg = {"configurable": {"thread_id": thread_id}, "max_concurrency": concurrency, "recursion_limit": 100}
    callbacks = make_trace_callbacks(thread_id)
    if callbacks:
        cfg["callbacks"] = callbacks
        cfg["run_name"] = "pr-review:run"
        cfg["metadata"] = {"langfuse_session_id": thread_id,
                           "langfuse_tags": ["pr-review", f"pr-{args.pr or thread_id}"]}
        log("[trace] langfuse tracing on")

    def run_graph():
        if args.replay_from:
            cfg["configurable"]["checkpoint_id"] = args.replay_from
            log(f"[replay] thread {thread_id} from checkpoint {args.replay_from} -- earlier nodes replay from cache")
            return graph.invoke(None, cfg)
        light = f" light={light_model}" if light_model else ""
        log(f"[run] thread {thread_id} {args.provider}/{model}{light} mode={mode} concurrency={concurrency}")
        return graph.invoke(
            {"pr_ref": str(args.pr), "provider": args.provider, "model": model,
             "light_model": light_model, "timeout": args.timeout, "mode": mode}, cfg)

    ok, err = True, None
    try:
        if callbacks:
            # langfuse v4: an enclosing observation IS the trace root -- it
            # names the trace and makes score_current_trace deterministic (the
            # CallbackHandler has no public trace-id accessor). Session/tags
            # ride the handler metadata above.
            final = run_traced(run_graph)
        else:
            final = run_graph()
        emit(final, thread_id, args.json)
    except Exception as e:
        ok, err = False, str(e)[:300]
        log(f"error: {err}")
        return 1
    finally:
        flush_traces()
        log_usage({"ts": datetime.now(timezone.utc).isoformat(), "ok": ok, "ms": int((time.time() - t0) * 1000),
                   "pr": args.pr, "thread": thread_id, "provider": args.provider, "model": model,
                   "mode": mode,
                   "replay_from": args.replay_from, "error": err})
    return 0


if __name__ == "__main__":
    sys.exit(main())
