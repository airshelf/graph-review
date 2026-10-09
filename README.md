<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo-light.svg" alt="graph-review logo: a spider web with one finding caught in it" width="128">
  </picture>
</p>

<h1 align="center">graph-review</h1>

<p align="center">
  Six AI reviewers read your pull request. Three more try to prove each finding wrong.
</p>

<p align="center">
  <a href="https://github.com/airshelf/graph-review/actions/workflows/test.yml"><img src="https://github.com/airshelf/graph-review/actions/workflows/test.yml/badge.svg?branch=main" alt="tests"></a>
  <a href="https://github.com/airshelf/graph-review/actions/workflows/test.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/airshelf/graph-review/badges/coverage.json" alt="coverage"></a>
  <a href="https://github.com/airshelf/graph-review/tags"><img src="https://img.shields.io/github/v/tag/airshelf/graph-review?label=version" alt="version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/airshelf/graph-review" alt="license"></a>
</p>

graph-review is a LangGraph multi-agent pull-request reviewer. It is the reviewer
that gates every PR at AirShelf: blockers make the check red, majors withhold
approval, and minors are advisory.

AirShelf runs it with `--provider azure --model gpt-6-sol` at reasoning effort medium.
In a blind, paired test on 2026-10-01, that model found 7 of 8 known defects. No other
arm of the test found more than 4. See [BENCHMARKS.md](BENCHMARKS.md).

The script reads GitHub and prints a review. It does not edit the branch, post
comments, or approve a PR. The optional CI workflows handle comments and approval.

## Why a graph

```text
fetch -> brief -> review -> verify -> report
```

Fetch builds shared context. The brief checks relevant external API or dependency
facts when web tools are available. Six reviewer dimensions and a product advisory
run in parallel. Three adversarial lenses check each blocker or major independently;
a majority refutation removes the claim. The report deduplicates, ranks, and renders
findings in Python. The advisory never gates a merge.

Reviewers can read files and search at the PR head. Verifiers get the claim, not the
shared context pack. They must check the evidence again. Named invariants come from
the target repo's instructions, not this repo's prompts.

## Install and quick start

Install Git, [uv](https://docs.astral.sh/uv/), and the
[GitHub CLI](https://cli.github.com/). Authenticate `gh` for the target repository.
The script declares its Python dependencies inline; uv prepares them on first run.
You also need access to a model that supports tool calls.

Clone the reviewer separately from the repository you want to review. Replace the
example paths, key, model, and PR reference below with your own values.

```sh
git clone https://github.com/airshelf/graph-review.git /path/to/graph-review
cd /path/to/target-checkout
gh auth login
export OPENAI_API_KEY='your-key'
export OPENAI_MODEL='your-model'
uv run /path/to/graph-review/review.py --pr '<number-or-head-branch>'
uv run /path/to/graph-review/review.py --pr '<number-or-head-branch>' --json
```

Run from inside the target Git checkout. Its root is resolved with
`git rev-parse --show-toplevel`, not from the reviewer's location. The checkout's
`origin` must point at the repository that owns the PR. Keep its history available.
`--model` overrides the selected model; `--provider` selects a
provider. Use `--help` for all options.

Reports, SQLite checkpoints, and `.state/usage.jsonl` live in `.state/` beside
`review.py`. These files can contain source and review text. Keep them private.

## Providers

`openai` is the default. Keys, endpoints, and projects come from environment
variables, with the local key-file exceptions listed below. Azure endpoints and
Vertex projects have no fallback value.

| Provider | Required configuration | Model and optional settings |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY`; `--model` or `OPENAI_MODEL` | `OPENAI_BASE_URL` for an OpenAI-compatible endpoint |
| `azure` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | Deployment via `--model` or `AZURE_OPENAI_DEPLOYMENT`; `AZURE_OPENAI_API_VERSION` |
| `foundry` | `AZURE_FOUNDRY_API_KEY`, `AZURE_FOUNDRY_ENDPOINT` | Deployment via `--model` or `AZURE_FOUNDRY_DEPLOYMENT`; `AZURE_FOUNDRY_API_VERSION` |
| `gemini` | `GOOGLE_GEMINI_API_KEY` or `GEMINI_API_KEY` | `--model`; uses the native Gemini client |
| `gemini-vertex` | `VERTEX_PROJECT` and Google Application Default Credentials | `--model`, `VERTEX_LOCATION` |
| `glm` | `GLM_API_KEY` or `~/.config/zai-key` | `--model`; uses the z.ai coding endpoint |
| `claude-vertex` | `VERTEX_PROJECT`, Google Application Default Credentials, and the model enabled in Vertex | `--model`, `VERTEX_LOCATION`; prompt caching stays enabled |

For Vertex, set up ADC locally with `gcloud auth application-default login` or
configure workload identity in CI. Choose a model or deployment available to your
account. Non-OpenAI defaults are listed in `PROVIDERS` in [review.py](review.py).

OpenAI and Azure share temperature and Responses API selection. GPT-6-class model
names select the Responses API. `OPENAI_RESPONSES_API` can override that choice;
`OPENAI_NO_TEMPERATURE=1` omits temperature for reasoning models. Use the matching
`AZURE_OPENAI_` prefix for Azure, including `AZURE_OPENAI_NO_TEMPERATURE=1`.
`OPENAI_REASONING_EFFORT` and `OPENAI_SERVICE_TIER` also have Azure equivalents.
An OpenAI-compatible service must support the API your model selects.

`--concurrency` or `REVIEW_CONCURRENCY` controls parallel work. `--light-model` or
`REVIEW_LIGHT_MODEL` selects a model on the same provider for the brief and advisory.
Without it, all agents use the main model. Lower concurrency when provider quotas
cause rate limits.

## Context and optional tools

The context pack contains changed-file bodies from the PR head, matched learnings,
and live instructions from the target checkout. It reads the whole `CLAUDE.md`, or
`AGENTS.md` if `CLAUDE.md` is absent. The section is labeled
`repo agent instructions (CLAUDE.md / AGENTS.md, live)`. Oversized instructions are
capped with a visible truncation marker. Other context has provider-specific caps.

Keep reusable lessons in `<target root>/learnings/*.md`, split into Markdown
sections with `##` headings. Mention the relevant filenames or path terms so the
matcher can find them. It selects sections by repeated path matches and skips the
index `README.md`. A missing learnings directory is silently ignored.

The pack also includes best-effort import reachability and stale-guard checks.
`reach.py` recognizes Next.js app routes and pages by default; override the entry
point pattern with `GRAPH_REVIEW_ENTRY_RE`. `stale_guards.py --judge` can use the
TypeSafe Jev API and requires `TYPESAFE_API_KEY`. The reviewer's checks stay
mechanical when that key is absent.

Optional web research expects `web_search.sh` and `scrape.sh` under
`PR_REVIEW_WEB_TOOLS`, or `~/.claude/tools` by default. The brief is skipped when
the search script is absent. Search uses `BRAVE_API_KEY` or
`~/.config/brave/api_key` for snippets, then falls back to the search script.
External facts remain unverified until a reviewer checks them against the pinned
dependency and code. Reviews still work without web tools.

`REVIEW_DATABASE_URL` enables the advisory's guarded database query tool. This
requires `psql` and a read-only database role limited to data safe to send to the
model. The tool also requests read-only transactions. Leave it unset if you do
not want database data sent to the model.

Tracing is optional. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and the
appropriate `LANGFUSE_HOST` to enable it. `LANGFUSE_TRACING_ENVIRONMENT` can label the
run; otherwise it is inferred from CI or local execution. Tracing failures do not
stop reviews. The trace mask reduces common personal-data exposure, but is not a
complete redaction system.

## GitHub Actions

Copy [graph-review.yml](examples/github/graph-review.yml) and
[graph-approve.yml](examples/github/graph-approve.yml) into `.github/workflows/` in
the target repository. The example pins the `v1.0.0` release tag. Pin a commit SHA
instead if you want a reference that can never move.

Set the provider's keys as repository secrets and its model, endpoint, and project
settings as repository variables used by the example. `GRAPH_REVIEW_PROVIDER` is
optional and defaults to `openai`. Vertex users must add ADC setup before the run.
For approval, install a GitHub App with pull-request write access on the target
repo. Create a `graph-approval` environment restricted to the default branch.
Store `GRAPH_APP_ID` and `GRAPH_APP_PRIVATE_KEY` only as that environment's secrets,
not repository-wide secrets. This keeps approval credentials out of PR workflows.
Set branch protection to dismiss stale approvals or require review of the latest
commit.

The review workflow skips drafts, checks out full history, and runs the reviewer
from the target checkout. It updates a single PR comment. Only blockers make the
review job fail. Provider failures warn and leave the check green.

The verdict artifact records the PR, reviewed head, and approval eligibility.
Majors and dead dimensions withhold approval, as do blockers. The separate
`workflow_run` approver runs from the default branch. It rejects forks and PRs
targeting other branches, checks the current head against the reviewed head, and
checks changed files through GitHub's API. Any change under `.github/workflows/` or
`.github/actions/` needs human approval, because a merged workflow could request the
approval environment. The approver does not run
PR code. A green check alone is not proof that a review completed.

These examples do not impose a per-PR review budget or head cap. A target repo can
add either to control cost and repeated review cycles. `--mode verification`
focuses a follow-up on the prior open blocker or major batch and its fix delta.
The workflow, not the model, must enforce any review budget.

## Replay

Each run prints its thread ID. Use the same target checkout and original provider
configuration when inspecting or replaying it. Pass `--provider` again if the
original run did not use `openai`.

```sh
uv run /path/to/graph-review/review.py --history --thread '<thread-id>'
uv run /path/to/graph-review/review.py --replay-from '<checkpoint-id>' --thread '<thread-id>'
```

Pick a checkpoint before the stage you changed. Earlier nodes reuse saved state;
later nodes run with the current code. A checkpoint before verification lets you
change verifier prompts without rerunning reviewers. To test reviewer prompts,
start before review. Replay can make new model calls and is not an offline mode.

Review discussion is fetched for continuity. The report marks findings as new,
still open, or resolved through deterministic comparison with the previous graph
comment. The model does not assign those labels.

## Loop-control walls

The tool wall refuses identical repeated calls. This limits repeated context, but
cannot force a model to stop. The convergence wall counts refused repeats and
tool rounds. At its limit it removes tools, asks for a final answer, and clears
any further tool calls from the response. Both removing and clearing are needed:
a model can emit a call even when tools are no longer offered.

Recursion limits remain a final backstop. A failed reviewer becomes a visible
coverage gap, not evidence of clean code. Budgets and prompts are constants in
[review.py](review.py); the convergence mechanism lives in
[convergence.py](convergence.py).

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Review completed; findings are data, not an exit status |
| `1` | Runtime failure |
| `2` | Invalid command usage |
| `3` | Missing prerequisite, with a fix printed |

Read the report's coverage notes even after a successful exit.

## Why not use it

Reviews are stochastic. Independent verification can miss a shared mistaken
assumption. Large diffs can exceed context caps, and a forced conclusion can be
shallow. The import and guard helpers are heuristics, not a language-wide analysis.

This needs GitHub access, model credentials, network access, and a budget. Source,
discussion, and any enabled database context are sent to the selected provider.
Check that this fits your data policy. Read-only tools and prompt rules are not a
complete defense against malicious content in a PR.

This tool cannot replace tests, static checks, or human review.
The fail-open CI policy is unsuitable if every provider outage must block merging.
See [BENCHMARKS.md](BENCHMARKS.md) for the model comparisons behind these defaults.

## License

[MIT](LICENSE).
