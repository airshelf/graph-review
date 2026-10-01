# Benchmarks

These are the model and reviewer comparisons behind graph-review's defaults. Every run was
made on real pull requests in a private production codebase. So we publish the numbers and
not the code or the PRs.

Read each table with its `n`. Most of these are small, paired runs made to answer one
question on one day. They are directional. Run-to-run variance on the same PR is often
larger than the gap between two models.

Terms used below:

- **Ground-truth hit.** A finding in a file where a real defect was later confirmed and fixed.
- **One-off finding.** A finding that appears in only one of several repeat runs of the same
  arm on the same PR. We use it as a proxy for noise.
- **Dead dimension.** A reviewer dimension that crashed or hit its recursion limit. It shows
  up in the report as a coverage gap. It is never counted as a clean result.
- **Wall time.** The whole review, from fetch to the rendered report.

## Which model should gate merges? (2026-10-01)

**Answer: `gpt-6-sol`.** It found 7 of 8 known defects. No other model found more than 4.
The cost is more majors per PR, so more PRs wait for a human approval.

Five arms on Azure OpenAI, 87 runs, all on one day. `gpt-6.1-sol high` is the same model as
`gpt-6.1-sol` at reasoning effort high. Every other arm ran at effort medium.

The method was blind and paired. Prior review threads were hidden from every arm. All arms
reviewed the same pinned PR head at the same time, so they met the same API load. Each arm
ran 3 reps per case.

| Case | Size | Role |
| --- | --- | --- |
| A | 16 files, +222/-37 | 5 known defects, a data-editing feature |
| B | 12 files, +700/-86 | 3 known defects, a dashboard metric change |
| C | 20 files, +1944/-557 | Control. Its original review found nothing. |
| D | 4 files, +33/-1 | A small fix that shipped a production blocker |

Cases A and B ran at the commit where the known defects still existed. The answer key is the
8 verified findings of the original CI review of those PRs. That review was made by
`gpt-5.6-terra`, so the key favours terra.

### Known-defect recall (3 reps per arm)

| Known defect | terra 5.6 | sol 6 | sol 6.1 | sol 6.1 high | astra 6 |
| --- | --- | --- | --- | --- | --- |
| Edit history shows the wrong value | 0/3 | 3/3 | 1/3 | 0/3 | 0/3 |
| A list of objects is saved as strings | 3/3 | 1/3 | 0/3 | 0/3 | 0/3 |
| A list of numbers or booleans is saved as strings | 0/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| An edited value keeps its old verification verdict | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Regeneration restores the unedited value | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| A card shows an old report as this week's | 0/3 | 3/3 | 1/3 * | 0/3 | 3/3 * |
| A second reader counts excluded sources | 3/3 | 1/3 | 0/3 | 3/3 | 0/3 |
| A stale line in an operations doc | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| **Distinct defects found, of 8** | 4 | **7** | 4 | 3 | 3 |
| **Run-level hits, of 24** | 11 | **16** | 8 | 9 | 9 |

\* These runs flagged only the tooltip label, not the stale report choice. They are counted
anyway.

### Consensus recall

This key does not depend on any one model. A cluster is a set of findings in one file within
25 lines of each other. It counts as a consensus issue when two different models raise it.
The two `gpt-6.1-sol` efforts count as one model.

| Slice | terra 5.6 | sol 6 | sol 6.1 | sol 6.1 high | astra 6 |
| --- | --- | --- | --- | --- | --- |
| Defect commits, 11 consensus issues | 7 | **10** | 5 | 6 | 4 |
| Final commits, 14 consensus issues | 8 | **12** | 6 | 7 | 5 |

### Speed, noise and load (all cases)

| | terra 5.6 | sol 6 | sol 6.1 | sol 6.1 high | astra 6 |
| --- | --- | --- | --- | --- | --- |
| Runs | 18 | 18 | 18 | 15 | 18 |
| Failed runs / dead dimensions | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Median wall time | 163 s | 174 s | 144 s | 274 s | **122 s** |
| Max wall time | 404 s | 472 s | **173 s** | 450 s | 206 s |
| One-off findings (noise proxy) | 4 | 5 | **0** | 3 | 1 |
| Majors per run, cases A to C | 2.4 | 2.9 | 1.2 | 2.5 | **0.8** |
| Honours the priority service tier | yes | yes | no | no | no |

What we read from it:

- `gpt-6-sol` has the best recall on every measure. Its cost is the most majors and the
  longest tail. A verified major withholds auto-approval, so that cost is human time.
- `gpt-6.1-sol` is faster and quieter, with fewer than half the majors. It found 4 of 8
  known defects, and it missed both data-corruption defects in every run. On 2026-10-01 Azure
  did not honour the priority tier for it: a request for `priority` came back as `default`.
- `gpt-6-astra` is the fastest and the quietest. It found 3 of 8.
- `gpt-6.1-sol` at effort high was slower than `gpt-6-sol` and found less.
- The 2026-09-23 speed gap below has closed. Today `gpt-6-sol` and terra ran at 174 s
  and 163 s.

### What no model caught

Case D shipped a production blocker. A URL getter threw on the framework's request proxy,
so every request answered HTTP 503. All five arms returned zero findings, in every rep. A
live probe after deploy caught it. A reviewer that reads code will not find every bug of this
class. Pair it with a deploy-time probe.

Caveats: 3 reps per case. Only 2 cases have known defects, 8 defects in total. Treat a gap of
one or two defects as noise. The 3-of-8 to 7-of-8 gap is the one we act on.


## GPT-6 Sol vs GPT-5.6 Terra (2026-09-23)

The question was whether to move the gate from `gpt-5.6-terra` to `gpt-6-sol`, both on Azure
OpenAI. The two arms ran concurrently on the same PRs. Earlier review comments were hidden from
both arms, so neither could anchor on a prior review.

| | gpt-5.6-terra | gpt-6-sol |
| --- | --- | --- |
| Ground-truth file hits, 2 known-defect PRs, 10 runs per arm pooled | 5 | 14 |
| One-off findings per PR, 5 reps (two PRs measured) | 2 and 2 | 2 and 2 |
| Recurring findings (3 or more of 5 reps) on one known-defect PR | 1 | 4 |
| Median wall time | 191 s | 341 s |
| Pairs where it was the slower arm | 4 of 14 | 10 of 14 |

29 reviews in total across 3 PRs: 2 with known defects and 1 control.

What we read from it:

- Sol found about three times as many ground-truth hits.
- Sol was not noisier in the sense that matters. One-off findings were equal. Its extra findings
  were ones that recurred across reps, and its most repeated ones sat in the ground-truth files.
- The slowdown comes from finding more. Each blocker or major gets three adversarial
  verifiers. Findings and seconds correlate at +0.73. At 3 findings each, both arms took about
  the same time (about 182 s against 191 s).
- The cost of more findings is more human approvals. A verified major withholds auto-approval.
- The speed gap did not hold. By 2026-10-01 the two medians were 174 s and 163 s.

The trap we hit first: GPT-6 deployments carry a default reasoning effort. Azure's
`chat/completions` endpoint rejects function tools together with reasoning. Every reviewer
binds tools, so every reviewer failed with HTTP 400 in about 2 seconds. Because the CI lane
fails open, the run reported a clean, blocker-free review of nothing. The fix is the Responses
API (`api-version` 2025-03-01-preview or later). `review.py` selects it for GPT-6-class models.

## Can a cheap judge predict which findings the verifier kills? (2026-09-28)

Three adversarial lenses check every blocker and major before it reaches the PR. We asked
whether a single cheap model call could predict their verdict and skip the expensive step.

Over 500 consecutive PRs the reviewers raised 1,978 blocker or major findings. The verifiers
kept 1,890 and killed 88 (4.5%). We scored all 88 killed findings and 230 random kept ones.
Each judge saw the finding's title, its `file:line` and the code at the reviewed commit.

| Judge | Context | AUC, kept vs killed |
| --- | --- | --- |
| TypeSafe Jev (one yes/no question) | 90-line window | 0.53 (95% CI 0.46 to 0.60) |
| gpt-4.1 (yes/no token odds) | 90-line window | 0.55 |
| TypeSafe Jev | whole file | 0.50 |
| gpt-4.1 | whole file | 0.52 |

Neither judge beats a coin. gpt-4.1 said yes to nearly everything (median probability 1.00).
42 of the 88 kill reasons cite a different file from the one the finding names. A judge with no
repository tools cannot see why those findings die. That is why the verifiers keep their tools.

## Provider and model history (July to September 2026)

These are single runs on one PR each (`n = 1`), unless the row says otherwise. They explain
why some providers are no longer the default. They are not durable verdicts.

| Setup | Result |
| --- | --- |
| `gemini-3.1-pro-preview` on Vertex | One reviewer took 271 s, plus Vertex 504 errors. Rejected on latency. |
| `gemini-3.5-flash` on Vertex | Reviewers took 262 to 345 s. One reviewer failed with "Model input cannot be empty". One verify lens hit the recursion limit. Rejected. |
| `gemini-3-flash-preview` on Vertex | Stable, but about 7 to 10 minutes per full review. It was the default until an Azure lane existed. |
| `gpt-5.4-mini` on Azure vs `gemini-3-flash-preview` on Vertex, same PR | 68 s with 0 dead dimensions, against 593 s with 3 dead dimensions. The Gemini run predates a fix to context truncation that targeted its failure mode, so this pair is not a clean verdict. |
| Ensemble: 3 dimensions on Azure Foundry, 2 on Gemini, 1 on Azure OpenAI, cross-family verify | Killed at 18.5 minutes with 2 dead dimensions. On the same PR, Azure alone took 218 s and Gemini alone 627 s. |
| Azure Foundry `grok-4.3` vs `deepseek-v4-pro`, same PR | grok-4.3: 224 s, 0 dead dimensions. deepseek-v4-pro: 1,365 s, 4 dead dimensions, 1 finding kept. |
| `glm-5.2` on the z.ai coding plan | The plan rate-limits 6 or more parallel streams. At concurrency 2 a full pass takes 30 to 60 minutes. |
| `gpt-6-astra` (reasoning off) vs `gpt-5.6-terra`, same PR (2026-09-07) | Both completed 6 of 6 reviewers. Both returned 3 raw findings and 0 blockers or majors after verification. Slowest node 42 s against 36 s. Each caught a different real issue that the other missed. |

The lesson from the ensemble row: an ensemble inherits the worst latency of its members and
every one of their failure modes. It combines findings well. It combines reliability badly.

## Other reviewers we compared

### open-code-review with gpt-5.6-luna vs graph-review with gpt-5.6-terra (2026-07-24)

5 PRs, one run each, findings judged by hand.

| PR | graph-review | open-code-review + luna |
| --- | --- | --- |
| 1 | Clean | One high-severity false positive. Its model did not know a recent GitHub Actions feature. graph-review's web brief did. |
| 2 | 2 majors | Found the same key major, plus 2 more. |
| 3 | Clean | One real bug that graph-review missed. |
| 4 | 3 cross-file wiring majors | Missed all three. Reported two robustness points instead. |
| 5 | 1 security major | Different, code-level findings. The two were complementary. |

open-code-review took 23 to 45 s per PR and about 786k tokens over the five. It is good at
per-file robustness and weak tests. It cannot gate. It produced a stale-knowledge false
positive at high severity, and it misses cross-file and approval-surface reasoning. We ran it
as a non-gating second lane for one day and stopped. Authors treated its findings as a work
queue. One PR took ten review rounds and nine fix commits in about 95 minutes.

### GLM harnesses for a second-opinion review (2026-07-15)

3 harnesses (claude-code-action, pi, opencode) by 2 models (glm-5.1, glm-5.2), over 5 PRs:
3 with confirmed bugs and 2 clean controls. One run per cell.

- Run-to-run variance was the biggest effect. The same harness, model, prompt and diff caught
  a bug in one run and buried it in the next.
- No harness and model pair won everywhere. Hits and misses flipped between PRs.
- glm-5.2 beat glm-5.1. glm-5.1 caught nothing that glm-5.2 missed.
- One schema bug was caught by no GLM cell at all.
- Wall time ranged from 59 s to 831 s per review.
