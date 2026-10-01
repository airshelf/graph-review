# Working on graph-review

Keep the flat layout. Run the reviewer from a target Git checkout, using an
absolute path to `review.py` when the reviewer lives elsewhere. Target context
comes from that checkout's Git root. State stays beside the script in `.state/`.

## File map

- `review.py`: CLI, provider configuration, prompts, context pack, graph, reports.
- `convergence.py`: agent-loop termination middleware and response helpers.
- `reach.py`: import reachability context.
- `stale_guards.py`: stale-guard checks and the optional external judge.
- `tests/`: behavior tests, including the real convergence-loop regression.
- `examples/github/`: review and default-branch approval workflows.

Prompts live as constants at the top of `review.py`. Edit `DIMENSIONS`,
`ADVISORY_PROMPT`, `LENSES`, `FINDINGS_FORMAT`, `BRIEF_PROMPT`, `REVIEWER_PROMPT`,
and `VERIFICATION_PROMPT` there. Keep generic
precision, severity, verification, and smallest-fix rules. Repo-specific rules
belong in the target repo's `CLAUDE.md` or `AGENTS.md`, not hardcoded prompts.

## Checks

From this repository, with dependencies available:

```sh
uv run --with pytest --with 'langchain>=1.3' --with 'langgraph>=0.6' python -m pytest tests
```

The standard-library syntax check needs no installation or network:

```sh
python3 -m py_compile review.py convergence.py reach.py stale_guards.py tests/*.py
```

Keep tests focused on behavior. Mock provider clients for offline unit tests.
Never put keys, local state, or private repository details in committed files.
