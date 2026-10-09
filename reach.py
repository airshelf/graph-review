#!/usr/bin/env python3
"""Which entry points does this diff reach, that the diff does not contain?

Reviewers read the PATCH. They cannot see what it REACHES, so a change to a shared
module can silently alter endpoints in files the diff never mentions. A page may
sit several imports away from a changed helper and still need a review.

No index, no tool, no committed artifact, nothing installed. It reads the checkout's
own JavaScript and TypeScript imports and walks them backwards. A committed lookup
table would go stale or need regeneration on every shared-module change, creating
avoidable merge conflicts.

Symbol-filtered on the first hop, FILE-LEVEL after that, deliberately. It reports
"these entry points import a changed module", not a proof that a changed function
executes there. It can OVER-report. For a reviewer hint that is the right error
direction: a name to check, never a finding.

Exit: 0 for a report, 2 for unresolvable refs. The reviewer and the human decide
whether a reachable entry point has a defect.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys

IMPORT = re.compile(r"""from\s+['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)""")
# Next.js app/pages entries at any depth, including a direct app/page entry.
# Projects with another router can provide their own pattern without editing code.
DEFAULT_ENTRY_RE = (r"(^|/)app/(?:.*/)?(route|page)\.(ts|tsx|js|jsx)$"
                    r"|(^|/)pages/.*\.(ts|tsx|js|jsx)$")
ENTRY = re.compile(os.environ.get("GRAPH_REVIEW_ENTRY_RE") or DEFAULT_ENTRY_RE)
SOURCE_EXTS = (".ts", ".tsx", ".js", ".jsx")
EXTS = ("", *SOURCE_EXTS, *(f"/index{ext}" for ext in SOURCE_EXTS))
PATHS_PAIR = re.compile(r'"([^"]+)"\s*:\s*\[\s*"([^"]+)"')
EXPORT_DEF = re.compile(r"^\s*export\s+(async\s+)?(?:function|const|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
IMPORT_CLAUSE = re.compile(r"import\s+([^;'\"]*?)\s*from\s+['\"]([^'\"]+)['\"]", re.S)


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout


def source_files() -> list[str]:
    return [f for f in sh("git", "ls-files", *(f"*{ext}" for ext in SOURCE_EXTS)).split()
            if "node_modules" not in f]


def resolve_spec(spec: str, src: str, aliases: dict[str, str], known: set[str]) -> str | None:
    if spec.startswith("."):
        base = os.path.normpath(os.path.join(os.path.dirname(src), spec))
    else:
        # Longest alias wins: `@/shared/` must beat `@/` when both are configured.
        hit = max((a for a in aliases if spec.startswith(a)), key=len, default=None)
        if hit is None:
            return None
        base = os.path.normpath(aliases[hit] + spec[len(hit):])
    for cand in (base + e for e in EXTS):
        if cand in known:
            return cand
    return None


def build_reverse(files: list[str], aliases: dict[str, str]) -> dict[str, set[str]]:
    """target file -> set of files importing it. One pass, no AST: an import specifier is
    unambiguous enough that a regex plus real path resolution gets the edges, and a parser
    would cost a dependency for the same result."""
    known = set(files)
    rev: dict[str, set[str]] = {}
    for f in files:
        try:
            src = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in IMPORT.finditer(src):
            tgt = resolve_spec(m.group(1) or m.group(2), f, aliases, known)
            if tgt and tgt != f:
                rev.setdefault(tgt, set()).add(f)
    return rev


def load_aliases() -> dict[str, str]:
    """Every tsconfig's `paths`, so a sub-project alias resolves. An alias may
    escape its sub-project root to import shared code; missing it silently drops
    all those importers out of the walk.

    Three traps, all of which failed SILENTLY and each of which cost a debugging round:

    1. Do NOT strip comments and `json.loads`. The alias PATTERNS contain `/*`
       (`"@/*": ["./*"]`), so a block-comment regex matches from the first pattern to the
       end of the file. Every alias can vanish while the graph still looks healthy.
    2. Scope the scan to the paths OBJECT. Splitting on `"paths"` and reading to EOF
       swallowed the sibling `exclude` and `include` keys as if they were aliases.
    3. The ROOT tsconfig must win. `git ls-files` returns paths alphabetically, so
       a nested config can come first and claim a repo-wide alias. Root imports
       would then resolve against the wrong directory and quietly disappear.
    """
    aliases: dict[str, str] = {}
    # Nested configs first, root last, so the root's own mapping overwrites a sub-project's
    # claim on the same prefix rather than losing the race to it.
    for cfg in sorted(sh("git", "ls-files", "*tsconfig.json").split(),
                      key=lambda c: (c.count("/") == 0, c)):
        try:
            raw = open(cfg, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        head = raw.split('"paths"', 1)
        if len(head) < 2:
            continue
        body, depth, end = head[1], 0, None
        for i, ch in enumerate(body):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            continue
        root = os.path.dirname(cfg)
        for pattern, target in PATHS_PAIR.findall(body[:end]):
            aliases[pattern.rstrip("*")] = (
                os.path.normpath(os.path.join(root, target.rstrip("*"))) + "/")
    return aliases


def changed_symbols(base: str, head: str, files: list[str]) -> dict[str, set[str]]:
    """file -> exported symbol names whose DEFINITION line the diff touches.

    Seeding with whole FILES can greatly over-report because many entry points
    transitively import shared modules. A long, mostly irrelevant list hides the
    few callers worth checking, so narrow the first hop to changed exports."""
    out: dict[str, set[str]] = {}
    for f in files:
        for line in sh("git", "diff", f"{base}...{head}", "--", f).splitlines():
            if line[:1] in "+-" and not line.startswith(("+++", "---")):
                m = EXPORT_DEF.match(line[1:])
                if m:
                    out.setdefault(f, set()).add(m.group(2))
    return out


def imports_symbol(importer: str, target: str, symbols: set[str], aliases: dict[str, str],
                   known: set[str]) -> bool:
    """Does `importer` actually NAME one of `symbols` in its import from `target`?

    This is the narrowing. A file importing `formatDate` from a module is untouched by a
    change to `buildSummary` in that same module, and file-level reachability cannot
    tell those apart."""
    try:
        src = open(importer, encoding="utf-8", errors="replace").read()
    except OSError:
        return False
    for m in IMPORT_CLAUSE.finditer(src):
        clause, spec = m.group(1), m.group(2)
        if resolve_spec(spec, importer, aliases, known) != target:
            continue
        if "*" in clause:            # namespace import: every symbol is in scope
            return True
        named = {n.split(" as ")[0].strip()
                 for n in clause.replace("{", " ").replace("}", " ").split(",")}
        if named & symbols:
            return True
    return False


def reach(seeds: dict[str, set[str]], rev: dict[str, set[str]], aliases: dict[str, str],
          known: set[str], depth: int = 8) -> set[str]:
    """Walk importers, but only those that actually name a changed symbol at the FIRST
    hop. Past hop one the module itself is implicated, so plain file reachability is
    correct from there on."""
    first: set[str] = set()
    for tgt, syms in seeds.items():
        for importer in rev.get(tgt, ()):
            if imports_symbol(importer, tgt, syms, aliases, known):
                first.add(importer)
    seen, frontier = set(first), set(first)
    for _ in range(depth):
        nxt = {f for cur in frontier for f in rev.get(cur, ())} - seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    # The base ref MUST resolve, and a failure here must EXIT NON-ZERO. Since this script
    # now always prints the zero summary, an unresolvable ref would otherwise print
    # "no JavaScript or TypeScript files changed ... 0" and look like a legitimate zero -- which both
    # poisons the denominator and breaks the caller's origin/<base> then <base> fallback.
    for ref in (a.base, a.head):
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                          capture_output=True, text=True).returncode != 0:
            print(f"reach: cannot resolve ref {ref!r}", file=sys.stderr)
            return 2

    changed = [f for f in sh("git", "diff", "--name-only", f"{a.base}...{a.head}").split()
               if f.endswith(SOURCE_EXTS)]
    if not changed:
        # Print the summary even here. A caller counting "how often was this non-empty"
        # needs the DENOMINATOR, and an early return with no summary makes a run that
        # genuinely reached nothing indistinguishable from one that never ran.
        print("reach: no JavaScript or TypeScript files changed")
        print("\n  *** ENTRY POINTS AFFECTED BUT NOT IN THE DIFF: 0 ***")
        return 0

    files = source_files()
    known = set(files)
    aliases = load_aliases()
    rev = build_reverse(files, aliases)
    present = [f for f in changed if f in known]
    seeds = changed_symbols(a.base, a.head, present)
    if not seeds:
        # Same reason as above: a real zero, not an absence. Only an execution FAILURE
        # should produce no summary at all.
        print("reach: no exported symbol definitions changed — nothing to trace")
        print("\n  *** ENTRY POINTS AFFECTED BUT NOT IN THE DIFF: 0 ***")
        return 0
    in_diff = set(changed)
    reached_all = reach(seeds, rev, aliases, known)
    reached = reached_all - in_diff
    unseen = sorted(f for f in reached if ENTRY.search(f))
    seen = sorted(f for f in reached_all & in_diff if ENTRY.search(f))

    if a.json:
        print(json.dumps({"changed": changed, "unseen_entry_points": unseen,
                          "entry_points_in_diff": seen}, indent=2))
        return 0

    print(f"reach: {len(changed)} JS/TS files changed, {len(reached)} files import them transitively")
    print(f"\n  *** ENTRY POINTS AFFECTED BUT NOT IN THE DIFF: {len(unseen)} ***")
    for f in unseen:
        print(f"      {f}")
    if not unseen:
        print("      (none)")
    if seen:
        print(f"\n  entry points reached AND already in the diff: {len(seen)}")
    print("\n  File-level reachability: these files import the changed ones, directly or")
    print("  transitively. Facts to CHECK, not findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
