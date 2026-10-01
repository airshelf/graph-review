"""Red-path proof for reach.py.

Two failure shapes, and both return ZERO with no error, which is the dangerous answer:

  A. The walker finds nothing because the diff genuinely reaches nothing. Correct.
  B. The walker finds nothing because module resolution broke. A silent, total lie.

Six resolver bugs during development all produced B. Each one made the tool report
"0 entry points affected" while running, exiting 0, and looking healthy. So the tests
below assert the RESOLVER's health directly, not only the walk's output -- a test that
only checked "the walk returns something" would have passed through most of them.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "reach.py"


def _load_reach():
    spec = importlib.util.spec_from_file_location("reach", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r = _load_reach()


@pytest.fixture
def aliases(tmp_path, monkeypatch):
    """A target checkout owns its aliases; the tool's own repo need not have any."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "service").mkdir()
    (tmp_path / "service" / "tsconfig.json").write_text('''{
      "compilerOptions": {"paths": {
        "@/*": ["./*"],
        "@/shared/*": ["../shared/*"]
      }},
      "include": ["**/*.ts"]
    }''')
    (tmp_path / "tsconfig.json").write_text('''{
      // Alias wildcards must not be mistaken for block comments.
      "compilerOptions": {"paths": {
        "@/*": ["./*"],
        "@components/*": ["./components/*"],
        "@utils/*": ["./utils/*"]
      }},
      "exclude": ["node_modules"],
      "include": ["**/*.ts", "**/*.tsx"]
    }''')
    monkeypatch.setattr(r, "sh", lambda *args: "service/tsconfig.json\ntsconfig.json\n")
    return r.load_aliases()


# ------------------------------------------------------------------ resolver health
def test_the_repo_wide_alias_survives_subproject_configs(aliases):
    """Alphabetical config order must not let a subproject claim the root alias.
    Otherwise imports silently resolve against the wrong directory."""
    assert "@/" in aliases, "the repo-wide @/ alias vanished — the walk will return 0 for everything"
    assert aliases["@/"] in ("./", "."), f"@/ should map to the repo root, got {aliases['@/']!r}"


def test_subproject_alias_is_kept_alongside_the_root_one(aliases):
    """A subproject alias can escape its own root to reach shared code. Losing it
    drops those importers out of the walk."""
    assert aliases.get("@/shared/") == "shared/", (
        f"subproject's @/shared/ alias lost, got {aliases.get('@/shared/')!r}")


def test_alias_table_holds_no_non_alias_keys(aliases):
    """Bug 4: splitting on `"paths"` and reading to EOF swallowed the sibling `exclude`
    and `include` keys as if they were aliases."""
    for junk in ("exclude", "include", "compilerOptions"):
        assert junk not in aliases, f"{junk!r} parsed as an alias — the paths scan overran its object"


def test_alias_patterns_containing_a_star_survive(aliases):
    """Bugs 1-3: the alias PATTERNS contain `/*`, so a block-comment regex ate from the
    first pattern to EOF, then a swallowed parse error hid the loss."""
    assert len(aliases) >= 4, (
        f"only {len(aliases)} aliases parsed — comment stripping likely ate the file")
    assert set(aliases) == {"@/", "@/shared/", "@components/", "@utils/"}


# ------------------------------------------------------------------ the walk itself
def test_only_importers_naming_a_changed_symbol_are_followed(tmp_path, monkeypatch):
    """The narrowing. A file importing `formatDate` is untouched by a change to
    `buildSummary` in the same module. File-level reachability cannot tell those apart
    and floods the reviewer with unrelated entry points."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "m.ts").write_text("export function buildSummary(){}\nexport function formatDate(){}\n")
    (tmp_path / "wants.ts").write_text("import { buildSummary } from './shared/m';\n")
    (tmp_path / "other.ts").write_text("import { formatDate } from './shared/m';\n")
    known = {"shared/m.ts", "wants.ts", "other.ts"}
    assert r.imports_symbol("wants.ts", "shared/m.ts", {"buildSummary"}, {}, known) is True
    assert r.imports_symbol("other.ts", "shared/m.ts", {"buildSummary"}, {}, known) is False


def test_namespace_import_counts_as_naming_every_symbol(tmp_path, monkeypatch):
    """`import * as m` puts every export in scope, so it must not be filtered out."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "m.ts").write_text("export function x(){}\n")
    (tmp_path / "ns.ts").write_text("import * as m from './shared/m';\n")
    assert r.imports_symbol("ns.ts", "shared/m.ts", {"x"}, {}, {"shared/m.ts", "ns.ts"}) is True


def test_reach_returns_nothing_when_no_importer_names_the_symbol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m.ts").write_text("export function changed(){}\n")
    (tmp_path / "u.ts").write_text("import { other } from './m';\n")
    assert r.reach({"m.ts": {"changed"}}, {"m.ts": {"u.ts"}}, {}, {"m.ts", "u.ts"}) == set()


def test_reach_walks_past_the_first_hop(tmp_path, monkeypatch):
    """Hop one is symbol-filtered; after that the module itself is implicated, so plain
    file reachability is correct. A walk that stopped at hop one would have missed
    a page several imports away from the changed helper."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m.ts").write_text("export function changed(){}\n")
    (tmp_path / "mid.ts").write_text("import { changed } from './m';\n")
    (tmp_path / "page.tsx").write_text("import { mid } from './mid';\n")
    rev = {"m.ts": {"mid.ts"}, "mid.ts": {"page.tsx"}}
    known = {"m.ts", "mid.ts", "page.tsx"}
    out = r.reach({"m.ts": {"changed"}}, rev, {}, known)
    assert out == {"mid.ts", "page.tsx"}, f"multi-hop expansion broken: {out}"


def test_entry_point_pattern_matches_routes_and_pages_only(monkeypatch):
    monkeypatch.delenv("GRAPH_REVIEW_ENTRY_RE", raising=False)
    entry = _load_reach().ENTRY
    for path in ("app/api/search/route.ts", "packages/service/app/v1/items/route.ts",
                 "src/app/[itemId]/details/page.tsx", "app/page.tsx", "app/route.js",
                 "src/app/account/page.jsx", "pages/index.js", "src/pages/items/[id].tsx",
                 "packages/site/pages/api/items.ts", "pages/help.jsx"):
        assert entry.search(path), f"Next.js entry point not recognized: {path}"
    for path in ("shared/summary.ts", "components/dashboard/Summary.tsx",
                 "src/app/account/layout.tsx", "src/notpages/index.ts", "page.tsx"):
        assert not entry.search(path), f"non-entry matched: {path}"


def test_entry_point_pattern_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GRAPH_REVIEW_ENTRY_RE", r"(^|/)handlers/.*\.js$")
    entry = _load_reach().ENTRY
    assert entry.search("src/handlers/items.js")
    assert not entry.search("app/api/items/route.ts"), "override must replace the default"


def test_javascript_source_files_participate_in_the_walk(monkeypatch):
    def listed(*args):
        assert args == ("git", "ls-files", "*.ts", "*.tsx", "*.js", "*.jsx")
        return "shared/data.js\nsrc/app/items/page.jsx\nnode_modules/vendor/index.js\n"

    monkeypatch.setattr(r, "sh", listed)
    assert r.source_files() == ["shared/data.js", "src/app/items/page.jsx"]
    known = {"shared/data.js", "components/index.jsx"}
    assert r.resolve_spec("./shared/data", "page.jsx", {}, known) == "shared/data.js"
    assert r.resolve_spec("./components", "page.jsx", {}, known) == "components/index.jsx"


# --------------------------------------------- denominator: a real zero vs an absence
def test_zero_summary_is_printed_on_every_success_path(tmp_path, monkeypatch):
    """A caller counting "how often was this non-empty" needs the DENOMINATOR. An early
    return with no summary makes a run that genuinely reached nothing look identical to
    one that never ran."""
    import subprocess, sys
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "a.md").write_text("doc\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "one"], cwd=repo, check=True)
    (repo / "b.md").write_text("more\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "two"], cwd=repo, check=True)

    out = subprocess.run([sys.executable, str(SCRIPT),
                          "--base", "HEAD~1", "--head", "HEAD"],
                         cwd=repo, capture_output=True, text=True)
    assert out.returncode == 0
    assert "NOT IN THE DIFF: 0" in out.stdout, \
        f"a non-source diff must still print the zero summary, got: {out.stdout!r}"


def test_unresolvable_base_ref_exits_nonzero_and_prints_no_summary(tmp_path):
    """An execution FAILURE must not look like a legitimate zero. Printing the summary
    unconditionally first made a bad ref read as "0 entry points", which both poisoned
    the denominator and broke review.py's origin/<base> then <base> fallback."""
    import subprocess, sys
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    out = subprocess.run([sys.executable, str(SCRIPT),
                          "--base", "definitely-not-a-ref", "--head", "HEAD"],
                         cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode != 0, "an unresolvable base ref must exit non-zero"
    assert "NOT IN THE DIFF" not in out.stdout, "a failure must print no zero summary"
