"""Pin for the empty-finding filter at reviewer ingestion.

THE REGRESSION. A reviewer with nothing to say sometimes fills the schema in
rather than returning `findings: []`. A model shipped
`{file: "None", line: 0, title: "None", why: "None"}` onto a real PR comment,
where it rendered as "**`None:0` -- None**" under Minors. Three
costs, in rising order of seriousness: it is noise in a human-read report; it
inflates the finding count the header advertises; and every kept finding fans
out to a 3-lens adversarial verify, so a placeholder burns three LLM calls
proving that nothing is nothing.

The invariant: a finding whose TITLE is a placeholder never enters the pipeline
-- not the report, not the verify fan-out, not the counts. Title only: it is the
field every renderer and the verify prompt depend on, so a placeholder there is
fatal regardless of the other fields, while a real title with a thin `why` is a
weak finding rather than an empty one and must survive (dropping it would
silently narrow the review, which is the failure mode this whole lane exists to
avoid).
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "review", Path(__file__).resolve().parents[1] / "review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


# --- the shapes that must die ----------------------------------------------


def test_schema_filled_with_placeholders_is_empty():
    """Every text field can contain the model's literal placeholder string."""
    assert review.is_empty_finding(
        {"file": "None", "line": 0, "title": "None", "why": "None",
         "failure": "None", "fix": "None", "severity": "minor"})


def test_json_null_title_is_empty():
    """`"title": null` decodes to Python None -- str() would make it "None"."""
    assert review.is_empty_finding({"title": None, "why": "something"})


def test_missing_title_is_empty():
    assert review.is_empty_finding({"why": "something"})


def test_placeholder_titles_are_empty():
    for t in ["", "  ", "none", "None", "N/A", "n/a", "null", "nil", "-", "--",
              "no findings", "No Issues"]:
        assert review.is_empty_finding({"title": t}), t


def test_non_dict_is_empty():
    """A model can emit a bare string or null in the findings array."""
    for junk in [None, "None", 0, [], "a string finding"]:
        assert review.is_empty_finding(junk), junk


# --- the shapes that must SURVIVE ------------------------------------------


def test_real_finding_survives():
    assert not review.is_empty_finding(
        {"file": "app/x.ts", "line": 10, "title": "off-by-one in the retry loop",
         "why": "…", "severity": "major"})


def test_thin_finding_with_a_real_title_survives():
    """A vague `why` is a WEAK finding, not an empty one. Dropping it here would
    narrow the review invisibly -- worse than the noise this filter removes."""
    assert not review.is_empty_finding(
        {"file": "a.py", "line": 1, "title": "this looks wrong", "why": "None"})


def test_title_merely_containing_none_survives():
    """Substring matching would eat real findings; the check is whole-value."""
    for t in ["None returned where a dict was expected",
              "n/a is rendered instead of the count",
              "nil check missing"]:
        assert not review.is_empty_finding({"title": t}), t


# --- the ingestion path itself (not just the predicate) --------------------


def test_ingestion_drops_the_empty_and_keeps_the_real():
    raw = [
        {"file": "a.ts", "line": "12", "title": "real problem", "severity": "major"},
        {"file": "None", "line": 0, "title": "None", "why": "None"},
        {"file": "b.ts", "line": 3, "title": "another", "severity": "nit"},
    ]
    findings, empty = review.normalize_findings(raw, "correctness")
    assert empty == 1
    assert [f["title"] for f in findings] == ["real problem", "another"]
    assert all(f["dimension"] == "correctness" for f in findings)
    assert findings[0]["line"] == 12, "line must be coerced to int"


def test_all_empty_yields_no_findings():
    """A 'no findings' review that filled the schema in."""
    findings, empty = review.normalize_findings(
        [{"title": "None", "why": "None"}], "simplicity-reuse")
    assert findings == [] and empty == 1


def test_unknown_severity_defaults_to_minor():
    findings, _ = review.normalize_findings(
        [{"title": "x", "severity": "catastrophic"}], "d")
    assert findings[0]["severity"] == "minor"


def test_unparseable_line_does_not_raise():
    findings, _ = review.normalize_findings([{"title": "x", "line": "L42"}], "d")
    assert findings[0]["line"] == 0


def test_cap_still_applies_and_counts_only_kept():
    raw = [{"title": f"f{i}"} for i in range(review.MAX_FINDINGS_PER_DIMENSION + 5)]
    findings, empty = review.normalize_findings(raw, "d")
    assert len(findings) == review.MAX_FINDINGS_PER_DIMENSION and empty == 0


def test_major_only_report_is_blocker_free():
    result = {"findings": [{"severity": "major", "file": "a.ts", "line": 1,
                            "title": "real advisory bug", "why": "why",
                            "suggestion": "fix", "dimensions": ["correctness"],
                            "status": "new"}],
              "stats": {"raw": 1, "kept": 1, "killed": 0, "downgraded": 0,
                        "meta": [], "gaps": 0, "cap_drops": 0, "resolved": None},
              "resolved": [], "killed": []}
    md = review.render_markdown({"number": 1, "title": "t"}, result, "")
    assert "Blocker-free." in md
    assert "### Major" in md


def test_non_list_output_is_survivable():
    assert review.normalize_findings(None, "d") == ([], 0)
    assert review.normalize_findings({"findings": []}, "d") == ([], 0)
