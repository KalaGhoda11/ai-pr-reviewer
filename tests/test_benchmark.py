"""Tests for the detector benchmark harness (:mod:`mlscan.benchmark`).

The metric layer is deliberately pure stdlib — no numpy, no pandas, no dataset —
so a wrong precision/recall is caught here rather than in a 15-minute corpus run
that nobody re-reads. These tests therefore run in the web-app CI job too, which
installs only requirements.txt.

The two tests that need the ML extras call ``pytest.importorskip`` inside the
test body (not at import time) so collection never fails where scikit-learn is
absent.
"""

import pytest

from mlscan.benchmark import (
    Row,
    benchmark_rows,
    binary_metrics,
    by_group,
    format_report,
    language_group,
    patched_metrics,
    per_cwe_metrics,
    rule_detect,
    run_detectors,
    stratified_sample,
)

SAFE = "safe"

YAML_VULN = "import yaml\ndef load(blob):\n    return yaml.load(blob)"
YAML_FIXED = "import yaml\ndef load(blob):\n    return yaml.safe_load(blob)"
OS_VULN = 'import os\ndef ping(host):\n    os.system("ping " + host)'
OS_FIXED = ('import subprocess\ndef ping(host):\n'
            '    subprocess.run(["ping", host], shell=False)')
BENIGN = "def add(a, b):\n    return a + b"
C_OVERFLOW = ("int main(int argc, char **argv) {\n    char buf[8];\n"
              "    strcpy(buf, argv[1]);\n    return 0;\n}")
EVAL_VULN = "def f(x):\n    return eval(x)"


# ---------------------------------------------------------------------------
# binary metrics
# ---------------------------------------------------------------------------

def test_binary_metrics_counts_and_scores():
    golds = [SAFE, SAFE, SAFE, "CWE-89", "CWE-79", "CWE-94"]
    detections = [set(), {"CWE-89"}, set(), {"CWE-89"}, {"CWE-502"}, set()]
    m = binary_metrics(golds, detections)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["n"] == 6 and m["n_vulnerable"] == 3 and m["n_safe"] == 3
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert m["flag_rate"] == pytest.approx(3 / 6, abs=1e-4)
    assert m["fire_rate_on_safe"] == pytest.approx(1 / 3, abs=1e-4)


def test_binary_metrics_is_set_membership_not_correctness():
    """The binary decision only asks "did it fire", not "did it name the right
    CWE" — naming is scored separately by ``per_cwe_metrics``."""
    m = binary_metrics(["CWE-89"], [{"MEMORY-OOB"}])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)


def test_binary_metrics_silent_detector_scores_zero_not_nan():
    m = binary_metrics([SAFE, "CWE-89"], [set(), set()])
    assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)
    assert m["fire_rate_on_safe"] == 0.0
    assert m["accuracy"] == 0.5


def test_binary_metrics_on_empty_input_does_not_divide_by_zero():
    m = binary_metrics([], [])
    assert m["n"] == 0 and m["f1"] == 0.0 and m["accuracy"] == 0.0


def test_binary_metrics_perfect_detector():
    golds = [SAFE, "CWE-89", "CWE-94"]
    m = binary_metrics(golds, [set(), {"CWE-89"}, {"CWE-94"}])
    assert (m["precision"], m["recall"], m["f1"]) == (1.0, 1.0, 1.0)
    assert m["fire_rate_on_safe"] == 0.0


# ---------------------------------------------------------------------------
# per-CWE metrics
# ---------------------------------------------------------------------------

def test_per_cwe_scores_every_named_cwe_one_vs_rest():
    golds = ["CWE-89", "CWE-89", SAFE]
    # Row 0 names the right CWE plus a spurious one; row 1 misses entirely.
    detections = [{"CWE-89", "CWE-79"}, set(), {"CWE-79"}]
    per = per_cwe_metrics(golds, detections)["classes"]
    assert per["CWE-89"]["support"] == 2
    assert (per["CWE-89"]["tp"], per["CWE-89"]["fn"], per["CWE-89"]["fp"]) == (1, 1, 0)
    assert per["CWE-89"]["precision"] == 1.0
    assert per["CWE-89"]["recall"] == 0.5
    # CWE-79 has no support here, so it is pure false positives.
    assert (per["CWE-79"]["support"], per["CWE-79"]["fp"]) == (0, 2)
    assert per["CWE-79"]["f1"] == 0.0


def test_per_cwe_macro_averages_only_over_supported_classes():
    """A sampled slice must not be punished for classes it does not contain."""
    golds = ["CWE-89", "CWE-94"]
    out = per_cwe_metrics(golds, [{"CWE-89"}, {"CWE-94"}])
    assert out["macro_over"] == ["CWE-89", "CWE-94"]
    assert out["macro_f1"] == 1.0


def test_per_cwe_covers_the_whole_taxonomy_even_when_absent():
    from mlscan.labels import TAXONOMY

    out = per_cwe_metrics([SAFE], [set()])
    assert set(out["classes"]) == set(TAXONOMY)
    assert out["macro_f1"] == 0.0 and out["macro_over"] == []


def test_per_cwe_includes_classes_only_the_detector_invents():
    out = per_cwe_metrics([SAFE], [{"CWE-1337"}])
    assert out["classes"]["CWE-1337"]["fp"] == 1


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

def test_by_group_splits_the_precision_denominator():
    golds = [SAFE, SAFE, "CWE-89"]
    detections = [{"CWE-94"}, set(), {"CWE-89"}]
    groups = by_group(golds, detections, ["noisy", "clean", "clean"])
    assert groups["noisy"]["fire_rate_on_safe"] == 1.0
    assert groups["clean"]["fire_rate_on_safe"] == 0.0
    assert groups["clean"]["recall"] == 1.0


def test_language_group_separates_the_c_family():
    assert language_group("C") == "c_family"
    assert language_group("cpp") == "c_family"
    assert language_group("C++") == "c_family"
    assert language_group("Python") == "non_c"
    assert language_group("C#") == "non_c"      # rules cover .NET, model does not
    assert language_group("") == "non_c"


# ---------------------------------------------------------------------------
# code_fixed (patched) metrics
# ---------------------------------------------------------------------------

def test_patched_metrics_counts_persistence_and_pure_false_positives():
    golds = ["CWE-89", "CWE-94", "CWE-79", "CWE-502"]
    on_vuln = [{"CWE-89"}, {"CWE-94"}, set(), {"CWE-502"}]
    on_fixed = [set(),      {"CWE-94"}, {"CWE-79"}, {"CWE-89"}]
    m = patched_metrics(golds, on_vuln, on_fixed)
    assert m["n_pairs"] == 4
    assert m["fired_on_fixed"] == 3
    assert m["fire_rate_on_fixed"] == 0.75
    # Row 2 fires only after the patch: nothing was removed, so it is an
    # unambiguous false positive.
    assert m["fires_only_on_fixed"] == 1
    # Rows 0, 1, 3 named the true CWE on the vulnerable side; only 0 and 3 went
    # silent on the patch.
    assert m["named_true_cwe_on_vulnerable"] == 3
    assert m["patch_sensitive"] == 2
    assert m["patch_sensitivity"] == pytest.approx(2 / 3, abs=1e-4)


def test_patched_metrics_without_any_true_cwe_hit_is_zero_not_error():
    m = patched_metrics(["CWE-89"], [set()], [set()])
    assert m["named_true_cwe_on_vulnerable"] == 0
    assert m["patch_sensitivity"] == 0.0


# ---------------------------------------------------------------------------
# rows and sampling
# ---------------------------------------------------------------------------

def test_has_patch_requires_a_vulnerable_row_and_a_real_change():
    assert Row(code="a", label="CWE-89", code_fixed="b").has_patch
    assert not Row(code="a", label="CWE-89", code_fixed="a").has_patch   # unchanged
    assert not Row(code="a", label="CWE-89", code_fixed="  ").has_patch  # empty
    assert not Row(code="a", label="CWE-89").has_patch
    assert not Row(code="a", label=SAFE, code_fixed="b").has_patch       # not a fix


def test_stratified_sample_keeps_every_class_and_the_prior():
    rows = ([Row(code="s", label=SAFE)] * 900
            + [Row(code="v", label="CWE-89")] * 90
            + [Row(code="r", label="CWE-200")] * 10)
    kept = stratified_sample(rows, 100, seed=1)
    labels = [r.label for r in kept]
    assert set(labels) == {SAFE, "CWE-89", "CWE-200"}
    assert 80 <= labels.count(SAFE) <= 95        # prior preserved, not balanced
    assert labels.count("CWE-200") >= 1          # rare class survives
    assert 90 <= len(kept) <= 110


def test_stratified_sample_is_deterministic_and_a_noop_when_large_enough():
    rows = [Row(code=str(i), label=SAFE if i % 2 else "CWE-89") for i in range(50)]
    assert stratified_sample(rows, 0) is rows
    assert stratified_sample(rows, 500) is rows
    a = [r.code for r in stratified_sample(rows, 10, seed=7)]
    b = [r.code for r in stratified_sample(rows, 10, seed=7)]
    assert a == b


# ---------------------------------------------------------------------------
# the rule detector (pure stdlib - runs everywhere)
# ---------------------------------------------------------------------------

def test_rule_detect_returns_the_named_cwe_set():
    assert rule_detect(YAML_VULN) == {"CWE-502"}
    assert rule_detect(YAML_FIXED) == frozenset()
    assert rule_detect(OS_VULN) == {"CWE-94"}
    assert rule_detect(BENIGN) == frozenset()


def test_hybrid_is_the_union_of_the_two_layers(monkeypatch):
    """``scanner.scan`` merges rule findings into the ML findings keyed on the
    CWE id, so the hybrid set is exactly the union. Stub the ML side so this
    holds without scikit-learn."""
    import mlscan.benchmark as bench

    monkeypatch.setattr(bench, "ml_detect",
                        lambda codes, threshold, progress=None:
                        [frozenset({"CWE-89"}) for _ in codes])
    out = run_detectors([YAML_VULN, BENIGN], bench.DETECTORS, 0.5)
    assert out["ml"] == [{"CWE-89"}, {"CWE-89"}]
    assert out["rules"] == [{"CWE-502"}, frozenset()]
    assert out["hybrid"] == [{"CWE-89", "CWE-502"}, {"CWE-89"}]


# ---------------------------------------------------------------------------
# end-to-end on synthetic rows (no dataset, no model)
# ---------------------------------------------------------------------------

def _synthetic_rows():
    return [
        Row(code=YAML_VULN, label="CWE-502", code_fixed=YAML_FIXED,
            language="Python", source="synthetic"),
        Row(code=OS_VULN, label="CWE-94", code_fixed=OS_FIXED,
            language="Python", source="synthetic"),
        # C memory bug. The rule engine gained C coverage (RX-C-STRCPY-FIXED),
        # so this is now a true positive; it stays flagged dup_of_train because
        # the unseen-slice tests below rely on exactly one duplicated row.
        Row(code=C_OVERFLOW, label="MEMORY-OOB", language="C",
            source="synthetic", dup_of_train=True),
        Row(code=BENIGN, label=SAFE, language="Python", source="synthetic"),
        # A mislabelled "safe" row of the kind source=labeled_dataset is full of.
        Row(code=EVAL_VULN, label=SAFE, language="Python", source="noisy_labels"),
    ]


def test_benchmark_rows_end_to_end_with_rules_only():
    payload = benchmark_rows(_synthetic_rows(), detectors=("rules",),
                             verify=0, quiet=True)
    assert payload["n_rows"] == 5
    assert payload["n_unseen"] == 4          # one row flagged dup_of_train
    assert payload["n_code_fixed_pairs"] == 2

    binary = payload["detectors"]["rules"]["all_rows"]["binary"]
    # 3 TP (yaml, os.system, C strcpy), 1 FP (the mislabelled-safe eval row),
    # 0 FN, 1 TN. The C row became a TP once the engine gained C coverage.
    assert (binary["tp"], binary["fp"], binary["fn"], binary["tn"]) == (3, 1, 0, 1)
    assert binary["recall"] == pytest.approx(1.0, abs=1e-4)

    # The pooled safe-row fire rate is driven entirely by the noisy source; the
    # per-source split is what makes that visible.
    by_source = payload["detectors"]["rules"]["all_rows"]["by_source"]
    assert by_source["noisy_labels"]["fire_rate_on_safe"] == 1.0
    assert by_source["synthetic"]["fire_rate_on_safe"] == 0.0

    # C is now covered by the regex tier (lower confidence than the AST tier).
    groups = payload["detectors"]["rules"]["all_rows"]["by_language_group"]
    assert groups["c_family"]["recall"] == 1.0
    assert groups["non_c"]["recall"] == 1.0

    # Both patches remove the construct that was flagged.
    patched = payload["detectors"]["rules"]["code_fixed"]
    assert patched["n_pairs"] == 2
    assert patched["fired_on_fixed"] == 0
    assert patched["patch_sensitivity"] == 1.0

    per_cwe = payload["detectors"]["rules"]["all_rows"]["per_cwe"]["classes"]
    assert per_cwe["CWE-502"]["recall"] == 1.0
    assert per_cwe["MEMORY-OOB"]["recall"] == 1.0
    assert per_cwe["CWE-94"]["fp"] == 1      # the mislabelled eval() row


def test_benchmark_rows_unseen_slice_excludes_duplicated_rows():
    payload = benchmark_rows(_synthetic_rows(), detectors=("rules",),
                             verify=0, quiet=True)
    res = payload["detectors"]["rules"]
    # The unseen slice must drop exactly the one dup_of_train row (the C sample)
    # and nothing else, so it reports 4 of the 5 rows. Recall is 1.0 on both
    # slices here: the rule engine has no false negatives on these fixtures.
    assert res["all_rows"]["binary"]["n"] == 5
    assert res["unseen_only"]["binary"]["n"] == 4
    assert res["unseen_only"]["binary"]["recall"] == 1.0
    # The dropped row was a true positive, so the unseen slice has one fewer TP.
    assert res["unseen_only"]["binary"]["tp"] == res["all_rows"]["binary"]["tp"] - 1


def test_headline_block_matches_the_detailed_metrics():
    payload = benchmark_rows(_synthetic_rows(), detectors=("rules",),
                             verify=0, quiet=True)
    head = payload["headline"]["rules"]
    detail = payload["detectors"]["rules"]
    assert head["binary_f1"] == detail["all_rows"]["binary"]["f1"]
    assert head["binary_recall"] == detail["all_rows"]["binary"]["recall"]
    assert head["binary_f1_unseen_only"] == detail["unseen_only"]["binary"]["f1"]
    assert head["patch_sensitivity"] == detail["code_fixed"]["patch_sensitivity"]
    assert "UPPER_BOUND" in " ".join(head)      # the caveat is un-droppable


def test_headline_tolerates_a_corpus_slice_with_no_patches():
    rows = [Row(code=BENIGN, label=SAFE), Row(code=YAML_VULN, label="CWE-502")]
    head = benchmark_rows(rows, detectors=("rules",), verify=0,
                          quiet=True)["headline"]["rules"]
    assert head["patch_sensitivity"] is None
    assert head["fire_rate_on_patched_code_UPPER_BOUND"] is None


def test_benchmark_payload_is_json_serialisable_and_reports_caveats():
    import json

    payload = benchmark_rows(_synthetic_rows(), detectors=("rules",),
                             verify=0, quiet=True)
    text = json.dumps(payload)              # frozensets would raise here
    assert "upper bound" in text.lower()
    assert payload["caveats"] and all(isinstance(c, str) for c in payload["caveats"])


def test_benchmark_rows_without_any_patch_pairs():
    rows = [Row(code=BENIGN, label=SAFE), Row(code=YAML_VULN, label="CWE-502")]
    payload = benchmark_rows(rows, detectors=("rules",), verify=0, quiet=True)
    assert payload["n_code_fixed_pairs"] == 0
    assert payload["detectors"]["rules"]["code_fixed"] is None


def test_format_report_renders_every_section():
    payload = benchmark_rows(_synthetic_rows(), detectors=("rules",),
                             verify=0, quiet=True)
    report = format_report(payload, "test")
    assert "all_rows" in report and "unseen_only" in report
    assert "code_fixed" in report and "UPPER BOUND" in report
    assert "Per-CWE F1" in report and "CWE-502" in report
    assert "Caveats:" in report


# ---------------------------------------------------------------------------
# ML-backed: the batched path must equal the shipped scanner
# ---------------------------------------------------------------------------

def test_batched_ml_detection_equals_scanner_scan():
    """The benchmark batches inference for speed; that must not become a second
    decision rule. Same guarantee ``verify_against_scanner`` enforces at runtime."""
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    from mlscan.scanner import DEFAULT_THRESHOLD, MODEL_PATH

    if not MODEL_PATH.exists():
        pytest.skip("trained model artifact not present")

    from mlscan.benchmark import ml_detect, run_detectors, verify_against_scanner

    codes = [YAML_VULN, OS_VULN, BENIGN, C_OVERFLOW, EVAL_VULN]
    named = run_detectors(codes, ("ml", "rules", "hybrid"), DEFAULT_THRESHOLD)
    assert verify_against_scanner(codes, named["ml"], named["hybrid"],
                                  DEFAULT_THRESHOLD, n=len(codes)) == len(codes)
    assert ml_detect(codes, DEFAULT_THRESHOLD) == named["ml"]


def test_verify_against_scanner_raises_when_the_paths_disagree():
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    from mlscan.scanner import DEFAULT_THRESHOLD, MODEL_PATH

    if not MODEL_PATH.exists():
        pytest.skip("trained model artifact not present")

    from mlscan.benchmark import verify_against_scanner

    with pytest.raises(RuntimeError, match="disagrees with scanner.scan"):
        verify_against_scanner([BENIGN], [frozenset({"CWE-89"})], None,
                               DEFAULT_THRESHOLD, n=1)
