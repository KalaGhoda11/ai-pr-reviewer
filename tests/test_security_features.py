"""Tests for the hand-crafted security-indicator transformer.

Fast and self-contained: no dataset, no trained model, no network. Each test
asserts that a specific snippet lights up the intended column AND that the
paired "safe" snippet does not — that contrast is the whole point of the block.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from scipy import sparse  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.pipeline import FeatureUnion  # noqa: E402
from sklearn.preprocessing import MaxAbsScaler  # noqa: E402

from mlscan.security_features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    SecurityFeatures,
    extract,
)


def feats(code: str) -> dict[str, float]:
    """Feature name -> value for one snippet."""
    return dict(zip(FEATURE_NAMES, extract(code)))


def assert_fires(code: str, *names: str) -> dict[str, float]:
    f = feats(code)
    for name in names:
        assert f[name] > 0.0, f"expected {name} to fire on: {code!r}"
    return f


def assert_quiet(code: str, *names: str) -> dict[str, float]:
    f = feats(code)
    for name in names:
        assert f[name] == 0.0, f"expected {name} to stay silent on: {code!r}"
    return f


# ---- transformer contract ------------------------------------------------

def test_feature_names_are_unique_and_match_width():
    assert len(set(FEATURE_NAMES)) == N_FEATURES
    names = SecurityFeatures().get_feature_names_out()
    assert len(names) == N_FEATURES
    assert all(n.startswith("sec__") for n in names)
    assert 25 <= N_FEATURES <= 46


def test_transform_returns_sparse_float32_matrix():
    X = SecurityFeatures().fit_transform(["strcpy(a, b);", "int add(int a){return a;}"])
    assert sparse.issparse(X)
    assert X.format == "csr"
    assert X.shape == (2, N_FEATURES)
    assert X.dtype == np.float32
    assert np.isfinite(X.toarray()).all()


def test_fit_is_stateless_and_estimator_is_clonable():
    t = SecurityFeatures()
    a = t.fit(["eval(x)"]).transform(["pickle.loads(b)"]).toarray()
    b = clone(t).fit(["totally different corpus"]).transform(["pickle.loads(b)"]).toarray()
    np.testing.assert_allclose(a, b)  # transform never depends on what was fit
    assert t.get_params()["max_chars"] == 4000


def test_composes_with_featureunion_and_maxabsscaler():
    from sklearn.pipeline import Pipeline

    corpus = ["strcpy(d, s);", "eval(x)", "def f():\n    return 1"]
    pipe = Pipeline([
        ("sec", SecurityFeatures()),
        ("scale", MaxAbsScaler()),
    ])
    X = pipe.fit_transform(corpus)
    assert X.shape == (3, N_FEATURES)
    assert sparse.issparse(X)          # MaxAbsScaler must not densify
    assert abs(X).max() <= 1.0 + 1e-6

    union = FeatureUnion([("sec", SecurityFeatures())])
    assert union.fit_transform(corpus).shape == (3, N_FEATURES)


def test_rejects_a_bare_string():
    with pytest.raises(ValueError):
        SecurityFeatures().fit_transform("eval(x)")


def test_degenerate_inputs_do_not_crash():
    X = SecurityFeatures().fit_transform(["", "\n\n", "é中文 // コメント", "a" * 50000])
    assert X.shape == (4, N_FEATURES)
    assert np.isfinite(X.toarray()).all()


def test_max_chars_truncates():
    tail = "eval(user)"
    padded = ("x = 1\n" * 2000) + tail
    assert SecurityFeatures(max_chars=100).fit_transform([padded]).toarray()[0][
        FEATURE_NAMES.index("inj_eval_exec")] == 0.0
    assert SecurityFeatures(max_chars=None).fit_transform([padded]).toarray()[0][
        FEATURE_NAMES.index("inj_eval_exec")] > 0.0


# ---- C / C++ memory safety ----------------------------------------------

def test_unsafe_vs_safe_c_string_functions():
    unsafe = assert_fires("void f(char *s){ char b[8]; strcpy(b, s); }",
                          "mem_unsafe_str_fn")
    assert unsafe["mem_safe_str_fn"] == 0.0

    safe = assert_fires("void f(char *s){ char b[8]; snprintf(b, sizeof(b), \"%s\", s); }",
                        "mem_safe_str_fn", "mem_sizeof")
    assert safe["mem_unsafe_str_fn"] == 0.0


def test_strncpy_does_not_count_as_strcpy():
    assert_quiet("strncpy(dst, src, n);", "mem_unsafe_str_fn")
    assert_quiet("vsprintf_is_not_here();", "mem_unsafe_str_fn")


def test_memcpy_and_alloca():
    assert_fires("memcpy(dst, src, len);", "mem_copy_fn")
    assert_fires("char *p = alloca(n);", "mem_alloca")
    assert_quiet("char *p = malloc(n);", "mem_alloca")


def test_alloc_and_free_pairing():
    f = assert_fires("void *p = malloc(n);\nfree(p);\nfree(p);",
                     "mem_alloc", "mem_free")
    assert f["mem_free"] > 0.0


def test_pointer_deref_with_and_without_null_check():
    unchecked = assert_fires("int g(struct s *p){ return p->len; }", "ptr_arrow")
    assert unchecked["null_check"] == 0.0

    checked = assert_fires(
        "int g(struct s *p){ if (p == NULL) return -1; return p->len; }",
        "ptr_arrow", "null_literal", "null_check")
    assert checked["null_check"] > 0.0


def test_index_arithmetic_and_off_by_one_loop():
    assert_fires("for (i = 0; i <= n; i++) buf[i + 1] = 0;",
                 "idx_arith", "loop_le_bound")
    assert_quiet("for (i = 0; i < n; i++) buf[i] = 0;",
                 "idx_arith", "loop_le_bound")


def test_bounds_check_is_negative_evidence():
    assert_fires("if (len < sizeof(buf)) memcpy(buf, src, len);",
                 "val_bounds_check", "mem_copy_fn")
    assert_quiet("memcpy(buf, src, len);", "val_bounds_check")


# ---- format strings ------------------------------------------------------

def test_format_string_literal_vs_non_literal():
    assert_fires("printf(user_supplied);", "fmt_nonliteral")
    assert_quiet('printf("%d items\\n", n);', "fmt_nonliteral")
    assert_fires('printf("%n", &count);', "fmt_pct_n")
    assert_quiet('printf("%s\\n", name);', "fmt_pct_n")


# ---- code / command injection -------------------------------------------

def test_eval_exec_detected_across_languages():
    assert_fires("def run(a):\n    return eval(a)", "inj_eval_exec")
    assert_fires("var fn = new Function(src);", "inj_eval_exec")
    # re.compile must NOT be mistaken for the builtin compile()
    assert_quiet("import re\npat = re.compile(r'^a+$')", "inj_eval_exec")


def test_os_command_and_shell_true():
    f = assert_fires("subprocess.call(cmd, shell=True)",
                     "inj_os_command", "inj_shell_true")
    assert f["inj_shell_true"] > 0.0
    assert_fires('os.system("ls " + d)', "inj_os_command")
    assert_fires('Runtime.getRuntime().exec(cmd);', "inj_os_command")
    assert_quiet("subprocess.run([\"ls\", d], shell=False)", "inj_shell_true")


# ---- SQL injection -------------------------------------------------------

def test_sql_concatenation_vs_parameterized():
    dynamic = assert_fires(
        'q = "SELECT * FROM accounts WHERE id = \'" + a + "\'"\ncur.execute(q)',
        "sql_keyword", "sql_dynamic", "gen_dyn_string")

    param = assert_fires(
        'cur.execute("SELECT * FROM accounts WHERE id = ?", (a,))',
        "sql_keyword", "sql_parameterized")
    assert param["sql_dynamic"] == 0.0
    assert dynamic["sql_parameterized"] == 0.0


def test_sql_fstring_is_dynamic():
    assert_fires('cur.execute(f"SELECT name FROM users WHERE id={a}")',
                 "sql_keyword", "sql_dynamic")


def test_english_prose_does_not_look_like_sql():
    assert_quiet("// select the next update and delete stale rows\nint n = 0;",
                 "sql_keyword", "sql_dynamic")


# ---- deserialization -----------------------------------------------------

def test_unsafe_vs_safe_deserialization():
    bad = assert_fires("import pickle\nobj = pickle.loads(blob)", "deser_unsafe")
    assert bad["deser_safe"] == 0.0

    good = assert_fires("import json\nobj = json.loads(blob)", "deser_safe")
    assert good["deser_unsafe"] == 0.0

    assert_fires("ObjectInputStream in = new ObjectInputStream(s);\nin.readObject();",
                 "deser_unsafe")
    assert_fires("$o = unserialize($_POST['d']);", "deser_unsafe", "taint_source")


def test_yaml_load_is_context_sensitive():
    assert_fires("cfg = yaml.load(stream)", "deser_unsafe")
    assert_quiet("cfg = yaml.load(stream, Loader=yaml.SafeLoader)", "deser_unsafe")
    assert_quiet("cfg = yaml.safe_load(stream)", "deser_unsafe")
    assert_fires("cfg = yaml.safe_load(stream)", "deser_safe")


# ---- XSS -----------------------------------------------------------------

def test_xss_sink_vs_escaping():
    sink = assert_fires("el.innerHTML = untrusted;", "xss_sink")
    assert sink["xss_escape"] == 0.0

    escaped = assert_fires("el.textContent = DOMPurify.sanitize(untrusted);",
                           "xss_escape")
    assert escaped["xss_sink"] == 0.0

    assert_fires("document.write(location.hash);", "xss_sink")
    assert_fires('echo "<b>" . $_GET["name"] . "</b>";',
                 "xss_sink", "taint_source", "gen_dyn_string")
    assert_fires('echo htmlspecialchars($n);', "xss_escape")


# ---- taint / validation --------------------------------------------------

def test_taint_sources_are_api_based():
    assert_fires("name = request.args['q']", "taint_source")
    assert_fires("String v = req.getParameter(\"q\");", "taint_source")
    assert_fires("path = sys.argv[1]", "taint_source")
    assert_quiet("total = a + b", "taint_source")


def test_path_traversal_literal():
    assert_fires('open(base + "../../etc/passwd")', "path_dotdot")
    assert_quiet('open(base + "data.txt")', "path_dotdot")


def test_regex_validation_and_unchecked_parse():
    assert_fires("if re.fullmatch(r'\\d{1,4}', v):\n    pass", "val_regex")
    assert_fires("int n = atoi(argv[1]);", "val_unchecked_parse")
    assert_fires("try:\n    x = 1\nexcept ValueError:\n    pass", "val_try_catch")


# ---- crypto / secrets / exposure ----------------------------------------

def test_weak_vs_strong_crypto():
    weak = assert_fires("h = hashlib.md5(pw).hexdigest()", "crypto_weak")
    assert weak["crypto_strong"] == 0.0

    strong = assert_fires("h = hashlib.sha256(pw).hexdigest()", "crypto_strong")
    assert strong["crypto_weak"] == 0.0

    assert_fires('Cipher.getInstance("DES/ECB/PKCS5Padding");', "crypto_weak")


def test_tls_verification_disabled():
    assert_fires("requests.get(url, verify=False)", "crypto_tls_off")
    assert_fires("ctx = ssl._create_unverified_context()", "crypto_tls_off")
    assert_quiet("requests.get(url, verify=True)", "crypto_tls_off")


def test_hardcoded_secret_and_high_entropy_literal():
    f = assert_fires('api_key = "AKIA5FZ8QW3RTYUIOPLK"',
                     "secret_hardcoded", "secret_high_entropy")
    assert f["secret_hardcoded"] > 0.0
    assert_fires('String password = "hunter2secret";', "secret_hardcoded")
    assert_quiet('greeting = "hello there world"', "secret_hardcoded",
                 "secret_high_entropy")


def test_weak_randomness_and_info_leak():
    assert_fires("token = random.randint(0, 999999)", "rand_weak")
    assert_quiet("token = secrets.token_hex(32)", "rand_weak")
    assert_fires("e.printStackTrace();", "info_leak_output")
    assert_fires("traceback.print_exc()", "info_leak_output")


# ---- robustness properties ----------------------------------------------

def test_identifier_renaming_leaves_the_vector_unchanged():
    """Length-preserving rename => byte-identical feature vector.

    The first snippet uses the exact giveaway identifiers that the synthetic
    rows in the corpus leak (``vulnerable_method``, ``user_input``); the second
    swaps them for neutral names of the same length, so any difference could
    only come from a pattern keying on a caller-chosen name.
    """
    leaky = (
        "def vulnerable_method(user_input):\n"
        "    query = \"SELECT * FROM users WHERE n='\" + user_input + \"'\"\n"
        "    cursor.execute(query)\n")
    neutral = (
        "def handle_row_values(identifier):\n"
        "    blobs = \"SELECT * FROM users WHERE n='\" + identifier + \"'\"\n"
        "    handle.execute(blobs)\n")
    assert len(leaky) == len(neutral)  # densities are therefore comparable
    assert extract(leaky) == extract(neutral)


def test_renaming_never_changes_which_signals_fire():
    """A rename that *does* change length may move magnitudes, not presence."""
    a = feats(
        "void unsafe_copy(char *user_supplied_buffer) {\n"
        "    char tmp[8];\n"
        "    strcpy(tmp, user_supplied_buffer);\n"
        "}\n")
    b = feats(
        "void q(char *z) {\n"
        "    char t[8];\n"
        "    strcpy(t, z);\n"
        "}\n")
    fired_a = {n for n in FEATURE_NAMES if a[n] > 0.0}
    fired_b = {n for n in FEATURE_NAMES if b[n] > 0.0}
    assert fired_a == fired_b
    assert "mem_unsafe_str_fn" in fired_a


def test_signals_are_length_normalised():
    one = feats("strcpy(d, s);\n")
    three = feats("strcpy(d, s);\n" * 3)
    assert three["mem_unsafe_str_fn"] == pytest.approx(
        one["mem_unsafe_str_fn"], abs=0.05)


def test_safe_code_lights_up_almost_nothing():
    f = feats("def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n")
    risky = [n for n in FEATURE_NAMES
             if not n.startswith(("gen_", "syntax_", "val_")) and f[n] > 0.0]
    assert risky == [], f"safe code fired: {risky}"
    assert f["syntax_script"] > 0.0
    assert f["gen_comment_ratio"] > 0.0


def test_c_and_script_syntax_markers_separate():
    c = feats("#include <string.h>\nstruct s { size_t n; };\nvoid f(struct s *p){ p->n = 0; }")
    py = feats("import os\ndef f(self):\n    return os.getcwd()")
    assert c["syntax_c"] > c["syntax_script"]
    assert py["syntax_script"] > py["syntax_c"]
