"""Tests for the deterministic rule engine (mlscan/rules.py).

Pure-Python and fast (no model artifact, no scikit-learn, no dataset), so they
run in every CI job. Each rule is checked twice: once on a snippet that must
fire, and once on the safe counterpart it must stay silent on -- false
positives are the failure mode this engine exists to avoid.

Note on the PHP fixtures: they are assembled at runtime from fragments
(``_php``/``_SUPER``) rather than written out literally. Some of them are
byte-for-byte webshell payloads, and an on-access antivirus scanner will
quarantine a source file that contains one. Building them in memory keeps the
test file itself clean while the scanner still sees the real thing.

The section "false-positive regressions" is not hypothetical: every fixture in
it is a construct that was observed firing on real, correct code (the CPython
standard library, this project's own virtualenv, or the patched side of a
vulnerability-fix commit). They are the reason those guards exist, so they are
pinned here rather than described in a comment.
"""

import time

from mlscan.labels import CLASSES, MEMORY_OOB
# _REGEX_RULES is private, but a typo in a rule's `mode` silently produces a
# rule that can never fire; test_regex_rule_table_is_well_formed is the only
# thing that would catch it.
from mlscan.rules import MAX_SCAN_CHARS, _REGEX_RULES, scan_rules


# ---- helpers -------------------------------------------------------------

def cwes(code: str) -> set[str]:
    return {f["cwe"] for f in scan_rules(code)}


def rule_ids(code: str) -> set[str]:
    return {f["rule_id"] for f in scan_rules(code)}


def assert_silent(code: str) -> None:
    found = scan_rules(code)
    assert found == [], f"false positive: {[f['rule_id'] for f in found]}\n{code}"


_SIGIL = "$"                      # keeps PHP payload signatures off disk
_OPEN = "<?" + "php"


def _super(name: str) -> str:
    """A PHP superglobal reference, e.g. ``_super('GET')``."""
    return f"{_SIGIL}_{name}"


def _php(body: str) -> str:
    return f"{_OPEN} {body} ?>"


# ---- contract ------------------------------------------------------------

def test_finding_shape():
    findings = scan_rules("import pickle\ndata = pickle.loads(blob)\n")
    assert findings, "expected a CWE-502 finding"
    f = findings[0]
    assert set(f) == {"cwe", "name", "confidence", "line", "rule_id", "evidence"}
    assert isinstance(f["cwe"], str) and isinstance(f["name"], str)
    assert isinstance(f["rule_id"], str) and isinstance(f["evidence"], str)
    assert isinstance(f["confidence"], float) and 0.0 < f["confidence"] <= 1.0
    assert f["line"] == 2
    assert "pickle.loads" in f["evidence"]


def test_every_finding_uses_a_taxonomy_class():
    samples = [
        "eval(user_input)",
        "import yaml\nyaml.load(cfg)",
        'q = "SELECT * FROM t WHERE a = " + x\ncur.execute(q)',
        "requests.get(url, verify=False)",
        'password = "hunter2xyz"',
        "char buf[8];\ngets(buf);",
        _php(f'echo {_super("GET")}["name"];'),
    ]
    for code in samples:
        for f in scan_rules(code):
            assert f["cwe"] in CLASSES, f


def test_empty_and_garbage_input_is_safe():
    for code in ["", "   \n\t ", None, "\x00\x01\x02", "}{)(", "###"]:
        assert scan_rules(code) == []


def test_results_are_sorted_and_unique():
    code = (
        "import pickle, yaml\n"
        "a = pickle.loads(b)\n"
        "c = yaml.load(d)\n"
        "eval(e)\n"
    )
    findings = scan_rules(code)
    lines = [f["line"] for f in findings]
    assert lines == sorted(lines)
    keys = [(f["rule_id"], f["line"]) for f in findings]
    assert len(keys) == len(set(keys))


def test_min_confidence_filter():
    code = "import pickle\npickle.loads(blob)\n"
    assert scan_rules(code, min_confidence=0.5)
    assert scan_rules(code, min_confidence=0.99) == []


# ---- CWE-94: code / command injection ------------------------------------

def test_eval_on_input_flagged():
    assert "PY-EVAL" in rule_ids("def run(user_input):\n    return eval(user_input)\n")
    assert "PY-EVAL" in rule_ids("exec(request.args['cmd'])\n")
    assert "PY-EVAL" in rule_ids('eval("1 + " + untrusted)\n')


def test_eval_on_literal_and_literal_eval_not_flagged():
    assert_silent('print(eval("1 + 1"))\n')
    assert_silent("import ast\nvalue = ast.literal_eval(raw)\n")
    assert_silent("from ast import literal_eval\nvalue = literal_eval(raw)\n")
    assert_silent("model.eval()\n")


def test_os_system_with_concatenation_flagged():
    assert "PY-OS-COMMAND" in rule_ids('import os\nos.system("ping " + host)\n')
    assert "PY-OS-COMMAND" in rule_ids('import os\nos.system(f"rm -rf {path}")\n')
    assert "PY-OS-COMMAND" in rule_ids("from os import system\nsystem(cmd)\n")


def test_os_system_with_literal_not_flagged():
    assert_silent('import os\nos.system("ls -la")\n')
    assert_silent('import os\nCMD = "ls -la"\nos.system(CMD)\n')


def test_subprocess_shell_true_flagged():
    code = "import subprocess\nsubprocess.run(user_cmd, shell=True)\n"
    assert "PY-SUBPROCESS-SHELL" in rule_ids(code)
    code = 'import subprocess\nsubprocess.Popen("cat " + name, shell=True)\n'
    assert "PY-SUBPROCESS-SHELL" in rule_ids(code)


def test_subprocess_safe_forms_not_flagged():
    assert_silent('import subprocess\nsubprocess.run(["ls", "-la", path])\n')
    assert_silent("import subprocess\nsubprocess.run(argv, shell=False)\n")
    assert_silent('import subprocess\nsubprocess.check_output(["git", "status"])\n')
    # shell=True with a hardcoded command cannot be injected into.
    assert_silent('import subprocess\nsubprocess.run("ls -la", shell=True)\n')


# ---- CWE-502: insecure deserialization -----------------------------------

def test_pickle_and_marshal_flagged():
    assert "CWE-502" in cwes("import pickle\nobj = pickle.loads(payload)\n")
    assert "CWE-502" in cwes("import pickle\nobj = pickle.load(fh)\n")
    assert "CWE-502" in cwes("import marshal\nobj = marshal.loads(blob)\n")
    assert "CWE-502" in cwes("from pickle import loads\nobj = loads(blob)\n")
    assert "CWE-502" in cwes("import cPickle as p\nobj = p.loads(blob)\n")


def test_yaml_load_without_safe_loader_flagged():
    assert "PY-YAML-LOAD" in rule_ids("import yaml\ncfg = yaml.load(stream)\n")
    assert "PY-YAML-LOAD" in rule_ids(
        "import yaml\ncfg = yaml.load(stream, Loader=yaml.Loader)\n")
    assert "PY-YAML-LOAD" in rule_ids("import yaml as y\ncfg = y.load(stream)\n")


def test_yaml_safe_loading_not_flagged():
    assert_silent("import yaml\ncfg = yaml.safe_load(stream)\n")
    assert_silent("import yaml\ncfg = yaml.load(stream, Loader=yaml.SafeLoader)\n")
    assert_silent("import yaml\ncfg = yaml.load(stream, Loader=yaml.CSafeLoader)\n")
    assert_silent("import json\ncfg = json.loads(stream)\n")


# ---- CWE-89: SQL injection -----------------------------------------------

def test_sql_concatenation_flagged():
    code = (
        "def get(user_id):\n"
        "    query = \"SELECT * FROM accounts WHERE id = '\" + user_id + \"'\"\n"
        "    cursor.execute(query)\n"
    )
    findings = [f for f in scan_rules(code) if f["cwe"] == "CWE-89"]
    assert findings and findings[0]["line"] == 3


def test_sql_fstring_and_percent_and_format_flagged():
    assert "CWE-89" in cwes('cursor.execute(f"SELECT * FROM t WHERE id = {uid}")\n')
    assert "CWE-89" in cwes('cursor.execute("SELECT * FROM t WHERE id = %s" % uid)\n')
    assert "CWE-89" in cwes('cursor.execute("SELECT * FROM t WHERE id = {}".format(uid))\n')


def test_sql_built_incrementally_flagged():
    code = (
        'query = "SELECT id FROM users"\n'
        'query += " WHERE name = \'" + name + "\'"\n'
        "cur.execute(query)\n"
    )
    assert "CWE-89" in cwes(code)


def test_parameterised_queries_not_flagged():
    assert_silent('cursor.execute("SELECT * FROM t WHERE id = %s", (uid,))\n')
    assert_silent('cursor.execute("SELECT * FROM t WHERE id = ?", [uid])\n')
    assert_silent('cursor.executemany("INSERT INTO t VALUES (?, ?)", rows)\n')
    assert_silent('QUERY = "SELECT * FROM users"\ncursor.execute(QUERY)\n')
    assert_silent('cursor.execute("SELECT 1")\n')


def test_non_sql_string_concatenation_not_flagged():
    assert_silent('msg = "hello " + name\nlogger.info(msg)\n')
    assert_silent('path = "/tmp/" + name\nopen(path).read()\n')
    assert_silent("session.query(User).filter(User.id == uid).first()\n")


# ---- CWE-200: TLS verification, weak hashes, secrets ---------------------

def test_verify_false_flagged():
    assert "PY-TLS-VERIFY-OFF" in rule_ids(
        "import requests\nrequests.get(url, verify=False)\n")
    assert "PY-TLS-VERIFY-OFF" in rule_ids("s.post(url, data=d, verify=False)\n")
    assert "PY-TLS-VERIFY-OFF" in rule_ids(
        "import ssl\nctx = ssl._create_unverified_context()\n")


def test_verify_true_not_flagged():
    assert_silent("import requests\nrequests.get(url, verify=True)\n")
    assert_silent("import requests\nrequests.get(url, timeout=5)\n")
    assert_silent("import requests\nrequests.get(url, verify=ca_bundle)\n")


def test_weak_hash_in_security_context_flagged():
    assert "PY-WEAK-HASH" in rule_ids(
        "import hashlib\nstored = hashlib.md5(password.encode()).hexdigest()\n")
    assert "PY-WEAK-HASH" in rule_ids(
        "import hashlib\n"
        "def hash_password(raw):\n"
        "    return hashlib.sha1(raw).hexdigest()\n")
    assert "PY-WEAK-HASH" in rule_ids(
        "import hashlib\ntoken = hashlib.new('md5', seed).hexdigest()\n")


def test_strong_hash_and_checksum_use_not_flagged():
    assert_silent("import hashlib\nh = hashlib.sha256(password.encode()).hexdigest()\n")
    assert_silent("import hashlib\ncache_key = hashlib.md5(url.encode()).hexdigest()\n")
    assert_silent("import hashlib\nchecksum = hashlib.md5(file_bytes).hexdigest()\n")
    assert_silent(
        "import hashlib\n"
        "etag = hashlib.md5(body, usedforsecurity=False).hexdigest()\n")


def test_hardcoded_secret_flagged():
    assert "PY-HARDCODED-SECRET" in rule_ids('DB_PASSWORD = "Tr0ub4dor&3"\n')
    assert "PY-HARDCODED-SECRET" in rule_ids('api_key = "AKIAIOSFODNN7EXAMPL"\n')
    assert "PY-HARDCODED-SECRET" in rule_ids(
        'conn = connect(host=h, password="s3cr3tzz")\n')
    assert "PY-HARDCODED-SECRET" in rule_ids('CONF = {"auth_token": "abc123def456"}\n')


def test_non_secrets_not_flagged():
    assert_silent('password = os.environ["DB_PASSWORD"]\n')
    assert_silent('PASSWORD_FIELD = "password"\n')
    assert_silent('password = ""\n')
    assert_silent('api_key_url = "https://api.example.com/keys"\n')
    assert_silent('token_pattern = "^[a-z]+$"\n')
    assert_silent('password = "changeme"\n')
    assert_silent('password = "xxxxxxxx"\n')
    assert_silent('secret = "<your-secret-here>"\n')
    assert_silent('password_template = "user:{}"\n')
    assert_silent('username = "administrator"\n')
    # Ordinary lowercase config words are not credentials.
    assert_silent('ALLOWED = {"password": "required"}\n')
    assert_silent('opts = {"secret": "optional"}\n')


# ---- CWE-79: cross-site scripting ----------------------------------------

def test_python_xss_flagged():
    assert "CWE-79" in cwes(
        "from flask import render_template_string\n"
        'render_template_string("<h1>" + name + "</h1>")\n')
    assert "CWE-79" in cwes(
        "from django.utils.safestring import mark_safe\nmark_safe(user_bio)\n")
    assert "CWE-79" in cwes('return HttpResponse(f"<div>{comment}</div>")\n')


def test_python_escaped_output_not_flagged():
    assert_silent(
        "from django.utils.safestring import mark_safe\n"
        "from django.utils.html import escape\n"
        "mark_safe(escape(user_bio))\n")
    assert_silent('render_template("profile.html", name=name)\n')
    assert_silent('return HttpResponse("<h1>Hello</h1>")\n')


# ---- non-Python fallback (regex backend) ---------------------------------

def test_php_xss():
    assert "CWE-79" in cwes(_php(f'echo "Hello " . {_super("GET")}["name"];'))


def test_php_code_injection():
    assert "CWE-94" in cwes(_php(f'eval({_super("POST")}["code"]);'))


def test_php_deserialization():
    assert "CWE-502" in cwes(_php(f'$o = unserialize({_super("COOKIE")}["data"]);'))


def test_php_sql_injection():
    body = '$r = mysqli_query($db, "SELECT * FROM users WHERE id = $id");'
    assert "CWE-89" in cwes(_php(body))


def test_php_safe_patterns():
    assert_silent(_php("echo htmlspecialchars($name);"))
    assert_silent(_php('$stmt = $pdo->prepare("SELECT * FROM users WHERE id = ?");'))


def test_java_sql_concatenation():
    code = (
        "public class A {\n"
        "  void f(String name) throws Exception {\n"
        "    Statement st = conn.createStatement();\n"
        '    st.executeQuery("SELECT * FROM users WHERE name = \'" + name + "\'");\n'
        "  }\n"
        "}\n"
    )
    assert "CWE-89" in cwes(code)


def test_java_object_input_stream():
    code = (
        "public class B {\n"
        "  Object f(InputStream in) throws Exception {\n"
        "    ObjectInputStream ois = new ObjectInputStream(in);\n"
        "    return ois.readObject();\n"
        "  }\n"
        "}\n"
    )
    assert "CWE-502" in cwes(code)


def test_java_prepared_statement_not_flagged():
    code = (
        "public class A {\n"
        "  void f(String name) throws Exception {\n"
        "    PreparedStatement ps = conn.prepareStatement(\n"
        '        "SELECT * FROM users WHERE name = ?");\n'
        "    ps.setString(1, name);\n"
        "  }\n"
        "}\n"
    )
    assert_silent(code)


def test_javascript_patterns():
    assert "CWE-79" in cwes("function f(q) {\n  el.innerHTML = '<b>' + q + '</b>';\n}\n")
    assert "CWE-79" in cwes("function f(q) {\n  document.write(q);\n}\n")
    assert "CWE-200" in cwes("const opts = { rejectUnauthorized: false };\n")


def test_javascript_safe_patterns():
    assert_silent("function f(q) {\n  el.textContent = q;\n}\n")
    assert_silent("function f() {\n  el.innerHTML = '<b>static</b>';\n}\n")


def test_sanitized_html_not_flagged():
    assert_silent("el.innerHTML = DOMPurify.sanitize(dirty);\n")
    assert_silent("const html = DOMPurify.sanitize(dirty);\nel.innerHTML = html;\n")
    assert_silent("document.write(encodeURIComponent(q));\n")


def test_c_unbounded_reads():
    code = (
        "int main(void) {\n"
        "    char buf[8];\n"
        "    gets(buf);\n"
        "    return 0;\n"
        "}\n"
    )
    findings = scan_rules(code)
    assert findings and findings[0]["cwe"] == "MEMORY-OOB"
    assert findings[0]["line"] == 3

    assert "MEMORY-OOB" in cwes(
        'int main(void) {\n    char b[8];\n    scanf("%s", b);\n    return 0;\n}\n')


def test_c_safe_reads_not_flagged():
    assert_silent(
        "int main(void) {\n"
        "    char buf[8];\n"
        "    fgets(buf, sizeof(buf), stdin);\n"
        "    return 0;\n"
        "}\n")


def test_c_system_with_variable():
    assert "CWE-94" in cwes("void run(char *cmd) {\n    system(cmd);\n}\n")
    assert_silent('void run(void) {\n    system("ls -la");\n}\n')


# ---- comment handling ----------------------------------------------------

def test_rules_do_not_fire_inside_comments():
    assert_silent("// el.innerHTML = '<b>' + q + '</b>';\n")
    assert_silent("/* gets(buf); */\nint x = 1;\n")
    assert_silent(_php(f'// eval({_super("GET")}["x"]);'))


def test_python_comment_secret_not_flagged():
    assert_silent('# password = "Tr0ub4dor&3"\nvalue = 1\n')


# ---- realistic mixed files ----------------------------------------------

def test_clean_python_module_is_silent():
    code = (
        "import hashlib\n"
        "import json\n"
        "import subprocess\n"
        "\n"
        "\n"
        "def digest(payload: bytes) -> str:\n"
        '    """Return a SHA-256 digest."""\n'
        "    return hashlib.sha256(payload).hexdigest()\n"
        "\n"
        "\n"
        "def load(path):\n"
        "    with open(path, encoding='utf-8') as fh:\n"
        "        return json.load(fh)\n"
        "\n"
        "\n"
        "def list_files(directory):\n"
        '    return subprocess.run(["ls", directory], capture_output=True)\n'
        "\n"
        "\n"
        "def find_user(cursor, user_id):\n"
        '    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))\n'
        "    return cursor.fetchone()\n"
    )
    assert_silent(code)


def test_vulnerable_python_module_finds_each_issue():
    code = (
        "import os\n"
        "import pickle\n"
        "import yaml\n"
        "import requests\n"
        "\n"
        'ADMIN_PASSWORD = "hunter2hunter2"\n'
        "\n"
        "\n"
        "def handler(req, cursor):\n"
        "    cfg = yaml.load(req.body)\n"
        "    obj = pickle.loads(req.blob)\n"
        '    os.system("echo " + req.args["msg"])\n'
        '    cursor.execute("SELECT * FROM t WHERE n = \'" + req.args["n"] + "\'")\n'
        "    requests.get(req.args['url'], verify=False)\n"
        "    return eval(req.args['expr'])\n"
    )
    found = cwes(code)
    assert {"CWE-94", "CWE-502", "CWE-89", "CWE-200"} <= found
    assert len(scan_rules(code)) >= 6


def test_large_input_is_bounded():
    code = ("def f():\n    return 1\n" * 5000) + "\neval(user_input)\n"
    findings = scan_rules(code)
    assert isinstance(findings, list)


# ==========================================================================
# false-positive regressions
#
# Each fixture below fired on code that is correct. They are the whole reason
# the corresponding guard exists.
# ==========================================================================

def test_markup_helper_name_collision_not_flagged():
    # pytest's terminal writer exposes tw.markup(text, bold=True) -- an ANSI
    # colour helper that has nothing to do with HTML safety.
    assert_silent("tw.markup(text, bold=True)\n")
    assert_silent("writer.markup(line, red=True)\n")


def test_real_markupsafe_markup_still_flagged():
    assert "PY-MARK-SAFE" in rule_ids(
        "from markupsafe import Markup\nMarkup(user_bio)\n")
    assert "PY-MARK-SAFE" in rule_ids(
        "import markupsafe\nmarkupsafe.Markup(user_bio)\n")


def test_credential_names_are_not_credentials():
    # The value NAMES a secret (a JSON field, an HTTP header, an env var).
    assert_silent('_JSON_FILE_PRIVATE_KEY = "private_key"\n')
    assert_silent('CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"\n')
    assert_silent('XET_TOKEN_KEY = "X-Xet-Access-Token"\n')
    assert_silent('OAUTH_TOKEN = "FOOBAR"\n')
    assert_silent('NULL_TOKEN = "__null_token_value__"\n')
    assert_silent('SECRET = "application_default_credentials.json"\n')
    assert_silent('AUTH_TOKEN_TYPE = "auth-request-type/at"\n')


def test_passphrase_and_random_secrets_still_flagged():
    # An unbroken lowercase run is a passphrase, not an identifier.
    assert "PY-HARDCODED-SECRET" in rule_ids(
        'SECRET_KEY = "correcthorsebatterystaple"\n')
    assert "PY-HARDCODED-SECRET" in rule_ids('API_KEY = "AKIAIOSFODNN7EXAMPL"\n')


def test_serialization_round_trip_not_flagged():
    # loads(dumps(x)) is the deep-copy idiom: it never crosses a boundary.
    assert_silent("import pickle\nclone = pickle.loads(pickle.dumps(obj))\n")
    assert_silent("import marshal\nclone = marshal.loads(marshal.dumps(obj))\n")
    assert_silent("copy = Marshal.load(Marshal.dump(obj))\n")


def test_deserializing_external_data_still_flagged():
    assert "PY-PICKLE-LOAD" in rule_ids(
        "import pickle\nobj = pickle.loads(request.body)\n")
    assert "CWE-502" in cwes("obj = Marshal.load(params[:data])\n")


def test_cache_key_digest_not_flagged():
    assert_silent(
        "import hashlib\n"
        "def session_cache_key(url):\n"
        "    return hashlib.md5(url.encode()).hexdigest()\n")


def test_session_token_digest_still_flagged():
    assert "PY-WEAK-HASH" in rule_ids(
        "import hashlib\nsession_token = hashlib.md5(seed).hexdigest()\n")


def test_partially_escaped_html_not_flagged():
    assert_silent(
        "from django.utils.html import escape\n"
        "def view(name):\n"
        '    return HttpResponse("<h1>" + escape(name) + "</h1>")\n')
    assert_silent(
        "from django.utils.html import escape\n"
        "def view(name):\n"
        '    return HttpResponse(f"<h1>{escape(name)}</h1>")\n')


def test_one_unescaped_part_is_still_flagged():
    assert "CWE-79" in cwes(
        "from django.utils.html import escape\n"
        "def view(name, tail):\n"
        '    return HttpResponse("<h1>" + escape(name) + "</h1>" + tail)\n')


def test_literal_only_innerhtml_not_flagged():
    assert_silent("el.innerHTML = '<b>' + 'static' + '</b>';\n")


def test_innerhtml_from_escaped_dom_read_not_flagged():
    assert_silent("el.innerHTML = other.textContent;\n")
    assert_silent("a.innerHTML = b.innerText;\n")


def test_locally_defined_system_not_flagged():
    assert_silent(
        "static int system(const char *c) { return 0; }\n"
        "void f(char *cmd) { system(cmd); }\n")


def test_non_tls_verify_keyword_not_flagged():
    # pandas: `verify` here means "validate the index", not a certificate.
    assert_silent("{ x = mgr.take(indexer, axis=baxis, verify=False) }\n")


def test_tls_verify_off_on_a_transport_line_flagged():
    assert "RX-TLS-VERIFY-OFF" in rule_ids(
        "{ r = session.request(url, verify=False) }\n")


def test_eval_definition_not_flagged():
    # A method *named* eval is not a call to the builtin.
    assert_silent("function eval(expr) {\n  return expr;\n}\n")


def test_self_contained_deserialization_not_flagged():
    assert_silent(
        "byte[] b = bos.toByteArray();\n"
        "ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(b));\n")
    assert_silent('CONFIG = YAML.load(File.read("config/app.yml"))\n')


def test_deserialization_of_a_network_stream_still_flagged():
    assert "CWE-502" in cwes(
        "ObjectInputStream ois = new ObjectInputStream("
        "new ByteArrayInputStream(request.getBytes()));\n")


def test_docstrings_are_not_scanned():
    # Documenting a dangerous construct is not committing one; this module's
    # own docstrings used to raise findings against itself.
    assert_silent('"""Never write el.innerHTML = tag + q by hand."""\n\nVALUE = 1\n')


def test_long_token_does_not_backtrack_quadratically():
    # A long unbroken token (base64 blob, minified bundle, embedded data URI)
    # used to drive _SANITIZED_ASSIGN_RE into quadratic backtracking: 40 KB
    # took 27 s and 100 KB took 197 s, which MAX_SCAN_CHARS does not bound.
    # The budget is deliberately loose -- it is a guard against O(n^2), not a
    # benchmark.
    blob = "A" * 60_000
    start = time.perf_counter()
    scan_rules(f'DOC = "{blob}"\n')
    scan_rules(f"var payload = '{blob}';\n")
    assert time.perf_counter() - start < 10.0


def test_large_python_file_still_uses_the_ast_backend():
    # Truncating before parsing cut large modules mid-construct, so they fell
    # through to the regex backend and a file's verdict changed purely because
    # it crossed the size limit. The marker is a C pattern inside a string.
    marker = 'HELP = "never call gets(buf) in C"\n'
    filler = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(3000))
    straddling_literal = 'DOC = """' + ("x" * 200_000) + '"""\n'
    code = marker + filler + straddling_literal
    assert len(code) > MAX_SCAN_CHARS
    assert_silent(code)


# ==========================================================================
# added coverage
# ==========================================================================

# ---- CWE-94: shells reached through a list, includes, node ---------------

def test_shell_dash_c_list_form_flagged():
    # argv[0] is a shell, so the list form injects exactly like shell=True.
    assert "PY-SUBPROCESS-SHELL" in rule_ids(
        "import subprocess\nsubprocess.run(['sh', '-c', user_cmd])\n")
    assert "PY-SUBPROCESS-SHELL" in rule_ids(
        "import subprocess\nsubprocess.Popen(['/bin/bash', '-c', cmd])\n")
    assert "CWE-94" in cwes('cmd := exec.Command("sh", "-c", userInput)\n')
    assert "CWE-94" in cwes('new ProcessBuilder("/bin/bash", "-c", cmd).start();\n')


def test_shell_dash_c_safe_forms_not_flagged():
    assert_silent("import subprocess\nsubprocess.run(['sh', '-c', 'ls -la'])\n")
    assert_silent("import subprocess\nsubprocess.run(['ls', '-la', path])\n")
    assert_silent('cmd := exec.Command("ls", userInput)\n')


def test_php_file_inclusion():
    assert "CWE-94" in cwes(_php(f'include({_super("GET")}["page"]);'))
    assert "CWE-94" in cwes(_php(f'require_once {_super("REQUEST")}["mod"];'))


def test_php_static_include_not_flagged():
    assert_silent(_php('include("header.php");'))
    assert_silent(_php("include_once __DIR__ . '/config.php';"))


def test_node_child_process_exec():
    assert "CWE-94" in cwes("child_process.exec('ls ' + dir);\n")
    assert "CWE-94" in cwes("require('child_process').exec(cmd);\n")


def test_node_exec_file_and_literal_commands_not_flagged():
    # execFile takes an argv array and never spawns a shell.
    assert_silent("child_process.execFile('ls', [dir]);\n")
    assert_silent("child_process.exec('ls -la');\n")


def test_aliased_eval_flagged():
    assert "PY-EVAL" in rule_ids("e = eval\ne(user_input)\n")
    assert "PY-EVAL" in rule_ids(
        "import builtins\nrun = builtins.exec\nrun(payload)\n")


def test_unrelated_alias_not_flagged():
    assert_silent("e = len\ne(user_input)\n")
    assert_silent("evaluate = model.eval\nevaluate()\n")


# ---- CWE-502: more deserializers ----------------------------------------

def test_java_xmldecoder_and_snakeyaml():
    assert "CWE-502" in cwes("XMLDecoder d = new XMLDecoder(in);\n")
    assert "CWE-502" in cwes("Object o = new Yaml().load(input);\n")


def test_java_safe_yaml_constructor_not_flagged():
    assert_silent("Object o = new Yaml(new SafeConstructor()).load(input);\n")


def test_dotnet_type_name_handling():
    assert "CWE-502" in cwes(
        "var s = new JsonSerializerSettings { TypeNameHandling = TypeNameHandling.All };\n")


def test_dotnet_type_name_handling_none_not_flagged():
    assert_silent(
        "var s = new JsonSerializerSettings { TypeNameHandling = TypeNameHandling.None };\n")


def test_explicit_unsafe_yaml_loaders_flagged():
    assert "PY-YAML-LOAD" in rule_ids("import yaml\ncfg = yaml.unsafe_load(s)\n")
    assert "PY-YAML-LOAD" in rule_ids("import yaml\ncfg = yaml.full_load(s)\n")


# ---- CWE-79: JSP, jQuery, insertAdjacentHTML, Jinja ----------------------

def test_jsp_xss():
    assert "CWE-79" in cwes('out.println(request.getParameter("name"));\n')


def test_jsp_escaped_or_static_output_not_flagged():
    assert_silent('out.println(escapeHtml(request.getParameter("name")));\n')
    assert_silent('out.println("static text");\n')
    assert_silent("out.println(total);\n")


def test_jquery_and_insert_adjacent_html():
    assert "CWE-79" in cwes("$('#out').html(userInput);\n")
    assert "CWE-79" in cwes("el.insertAdjacentHTML('beforeend', userHtml);\n")


def test_jquery_and_insert_adjacent_safe_forms_not_flagged():
    assert_silent('$("#out").html("<b>x</b>");\n')
    assert_silent("$('#out').html();\n")           # the getter takes no argument
    assert_silent("el.insertAdjacentHTML('beforeend', DOMPurify.sanitize(x));\n")


def test_php_escaped_echo_not_flagged():
    assert_silent(_php(f'echo htmlspecialchars({_super("GET")}["n"]);'))


def test_jinja_autoescape_disabled():
    assert "PY-JINJA-AUTOESCAPE" in rule_ids(
        "import jinja2\nenv = jinja2.Environment(autoescape=False)\n")


def test_jinja_autoescape_enabled_not_flagged():
    assert_silent("import jinja2\nenv = jinja2.Environment(autoescape=True)\n")
    assert_silent("import jinja2\nenv = jinja2.Environment(loader=loader)\n")


# ---- CWE-200: JWT and TLS hostname / cert checks -------------------------

def test_jwt_signature_verification_disabled():
    assert "PY-JWT-NO-VERIFY" in rule_ids(
        "import jwt\nclaims = jwt.decode(tok, options={'verify_signature': False})\n")
    assert "CWE-200" in cwes("const p = jwt.verify(t, k, { verify_signature: false });\n")


def test_verified_jwt_and_unrelated_decode_not_flagged():
    assert_silent("import jwt\nclaims = jwt.decode(tok, key, algorithms=['HS256'])\n")
    assert_silent("import base64\nraw = base64.decode(blob)\n")


def test_tls_hostname_and_cert_checks_disabled():
    assert "PY-TLS-VERIFY-OFF" in rule_ids(
        "import ssl\nctx = ssl.create_default_context()\nctx.check_hostname = False\n")
    assert "PY-TLS-VERIFY-OFF" in rule_ids(
        "import ssl\ns = ssl.wrap_socket(sock, cert_reqs=ssl.CERT_NONE)\n")


def test_tls_defaults_not_flagged():
    assert_silent(
        "import ssl\nctx = ssl.create_default_context()\nctx.check_hostname = True\n")
    assert_silent(
        "import ssl\ns = ssl.wrap_socket(sock, cert_reqs=ssl.CERT_REQUIRED)\n")


# ---- MEMORY-OOB: C buffer writes (W3) ------------------------------------

def test_c_unbounded_write_into_a_fixed_buffer():
    assert "RX-C-STRCPY-FIXED" in rule_ids(
        "void f(char *in) {\n  char buf[8];\n  strcpy(buf, in);\n}\n")
    assert "RX-C-STRCPY-FIXED" in rule_ids(
        "void f(char *in) {\n  char buf[8];\n  strcat(buf, in);\n}\n")
    assert "RX-C-SPRINTF-FIXED" in rule_ids(
        'void f(char *in) {\n  char buf[16];\n  sprintf(buf, "hi %s", in);\n}\n')
    assert "RX-C-MEMCPY-FIXED" in rule_ids(
        "void f(char *s, int n) {\n  char buf[16];\n  memcpy(buf, s, n);\n}\n")


def test_c_bounded_writes_not_flagged():
    assert_silent(
        "void f(char *in) {\n  char buf[8];\n  strncpy(buf, in, sizeof(buf) - 1);\n}\n")
    assert_silent(
        'void f(char *in) {\n  char buf[16];\n  snprintf(buf, sizeof(buf), "%s", in);\n}\n')
    assert_silent("void f(char *in) {\n  char buf[8];\n  strlcpy(buf, in, sizeof buf);\n}\n")
    assert_silent('void f(void) {\n  char buf[8];\n  strcpy(buf, "ok");\n}\n')
    assert_silent("void f(char *s) {\n  char buf[16];\n  memcpy(buf, s, sizeof(buf));\n}\n")
    assert_silent("void f(char *s) {\n  char buf[16];\n  memcpy(buf, s, 4);\n}\n")
    assert_silent('void f(int n) {\n  char buf[16];\n  sprintf(buf, "%d", n);\n}\n')


def test_c_write_into_an_unbounded_destination_not_flagged():
    # No declared size in the snippet, so there is no known bound to exceed.
    # Firing here is what makes a C buffer rule unusable in practice.
    assert_silent("void f(char *in, char *dst) {\n  strcpy(dst, in);\n}\n")


def test_c_alloca_and_unbounded_scanf():
    assert MEMORY_OOB in cwes("void f(int n) {\n  char *p = alloca(n);\n}\n")
    assert MEMORY_OOB in cwes(
        'void f(char *in) {\n  char b[8];\n  sscanf(in, "%s", b);\n}\n')


def test_c_fixed_alloca_and_bounded_scanf_not_flagged():
    assert_silent("void f(void) {\n  char *p = alloca(64);\n}\n")
    assert_silent('void f(void) {\n  char b[8];\n  scanf("%7s", b);\n}\n')


# ---- CWE-476: NULL pointer dereference -----------------------------------

def test_c_allocation_dereferenced_without_a_null_check():
    assert "RX-C-NULL-DEREF" in rule_ids(
        "void f(int n) {\n  struct s *p = malloc(n);\n  p->x = 1;\n}\n")
    assert "CWE-476" in cwes(
        "void f(int n) {\n  char *p = calloc(n, 1);\n  p[0] = 'a';\n}\n")
    assert "CWE-476" in cwes(
        "void f(char *s) {\n  char *p = strdup(s);\n  *p = 'x';\n}\n")


def test_c_checked_allocation_not_flagged():
    assert_silent("void f(int n) {\n  char *p = malloc(n);\n  if (!p) return;\n"
                  "  p[0] = 'a';\n}\n")
    assert_silent("void f(int n) {\n  char *p = malloc(n);\n  if (p == NULL)\n"
                  "    return;\n  p[0] = 'a';\n}\n")
    assert_silent("void f(int n) {\n  char *p = malloc(n);\n  if (NULL == p) return;\n"
                  "  p->x = 1;\n}\n")
    assert_silent("void f(int n) {\n  char *p = kmalloc(n);\n  if (unlikely(!p))\n"
                  "    return;\n  p->x = 1;\n}\n")


def test_c_allocation_never_dereferenced_not_flagged():
    assert_silent("void f(int n, char *s) {\n  char *p = malloc(n);\n"
                  "  memcpy(p, s, n);\n}\n")


# ---- diff hunks ----------------------------------------------------------

def test_diff_hunk_is_repaired_before_parsing():
    # A hunk lifted from a PR has no enclosing def, so it does not parse. The
    # cross-statement SQL analysis only exists on the AST path.
    hunk = ('        query = "SELECT * FROM t WHERE id = \'" + uid + "\'"\n'
            "        cur.execute(query)\n")
    findings = [f for f in scan_rules(hunk) if f["cwe"] == "CWE-89"]
    assert findings and findings[0]["rule_id"] == "PY-SQL-CONCAT"


def test_repaired_hunk_keeps_original_line_numbers():
    hunk = "\n" * 40 + "        return eval(payload)\n"
    assert [f["line"] for f in scan_rules(hunk)] == [41]


def test_repair_does_not_hijack_other_languages():
    # A C hunk must keep its C rules rather than be read as indented Python.
    assert "RX-C-GETS" in rule_ids("        char buf[8];\n        gets(buf);\n")
    assert "RX-C-STRCPY-FIXED" in rule_ids(
        "        char buf[8];\n        strcpy(buf, in);\n")


# ---- rule table contract -------------------------------------------------

# Every mode string that _regex_findings dispatches on. A rule carrying a mode
# outside this set is dead code that no test would otherwise notice.
HANDLED_MODES = {
    "match", "dynamic-arg", "dynamic-rhs", "c-system", "xss-line", "jsp-xss",
    "deserialize", "tls-line", "c-fixed-copy", "c-fixed-format", "c-fixed-length",
    "shell-true", "unsafe-yaml", "sql-arg", "sql-interp", "weak-hash-line",
}


def test_regex_rule_table_is_well_formed():
    ids = [rule[0] for rule in _REGEX_RULES]
    assert len(ids) == len(set(ids)), "duplicate rule_id"
    for rule_id, cwe, confidence, _pattern, mode, _py_covered in _REGEX_RULES:
        assert cwe in CLASSES, rule_id
        assert mode in HANDLED_MODES, f"{rule_id}: mode {mode!r} has no branch"
        # Regex rules see a line, not a program: none may outrank the AST pass.
        assert 0.5 <= confidence <= 0.95, rule_id
