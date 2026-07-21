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
"""

from mlscan.labels import CLASSES
from mlscan.rules import scan_rules


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
