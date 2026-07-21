"""Deterministic rules for unambiguous vulnerabilities.

The ML classifier in :mod:`mlscan.scanner` is a bag-of-n-grams model trained on
a corpus that is ~92% C, so it under-fires on textbook one-line defects in other
languages (``pickle.loads``, ``yaml.load``, ``shell=True``). This module is the
complement: a small set of hand-written rules that fire only on constructs that
are vulnerable *by definition*, and stay silent otherwise.

The design bias throughout is **precision over recall**. A false positive costs
a reviewer's trust; a miss is simply left to the model. Every rule therefore
requires positive evidence that the dangerous value is attacker-controllable
(non-literal), and every rule has an explicit safe counterpart it must not fire
on -- ``ast.literal_eval``, ``yaml.safe_load``, ``hashlib.sha256``,
``subprocess.run([...])``, parameterised queries, and so on.

Two analysis backends:

* **Python** is parsed with :mod:`ast`. Import aliases are resolved and
  single-assignment string constants are folded, which is what makes it
  possible to separate ``cursor.execute("... %s", (uid,))`` (safe) from
  ``cursor.execute("... %s" % uid)`` (CWE-89).
* **Everything else** -- C, PHP, Java, JavaScript, or a truncated snippet that
  does not parse -- falls back to bounded regexes run over a copy of the source
  with comments blanked out, so a rule never fires on commented-out code.

>>> from mlscan.rules import scan_rules
>>> [f["rule_id"] for f in scan_rules("import yaml\\nyaml.load(data)")]
['PY-YAML-LOAD']
>>> scan_rules("import yaml\\nyaml.safe_load(data)")
[]
"""

from __future__ import annotations

import ast
import bisect
import re
from collections import Counter
from dataclasses import dataclass, field

from mlscan.labels import MEMORY_OOB, TAXONOMY

# Very large inputs are truncated: the regex pass is linear but there is no
# point scanning a minified bundle, and line numbers past the cut are useless.
MAX_SCAN_CHARS = 200_000

# Evidence is a single trimmed source line, for display next to the finding.
MAX_EVIDENCE_CHARS = 160


# --------------------------------------------------------------------------
# shared vocabulary
# --------------------------------------------------------------------------

# A statement is "SQL" when its literal fragments read like a statement, not
# merely when they contain the word SELECT. Quantifiers are bounded so the
# regex cannot backtrack pathologically on a long minified line.
_SQL_RE = re.compile(
    r"(?is)\b(?:"
    r"select\b.{0,400}?\bfrom\b"
    r"|insert\s+into\b"
    r"|update\b.{0,200}?\bset\b"
    r"|delete\s+from\b"
    r"|(?:drop|create|alter|truncate)\s+table\b"
    r"|union\s+(?:all\s+)?select\b"
    r")"
)

_HTML_TAG_RE = re.compile(
    r"(?i)<\s*(?:script|iframe|img|svg|div|span|p|h[1-6]|a|body|table|tr|td|"
    r"li|ul|ol|input|form|button|br|b|i|strong|em|pre|code)\b|</\s*[a-z][a-z0-9]*\s*>"
)

# Identifier words that make a hash call a *security* hash rather than a
# checksum. Deliberately excludes bare "key" (cache_key = md5(url) is common
# and harmless) -- see ``_is_security_context``.
_SECURITY_WORDS = {
    "password", "passwords", "passwd", "pwd", "passphrase", "secret", "secrets",
    "token", "tokens", "credential", "credentials", "cred", "creds", "auth",
    "authenticate", "authentication", "signature", "hmac", "salt", "apikey",
    "login", "otp", "csrf", "nonce", "cookie", "session",
}

# Words that make a *name* look like it holds a secret literal.
_SECRET_NAME_WORDS = {
    "password", "passwd", "pwd", "passphrase", "secret", "token", "apikey",
    "credential", "credentials",
}
# ("key" alone is far too generic; it only counts next to one of these.)
_KEY_QUALIFIERS = {"api", "access", "private", "secret", "signing", "encryption",
                   "consumer", "license", "auth"}
# Names ending in one of these describe a secret rather than being one.
_NON_SECRET_SUFFIXES = {
    "name", "field", "label", "url", "uri", "path", "header", "env", "prefix",
    "suffix", "pattern", "regex", "placeholder", "hint", "msg", "message",
    "error", "type", "column", "col", "param", "file", "dir", "format",
    "template", "prompt", "desc", "description", "doc", "docs", "title", "text",
    "prop", "attr", "arg", "flag", "enabled", "required", "length", "len",
}

_PLACEHOLDER_RE = re.compile(
    r"(?i)^(?:"
    r"x+|\*+|\.+|-+|_+|0+|1234\d*|"
    r"none|null|nil|nan|true|false|todo|tbd|n/?a|unset|empty|default|hidden|"
    r"redacted|change_?me|change_?it|your[-_]?\w*|my[-_]?\w*|some[-_]?\w*|"
    r"test\w*|dummy\w*|fake\w*|example\w*|sample\w*|placeholder\w*|"
    r"password\d*|passwd\d*|secret\d*|token\d*|api_?key\d*|credentials?"
    r")$"
)

_IDENT_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")


def _words(identifier: str) -> list[str]:
    """Split ``api_keyValue`` into ``['api', 'key', 'Value']``, lowercased."""
    return [w.lower() for w in _IDENT_WORD_RE.findall(identifier or "")]


def _is_security_context(names) -> bool:
    """True when any identifier in ``names`` implies a security use."""
    bag: set[str] = set()
    for n in names:
        bag.update(_words(n))
    if bag & _SECURITY_WORDS:
        return True
    return "key" in bag and bool(bag & _KEY_QUALIFIERS)


def _looks_like_secret_name(name: str) -> bool:
    words = _words(name)
    if not words:
        return False
    if words[-1] in _NON_SECRET_SUFFIXES:
        return False
    bag = set(words)
    if bag & _SECRET_NAME_WORDS:
        return True
    return "key" in bag and bool(bag & _KEY_QUALIFIERS)


def _looks_like_secret_value(value) -> bool:
    """Reject placeholders, URLs, format strings and anything too short."""
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not 6 <= len(v) <= 200:
        return False
    if any(c.isspace() for c in v):
        return False
    if _PLACEHOLDER_RE.match(v):
        return False
    if v.lower().startswith(("http://", "https://", "/", "./", "../", "~/")):
        return False
    # Format placeholders / interpolation / templating are not literal secrets.
    if any(c in v for c in "{}%$<>"):
        return False
    if len(set(v)) < 3:
        return False
    # A light character-class check, the cheap stand-in for entropy: real
    # credentials mix cases/digits/symbols. Without it, ordinary config words
    # (`{"password": "required"}`) read as secrets. A long all-lowercase value
    # still passes, so passphrases are not lost.
    has_variety = (any(c.isdigit() for c in v) or any(c.isupper() for c in v)
                   or any(not c.isalnum() for c in v))
    return has_variety or len(v) >= 16


# --------------------------------------------------------------------------
# finding construction
# --------------------------------------------------------------------------

def _trim(text: str) -> str:
    one_line = " ".join(str(text).split())
    if len(one_line) > MAX_EVIDENCE_CHARS:
        one_line = one_line[: MAX_EVIDENCE_CHARS - 3] + "..."
    return one_line


def _finding(cwe: str, rule_id: str, confidence: float, line, evidence: str) -> dict:
    """Build one finding dict; ``name`` is taken from the shared taxonomy."""
    name = TAXONOMY[cwe][0] if cwe in TAXONOMY else cwe
    return {
        "cwe": cwe,
        "name": name,
        "confidence": round(float(confidence), 3),
        "line": int(line) if line else None,
        "rule_id": rule_id,
        "evidence": _trim(evidence),
    }


def _src_line(lines: list[str], lineno) -> str:
    if lineno and 1 <= lineno <= len(lines):
        return lines[lineno - 1]
    return ""


# --------------------------------------------------------------------------
# Python AST backend
# --------------------------------------------------------------------------

@dataclass
class _PyCtx:
    """Everything the Python rules need beyond the tree itself."""

    lines: list[str]
    aliases: dict[str, str] = field(default_factory=dict)
    const_names: set[str] = field(default_factory=set)
    const_str: dict[str, str] = field(default_factory=dict)
    sql_vars: dict[str, tuple[int, str]] = field(default_factory=dict)


def _dotted(node) -> str | None:
    """``os.path.join`` for an Attribute/Name chain, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _attr_tail(node) -> str:
    """Last component of a call target, even when the base is an expression."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _canonical(node, ctx: _PyCtx) -> str | None:
    """Dotted name with import aliases resolved (``y.load`` -> ``yaml.load``)."""
    dotted = _dotted(node)
    if dotted is None:
        return None
    head, sep, rest = dotted.partition(".")
    target = ctx.aliases.get(head)
    if target:
        return f"{target}.{rest}" if sep else target
    return dotted


def _collect_aliases(tree) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:  # `import os.path` binds `os`
                    aliases[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                if a.name != "*":
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def _target_names(node) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in node.elts:
            out.extend(_target_names(elt))
        return out
    return []


def _is_static(node, ctx: _PyCtx | None = None) -> bool:
    """True when the expression is built purely from literal constants."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_static(e, ctx) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_static(k, ctx) for k in node.keys) and all(
            _is_static(v, ctx) for v in node.values
        )
    if isinstance(node, ast.BinOp):
        return _is_static(node.left, ctx) and _is_static(node.right, ctx)
    if isinstance(node, ast.UnaryOp):
        return _is_static(node.operand, ctx)
    if isinstance(node, ast.JoinedStr):
        return all(isinstance(v, ast.Constant) for v in node.values)
    if isinstance(node, ast.Name):
        return ctx is not None and node.id in ctx.const_names
    return False


def _collect_constants(tree) -> tuple[set[str], dict[str, str]]:
    """Names bound exactly once to a literal, plus their string values.

    Conservative on purpose: a name that is also a parameter, loop variable,
    ``with`` target, function name or augmented-assignment target is never
    treated as constant.
    """
    static_vals: dict[str, ast.AST] = {}
    rebound: set[str] = set()
    bindings: Counter = Counter()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _target_names(target):
                    bindings[name] += 1
                    if _is_static(node.value):
                        static_vals.setdefault(name, node.value)
                    else:
                        rebound.add(name)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            for name in _target_names(node.target):
                bindings[name] += 1
                if _is_static(node.value):
                    static_vals.setdefault(name, node.value)
                else:
                    rebound.add(name)
        elif isinstance(node, ast.AugAssign):
            rebound.update(_target_names(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            rebound.update(_target_names(node.target))
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            rebound.update(_target_names(node.optional_vars))
        elif isinstance(node, ast.NamedExpr):
            rebound.update(_target_names(node.target))
        elif isinstance(node, ast.arg):
            rebound.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rebound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            rebound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            rebound.update(node.names)

    const_names = {n for n in static_vals if n not in rebound and bindings[n] == 1}
    const_str = {
        n: static_vals[n].value
        for n in const_names
        if isinstance(static_vals[n], ast.Constant)
        and isinstance(static_vals[n].value, str)
    }
    return const_names, const_str


def _string_parts(node, ctx: _PyCtx, depth: int = 0) -> tuple[str, bool]:
    """``(literal_text, has_dynamic_part)`` for a string-building expression.

    The literal text is what a SQL/HTML pattern is matched against; the flag is
    what separates ``"... %s" % uid`` from ``"... %s"``.
    """
    if depth > 12:
        return "", True
    if isinstance(node, ast.Constant):
        return (node.value, False) if isinstance(node.value, str) else ("", False)
    if isinstance(node, ast.Name):
        if node.id in ctx.const_str:
            return ctx.const_str[node.id], False
        return "", True
    if isinstance(node, ast.JoinedStr):
        text, dynamic = "", False
        for value in node.values:
            part, part_dyn = _string_parts(value, ctx, depth + 1)
            text += part
            dynamic = dynamic or part_dyn
        return text, dynamic
    if isinstance(node, ast.FormattedValue):
        return "", True
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            left, ldyn = _string_parts(node.left, ctx, depth + 1)
            right, rdyn = _string_parts(node.right, ctx, depth + 1)
            return left + right, ldyn or rdyn
        if isinstance(node.op, ast.Mod):  # "..." % params
            left, ldyn = _string_parts(node.left, ctx, depth + 1)
            return left, ldyn or not _is_static(node.right, ctx)
        return "", True
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "format":
            text, dynamic = _string_parts(func.value, ctx, depth + 1)
            return text, dynamic or bool(node.args) or bool(node.keywords)
        if isinstance(func, ast.Attribute) and func.attr == "join":
            text, _ = _string_parts(func.value, ctx, depth + 1)
            return text, True
        return "", True
    return "", True


def _dynamic_sql(node, ctx: _PyCtx) -> str | None:
    """Literal text of a *dynamically built* SQL string, else None."""
    text, dynamic = _string_parts(node, ctx)
    if dynamic and text and _SQL_RE.search(text):
        return text
    return None


def _collect_sql_vars(tree, ctx: _PyCtx) -> dict[str, tuple[int, str]]:
    """Variables that hold a SQL string assembled from non-literal pieces."""
    sql_vars: dict[str, tuple[int, str]] = {}
    seen_text: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            name = node.targets[0].id
            text, dynamic = _string_parts(node.value, ctx)
            seen_text[name] = text
            if dynamic and text and _SQL_RE.search(text):
                sql_vars[name] = (node.lineno, text)
            else:
                sql_vars.pop(name, None)
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.op, ast.Add)
        ):
            name = node.target.id
            text, dynamic = _string_parts(node.value, ctx)
            combined = seen_text.get(name, "") + text
            seen_text[name] = combined
            if (dynamic or name in sql_vars) and _SQL_RE.search(combined):
                sql_vars[name] = (node.lineno, combined)
    return sql_vars


def _link_parents(tree) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._mlscan_parent = parent  # noqa: SLF001


def _context_names(node) -> list[str]:
    """Identifiers in the enclosing statement plus the enclosing def name."""
    names: list[str] = []
    stmt = node
    while stmt is not None and not isinstance(stmt, ast.stmt):
        stmt = getattr(stmt, "_mlscan_parent", None)
    if stmt is not None:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
            elif isinstance(sub, ast.Attribute):
                names.append(sub.attr)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                names.append(sub.value)
    scope = node
    while scope is not None:
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(scope.name)
            break
        scope = getattr(scope, "_mlscan_parent", None)
    return names


def _kwarg(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_false(node) -> bool:
    return isinstance(node, ast.Constant) and node.value in (False, 0)


def _is_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


# --- individual Python rules ---------------------------------------------

_EVAL_NAMES = {"eval", "exec", "builtins.eval", "builtins.exec"}
_SHELL_ALWAYS = {"os.system", "os.popen", "subprocess.getoutput",
                 "subprocess.getstatusoutput", "commands.getoutput",
                 "commands.getstatusoutput"}
_SUBPROCESS_CALLS = {"subprocess.run", "subprocess.call", "subprocess.Popen",
                     "subprocess.check_output", "subprocess.check_call"}
_PICKLE_MODULES = {"pickle", "cPickle", "_pickle", "dill", "cloudpickle", "marshal"}
_SAFE_YAML_LOADERS = {"SafeLoader", "CSafeLoader", "BaseLoader", "CBaseLoader"}
_WEAK_HASHES = {"md5", "sha1", "md4", "sha"}
_ESCAPE_FUNCS = {"escape", "escape_html", "conditional_escape", "clean",
                 "sanitize", "bleach", "quote", "quoteattr", "striptags"}
_SQL_SINKS = {"execute", "executemany", "executescript", "execute_sql",
              "executequery", "executeupdate", "exec_driver_sql", "raw",
              "query", "read_sql", "read_sql_query", "execute_string"}
# Deliberately narrow: `f.write("<h1>" + t)` is a template generator, not XSS.
_HTML_SINKS = {"httpresponse", "htmlresponse", "make_response", "response"}
_HTTP_VERBS = {"get", "post", "put", "delete", "patch", "head", "options",
               "request", "session", "client", "asyncclient"}


def _py_rule_eval(node: ast.Call, ctx: _PyCtx, out: list) -> None:
    """CWE-94: eval()/exec() on anything that is not a literal."""
    canonical = _canonical(node.func, ctx)
    if canonical not in _EVAL_NAMES or not node.args:
        return
    if _is_static(node.args[0], ctx):
        return
    out.append(_finding("CWE-94", "PY-EVAL", 0.95, node.lineno,
                        _src_line(ctx.lines, node.lineno)))


def _py_rule_shell(node: ast.Call, ctx: _PyCtx, out: list) -> None:
    """CWE-94: a shell command assembled from non-literal input."""
    canonical = _canonical(node.func, ctx)
    if canonical is None:
        return
    if canonical in _SHELL_ALWAYS:
        if node.args and not _is_static(node.args[0], ctx):
            out.append(_finding("CWE-94", "PY-OS-COMMAND", 0.92, node.lineno,
                                _src_line(ctx.lines, node.lineno)))
        return
    if canonical in _SUBPROCESS_CALLS and _is_true(_kwarg(node, "shell")):
        # shell=True with a hardcoded command is not injectable.
        if node.args and not _is_static(node.args[0], ctx):
            out.append(_finding("CWE-94", "PY-SUBPROCESS-SHELL", 0.92, node.lineno,
                                _src_line(ctx.lines, node.lineno)))


def _py_rule_deserialize(node: ast.Call, ctx: _PyCtx, out: list) -> None:
    """CWE-502: pickle / marshal / unsafe yaml.load."""
    canonical = _canonical(node.func, ctx)
    if canonical is None:
        return
    module, _, func = canonical.rpartition(".")
    root = module.split(".")[0] if module else ""

    if root in _PICKLE_MODULES and func in {"load", "loads"}:
        if node.args and _is_static(node.args[0], ctx):
            return  # deserialising a literal payload is not attacker-controlled
        conf = 0.9 if func == "loads" else 0.85
        out.append(_finding("CWE-502", "PY-PICKLE-LOAD", conf, node.lineno,
                            _src_line(ctx.lines, node.lineno)))
        return

    if canonical in {"jsonpickle.decode", "shelve.open"}:
        out.append(_finding("CWE-502", "PY-PICKLE-LOAD", 0.8, node.lineno,
                            _src_line(ctx.lines, node.lineno)))
        return

    if canonical in {"yaml.load", "ruamel.yaml.load"}:
        loader = _kwarg(node, "Loader")
        if loader is None and len(node.args) >= 2:
            loader = node.args[1]
        if loader is not None and _attr_tail(loader) in _SAFE_YAML_LOADERS:
            return  # yaml.load(x, Loader=yaml.SafeLoader) is safe
        out.append(_finding("CWE-502", "PY-YAML-LOAD", 0.9, node.lineno,
                            _src_line(ctx.lines, node.lineno)))


def _py_rule_sql(node: ast.Call, ctx: _PyCtx, out: list) -> None:
    """CWE-89: a concatenated / interpolated query handed to an executor."""
    if _attr_tail(node.func).lower() not in _SQL_SINKS:
        return
    candidates = list(node.args)
    for arg in list(node.args):
        if isinstance(arg, ast.Call):  # sqlalchemy: execute(text("..." + x))
            candidates.extend(arg.args)

    for arg in candidates:
        text = _dynamic_sql(arg, ctx)
        if text:
            out.append(_finding("CWE-89", "PY-SQL-CONCAT", 0.93, node.lineno,
                                _src_line(ctx.lines, node.lineno)))
            return
        if isinstance(arg, ast.Name) and arg.id in ctx.sql_vars:
            assign_line, _ = ctx.sql_vars[arg.id]
            evidence = (f"{_src_line(ctx.lines, assign_line).strip()} -> "
                        f"{_src_line(ctx.lines, node.lineno).strip()}")
            out.append(_finding("CWE-89", "PY-SQL-CONCAT", 0.9, node.lineno, evidence))
            return


def _py_rule_tls(node: ast.Call, ctx: _PyCtx, out: list) -> None:
    """CWE-200: TLS certificate verification disabled."""
    canonical = _canonical(node.func, ctx)
    if canonical in {"ssl._create_unverified_context", "ssl._https_verify_certificates"}:
        out.append(_finding("CWE-200", "PY-TLS-VERIFY-OFF", 0.9, node.lineno,
                            _src_line(ctx.lines, node.lineno)))
        return
    verify = _kwarg(node, "verify")
    if verify is None or not _is_false(verify):
        return
    root = (canonical or "").split(".")[0]
    if _attr_tail(node.func).lower() in _HTTP_VERBS or root in {
        "requests", "httpx", "aiohttp", "urllib3",
    }:
        out.append(_finding("CWE-200", "PY-TLS-VERIFY-OFF", 0.9, node.lineno,
                            _src_line(ctx.lines, node.lineno)))


def _py_rule_weak_hash(node: ast.Call, ctx: _PyCtx, out: list) -> None:
    """CWE-200: MD5/SHA-1 in a security context (not as a checksum)."""
    canonical = _canonical(node.func, ctx)
    if canonical is None:
        return
    module, _, func = canonical.rpartition(".")
    algo = func.lower()
    if module == "hashlib" and algo == "new":
        if not node.args or not isinstance(node.args[0], ast.Constant):
            return
        algo = str(node.args[0].value).lower().replace("-", "")
    elif module != "hashlib" or algo not in _WEAK_HASHES:
        return
    if algo not in _WEAK_HASHES:
        return
    if _is_false(_kwarg(node, "usedforsecurity")):
        return  # explicitly declared non-security use
    if not _is_security_context(_context_names(node)):
        return
    out.append(_finding("CWE-200", "PY-WEAK-HASH", 0.85, node.lineno,
                        _src_line(ctx.lines, node.lineno)))


def _is_escaped(node) -> bool:
    return isinstance(node, ast.Call) and _attr_tail(node.func).lower() in _ESCAPE_FUNCS


def _py_rule_xss(node: ast.Call, ctx: _PyCtx, out: list) -> None:
    """CWE-79: untrusted data rendered as HTML without escaping."""
    canonical = _canonical(node.func, ctx)
    tail = _attr_tail(node.func)
    lower_tail = tail.lower()

    if lower_tail in {"render_template_string", "mark_safe", "markup"} and node.args:
        arg = node.args[0]
        if _is_static(arg, ctx) or _is_escaped(arg):
            return
        rule = "PY-TEMPLATE-INJECTION" if lower_tail == "render_template_string" \
            else "PY-MARK-SAFE"
        out.append(_finding("CWE-79", rule, 0.88, node.lineno,
                            _src_line(ctx.lines, node.lineno)))
        return

    if lower_tail in _HTML_SINKS or (canonical or "").startswith("flask."):
        for arg in node.args:
            if _is_escaped(arg):
                continue
            text, dynamic = _string_parts(arg, ctx)
            if dynamic and text and _HTML_TAG_RE.search(text):
                out.append(_finding("CWE-79", "PY-HTML-CONCAT", 0.85, node.lineno,
                                    _src_line(ctx.lines, node.lineno)))
                return


def _py_rule_secrets(tree, ctx: _PyCtx, out: list) -> None:
    """CWE-200: a credential written straight into the source."""

    def report(name, value, lineno):
        if _looks_like_secret_name(name) and _looks_like_secret_value(value):
            if value.strip().lower() == name.strip().lower():
                return
            out.append(_finding("CWE-200", "PY-HARDCODED-SECRET", 0.85, lineno,
                                _src_line(ctx.lines, lineno)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                names = _target_names(target)
                if isinstance(target, ast.Attribute):
                    names = [target.attr]
                for name in names:
                    report(name, node.value.value, node.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
            name = _attr_tail(node.target)
            report(name, node.value.value, node.lineno)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and isinstance(kw.value, ast.Constant):
                    report(kw.arg, kw.value.value, node.lineno)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                        and isinstance(value, ast.Constant)):
                    report(key.value, value.value, getattr(key, "lineno", node.lineno))


_CALL_RULES = (
    _py_rule_eval,
    _py_rule_shell,
    _py_rule_deserialize,
    _py_rule_sql,
    _py_rule_tls,
    _py_rule_weak_hash,
    _py_rule_xss,
)


def _try_parse(code: str):
    try:
        return ast.parse(code)
    except Exception:  # SyntaxError, ValueError, RecursionError, MemoryError...
        return None


def _python_findings(tree, lines: list[str]) -> list[dict]:
    const_names, const_str = _collect_constants(tree)
    ctx = _PyCtx(lines=lines, aliases=_collect_aliases(tree),
                 const_names=const_names, const_str=const_str)
    ctx.sql_vars = _collect_sql_vars(tree, ctx)
    _link_parents(tree)

    out: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for rule in _CALL_RULES:
                rule(node, ctx, out)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            # ctx.verify_mode = ssl.CERT_NONE
            if _canonical(node.value, ctx) == "ssl.CERT_NONE":
                out.append(_finding("CWE-200", "PY-TLS-VERIFY-OFF", 0.85, node.lineno,
                                    _src_line(lines, node.lineno)))
    _py_rule_secrets(tree, ctx, out)
    return out


# --------------------------------------------------------------------------
# regex backend (non-Python, or Python that does not parse)
# --------------------------------------------------------------------------

# Anything that can start a string or a comment in the languages we see.
_MASK_SCAN_RE = re.compile(r"""//|/\*|['"`#]""")

# A backtick further than this from its partner is not a template literal, just
# a stray character (prose, a regex class, a Markdown fence). Guessing wrong
# here used to desynchronise the whole scan and leak comments back in.
_MAX_TEMPLATE_SPAN = 400


def _mask_comments(code: str) -> str:
    """Blank out comments, preserving offsets so line numbers stay valid.

    String literals are deliberately left intact -- the SQL and HTML rules
    match *inside* strings. Handles ``//``, ``/* */`` and ``#`` comments,
    Python triple-quoted strings, and JS template literals, and skips quoted
    regions so ``"http://x"`` is not mistaken for a comment.
    """
    out = list(code)
    n = len(code)
    i = 0
    while i < n:
        match = _MASK_SCAN_RE.search(code, i)
        if match is None:
            break
        i = match.start()
        token = match.group(0)

        if token in ("'", '"'):
            triple = code[i:i + 3]
            if triple in ('"""', "'''"):
                end = code.find(triple, i + 3)
                i = n if end == -1 else end + 3
                continue
            j = i + 1
            while j < n:
                ch = code[j]
                if ch == "\\":
                    j += 2
                    continue
                j += 1
                if ch == token or ch == "\n":  # unterminated literals end at EOL
                    break
            i = j
        elif token == "`":
            end = code.find("`", i + 1)
            i = i + 1 if end == -1 or end - i > _MAX_TEMPLATE_SPAN else end + 1
        elif token == "/*":
            end = code.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out[i:end] = [c if c == "\n" else " " for c in code[i:end]]
            i = end
        else:  # "//" or "#" -- to end of line
            end = code.find("\n", i)
            end = n if end == -1 else end
            out[i:end] = [" "] * (end - i)
            i = end
    return "".join(out)


_NEWLINE_RE = re.compile(r"\n")


def _line_starts(code: str) -> list[int]:
    return [0] + [m.end() for m in _NEWLINE_RE.finditer(code)]


def _line_of(offset: int, starts: list[int]) -> int:
    return bisect.bisect_right(starts, offset)


def _arg_region(code: str, paren: int, limit: int = 400) -> str:
    """Text inside the call parenthesis at ``paren`` (bounded, paren-aware)."""
    depth, i = 0, paren
    end = min(len(code), paren + limit)
    chunks: list[str] = []
    while i < end:
        ch = code[i]
        if ch == "(":
            depth += 1
            if depth == 1:
                i += 1
                continue
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        chunks.append(ch)
        i += 1
    return "".join(chunks)


_QUOTED_ONLY_RE = re.compile(
    r"""^(?:"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`[^`$\n]*`)$"""
)


def _looks_dynamic(argument: str) -> bool:
    """True unless the argument region is a single literal string/number."""
    arg = argument.strip()
    if not arg:
        return False
    if _QUOTED_ONLY_RE.match(arg):
        return False
    if re.fullmatch(r"[-+]?[0-9.]+[fFlLuU]*", arg):
        return False
    return bool(re.search(r"[A-Za-z_$]", arg))


# (rule_id, cwe, confidence, pattern, mode, skip_when_python_parsed)
#   mode "match"       -> the pattern alone is the evidence
#   mode "dynamic-arg" -> also require the call argument to be non-literal
_REGEX_RULES = (
    # ---- CWE-94: code / command injection ----
    ("RX-PHP-EVAL", "CWE-94", 0.95,
     re.compile(r"\beval\s*\(\s*\$"), "match", False),
    ("RX-PHP-EXEC", "CWE-94", 0.9,
     re.compile(r"\b(?:system|shell_exec|passthru|proc_open|popen|pcntl_exec)"
                r"\s*\([^)\n]{0,120}\$\w"), "match", False),
    ("RX-PHP-ASSERT", "CWE-94", 0.9,
     re.compile(r"\bassert\s*\(\s*\$_(?:GET|POST|REQUEST|COOKIE)"), "match", False),
    ("RX-JAVA-EXEC", "CWE-94", 0.88,
     re.compile(r"getRuntime\s*\(\s*\)\s*\.\s*exec\s*\("), "dynamic-arg", False),
    ("RX-C-SYSTEM", "CWE-94", 0.8,
     re.compile(r"\bsystem\s*\(\s*(?![\"'])[A-Za-z_]\w*\s*\)"), "match", True),
    ("RX-EVAL", "CWE-94", 0.9,
     re.compile(r"(?<![\w.])eval\s*\("), "dynamic-arg", True),
    ("RX-JS-FUNCTION-CTOR", "CWE-94", 0.8,
     re.compile(r"\bnew\s+Function\s*\("), "dynamic-arg", True),
    ("RX-PY-SHELL-TRUE", "CWE-94", 0.85,
     re.compile(r"\bsubprocess\.\w+\s*\("), "shell-true", True),
    ("RX-PY-OS-SYSTEM", "CWE-94", 0.88,
     re.compile(r"\bos\.(?:system|popen)\s*\("), "dynamic-arg", True),

    # ---- CWE-502: insecure deserialization ----
    ("RX-PHP-UNSERIALIZE", "CWE-502", 0.9,
     re.compile(r"\bunserialize\s*\(\s*\$"), "match", False),
    ("RX-JAVA-OBJECTINPUTSTREAM", "CWE-502", 0.88,
     re.compile(r"\bnew\s+ObjectInputStream\s*\("), "match", False),
    ("RX-DOTNET-BINARYFORMATTER", "CWE-502", 0.85,
     re.compile(r"\bnew\s+BinaryFormatter\s*\("), "match", False),
    ("RX-RUBY-MARSHAL", "CWE-502", 0.9,
     re.compile(r"\bMarshal\.load\s*\("), "match", False),
    ("RX-RUBY-YAML-LOAD", "CWE-502", 0.82,
     re.compile(r"\bYAML\.load\s*\("), "match", False),
    ("RX-PY-PICKLE-LOAD", "CWE-502", 0.88,
     re.compile(r"\b(?:pickle|cPickle|_pickle|dill|cloudpickle|marshal)"
                r"\.loads?\s*\("), "match", True),
    ("RX-PY-YAML-LOAD", "CWE-502", 0.88,
     re.compile(r"\byaml\.load\s*\("), "unsafe-yaml", True),

    # ---- CWE-89: SQL injection ----
    ("RX-SQL-CONCAT", "CWE-89", 0.9,
     re.compile(r"(?i)\b(?:execute|executeQuery|executeUpdate|executeLargeUpdate|"
                r"prepareStatement|createStatement|mysql_query|mysqli_query|"
                r"pg_query|rawQuery|execSQL|query)\s*\("), "sql-arg", True),
    ("RX-SQL-INTERP", "CWE-89", 0.88,
     re.compile(r"""(?is)["'`][^"'`\n]{0,300}?["'`]"""), "sql-interp", True),

    # ---- CWE-79: cross-site scripting ----
    ("RX-PHP-ECHO-INPUT", "CWE-79", 0.93,
     re.compile(r"(?:\b(?:echo|print)\b|<\?=)[^;\n]{0,200}"
                r"\$_(?:GET|POST|REQUEST|COOKIE)"), "match", False),
    ("RX-JS-INNERHTML", "CWE-79", 0.85,
     re.compile(r"\.(?:innerHTML|outerHTML)\s*=\s*([^;\n]{1,200})"), "dynamic-rhs", False),
    ("RX-JS-DOCUMENT-WRITE", "CWE-79", 0.85,
     re.compile(r"\bdocument\.write(?:ln)?\s*\("), "dynamic-arg", False),
    ("RX-REACT-DANGEROUS-HTML", "CWE-79", 0.78,
     re.compile(r"dangerouslySetInnerHTML\s*=\s*\{\{\s*__html\s*:\s*([^}\n]{1,200})"),
     "dynamic-rhs", False),

    # ---- CWE-200: information exposure ----
    ("RX-TLS-VERIFY-OFF", "CWE-200", 0.85,
     re.compile(r"\bverify\s*=\s*False\b"), "match", True),
    ("RX-TLS-CURL-OFF", "CWE-200", 0.9,
     re.compile(r"CURLOPT_SSL_VERIFY(?:PEER|HOST)\s*,\s*(?:0|false|FALSE)\b"),
     "match", False),
    ("RX-TLS-GO-SKIP", "CWE-200", 0.9,
     re.compile(r"InsecureSkipVerify\s*:\s*true\b"), "match", False),
    ("RX-TLS-NODE-REJECT", "CWE-200", 0.9,
     re.compile(r"rejectUnauthorized\s*:\s*false\b"), "match", False),
    ("RX-JAVA-WEAK-HASH", "CWE-200", 0.85,
     re.compile(r"""MessageDigest\.getInstance\s*\(\s*["'](?:MD5|SHA-?1)["']"""),
     "match", False),
    ("RX-PHP-WEAK-HASH", "CWE-200", 0.85,
     re.compile(r"(?i)\b(?:md5|sha1)\s*\(\s*\$(?:\w*(?:pass|pwd|secret|token|"
                r"cred|salt|auth)\w*)"), "match", False),
    ("RX-PY-WEAK-HASH", "CWE-200", 0.82,
     re.compile(r"\bhashlib\.(?:md5|sha1)\s*\("), "weak-hash-line", True),

    # ---- MEMORY-OOB: unbounded reads (textbook, never intentional) ----
    ("RX-C-GETS", MEMORY_OOB, 0.9,
     re.compile(r"(?<![\w.>])gets\s*\("), "match", True),
    ("RX-C-SCANF-S", MEMORY_OOB, 0.85,
     re.compile(r"(?<![\w.>])scanf\s*\(\s*\"[^\"\n]*%s"), "match", True),
)

_SQL_DYNAMIC_RE = re.compile(r"""["'`]\s*[+.]|[+.]\s*["'`]|\$\{|\#\{|\$\w""")
_SQL_INTERP_RE = re.compile(r"\$\{?\w|#\{\w|%\(\w+\)s\s*%|\bf[\"']")

# Escaping helpers used by the XSS rules: their output is safe to inject.
_SANITIZER = (r"(?:DOMPurify\.sanitize|sanitizeHtml|sanitize|escapeHtml|escapeHTML|"
              r"encodeURIComponent|htmlEscape|htmlspecialchars|purify|xssFilter)")
_SANITIZER_CALL_RE = re.compile(rf"(?i)\b{_SANITIZER}\s*\(")
_SANITIZED_ASSIGN_RE = re.compile(
    rf"(?i)([A-Za-z_$][\w$]*)\s*=\s*[^;\n]{{0,80}}?\b{_SANITIZER}\s*\(")
_BARE_IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*")


def _is_sanitized(expression: str, sanitized_names: set[str]) -> bool:
    """True when the value being injected has demonstrably been escaped."""
    if _SANITIZER_CALL_RE.search(expression):
        return True
    ident = expression.strip().rstrip(";").strip()
    return bool(_BARE_IDENT_RE.fullmatch(ident)) and ident in sanitized_names


def _regex_findings(code: str, lines: list[str], python_ok: bool) -> list[dict]:
    masked = _mask_comments(code)
    starts = _line_starts(code)
    out: list[dict] = []
    seen: set[tuple[str, int]] = set()

    def emit(rule_id, cwe, conf, offset):
        lineno = _line_of(offset, starts)
        if (rule_id, lineno) in seen:
            return
        seen.add((rule_id, lineno))
        out.append(_finding(cwe, rule_id, conf, lineno, _src_line(lines, lineno)))

    has_sql = bool(_SQL_RE.search(masked))
    sanitized = set(_SANITIZED_ASSIGN_RE.findall(masked))
    for rule_id, cwe, conf, pattern, mode, py_covered in _REGEX_RULES:
        if python_ok and py_covered:
            continue  # the AST pass already made a more precise decision
        if mode in ("sql-arg", "sql-interp") and not has_sql:
            continue  # cheap prefilter: no SQL statement anywhere in the file
        for match in pattern.finditer(masked):
            if mode == "match":
                emit(rule_id, cwe, conf, match.start())
            elif mode == "dynamic-arg":
                paren = masked.find("(", match.end() - 1)
                if paren == -1:
                    continue
                region = _arg_region(masked, paren)
                if not _looks_dynamic(region):
                    continue
                if cwe == "CWE-79" and _is_sanitized(region, sanitized):
                    continue
                emit(rule_id, cwe, conf, match.start())
            elif mode == "dynamic-rhs":
                rhs = match.group(1)
                if _looks_dynamic(rhs) and not _is_sanitized(rhs, sanitized):
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "shell-true":
                region = _arg_region(masked, masked.find("(", match.end() - 1))
                if re.search(r"\bshell\s*=\s*True\b", region):
                    head = region.split(",", 1)[0]
                    if _looks_dynamic(head):
                        emit(rule_id, cwe, conf, match.start())
            elif mode == "unsafe-yaml":
                region = _arg_region(masked, masked.find("(", match.end() - 1))
                if not re.search(r"(?:Safe|Base)Loader", region):
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "sql-arg":
                region = _arg_region(masked, masked.find("(", match.end() - 1))
                if _SQL_RE.search(region) and _SQL_DYNAMIC_RE.search(region):
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "sql-interp":
                literal = match.group(0)
                if _SQL_RE.search(literal) and _SQL_INTERP_RE.search(literal):
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "weak-hash-line":
                lineno = _line_of(match.start(), starts)
                if _is_security_context(re.findall(r"[A-Za-z_]\w*",
                                                   _src_line(lines, lineno))):
                    emit(rule_id, cwe, conf, match.start())

    if not python_ok:
        # For parseable Python the AST rule is strictly better: it knows an
        # assignment from a default argument and resolves the real value.
        out.extend(_regex_secret_findings(masked, lines, starts))

    # Several patterns describe the same defect in different dialects
    # (`eval($x)` matches both RX-PHP-EVAL and RX-EVAL). Keep the most
    # confident one per (cwe, line) so a single defect is reported once.
    best: dict[tuple[str, int | None], dict] = {}
    for finding in out:
        key = (finding["cwe"], finding["line"])
        if key not in best or finding["confidence"] > best[key]["confidence"]:
            best[key] = finding
    return list(best.values())


_SECRET_ASSIGN_RE = re.compile(
    r"""(?i)(?<![\w.])([A-Za-z_][A-Za-z0-9_]{2,40})\s*(?:=|:|=>|:=)\s*"""
    r"""(["'])([^"'\n]{6,200})\2"""
)


def _regex_secret_findings(masked: str, lines: list[str],
                           starts: list[int]) -> list[dict]:
    """CWE-200: hardcoded credentials, in any language."""
    out: list[dict] = []
    seen: set[int] = set()
    for match in _SECRET_ASSIGN_RE.finditer(masked):
        name, _, value = match.group(1), match.group(2), match.group(3)
        if not _looks_like_secret_name(name) or not _looks_like_secret_value(value):
            continue
        if value.lower() == name.lower():
            continue
        lineno = _line_of(match.start(), starts)
        if lineno in seen:
            continue
        seen.add(lineno)
        out.append(_finding("CWE-200", "RX-HARDCODED-SECRET", 0.85, lineno,
                            _src_line(lines, lineno)))
    return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def scan_rules(code: str, min_confidence: float = 0.0) -> list[dict]:
    """Run every rule over ``code`` and return the findings.

    Each finding is a dict with ``cwe``, ``name``, ``confidence`` (0-1),
    ``line`` (1-based, or None), ``rule_id`` and ``evidence`` (the offending
    source line). The list is ordered by line, then by descending confidence,
    and is empty when nothing is certain enough to report.
    """
    if not isinstance(code, str) or not code.strip():
        return []
    if len(code) > MAX_SCAN_CHARS:
        code = code[:MAX_SCAN_CHARS]

    lines = code.splitlines()
    tree = _try_parse(code)

    findings: list[dict] = []
    emitted: set[tuple[str, int]] = set()      # (rule_id, line)
    covered: set[tuple[str, int]] = set()      # (cwe, line) claimed by the AST

    if tree is not None:
        for finding in _python_findings(tree, lines):
            key = (finding["rule_id"], finding["line"])
            if key in emitted:
                continue
            emitted.add(key)
            covered.add((finding["cwe"], finding["line"]))
            findings.append(finding)

    for finding in _regex_findings(code, lines, python_ok=tree is not None):
        key = (finding["rule_id"], finding["line"])
        if key in emitted or (finding["cwe"], finding["line"]) in covered:
            continue
        emitted.add(key)
        findings.append(finding)

    findings = [f for f in findings if f["confidence"] >= min_confidence]
    findings.sort(key=lambda f: (f["line"] or 0, -f["confidence"], f["rule_id"]))
    return findings
