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
  ``cursor.execute("... %s" % uid)`` (CWE-89). A snippet that is a bare diff
  hunk (uniformly indented, no enclosing ``def``) is repaired before parsing --
  see :func:`_try_parse` -- because the AST path is strictly more precise than
  the regex path and would otherwise be skipped on almost every hunk.
* **Everything else** -- C, PHP, Java, JavaScript, Go, Ruby, C#, or a snippet
  that does not parse -- falls back to bounded regexes run over a copy of the
  source with comments blanked out, so a rule never fires on commented-out code.
  Regex rules carry deliberately *lower* confidence than their AST equivalents:
  they see a line, not a program.

Known boundaries, stated rather than hidden:

* C coverage is limited to defects whose bound is visible in the snippet -- an
  unbounded write into a **locally declared fixed-size array**, or a heap
  pointer dereferenced with no NULL check in the following lines. Rules that
  would need to know whether a length is attacker-controlled (bare ``memcpy``,
  ``malloc`` arithmetic) are deliberately absent: measured on this project's
  corpus they run at roughly chance precision, which is worse than silence.
* CWE-20 (improper input validation) and CWE-476 outside C have no rules. They
  are residual categories in the corpus, not lexical patterns.

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
import textwrap
from collections import Counter
from dataclasses import dataclass, field

from mlscan.labels import MEMORY_OOB, TAXONOMY

# Very large inputs are truncated: the regex pass is linear but there is no
# point scanning a minified bundle, and line numbers past the cut are useless.
MAX_SCAN_CHARS = 200_000

# The AST pass gets a far larger budget than the regex pass, and is applied to
# the *untruncated* source. Truncating first used to cut a large module mid
# construct ("unterminated triple-quoted string"), which silently downgraded it
# to the coarse regex backend -- so a file's verdict changed purely because it
# crossed 200 KB.
MAX_PARSE_CHARS = 2_000_000

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
    "login", "otp", "csrf",
}

# "session" and "cookie" used to sit in the set above and were the single
# largest weak-hash false-positive family in ordinary code, because they also
# name caches: `def session_cache_key(url): return md5(url)` is a cache key, not
# a credential. They now only count when nothing else in the statement says
# "checksum".
_WEAK_SECURITY_WORDS = {"session", "cookie", "nonce"}

# Words that mean "this digest identifies content", which is the legitimate
# non-cryptographic use of MD5 that the rule must not flag.
_CHECKSUM_WORDS = {
    "cache", "cached", "checksum", "etag", "fingerprint", "digest", "hash",
    "dedup", "dedupe", "bucket", "shard", "url", "uri", "path", "filename",
    "file", "content", "body", "chunk", "blob", "avatar", "gravatar", "color",
    "colour", "seed", "version",
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

# A value shaped like an *identifier* -- letter words joined by -_./ with no
# digit -- is naming a credential, not holding one. This is what separates
# `PRIVATE_KEY = "private_key"` (a JSON field name), `TOKEN_HEADER =
# "X-Xet-Access-Token"` (an HTTP header name) and `CRED_ENV =
# "GOOGLE_APPLICATION_CREDENTIALS"` (an env-var name) from `key = "AKIA...7EX"`.
# Every real credential in the corpus mixes in a digit or a non-separator
# symbol, so this costs no recall.
_NAME_SHAPED_RE = re.compile(r"^[-_./]*[A-Za-z]+(?:[-_./]+[A-Za-z]+)*[-_./]*$")
_SEPARATOR_RE = re.compile(r"[-_./]")


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
    if "key" in bag and bag & _KEY_QUALIFIERS:
        return True
    # An ambiguous word ("session", "cookie") counts only when the statement
    # does not also read as content addressing.
    return bool(bag & _WEAK_SECURITY_WORDS) and not (bag & _CHECKSUM_WORDS)


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
    if _NAME_SHAPED_RE.match(v):
        # Letter words joined by separators: a field/header/env-var NAME. The
        # one exception is a long unbroken run of letters, which is what a
        # passphrase looks like ("correcthorsebatterystaple").
        if _SEPARATOR_RE.search(v) or len(v) < 16:
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
    eval_aliases: dict[str, str] = field(default_factory=dict)


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
    return _resolve_dotted(_dotted(node), ctx.aliases)


_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _collect_scope(tree) -> tuple[dict[str, str], dict[str, str], set[int]]:
    """Import aliases, one-hop eval aliases and docstring lines, in one walk.

    ``ast.walk`` dominates this module's runtime on real files, so the three
    whole-tree scans these used to need are fused into one.

    * *aliases* resolve ``y.load`` back to ``yaml.load``.
    * *eval_aliases* catch ``e = eval``; only a bare ``name = <dotted name>``
      counts, which is narrow enough to be unambiguous (nothing legitimate
      aliases ``eval``) while catching the obfuscation that otherwise walks
      straight past :func:`_py_rule_eval`.
    * *doc_lines* are prose that the regex backend would otherwise match
      security patterns inside -- this module's own documentation of
      ``el.innerHTML = "<b>" + q`` used to raise RX-JS-INNERHTML against itself.
    """
    aliases: dict[str, str] = {}
    candidates: list[tuple[str, ast.AST]] = []
    doc_lines: set[int] = set()

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
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)
              and isinstance(node.value, (ast.Name, ast.Attribute))):
            candidates.append((node.targets[0].id, node.value))
        elif isinstance(node, _DOCSTRING_OWNERS):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    end = getattr(value, "end_lineno", None) or value.lineno
                    doc_lines.update(range(value.lineno, end + 1))

    eval_aliases = {}
    for name, value in candidates:
        canonical = _resolve_dotted(_dotted(value), aliases)
        if canonical in _EVAL_NAMES:
            eval_aliases[name] = canonical
    return aliases, eval_aliases, doc_lines


def _resolve_dotted(dotted: str | None, aliases: dict[str, str]) -> str | None:
    if dotted is None:
        return None
    head, sep, rest = dotted.partition(".")
    target = aliases.get(head)
    if target:
        return f"{target}.{rest}" if sep else target
    return dotted


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
# yaml.load without a safe Loader, plus its two explicit unsafe spellings.
_UNSAFE_YAML_CALLS = {"yaml.unsafe_load", "yaml.full_load",
                      "ruamel.yaml.unsafe_load"}
# Programs that are a shell, so `[prog, "-c", cmd]` is shell injection even
# though the caller passed a list and shell=False.
_SHELL_PROGRAMS = {"sh", "bash", "zsh", "ksh", "dash", "csh", "tcsh",
                   "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh"}
_SHELL_C_FLAGS = {"-c", "/c", "/C", "-Command", "-command"}
# Modules whose ``Markup`` really is the "trust this HTML" escape hatch. Without
# this gate the bare attribute tail collides with unrelated helpers -- pytest's
# ANSI colour writer exposes ``tw.markup(text, bold=True)``.
_MARKUP_MODULES = ("markupsafe", "jinja2", "flask", "django.utils.safestring",
                   "webhelpers.html")
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
    if canonical not in _EVAL_NAMES and canonical not in ctx.eval_aliases:
        return
    if not node.args or _is_static(node.args[0], ctx):
        return
    out.append(_finding("CWE-94", "PY-EVAL", 0.95, node.lineno,
                        _src_line(ctx.lines, node.lineno)))


def _shell_list_command(node: ast.Call, ctx: _PyCtx):
    """The dynamic command in ``[<shell>, "-c", cmd]``, else None.

    ``subprocess.run(["ls", path])`` is the safe idiom precisely because argv is
    handed to ``execve`` untouched -- but that argument evaporates when argv[0]
    *is* a shell, which is the one list form that still injects.
    """
    if not node.args:
        return None
    argv = node.args[0]
    if not isinstance(argv, (ast.List, ast.Tuple)) or len(argv.elts) < 3:
        return None
    head = argv.elts[:2]
    if not all(isinstance(e, ast.Constant) and isinstance(e.value, str)
               for e in head):
        return None
    program = head[0].value.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if program not in _SHELL_PROGRAMS or head[1].value not in _SHELL_C_FLAGS:
        return None
    if all(_is_static(e, ctx) for e in argv.elts[2:]):
        return None  # a hardcoded script cannot be injected into
    return argv.elts[2]


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
    if canonical not in _SUBPROCESS_CALLS:
        return
    if _is_true(_kwarg(node, "shell")):
        # shell=True with a hardcoded command is not injectable.
        if node.args and not _is_static(node.args[0], ctx):
            out.append(_finding("CWE-94", "PY-SUBPROCESS-SHELL", 0.92, node.lineno,
                                _src_line(ctx.lines, node.lineno)))
        return
    if _shell_list_command(node, ctx) is not None:
        out.append(_finding("CWE-94", "PY-SUBPROCESS-SHELL", 0.9, node.lineno,
                            _src_line(ctx.lines, node.lineno)))


def _is_roundtrip_dump(node, ctx: _PyCtx) -> bool:
    """True for ``pickle.loads(pickle.dumps(x))`` -- the deep-copy idiom.

    Serialising a value this process just produced and reading it straight back
    never crosses a trust boundary. It is also the single most common shape of
    ``pickle.loads`` in real code (test suites round-trip estimators with it).
    """
    if not isinstance(node, ast.Call):
        return False
    canonical = _canonical(node.func, ctx) or ""
    module, _, func = canonical.rpartition(".")
    root = module.split(".")[0] if module else ""
    return func in {"dump", "dumps"} and root in _PICKLE_MODULES


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
        if node.args and _is_roundtrip_dump(node.args[0], ctx):
            return
        conf = 0.9 if func == "loads" else 0.85
        out.append(_finding("CWE-502", "PY-PICKLE-LOAD", conf, node.lineno,
                            _src_line(ctx.lines, node.lineno)))
        return

    if canonical in {"jsonpickle.decode", "shelve.open"}:
        out.append(_finding("CWE-502", "PY-PICKLE-LOAD", 0.8, node.lineno,
                            _src_line(ctx.lines, node.lineno)))
        return

    if canonical in _UNSAFE_YAML_CALLS:
        # `unsafe_load` / `full_load` take no Loader argument: the name is the
        # decision.
        out.append(_finding("CWE-502", "PY-YAML-LOAD", 0.9, node.lineno,
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
    cert_reqs = _kwarg(node, "cert_reqs")
    if cert_reqs is not None and _canonical(cert_reqs, ctx) == "ssl.CERT_NONE":
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


def _all_dynamic_parts_escaped(node, ctx: _PyCtx, depth: int = 0) -> bool:
    """True when *every* non-literal piece of a string expression is escaped.

    Testing only the top node -- as this rule used to -- misses the common
    partial form ``"<h1>" + escape(name) + "</h1>"``, where the dynamic part is
    escaped but the expression as a whole is a concatenation.
    """
    if depth > 12:
        return False
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in ctx.const_names
    if isinstance(node, ast.JoinedStr):
        return all(_all_dynamic_parts_escaped(v, ctx, depth + 1) for v in node.values)
    if isinstance(node, ast.FormattedValue):
        return _all_dynamic_parts_escaped(node.value, ctx, depth + 1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (_all_dynamic_parts_escaped(node.left, ctx, depth + 1)
                and _all_dynamic_parts_escaped(node.right, ctx, depth + 1))
    if isinstance(node, ast.Call):
        return _is_escaped(node)
    return False


def _resolves_to(canonical: str | None, prefixes) -> bool:
    """True when a canonical dotted name comes from one of ``prefixes``."""
    if not canonical:
        return False
    return any(canonical == p or canonical.startswith(p + ".") for p in prefixes)


def _py_rule_xss(node: ast.Call, ctx: _PyCtx, out: list) -> None:
    """CWE-79: untrusted data rendered as HTML without escaping."""
    canonical = _canonical(node.func, ctx)
    tail = _attr_tail(node.func)
    lower_tail = tail.lower()

    # jinja2.Environment(autoescape=False): every template rendered by this
    # environment interpolates raw. Only the *explicit* False is flagged.
    if lower_tail == "environment" and _is_false(_kwarg(node, "autoescape")):
        out.append(_finding("CWE-79", "PY-JINJA-AUTOESCAPE", 0.85, node.lineno,
                            _src_line(ctx.lines, node.lineno)))
        return

    if lower_tail in {"render_template_string", "mark_safe", "markup"} and node.args:
        if lower_tail == "markup" and not _resolves_to(canonical, _MARKUP_MODULES):
            return
        arg = node.args[0]
        if _is_static(arg, ctx) or _all_dynamic_parts_escaped(arg, ctx):
            return
        rule = "PY-TEMPLATE-INJECTION" if lower_tail == "render_template_string" \
            else "PY-MARK-SAFE"
        out.append(_finding("CWE-79", rule, 0.88, node.lineno,
                            _src_line(ctx.lines, node.lineno)))
        return

    if lower_tail in _HTML_SINKS or (canonical or "").startswith("flask."):
        for arg in node.args:
            if _all_dynamic_parts_escaped(arg, ctx):
                continue
            text, dynamic = _string_parts(arg, ctx)
            if dynamic and text and _HTML_TAG_RE.search(text):
                out.append(_finding("CWE-79", "PY-HTML-CONCAT", 0.85, node.lineno,
                                    _src_line(ctx.lines, node.lineno)))
                return


def _py_rule_jwt(node: ast.Call, ctx: _PyCtx, out: list) -> None:
    """CWE-200: a JWT decoded without verifying its signature.

    ``jwt.decode(tok, options={"verify_signature": False})`` turns a signed
    assertion into an attacker-supplied dict; there is no benign reading of it
    outside a test.
    """
    if _attr_tail(node.func) != "decode":
        return
    canonical = _canonical(node.func, ctx) or ""
    if "jwt" not in canonical.lower() and "jose" not in canonical.lower():
        return
    disabled = _is_false(_kwarg(node, "verify"))
    options = _kwarg(node, "options")
    if isinstance(options, ast.Dict):
        for key, value in zip(options.keys, options.values):
            if (isinstance(key, ast.Constant) and key.value == "verify_signature"
                    and _is_false(value)):
                disabled = True
    if disabled:
        out.append(_finding("CWE-200", "PY-JWT-NO-VERIFY", 0.9, node.lineno,
                            _src_line(ctx.lines, node.lineno)))


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
    _py_rule_jwt,
    _py_rule_weak_hash,
    _py_rule_xss,
)


# A snippet that reads like Python. Used only to decide whether a *repair* of a
# non-parsing snippet is worth attempting -- guessing wrong on a C or Java
# fragment would suppress the regex rules that actually cover it.
_PY_HINT_RE = re.compile(
    r"(?m)^\s*(?:def|class|import|from|elif|except|async\s+def|@)\b"
    r"|\bself\s*\.|\bNone\b|\bTrue\b|\bFalse\b|:\s*$"
)
_NON_PY_HINT_RE = re.compile(
    r"(?m)^\s*#include\b|\{\s*$|^\s*\}|;\s*$|\bfunction\b|\bvar\s|\blet\s"
    r"|\bconst\s|\bpublic\s|\bnew\s+\w+\s*\(|\$\w|#\{|=>|::|<%|\bend\s*$"
)


def _looks_like_python(code: str) -> bool:
    """Whether a *non-parsing* snippet is worth trying to repair as Python.

    A hunk carries few keywords, so absence of the C-family markers counts as
    much as presence of the Python ones -- `query = "..." + uid` has neither.
    """
    other = len(_NON_PY_HINT_RE.findall(code))
    if other == 0:
        return True
    return len(_PY_HINT_RE.findall(code)) >= other


def _try_parse(code: str):
    """Parse ``code``, repairing a bare diff hunk if that is what it is.

    A hunk lifted out of a unified diff is a *fragment*: uniformly indented and
    missing its enclosing ``def``. Measured on realistic 12-line windows, 88% of
    them fail :func:`ast.parse`, which silently drops the whole file onto the
    coarse regex backend -- the backend that cannot do the cross-statement flow
    analysis the SQL rule depends on. Two repairs recover most of them:

    1. dedent to the common indent (line count, and therefore every line
       number, is unchanged);
    2. failing that, wrap in ``if True:`` and shift the tree back by one line.

    Both are gated on the text actually looking like Python, so a C or Java
    fragment that happens to become parseable is not mistaken for Python.
    """
    try:
        return ast.parse(code)
    except Exception:  # SyntaxError, ValueError, RecursionError, MemoryError...
        pass
    if not _looks_like_python(code):
        return None
    try:
        return ast.parse(textwrap.dedent(code))
    except Exception:
        pass
    try:
        tree = ast.parse("if True:\n" + code)
    except Exception:
        return None
    ast.increment_lineno(tree, -1)
    return tree


def _python_findings(tree, lines: list[str], code: str) -> tuple[list[dict], set[int]]:
    """Findings from the AST backend, plus the docstring lines it saw."""
    const_names, const_str = _collect_constants(tree)
    aliases, eval_aliases, doc_lines = _collect_scope(tree)
    ctx = _PyCtx(lines=lines, aliases=aliases,
                 const_names=const_names, const_str=const_str,
                 eval_aliases=eval_aliases)
    ctx.sql_vars = _collect_sql_vars(tree, ctx)
    # Parent links exist only for _context_names, which only the weak-hash rule
    # uses -- and that rule requires the call to resolve to the hashlib module,
    # which cannot happen unless the source names it. Skipping the link pass is
    # worth ~15% of this module's AST time on real files.
    if "hashlib" in code:
        _link_parents(tree)

    out: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for rule in _CALL_RULES:
                rule(node, ctx, out)
        elif isinstance(node, ast.Assign):
            # ctx.verify_mode = ssl.CERT_NONE / ctx.check_hostname = False
            if (isinstance(node.value, ast.Attribute)
                    and _canonical(node.value, ctx) == "ssl.CERT_NONE"):
                out.append(_finding("CWE-200", "PY-TLS-VERIFY-OFF", 0.85, node.lineno,
                                    _src_line(lines, node.lineno)))
            elif _is_false(node.value) and any(
                    _attr_tail(t).lower() == "check_hostname" for t in node.targets):
                out.append(_finding("CWE-200", "PY-TLS-VERIFY-OFF", 0.85, node.lineno,
                                    _src_line(lines, node.lineno)))
    _py_rule_secrets(tree, ctx, out)
    return out, doc_lines


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


# String literals, so an identifier can be looked for *outside* the quotes.
# A template literal is only stripped when it interpolates nothing: `${x}` is
# exactly the dynamic part we are looking for.
_STRING_LITERAL_RE = re.compile(
    r'"(?:[^"\\\n]|\\.)*"'
    r"|'(?:[^'\\\n]|\\.)*'"
    r"|`(?:[^`\\$]|\\.|\$(?!\{))*`"
)
_NUMERIC_RE = re.compile(r"[-+]?[0-9.]+[fFlLuU]*")
_IDENTIFIER_HINT_RE = re.compile(r"[A-Za-z_$]")


def _looks_dynamic(argument: str) -> bool:
    """True unless the argument region is built only from literals.

    Stripping the literals first is what separates ``el.innerHTML = "<b>" +
    "static" + "</b>"`` (a constant that merely contains letters) from
    ``el.innerHTML = "<b>" + q``. Testing the raw text for a letter, as this
    used to, calls both of them dynamic.
    """
    arg = argument.strip()
    if not arg:
        return False
    if _NUMERIC_RE.fullmatch(arg):
        return False
    return bool(_IDENTIFIER_HINT_RE.search(_STRING_LITERAL_RE.sub("", arg)))


def _split_args(region: str) -> list[str]:
    """Split a call region on top-level commas, respecting quotes and nesting."""
    args: list[str] = []
    current: list[str] = []
    depth = 0
    quote = ""
    escaped = False
    for ch in region:
        if quote:
            current.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            continue
        if ch in "\"'`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append("".join(current))
            current = []
            continue
        current.append(ch)
    args.append("".join(current))
    return [a.strip() for a in args]


# (rule_id, cwe, confidence, pattern, mode, skip_when_python_parsed)
#   mode "match"       -> the pattern alone is the evidence
#   mode "dynamic-arg" -> also require the call argument to be non-literal
#
# Confidences are deliberately below their AST counterparts (PY-* runs 0.85-0.95,
# RX-* runs 0.6-0.93): a regex sees one line, so it cannot tell a call from a
# definition, or an attacker-controlled buffer from a local one. They are
# ordinal hints for ranking, NOT calibrated probabilities.
_REGEX_RULES = (
    # ---- CWE-94: code / command injection ----
    ("RX-PHP-EVAL", "CWE-94", 0.95,
     re.compile(r"\beval\s*\(\s*\$"), "match", False),
    ("RX-PHP-EXEC", "CWE-94", 0.9,
     re.compile(r"\b(?:system|shell_exec|passthru|proc_open|popen|pcntl_exec)"
                r"\s*\([^)\n]{0,120}\$\w"), "match", False),
    ("RX-PHP-ASSERT", "CWE-94", 0.9,
     re.compile(r"\bassert\s*\(\s*\$_(?:GET|POST|REQUEST|COOKIE)"), "match", False),
    ("RX-PHP-INCLUDE-INPUT", "CWE-94", 0.9,
     re.compile(r"\b(?:include|require)(?:_once)?\s*\(?\s*[^;\n]{0,80}?"
                r"\$_(?:GET|POST|REQUEST|COOKIE)"), "match", False),
    ("RX-JAVA-EXEC", "CWE-94", 0.88,
     re.compile(r"getRuntime\s*\(\s*\)\s*\.\s*exec\s*\("), "dynamic-arg", False),
    # argv[0] is a shell, so the list form injects exactly like shell=True.
    # Covers subprocess.run(["sh","-c",x]), exec.Command("sh","-c",x),
    # ProcessBuilder("bash","-c",x) and child_process.spawn("sh",["-c",x]).
    ("RX-SHELL-DASH-C", "CWE-94", 0.85,
     re.compile(r"""["'](?:/(?:usr/)?bin/)?(?:sh|bash|zsh|ksh|dash|cmd(?:\.exe)?)["']"""
                r"""\s*,\s*\[?\s*["'](?:-c|/c|/C)["']\s*,\s*([^,)\n\]]{1,200})"""),
     "dynamic-rhs", False),
    ("RX-NODE-EXEC", "CWE-94", 0.85,
     re.compile(r"""(?:\b(?:child_process|childProcess|cp)"""
                r"""|require\s*\(\s*["']child_process["']\s*\))"""
                r"""\s*\.\s*exec(?:Sync)?\s*\("""), "dynamic-arg", False),
    ("RX-C-SYSTEM", "CWE-94", 0.8,
     re.compile(r"\bsystem\s*\(\s*(?![\"'])[A-Za-z_]\w*\s*\)"), "c-system", True),
    # The lookbehinds keep the rule off `def eval(...)` / `function eval(...)`,
    # which is a *definition* of an unrelated method, not a call.
    ("RX-EVAL", "CWE-94", 0.9,
     re.compile(r"(?<![\w.])(?<!def )(?<!function )eval\s*\("), "dynamic-arg", True),
    ("RX-JS-FUNCTION-CTOR", "CWE-94", 0.8,
     re.compile(r"\bnew\s+Function\s*\("), "dynamic-arg", True),
    ("RX-PY-SHELL-TRUE", "CWE-94", 0.85,
     re.compile(r"\bsubprocess\.\w+\s*\("), "shell-true", True),
    ("RX-PY-OS-SYSTEM", "CWE-94", 0.88,
     re.compile(r"\bos\.(?:system|popen)\s*\("), "dynamic-arg", True),

    # ---- CWE-502: insecure deserialization ----
    ("RX-PHP-UNSERIALIZE", "CWE-502", 0.9,
     re.compile(r"\bunserialize\s*\(\s*\$"), "match", False),
    ("RX-JAVA-OBJECTINPUTSTREAM", "CWE-502", 0.78,
     re.compile(r"\bnew\s+ObjectInputStream\s*\("), "deserialize", False),
    ("RX-JAVA-XMLDECODER", "CWE-502", 0.88,
     re.compile(r"\bnew\s+XMLDecoder\s*\("), "deserialize", False),
    # An empty SnakeYAML constructor is the unsafe default; passing a
    # SafeConstructor makes the pattern not match at all.
    ("RX-JAVA-SNAKEYAML", "CWE-502", 0.8,
     re.compile(r"\bnew\s+Yaml\s*\(\s*\)\s*\.\s*load(?:As|All)?\s*\("),
     "deserialize", False),
    ("RX-DOTNET-BINARYFORMATTER", "CWE-502", 0.72,
     re.compile(r"\bnew\s+(?:BinaryFormatter|NetDataContractSerializer|LosFormatter"
                r"|SoapFormatter|ObjectStateFormatter)\s*\("), "deserialize", False),
    ("RX-DOTNET-TYPENAMEHANDLING", "CWE-502", 0.8,
     re.compile(r"TypeNameHandling\s*=\s*TypeNameHandling\.(?:All|Objects|Auto|Arrays)\b"),
     "match", False),
    ("RX-RUBY-MARSHAL", "CWE-502", 0.82,
     re.compile(r"\bMarshal\.load\s*\("), "deserialize", False),
    ("RX-RUBY-YAML-LOAD", "CWE-502", 0.75,
     re.compile(r"\bYAML\.load\s*\("), "deserialize", False),
    ("RX-PY-PICKLE-LOAD", "CWE-502", 0.88,
     re.compile(r"\b(?:pickle|cPickle|_pickle|dill|cloudpickle|marshal)"
                r"\.loads?\s*\("), "deserialize", True),
    ("RX-PY-YAML-LOAD", "CWE-502", 0.88,
     re.compile(r"\byaml\.(?:load|unsafe_load|full_load)\s*\("), "unsafe-yaml", True),

    # ---- CWE-89: SQL injection ----
    ("RX-SQL-CONCAT", "CWE-89", 0.9,
     re.compile(r"(?i)\b(?:execute|executeQuery|executeUpdate|executeLargeUpdate|"
                r"prepareStatement|createStatement|prepare|mysql_query|mysqli_query|"
                r"mysql_real_query|pg_query|PQexec|sqlite3_exec|rawQuery|execSQL|"
                r"find_by_sql|query|queryRow|queryContext|exec|execContext|where|"
                r"queryForObject|createQuery)\s*\("),
     "sql-arg", True),
    ("RX-SQL-INTERP", "CWE-89", 0.88,
     re.compile(r"""(?is)["'`][^"'`\n]{0,300}?["'`]"""), "sql-interp", True),

    # ---- CWE-79: cross-site scripting ----
    ("RX-PHP-ECHO-INPUT", "CWE-79", 0.93,
     re.compile(r"(?:\b(?:echo|print)\b|<\?=)[^;\n]{0,200}"
                r"\$_(?:GET|POST|REQUEST|COOKIE)"), "xss-line", False),
    ("RX-JSP-XSS", "CWE-79", 0.88,
     re.compile(r"\bout\s*\.\s*print(?:ln)?\s*\("), "jsp-xss", False),
    ("RX-JS-INNERHTML", "CWE-79", 0.85,
     re.compile(r"\.(?:innerHTML|outerHTML)\s*=\s*([^;\n]{1,200})"), "dynamic-rhs", False),
    ("RX-JS-INSERT-HTML", "CWE-79", 0.82,
     re.compile(r"\.insertAdjacentHTML\s*\("), "dynamic-arg", False),
    ("RX-JQUERY-HTML", "CWE-79", 0.8,
     re.compile(r"\$\s*\([^)\n]{0,120}\)\s*\.\s*html\s*\("), "dynamic-arg", False),
    ("RX-JS-DOCUMENT-WRITE", "CWE-79", 0.85,
     re.compile(r"\bdocument\.write(?:ln)?\s*\("), "dynamic-arg", False),
    ("RX-REACT-DANGEROUS-HTML", "CWE-79", 0.78,
     re.compile(r"dangerouslySetInnerHTML\s*=\s*\{\{\s*__html\s*:\s*([^}\n]{1,200})"),
     "dynamic-rhs", False),
    ("RX-JINJA-AUTOESCAPE", "CWE-79", 0.85,
     re.compile(r"\bautoescape\s*=\s*(?:False|false)\b"), "match", True),

    # ---- CWE-200: information exposure ----
    # `verify=False` is also an ordinary kwarg name on unrelated APIs
    # (`mgr.take(idx, verify=False)`), so the line must mention transport.
    ("RX-TLS-VERIFY-OFF", "CWE-200", 0.8,
     re.compile(r"\bverify\s*=\s*(?:False|false)\b"), "tls-line", True),
    ("RX-TLS-CERT-NONE", "CWE-200", 0.85,
     re.compile(r"\b(?:verify_mode|cert_reqs)\s*[=:]\s*(?:ssl\.)?CERT_NONE\b"),
     "match", True),
    ("RX-TLS-CHECK-HOSTNAME", "CWE-200", 0.85,
     re.compile(r"\bcheck_hostname\s*=\s*(?:False|false)\b"), "match", True),
    ("RX-TLS-CURL-OFF", "CWE-200", 0.9,
     re.compile(r"CURLOPT_SSL_VERIFY(?:PEER|HOST)\s*,\s*(?:0|false|FALSE)\b"),
     "match", False),
    ("RX-TLS-GO-SKIP", "CWE-200", 0.9,
     re.compile(r"InsecureSkipVerify\s*:\s*true\b"), "match", False),
    ("RX-TLS-NODE-REJECT", "CWE-200", 0.9,
     re.compile(r"rejectUnauthorized\s*:\s*false\b"), "match", False),
    ("RX-JWT-NO-VERIFY", "CWE-200", 0.88,
     re.compile(r"""["']?verify_signature["']?\s*[:=]\s*(?:False|false|0)\b"""),
     "match", True),
    ("RX-JAVA-WEAK-HASH", "CWE-200", 0.85,
     re.compile(r"""MessageDigest\.getInstance\s*\(\s*["'](?:MD5|SHA-?1)["']"""),
     "match", False),
    ("RX-PHP-WEAK-HASH", "CWE-200", 0.85,
     re.compile(r"(?i)\b(?:md5|sha1)\s*\(\s*\$(?:\w*(?:pass|pwd|secret|token|"
                r"cred|salt|auth)\w*)"), "match", False),
    ("RX-PY-WEAK-HASH", "CWE-200", 0.82,
     re.compile(r"\bhashlib\.(?:md5|sha1)\s*\("), "weak-hash-line", True),

    # ---- MEMORY-OOB ----
    # Unbounded reads: textbook, never intentional. Rare in modern corpora, but
    # they cost nothing and are correct by definition.
    ("RX-C-GETS", MEMORY_OOB, 0.9,
     re.compile(r"(?<![\w.>])gets\s*\("), "match", True),
    ("RX-C-SCANF-S", MEMORY_OOB, 0.85,
     re.compile(r"(?<![\w.>])(?:f|s)?scanf\s*\([^;\n]{0,120}?\"[^\"\n]*%s"),
     "match", True),
    # Unbounded writes into a buffer whose size is declared right there. The
    # fixed-size local destination is load-bearing: without it these are the
    # highest-volume, lowest-precision rules one can write for C.
    ("RX-C-STRCPY-FIXED", MEMORY_OOB, 0.72,
     re.compile(r"(?<![\w.>])(?:strcpy|strcat|wcscpy|wcscat|lstrcpy|lstrcat)\s*\("),
     "c-fixed-copy", True),
    ("RX-C-SPRINTF-FIXED", MEMORY_OOB, 0.72,
     re.compile(r"(?<![\w.>])(?:sprintf|vsprintf|swprintf)\s*\("),
     "c-fixed-format", True),
    ("RX-C-MEMCPY-FIXED", MEMORY_OOB, 0.62,
     re.compile(r"(?<![\w.>])(?:memcpy|memmove|bcopy|strncat)\s*\("),
     "c-fixed-length", True),
    ("RX-C-ALLOCA", MEMORY_OOB, 0.7,
     re.compile(r"(?<![\w.>])(?:alloca|_alloca)\s*\("), "dynamic-arg", True),
)

_SQL_DYNAMIC_RE = re.compile(r"""["'`]\s*[+.]|[+.]\s*["'`]|\$\{|\#\{|\$\w""")
_SQL_INTERP_RE = re.compile(r"\$\{?\w|#\{\w|%\(\w+\)s\s*%|\bf[\"']")

# Escaping helpers used by the XSS rules: their output is safe to inject.
_SANITIZER = (r"(?:DOMPurify\.sanitize|sanitizeHtml|sanitize|escapeHtml|escapeHTML|"
              r"encodeURIComponent|htmlEscape|htmlspecialchars|purify|xssFilter)")
_SANITIZER_CALL_RE = re.compile(rf"(?i)\b{_SANITIZER}\s*\(")
# The leading lookbehind and the bounded identifier are load-bearing, not
# style: without them the greedy `[\w$]*` is retried at every offset inside a
# long token and backtracks the whole way each time. A 40 KB base64 blob or
# minified bundle in a string literal took 27 s to scan -- quadratic, and past
# what MAX_SCAN_CHARS bounds. With them the same input is ~1 ms.
_SANITIZED_ASSIGN_RE = re.compile(
    rf"(?i)(?<![\w$])([A-Za-z_$][\w$]{{0,63}})\s*=\s*[^;\n]{{0,80}}?"
    rf"\b{_SANITIZER}\s*\(")
_BARE_IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*")
# Reading one of these yields text the DOM has already escaped. Copying it into
# innerHTML is a deliberate precision trade: it is the dominant benign shape of
# `el.innerHTML = other.<something>` in real front-end code.
_ESCAPED_DOM_READ_RE = re.compile(
    r"^[\w$.\[\]]*\.\s*(?:textContent|innerText|nodeValue|value)\s*;?$")

# Servlet-API readers of attacker-supplied request data.
_REQUEST_PARAM_RE = re.compile(
    r"\brequest\s*\.\s*(?:getParameter|getHeader|getQueryString|getCookies)\s*\(")

# A `verify=False` is only about TLS when the surrounding line is.
_TLS_CONTEXT_RE = re.compile(
    r"(?i)\b(?:requests|httpx|aiohttp|urllib3?|urlopen|session|ssl|tls|https?|"
    r"url|uri|cert|certs|certificate|ca_bundle|cafile|capath|verify_ssl|client|"
    r"api|endpoint|webhook|socket|connection|conn|curl|fetch|post|put|patch)\b")

# `static int system(const char *c)` -- a local shadow, not libc's system().
_C_SYSTEM_DEF_RE = re.compile(
    r"(?m)^\s*(?:static\s+|extern\s+|inline\s+|const\s+)*"
    r"(?:int|void|char|long|short|unsigned|signed|size_t|ssize_t|bool|BOOL)\s*\**\s*"
    r"system\s*\(")

# Deserialising something this code just serialised, or a file at a literal
# path shipped with the program, does not cross a trust boundary. These are the
# two shapes behind most surviving deserialization hits on *patched* code.
_LOCAL_SOURCE_RE = re.compile(
    r"""(?i)\.toByteArray\s*\(\s*\)"""
    r"""|\bnew\s+ByteArrayInputStream\s*\(\s*[\w.]*\s*\)"""
    r"""|\b(?:Marshal|YAML|JSON|pickle|cPickle|dill|cloudpickle|marshal)"""
    r"""\s*\.\s*dumps?\s*\("""
    r"""|\b(?:File\.read|File\.open|File\.new|IO\.read)\s*\(\s*["']"""
    r"""|\bnew\s+FileInputStream\s*\(\s*["']""")


def _is_sanitized(expression: str, sanitized_names: set[str]) -> bool:
    """True when the value being injected has demonstrably been escaped."""
    if _SANITIZER_CALL_RE.search(expression):
        return True
    stripped = expression.strip().rstrip(";").strip()
    if _ESCAPED_DOM_READ_RE.match(stripped):
        return True
    return bool(_BARE_IDENT_RE.fullmatch(stripped)) and stripped in sanitized_names


# --- C buffer analysis ----------------------------------------------------

# `char buf[64];` at statement position. Anchoring at the start of a line keeps
# parameters (`void f(char buf[8])`) out: those are pointers, not storage.
_C_FIXED_ARRAY_RE = re.compile(
    r"(?m)^[^\S\n]*(?:(?:static|const|volatile|register|unsigned|signed|struct|"
    r"auto)\s+)*"
    r"(?:char|wchar_t|u?int\d{0,2}_t|byte|BYTE|TCHAR|WCHAR|short|int|long|float|"
    r"double|size_t)\s+\**\s*([A-Za-z_]\w{0,63})\s*\[\s*[^\]\n]{1,64}\]")

_SIZEOF_RE = re.compile(r"\bsizeof\b")


def _c_fixed_arrays(masked: str) -> set[str]:
    """Names declared as fixed-size local arrays in this snippet."""
    return set(_C_FIXED_ARRAY_RE.findall(masked))


def _is_bare_name(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_]\w*", text.strip()))


# A pointer straight out of the allocator. `p = malloc(n)` on one line and a
# dereference a few lines later with no NULL test between them is CWE-476.
_C_ALLOC_RE = re.compile(
    r"(?<![\w.>])([A-Za-z_]\w{0,63})\s*=\s*(?:\([^)\n]{0,60}\)\s*)?"
    r"(?:malloc|calloc|realloc|strdup|strndup|kmalloc|kzalloc|g_malloc|xmalloc)"
    r"\s*\(")

# How far after the allocation a dereference still counts as "unchecked".
_NULL_DEREF_WINDOW = 6


def _c_null_deref_findings(masked: str, lines: list[str]) -> list[dict]:
    """CWE-476: allocator result dereferenced without a NULL check."""
    out: list[dict] = []
    masked_lines = masked.splitlines()
    for index, line in enumerate(masked_lines):
        for match in _C_ALLOC_RE.finditer(line):
            name = re.escape(match.group(1))
            checked = re.compile(
                rf"!\s*{name}\b|\b{name}\s*[=!]=\s*(?:NULL|0|nullptr)"
                rf"|(?:NULL|nullptr|0)\s*[=!]=\s*{name}\b"
                rf"|\bif\s*\(\s*{name}\s*\)|\bassert\s*\(\s*{name}\b"
                rf"|\b(?:un)?likely\s*\(\s*!?\s*{name}\b|\bIS_ERR\w*\s*\(\s*{name}\b")
            deref = re.compile(rf"\*\s*{name}\b|\b{name}\s*->|\b{name}\s*\[")
            for offset in range(1, _NULL_DEREF_WINDOW + 1):
                if index + offset >= len(masked_lines):
                    break
                following = masked_lines[index + offset]
                if checked.search(following):
                    break  # guarded -- nothing to report
                if deref.search(following):
                    lineno = index + offset + 1
                    out.append(_finding("CWE-476", "RX-C-NULL-DEREF", 0.6, lineno,
                                        _src_line(lines, lineno)))
                    break
    return out


_NON_NEWLINE_RE = re.compile(r"[^\n]")


def _blank_lines(masked: str, starts: list[int], linenos) -> str:
    """Replace whole lines with spaces, preserving every byte offset."""
    pieces: list[str] = []
    cursor = 0
    for lineno in sorted(linenos):
        if not 1 <= lineno <= len(starts):
            continue
        begin = starts[lineno - 1]
        end = starts[lineno] if lineno < len(starts) else len(masked)
        if begin < cursor:
            continue
        pieces.append(masked[cursor:begin])
        pieces.append(_NON_NEWLINE_RE.sub(" ", masked[begin:end]))
        cursor = end
    pieces.append(masked[cursor:])
    return "".join(pieces)


def _regex_findings(code: str, lines: list[str], python_ok: bool,
                    doc_lines=()) -> list[dict]:
    masked = _mask_comments(code)
    starts = _line_starts(code)
    if doc_lines:
        masked = _blank_lines(masked, starts, doc_lines)
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
    fixed_arrays: set[str] | None = None
    system_shadowed = bool(_C_SYSTEM_DEF_RE.search(masked))

    for rule_id, cwe, conf, pattern, mode, py_covered in _REGEX_RULES:
        if python_ok and py_covered:
            continue  # the AST pass already made a more precise decision
        if mode in ("sql-arg", "sql-interp") and not has_sql:
            continue  # cheap prefilter: no SQL statement anywhere in the file
        if mode.startswith("c-fixed"):
            if fixed_arrays is None:
                fixed_arrays = _c_fixed_arrays(masked)
            if not fixed_arrays:
                continue  # no declared buffer -> no known bound -> no rule
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
            elif mode == "c-system":
                if not system_shadowed:
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "xss-line":
                # `echo htmlspecialchars($_GET['x'])` is the fixed form.
                lineno = _line_of(match.start(), starts)
                if not _SANITIZER_CALL_RE.search(_src_line(lines, lineno)):
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "jsp-xss":
                region = _arg_region(masked, masked.find("(", match.end() - 1))
                if (_REQUEST_PARAM_RE.search(region)
                        and not _is_sanitized(region, sanitized)):
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "deserialize":
                region = _arg_region(masked, masked.find("(", match.end() - 1))
                if not _LOCAL_SOURCE_RE.search(region):
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "tls-line":
                lineno = _line_of(match.start(), starts)
                if _TLS_CONTEXT_RE.search(_src_line(lines, lineno)):
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "c-fixed-copy":
                # strcpy(dst, src): dst has a known size, src does not.
                args = _split_args(_arg_region(masked, masked.find("(", match.end() - 1)))
                if (len(args) >= 2 and args[0] in fixed_arrays
                        and _looks_dynamic(args[1])):
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "c-fixed-format":
                # sprintf(dst, "...%s...", src): %s writes without a bound.
                args = _split_args(_arg_region(masked, masked.find("(", match.end() - 1)))
                if (len(args) >= 3 and args[0] in fixed_arrays
                        and re.search(r'"[^"\n]*%s', args[1])
                        and any(_looks_dynamic(a) for a in args[2:])):
                    emit(rule_id, cwe, conf, match.start())
            elif mode == "c-fixed-length":
                # memcpy(dst, src, n): a length not derived from sizeof(dst).
                args = _split_args(_arg_region(masked, masked.find("(", match.end() - 1)))
                if (len(args) >= 3 and args[0] in fixed_arrays
                        and not _SIZEOF_RE.search(args[2])
                        and not _NUMERIC_RE.fullmatch(args[2])
                        and _looks_dynamic(args[2])):
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
        # Multi-line C analysis, so it cannot live in the single-pattern table.
        out.extend(_c_null_deref_findings(masked, lines))

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

    lines = code.splitlines()
    # Parse the WHOLE source. Truncating first cuts a large module mid-construct
    # so ast.parse fails, which silently downgrades it to the regex backend and
    # re-enables the coarse rules the AST pass exists to replace.
    tree = _try_parse(code) if len(code) <= MAX_PARSE_CHARS else None
    # The regex pass gets the truncated text; a prefix, so line numbers align.
    scanned = code[:MAX_SCAN_CHARS] if len(code) > MAX_SCAN_CHARS else code

    findings: list[dict] = []
    emitted: set[tuple[str, int]] = set()      # (rule_id, line)
    covered: set[tuple[str, int]] = set()      # (cwe, line) claimed by the AST

    doc_lines: set[int] = set()
    if tree is not None:
        python_findings, doc_lines = _python_findings(tree, lines, code)
        for finding in python_findings:
            key = (finding["rule_id"], finding["line"])
            if key in emitted:
                continue
            emitted.add(key)
            covered.add((finding["cwe"], finding["line"]))
            findings.append(finding)

    for finding in _regex_findings(scanned, lines, python_ok=tree is not None,
                                   doc_lines=doc_lines):
        key = (finding["rule_id"], finding["line"])
        if key in emitted or (finding["cwe"], finding["line"]) in covered:
            continue
        emitted.add(key)
        findings.append(finding)

    findings = [f for f in findings if f["confidence"] >= min_confidence]
    findings.sort(key=lambda f: (f["line"] or 0, -f["confidence"], f["rule_id"]))
    return findings
