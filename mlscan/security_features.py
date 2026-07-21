"""Hand-crafted security signals: a dense companion to the TF-IDF blocks.

``SecurityFeatures`` is a stateless scikit-learn transformer that turns raw code
text into a small, fixed set of interpretable numbers — "does this call an
unsafe C string function?", "is a SQL keyword next to string concatenation?",
"is TLS verification switched off?" — that a bag-of-n-grams view either dilutes
or cannot express at all.

Design rules (each one is load-bearing):

* **Rename-robust.** Every pattern keys on *API names, keywords, operators and
  literal structure* — ``strcpy(``, ``pickle.loads(``, ``shell=True``, ``->``,
  ``[i+1]``. Nothing keys on a caller-chosen identifier such as ``user_input``
  or ``vulnerable_method``, so renaming variables cannot move the vector. The
  one deliberate exception is the *secret* detector, where names like
  ``password``/``api_key`` are the signal itself, not an artefact.
* **Length-normalised.** Counts become "hits per 100 characters", then
  ``log1p``. Without this the block would just re-learn the length artefact
  (vulnerable samples in the corpus run longer than safe ones).
* **Paired positive/negative evidence.** ``strncpy`` is as informative as
  ``strcpy``; ``json.loads`` as informative as ``pickle.loads``. Each risky
  family therefore ships with its safe counterpart so a linear model can learn
  the contrast rather than the mere topic.
* **Cheap.** ``re`` + ``numpy`` + ``scipy`` only. All patterns are precompiled
  and length-bounded — no catastrophic backtracking, no AST/tree-sitter parsing
  (the corpus is 92% C, truncated mid-function, and often not parseable).

Output is a ``scipy.sparse.csr_matrix`` (float32) so it can be ``hstack``-ed or
dropped straight into a :class:`~sklearn.pipeline.FeatureUnion` alongside
:func:`mlscan.features.build_vectorizer`. Follow it with ``MaxAbsScaler`` —
never ``StandardScaler``, which would densify the whole matrix.

>>> from mlscan.security_features import SecurityFeatures
>>> X = SecurityFeatures().fit_transform(["strcpy(dst, src);"])
>>> X.shape[1] == len(SecurityFeatures().get_feature_names_out())
True
"""

from __future__ import annotations

import math
import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

# Matches mlscan.data.MAX_CODE_CHARS: keeps the dense block identical at train
# and at serve time, and bounds regex cost on the 240k-char outliers.
DEFAULT_MAX_CHARS = 4000

# --------------------------------------------------------------------------
# Memory safety (C/C++) — the corpus is ~92% C, so this group carries most of
# the MEMORY-OOB and CWE-476 signal.
# --------------------------------------------------------------------------

_P_UNSAFE_STR = re.compile(
    r"\b(?:strcpy|strcat|sprintf|vsprintf|gets|stpcpy|wcscpy|wcscat|lstrcpy"
    r"|StrCpy|StrCat|scanf)\s*\(")
_P_SAFE_STR = re.compile(
    r"\b(?:strncpy|strncat|snprintf|vsnprintf|_snprintf|strlcpy|strlcat"
    r"|strnlen|fgets|strncmp|g_strlcpy)\s*\(")
_P_MEMCPY = re.compile(
    r"\b(?:memcpy|memmove|memset|bcopy|bzero|wmemcpy|CopyMemory|RtlCopyMemory"
    r"|__builtin_memcpy)\s*\(")
_P_ALLOCA = re.compile(r"\b(?:alloca|_alloca|_malloca)\s*\(")
_P_ALLOC = re.compile(
    r"\b(?:malloc|calloc|realloc|kmalloc|kzalloc|vmalloc|xmalloc|g_malloc"
    r"|g_new0?|HeapAlloc)\s*\("
    r"|\bnew\s+[A-Za-z_]\w*\s*[\(\[]")
_P_FREE = re.compile(
    r"\b(?:free|kfree|vfree|g_free|xfree|HeapFree)\s*\("
    r"|\bdelete\s*(?:\[\s*\])?\s*[A-Za-z_]")
_P_SIZEOF = re.compile(r"\bsizeof\s*[\(\w]")
_P_ARROW = re.compile(r"->")
_P_DEREF_EXPR = re.compile(r"\*\s*\(")
_P_NULL_LIT = re.compile(r"\b(?:NULL|nullptr|Py_None)\b")
_P_NULL_CHECK = re.compile(
    r"(?:[!=]=\s*(?:NULL|nullptr|0)\b)"
    r"|(?:\b(?:NULL|nullptr)\s*[!=]=)"
    r"|(?:\bif\s*\(\s*!\s*[A-Za-z_])"
    r"|(?:\b(?:IS_ERR|IS_ERR_OR_NULL|unlikely|likely)\s*\(\s*!)"
    r"|(?:\bis\s+(?:not\s+)?None\b)")
# Subscript containing arithmetic: buf[i+1], p[len-1], a[-1].
_P_IDX_ARITH = re.compile(r"\[[^\]\n]*[+\-][^\]\n]*\]")
# Loop guarded with <= — the classic off-by-one.
_P_LOOP_LE = re.compile(r"\b(?:for|while)\b[^\n]{0,120}<=")

# --------------------------------------------------------------------------
# Format strings
# --------------------------------------------------------------------------

# printf-family whose *format* argument is an identifier, not a literal.
_P_FMT_NONLIT = re.compile(
    r"\b(?:printf|vprintf|syslog|vsyslog)\s*\(\s*(?![\"'L)])"
    r"|\b(?:fprintf|vfprintf|sprintf|asprintf)\s*\([^,()\n]{0,60},\s*(?![\"'L])"
    r"|\b(?:snprintf|vsnprintf)\s*\([^,()\n]{0,60},[^,()\n]{0,60},\s*(?![\"'L])")
# %n is a write primitive, not a formatting directive.
_P_FMT_PCT_N = re.compile(r"%[0-9.*\-+ #]*n(?![A-Za-z0-9_])")

# --------------------------------------------------------------------------
# Injection: code / OS command / SQL
# --------------------------------------------------------------------------

_P_EVAL = re.compile(
    r"\b(?:eval|exec|execfile)\s*\("
    r"|\bnew\s+Function\s*\("
    r"|\b(?:instance_eval|class_eval|module_eval|create_function)\b"
    r"|\bsetTimeout\s*\(\s*[\"']"
    r"|\bexecScript\s*\(")
_P_OS_CMD = re.compile(
    r"\bos\.(?:system|popen|spawn\w*|exec\w*)\s*\("
    r"|\bsubprocess\.(?:call|run|Popen|check_output|check_call|getoutput)\s*\("
    r"|\b(?:popen|_popen|system|shell_exec|passthru|proc_open|pcntl_exec)\s*\("
    r"|\bexec[lv][pe]{0,2}\s*\("
    r"|Runtime\.getRuntime\s*\(\s*\)\s*\.\s*exec\s*\("
    r"|\bProcessBuilder\s*\("
    r"|\bchild_process\b|\bexecSync\s*\(|\bspawnSync\s*\(")
_P_SHELL_TRUE = re.compile(
    r"shell\s*=\s*True|shell\s*:\s*true"
    r"|[\"']/bin/(?:sh|bash)[\"']\s*,\s*[\"']-c[\"']")

# Require real SQL shape (SELECT..FROM, UPDATE..SET) so English prose in
# comments does not fire the detector.
_P_SQL_KW = re.compile(
    r"\bSELECT\b[^\n;]{0,200}?\bFROM\b"
    r"|\bINSERT\s+INTO\b"
    r"|\bUPDATE\b[^\n;]{0,200}?\bSET\b"
    r"|\bDELETE\s+FROM\b"
    r"|\bDROP\s+TABLE\b"
    r"|\bUNION\b(?:\s+ALL)?\s+\bSELECT\b",
    re.IGNORECASE)
# Dynamic string construction: concatenation, interpolation, templating.
_P_DYN_STR = re.compile(
    r"[\"']\s*\+"
    r"|\+\s*[\"']"
    r"|\.\s*format\s*\("
    r"|\bf[\"']"
    r"|[\"']\s*%\s*[\(\w]"
    r"|\$\{"
    r"|[\"']\s*\.\s*\$"
    r"|`[^`\n]{0,200}\$\{"
    r"|\.\s*(?:append|concat)\s*\(\s*[\"']?")
# Evidence the query is parameterised rather than interpolated.
_P_SQL_PARAM = re.compile(
    r"\b(?:execute|executemany|query|prepare|prepareStatement|createQuery)"
    r"\s*\([^;\n]{0,200},"
    r"|\b(?:PreparedStatement|bindParam|bindValue|setString|setInt|setLong)\b"
    r"|Parameters\.Add\w*\s*\("
    r"|[\"'][^\"'\n]{0,200}\?[^\"'\n]{0,200}[\"']\s*,")

# --------------------------------------------------------------------------
# Deserialization
# --------------------------------------------------------------------------

_P_DESER_UNSAFE = re.compile(
    r"\b(?:pickle|cPickle|_pickle|dill|cloudpickle)\.(?:loads?|Unpickler)\s*\("
    r"|\bmarshal\.loads?\s*\("
    r"|\btorch\.load\s*\("
    r"|\bObjectInputStream\b|\breadObject\s*\(|\bXMLDecoder\b"
    r"|\bunserialize\s*\(|\bBinaryFormatter\b|\bNetDataContractSerializer\b"
    r"|\bTypeNameHandling\b|\bjsonpickle\b"
    r"|\bYAML\.(?:load|unsafe_load)\b|\bMarshal\.load\s*\(")
_P_YAML_LOAD = re.compile(r"\byaml\.(?:load|unsafe_load|full_load)\s*\(")
_P_SAFE_LOADER = re.compile(r"SafeLoader|CSafeLoader|BaseLoader", re.IGNORECASE)
_P_DESER_SAFE = re.compile(
    r"\byaml\.safe_load\s*\(|SafeLoader"
    r"|\bjson\.loads?\s*\(|\bJSON\.parse\s*\("
    r"|\bhmac\.compare_digest\s*\(|\bDataContractJsonSerializer\b")

# --------------------------------------------------------------------------
# XSS / output encoding
# --------------------------------------------------------------------------

_P_XSS_SINK = re.compile(
    r"\.innerHTML\s*[+]?=|\.outerHTML\s*[+]?="
    r"|document\.write(?:ln)?\s*\("
    r"|insertAdjacentHTML\s*\("
    r"|dangerouslySetInnerHTML"
    r"|\.\s*html\s*\(\s*[A-Za-z_$]"
    r"|\bv-html\b|\bng-bind-html\b"
    r"|\brender_template_string\s*\("
    r"|\bmark_safe\s*\(|\|\s*safe\b|\bHtml\.Raw\s*\("
    r"|\becho\s+[^;\n]{0,80}\$"
    r"|\bResponse\.Write\s*\("
    r"|autoescape\s*=\s*False")
_P_XSS_ESCAPE = re.compile(
    r"\bhtmlspecialchars\s*\(|\bhtmlentities\s*\(|\bescapeHtml\b|\bescape_html\b"
    r"|\bDOMPurify\b|\bsanitiz\w*\s*\(|\bencodeURIComponent\s*\("
    r"|\bhtml\.escape\s*\(|\bcgi\.escape\s*\(|\bbleach\.clean\s*\("
    r"|\bstrip_tags\s*\(|\bHtmlEncode\s*\(|\bAntiXss\b",
    re.IGNORECASE)

# --------------------------------------------------------------------------
# Taint sources / validation
# --------------------------------------------------------------------------

_P_TAINT = re.compile(
    r"\$_(?:GET|POST|REQUEST|COOKIE|SERVER|FILES|ENV)\b"
    r"|\brequest\.(?:args|form|GET|POST|values|params|data|json|body|query"
    r"|cookies|headers)\b"
    r"|\breq\.(?:query|body|params|headers|cookies)\b"
    r"|\bgetParameter\s*\(|\bgetHeader\s*\(|\bgetQueryString\s*\("
    r"|\bsys\.argv\b|\bprocess\.argv\b|\bargv\s*\["
    r"|\binput\s*\(|\braw_input\s*\(|\breadLine\s*\(|\bConsole\.ReadLine\s*\("
    r"|\bgetenv\s*\(|\bos\.environ\b"
    r"|\bscanf\s*\(|\brecv(?:from)?\s*\("
    r"|\bRequest\.(?:QueryString|Form|Params)\b")
_P_DOTDOT = re.compile(r"\.\.[\\/]")
_P_COND_LINE = re.compile(r"\b(?:if|while|elif|assert|require)\b")
_P_CMP = re.compile(r"[<>]=?|==|!=")
_P_SIZE_TOKEN = re.compile(
    r"\b(?:len|strlen|wcslen|sizeof|size|length|count|capacity|nmemb|bufsize"
    r"|buflen|remaining)\b"
    r"|\.length\b|\.size\s*\(\s*\)"
    r"|\b(?:MAX|MIN|LIMIT|LEN|SIZE)_?[A-Z0-9_]*\b")
_P_TRYCATCH = re.compile(r"\b(?:try|catch|except|finally|rescue)\b")
_P_REGEX_VALIDATE = re.compile(
    r"\bre\.(?:match|fullmatch|search|compile)\s*\(|\bpreg_match\s*\("
    r"|\.matches\s*\(|Pattern\.compile\s*\(|\bnew\s+Regex\s*\("
    r"|\bRegex\.IsMatch\s*\(|\bstartsWith\s*\(|\bendswith\s*\(")
_P_PARSE = re.compile(
    r"\b(?:atoi|atol|atoll|atof|strtoul?|strtoll?)\s*\("
    r"|\bparseInt\s*\(|\bparseFloat\s*\(|\bInteger\.parse\w+\s*\("
    r"|\b(?:int|float)\s*\(\s*[A-Za-z_]"
    r"|\bConvert\.To\w+\s*\(")

# --------------------------------------------------------------------------
# Crypto / secrets / information exposure
# --------------------------------------------------------------------------

_P_WEAK_HASH = re.compile(r"\b(?:md5|md4|md2|sha1|sha-1)\b", re.IGNORECASE)
_P_WEAK_CIPHER = re.compile(
    r"\b(?:DES|3DES|TripleDES|RC2|RC4|ARCFOUR|Blowfish|ECB)\b")
_P_CRYPTO_STRONG = re.compile(
    r"\b(?:sha256|sha384|sha512|sha3|blake2\w*|bcrypt|scrypt|argon2|pbkdf2\w*"
    r"|hmac|AES|GCM|ChaCha20|Ed25519|X25519)\b",
    re.IGNORECASE)
_P_TLS_OFF = re.compile(
    r"verify\s*=\s*False"
    r"|CERT_NONE|_create_unverified_context"
    r"|InsecureSkipVerify\s*:\s*true"
    r"|rejectUnauthorized\s*:\s*false"
    r"|SSL_VERIFY_NONE|ALLOW_ALL_HOSTNAME_VERIFIER|TrustAllCerts"
    r"|setHostnameVerifier\s*\("
    r"|ServerCertificateValidationCallback"
    r"|CURLOPT_SSL_VERIFY\w*\s*,\s*(?:0|false|FALSE)")
# Secret-ish *key* names bound to a literal. Names ARE the signal here.
_P_SECRET = re.compile(
    r"\b(?:pass(?:wo?rd)?|passwd|pwd|secret|token|api[_-]?key|access[_-]?key"
    r"|private[_-]?key|client[_-]?secret|credential\w*|auth[_-]?token"
    r"|bearer|salt|encryption[_-]?key)\b\s*(?:=|:|=>|:=)\s*[\"'][^\"'\n]{4,}[\"']",
    re.IGNORECASE)
_P_STR_LIT = re.compile(r"\"([^\"\\\n]{12,})\"|'([^'\\\n]{12,})'")
_P_RAND_WEAK = re.compile(
    r"\brandom\.(?:random|randint|randrange|choice|seed|shuffle|sample)\s*\("
    r"|\bMath\.random\s*\(|\brand\s*\(\s*\)|\bsrand\s*\(|\bmt_rand\s*\("
    r"|\bnew\s+Random\s*\(")
_P_INFO_LEAK = re.compile(
    r"\bprintStackTrace\s*\(|\btraceback\.(?:print_exc|format_exc)\s*\("
    r"|\.getMessage\s*\(\s*\)"
    r"|\bconsole\.(?:log|debug|error|warn)\s*\("
    r"|\bSystem\.(?:out|err)\.print\w*\s*\("
    r"|\bvar_dump\s*\(|\bprint_r\s*\(|\bphpinfo\s*\("
    r"|\b(?:debug|DEBUG)\s*=\s*(?:True|true|1)\b|\bdisplay_errors\b")

# --------------------------------------------------------------------------
# Generic structure / syntax-derived language markers
#
# The dataset's own ``language`` column is mislabelled and leaks the source
# corpus (hence the label). These markers are derived from syntax only.
# --------------------------------------------------------------------------

_P_COMMENT = re.compile(
    r"//[^\n]*"
    r"|/\*.*?\*/"
    r"|^[ \t]*#(?!\s*(?:include|define|ifdef|ifndef|endif|pragma|if\b|else"
    r"|elif|undef|error|import))[^\n]*"
    r"|\"\"\".*?\"\"\"",
    re.S | re.M)
_P_SYNTAX_C = re.compile(
    r"#include\b|#define\b|->|::"
    r"|\b(?:struct|unsigned|size_t|uint\d+_t|const\s+char|void\s*\*)\b")
_P_SYNTAX_SCRIPT = re.compile(
    r"\bdef\s+\w+\s*\(|\bself\b|\belif\b|\bimport\s+\w|\bfrom\s+\w+\s+import\b"
    r"|<\?php|\$\w+\s*=|\bfunction\s*\(|=>"
    r"|\b(?:var|let|const)\s+\w+\s*=|\becho\b|\brequire\s*\(")
_P_SYNTAX_JVM = re.compile(
    r"\b(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+\w+\s*\("
    r"|\bSystem\.|\busing\s+System\b|\bnamespace\s+\w|\bpackage\s+[\w.]+;"
    r"|\bnew\s+[A-Z]\w*\s*\(")


#: Column order of the matrix returned by :meth:`SecurityFeatures.transform`.
FEATURE_NAMES: tuple[str, ...] = (
    # memory safety (C/C++)
    "mem_unsafe_str_fn",
    "mem_safe_str_fn",
    "mem_copy_fn",
    "mem_alloca",
    "mem_alloc",
    "mem_free",
    "mem_sizeof",
    "ptr_arrow",
    "ptr_deref_expr",
    "null_literal",
    "null_check",
    "idx_arith",
    "loop_le_bound",
    # format strings
    "fmt_nonliteral",
    "fmt_pct_n",
    # injection
    "inj_eval_exec",
    "inj_os_command",
    "inj_shell_true",
    "sql_keyword",
    "sql_dynamic",
    "sql_parameterized",
    "deser_unsafe",
    "deser_safe",
    "xss_sink",
    "xss_escape",
    # taint sources / validation
    "taint_source",
    "path_dotdot",
    "val_bounds_check",
    "val_try_catch",
    "val_regex",
    "val_unchecked_parse",
    # crypto / secrets / exposure
    "crypto_weak",
    "crypto_strong",
    "crypto_tls_off",
    "secret_hardcoded",
    "secret_high_entropy",
    "rand_weak",
    "info_leak_output",
    # generic structure
    "gen_log_len",
    "gen_log_lines",
    "gen_comment_ratio",
    "gen_dyn_string",
    "syntax_c",
    "syntax_script",
    "syntax_jvm",
)

N_FEATURES = len(FEATURE_NAMES)


def _count(pattern: re.Pattern[str], text: str) -> int:
    """Number of non-overlapping matches (no intermediate list)."""
    return sum(1 for _ in pattern.finditer(text))


def _dens(count: int, n_chars: int) -> float:
    """Hits per 100 characters, log-compressed — length-invariant by design."""
    return math.log1p(100.0 * count / n_chars)


def _looks_high_entropy(literal: str) -> bool:
    """True for base64/hex/token-shaped literals (candidate embedded secrets)."""
    if len(literal) < 16 or " " in literal:
        return False
    if len(literal) >= 32 and all(c in "0123456789abcdefABCDEF" for c in literal):
        return True
    kinds = (any(c.islower() for c in literal)
             + any(c.isupper() for c in literal)
             + any(c.isdigit() for c in literal))
    return kinds >= 2 and all(c.isalnum() or c in "+/=_-." for c in literal)


def _sql_signals(text: str) -> tuple[int, int]:
    """Return (n SQL statements, n of those built by string concatenation)."""
    n_kw = n_dyn = 0
    limit = len(text)
    for m in _P_SQL_KW.finditer(text):
        n_kw += 1
        lo = max(0, m.start() - 80)
        hi = min(limit, m.end() + 160)
        if _P_DYN_STR.search(text, lo, hi):
            n_dyn += 1
    return n_kw, n_dyn


def _unsafe_yaml_loads(text: str) -> int:
    """``yaml.load(...)`` calls with no SafeLoader in the following window."""
    n = 0
    limit = len(text)
    for m in _P_YAML_LOAD.finditer(text):
        window = text[m.end():min(limit, m.end() + 120)]
        if not _P_SAFE_LOADER.search(window):
            n += 1
    return n


def _bounds_checks(text: str) -> int:
    """Conditionals that compare against a size/length quantity."""
    n = 0
    for line in text.split("\n"):
        if (_P_COND_LINE.search(line)
                and _P_CMP.search(line)
                and _P_SIZE_TOKEN.search(line)):
            n += 1
    return n


def _comment_chars(text: str) -> int:
    return sum(len(m.group(0)) for m in _P_COMMENT.finditer(text))


def _high_entropy_literals(text: str) -> int:
    n = 0
    for m in _P_STR_LIT.finditer(text):
        if _looks_high_entropy(m.group(1) or m.group(2) or ""):
            n += 1
    return n


def extract(code: str) -> list[float]:
    """Extract the raw feature row for one snippet, in ``FEATURE_NAMES`` order.

    Exposed separately from the transformer so the CLI / explanation layer can
    show which signals fired without instantiating an estimator.
    """
    text = code if isinstance(code, str) else str(code)
    n_chars = max(len(text), 1)
    n_lines = text.count("\n") + 1

    n_sql_kw, n_sql_dyn = _sql_signals(text)
    n_deser_unsafe = _count(_P_DESER_UNSAFE, text) + _unsafe_yaml_loads(text)
    n_weak_crypto = _count(_P_WEAK_HASH, text) + _count(_P_WEAK_CIPHER, text)

    return [
        # memory safety
        _dens(_count(_P_UNSAFE_STR, text), n_chars),
        _dens(_count(_P_SAFE_STR, text), n_chars),
        _dens(_count(_P_MEMCPY, text), n_chars),
        _dens(_count(_P_ALLOCA, text), n_chars),
        _dens(_count(_P_ALLOC, text), n_chars),
        _dens(_count(_P_FREE, text), n_chars),
        _dens(_count(_P_SIZEOF, text), n_chars),
        _dens(_count(_P_ARROW, text), n_chars),
        _dens(_count(_P_DEREF_EXPR, text), n_chars),
        _dens(_count(_P_NULL_LIT, text), n_chars),
        _dens(_count(_P_NULL_CHECK, text), n_chars),
        _dens(_count(_P_IDX_ARITH, text), n_chars),
        _dens(_count(_P_LOOP_LE, text), n_chars),
        # format strings
        _dens(_count(_P_FMT_NONLIT, text), n_chars),
        _dens(_count(_P_FMT_PCT_N, text), n_chars),
        # injection
        _dens(_count(_P_EVAL, text), n_chars),
        _dens(_count(_P_OS_CMD, text), n_chars),
        _dens(_count(_P_SHELL_TRUE, text), n_chars),
        _dens(n_sql_kw, n_chars),
        _dens(n_sql_dyn, n_chars),
        _dens(_count(_P_SQL_PARAM, text), n_chars),
        _dens(n_deser_unsafe, n_chars),
        _dens(_count(_P_DESER_SAFE, text), n_chars),
        _dens(_count(_P_XSS_SINK, text), n_chars),
        _dens(_count(_P_XSS_ESCAPE, text), n_chars),
        # taint / validation
        _dens(_count(_P_TAINT, text), n_chars),
        _dens(_count(_P_DOTDOT, text), n_chars),
        _dens(_bounds_checks(text), n_chars),
        _dens(_count(_P_TRYCATCH, text), n_chars),
        _dens(_count(_P_REGEX_VALIDATE, text), n_chars),
        _dens(_count(_P_PARSE, text), n_chars),
        # crypto / secrets / exposure
        _dens(n_weak_crypto, n_chars),
        _dens(_count(_P_CRYPTO_STRONG, text), n_chars),
        _dens(_count(_P_TLS_OFF, text), n_chars),
        _dens(_count(_P_SECRET, text), n_chars),
        _dens(_high_entropy_literals(text), n_chars),
        _dens(_count(_P_RAND_WEAK, text), n_chars),
        _dens(_count(_P_INFO_LEAK, text), n_chars),
        # generic structure (the only length-aware columns, deliberately few)
        math.log1p(n_chars) / 10.0,
        math.log1p(n_lines) / 10.0,
        min(_comment_chars(text) / n_chars, 1.0),
        _dens(_count(_P_DYN_STR, text), n_chars),
        _dens(_count(_P_SYNTAX_C, text), n_chars),
        _dens(_count(_P_SYNTAX_SCRIPT, text), n_chars),
        _dens(_count(_P_SYNTAX_JVM, text), n_chars),
    ]


class SecurityFeatures(BaseEstimator, TransformerMixin):
    """Turn raw code strings into a dense block of security indicators.

    Stateless: :meth:`fit` only validates its input, so the transformer is safe
    to fit on train and reuse verbatim at inference (no train/serve skew).

    Parameters
    ----------
    max_chars:
        Truncate each snippet before extraction. Defaults to
        ``DEFAULT_MAX_CHARS`` (4000) to match :data:`mlscan.data.MAX_CODE_CHARS`
        and to bound regex cost on pathological inputs. Pass ``None`` to
        disable.
    dtype:
        Output dtype; ``float32`` halves memory versus the sklearn default.
    """

    def __init__(self, max_chars: int | None = DEFAULT_MAX_CHARS,
                 dtype=np.float32) -> None:
        self.max_chars = max_chars
        self.dtype = dtype

    # -- sklearn API -------------------------------------------------------

    def fit(self, X, y=None) -> "SecurityFeatures":
        """No-op fit (the transform is a pure function of the text)."""
        self._check_input(X)
        self.n_features_out_ = N_FEATURES
        return self

    def transform(self, X) -> sparse.csr_matrix:
        """Return a ``(n_samples, N_FEATURES)`` CSR matrix of indicators."""
        docs = self._check_input(X)
        limit = self.max_chars
        rows = [
            extract(doc if limit is None else doc[:limit])
            for doc in docs
        ]
        dense = np.asarray(rows, dtype=self.dtype).reshape(len(rows), N_FEATURES)
        return sparse.csr_matrix(dense)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the ``sec__``-prefixed column names, in matrix order."""
        return np.asarray([f"sec__{n}" for n in FEATURE_NAMES], dtype=object)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _check_input(X) -> list[str]:
        if isinstance(X, str):
            raise ValueError(
                "SecurityFeatures expects an iterable of code strings, "
                "not a single string."
            )
        return [x if isinstance(x, str) else "" if x is None else str(x)
                for x in X]
