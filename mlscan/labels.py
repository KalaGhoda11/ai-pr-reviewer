"""The vulnerability taxonomy the scanner classifies into.

Eight vulnerability classes plus a "safe" class (9 total). Each was chosen to be
both recognizable AND well-supported in the training data.

The three out-of-bounds memory CWEs (CWE-119 buffer overflow, CWE-787
out-of-bounds write, CWE-125 out-of-bounds read) are collapsed into a single
``MEMORY-OOB`` class. They describe the same underlying defect at different
granularities, the dataset labels them inconsistently, and keeping them apart
cost far more than it bought: the previous 11-class model scored F1 0.199 on
CWE-787 and 0.429 on CWE-125 while confusing them with each other. ``fold_cwe``
/ ``CWE_MERGE_MAP`` perform that collapse.
"""

SAFE = "safe"

# The merged out-of-bounds memory class (see module docstring).
MEMORY_OOB = "MEMORY-OOB"

# Raw dataset CWE id -> the taxonomy class it folds into. Only the three memory
# CWEs are merged; every other class maps to itself and is absent from this map.
CWE_MERGE_MAP = {
    "CWE-119": MEMORY_OOB,  # Improper restriction of ops within a buffer
    "CWE-787": MEMORY_OOB,  # Out-of-bounds write
    "CWE-125": MEMORY_OOB,  # Out-of-bounds read
}

# Class id -> (short name, OWASP 2021 category, one-line description)
TAXONOMY = {
    "CWE-89": ("SQL Injection", "A03: Injection",
               "Untrusted input concatenated into an SQL query."),
    "CWE-79": ("Cross-Site Scripting (XSS)", "A03: Injection",
               "Untrusted input rendered into a page without escaping."),
    "CWE-94": ("Code Injection", "A03: Injection",
               "Untrusted input evaluated/executed as code (eval/exec)."),
    "CWE-502": ("Insecure Deserialization", "A08: Data Integrity Failures",
                "Deserializing untrusted data (pickle, yaml.load, etc.)."),
    "CWE-20": ("Improper Input Validation", "A03: Injection",
               "External input used without adequate validation."),
    "CWE-200": ("Information Exposure", "A01: Broken Access Control",
                "Sensitive information disclosed to an unauthorized actor."),
    "CWE-476": ("NULL Pointer Dereference", "A06: Vulnerable Components",
                "Dereferencing a pointer that may be NULL."),
    MEMORY_OOB: ("Out-of-bounds Memory Access", "A06: Vulnerable Components",
                 "Reading or writing outside the bounds of a buffer "
                 "(merges CWE-119, CWE-125, CWE-787)."),
}

# The exact set of classes the model predicts (order not significant).
CLASSES = [SAFE] + list(TAXONOMY.keys())


def fold_cwe(cwe_id) -> str | None:
    """Fold a raw dataset CWE id onto a taxonomy class.

    Returns the class id (e.g. ``"CWE-125"`` -> ``"MEMORY-OOB"``), or ``None``
    when the CWE is outside the taxonomy (caller decides to drop it).
    """
    if cwe_id is None:
        return None
    key = str(cwe_id).strip().upper()
    if not key or key in {"NAN", "NONE", ""}:
        return None
    key = CWE_MERGE_MAP.get(key, key)
    return key if key in TAXONOMY else None


def describe(label: str) -> dict:
    """Return a human-friendly description of a predicted label.

    ``cwe`` holds the class id, which is a real CWE id for the single-CWE
    classes and ``"MEMORY-OOB"`` for the merged memory class. Pre-merge ids
    (e.g. from an older 11-class artifact) are folded automatically.
    """
    if label == SAFE:
        return {"cwe": None, "name": "No vulnerability detected",
                "owasp": None, "description": "The model found no known vulnerability pattern."}
    label = CWE_MERGE_MAP.get(label, label)
    name, owasp, desc = TAXONOMY[label]
    return {"cwe": label, "name": name, "owasp": owasp, "description": desc}
