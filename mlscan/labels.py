"""The vulnerability taxonomy the scanner classifies into.

Ten well-known vulnerability classes (CWE) plus a "safe" class. Each was chosen
to be both recognizable AND well-supported (>=350 samples) in the training data.
"""

SAFE = "safe"

# CWE id -> (short name, OWASP 2021 category, one-line description)
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
    "CWE-119": ("Buffer Overflow", "A06: Vulnerable Components",
                "Operations on a memory buffer outside its bounds."),
    "CWE-787": ("Out-of-bounds Write", "A06: Vulnerable Components",
                "Writing past the end (or before the start) of a buffer."),
    "CWE-125": ("Out-of-bounds Read", "A06: Vulnerable Components",
                "Reading past the end (or before the start) of a buffer."),
    "CWE-476": ("NULL Pointer Dereference", "A06: Vulnerable Components",
                "Dereferencing a pointer that may be NULL."),
    "CWE-200": ("Information Exposure", "A01: Broken Access Control",
                "Sensitive information disclosed to an unauthorized actor."),
}

# The exact set of classes the model predicts (order not significant).
CLASSES = [SAFE] + list(TAXONOMY.keys())


def describe(label: str) -> dict:
    """Return a human-friendly description of a predicted label."""
    if label == SAFE:
        return {"cwe": None, "name": "No vulnerability detected",
                "owasp": None, "description": "The model found no known vulnerability pattern."}
    name, owasp, desc = TAXONOMY[label]
    return {"cwe": label, "name": name, "owasp": owasp, "description": desc}
