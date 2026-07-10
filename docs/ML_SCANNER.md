# ML Vulnerability Scanner (`mlscan`)

A **standalone, offline** machine-learning classifier for common code
vulnerabilities. Unlike the LLM reviewer, it uses **no API keys and no network**
at inference time — a trained scikit-learn model runs locally in milliseconds.

It classifies a code snippet into one of **10 well-known vulnerability classes**
(by CWE) or `safe`.

## Why this exists

The main reviewer calls an LLM (Gemini). This module adds a second, independent
AI technique — a **supervised model trained on a real labelled dataset** — that
is self-contained, deterministic, free to run, and works with no external
service. Together they show two different AI-engineering approaches.

## The data

- **Dataset:** [`ayshajavd/code-security-vulnerability-dataset`](https://huggingface.co/datasets/ayshajavd/code-security-vulnerability-dataset)
  — 175,419 real code samples labelled with CWE / OWASP / vulnerability flag,
  pre-split into train / validation / test.
- **Honest caveats:** the corpus is ~92% C/C++, and its `language` column is
  unreliable (samples are sometimes mislabelled), so the model is **language-
  agnostic** rather than Python-specific, and is trained on the reliable **CWE
  label**. It is heavily skewed toward `safe`, so the training split is
  **balanced by down-sampling `safe`**; validation/test keep their natural
  distribution for an honest evaluation.

## The 10 classes (+ `safe`)

Chosen to be both recognizable and well-supported (≥350 samples):

| CWE | Name | OWASP 2021 |
|-----|------|------------|
| CWE-89 | SQL Injection | A03: Injection |
| CWE-79 | Cross-Site Scripting (XSS) | A03: Injection |
| CWE-94 | Code Injection | A03: Injection |
| CWE-502 | Insecure Deserialization | A08: Data Integrity |
| CWE-20 | Improper Input Validation | A03: Injection |
| CWE-119 | Buffer Overflow | A06: Vulnerable Components |
| CWE-787 | Out-of-bounds Write | A06 |
| CWE-125 | Out-of-bounds Read | A06 |
| CWE-476 | NULL Pointer Dereference | A06 |
| CWE-200 | Information Exposure | A01: Broken Access Control |

## Approach

1. **Features** (`mlscan/features.py`): TF-IDF combining **word 1–2 grams**
   (captures API names like `os.system`, `eval`) and **character 3–5 grams**
   (captures symbol/operator patterns — string concatenation in SQL, format
   strings — that a word tokenizer drops). ~50,000 features.
2. **Model selection** (`mlscan/train.py`): two candidates are trained on the
   same features and the higher validation macro-F1 wins:
   | Model | Validation macro-F1 |
   |-------|--------------------:|
   | Logistic Regression (balanced) | 0.464 |
   | **LightGBM (winner)** | **0.574** |

## Results (held-out test split, 17,121 samples)

- **Accuracy: 0.917** · weighted-F1: 0.925 · macro-F1: 0.581

| Class | F1 | | Class | F1 |
|-------|---:|-|-------|---:|
| safe | 0.965 | | CWE-476 | 0.558 |
| CWE-502 | 0.766 | | CWE-119 | 0.506 |
| CWE-89 | 0.644 | | CWE-20 | 0.491 |
| CWE-94 | 0.634 | | CWE-125 | 0.429 |
| CWE-200 | 0.627 | | CWE-787 | 0.199 |
| CWE-79 | 0.569 | | | |

(Full report in `mlscan/model/metrics.json`.) CWE-787 is weakest because it
overlaps heavily with the other out-of-bounds memory classes.

## Usage

```bash
pip install -r requirements-ml.txt

# scan a file
python -m mlscan path/to/file.py

# scan a snippet
python -m mlscan --code "def r(x): return eval(x)"

# machine-readable
python -m mlscan path/to/file.py --json
```
Exit code is `1` when a vulnerability is flagged, `0` when clean — usable as a
CI gate.

```python
from mlscan.scanner import scan
result = scan(source_code)           # {"is_vulnerable": ..., "findings": [...]}
```

## Design decisions

- **Precision over recall:** a finding is only reported at ≥ 0.50 confidence, so
  trivially-safe code isn't flagged (fewer false positives). The trade-off is
  some missed low-signal cases (e.g. XSS on a bare string concat).
- **Balanced training** so the model actually predicts vulnerabilities instead
  of always answering `safe`.
- **Model artifact committed** (`mlscan/model/vuln_clf.joblib`, ~15 MB) so the
  scanner and its tests run without retraining.

## Limitations (stated honestly)

- Trained predominantly on C/C++; it generalizes to obvious cross-language
  patterns (SQLi, `eval`, `os.system`) but is not a Python-specific tool.
- It is a **probabilistic classifier, not a rule-based SAST engine** — treat its
  output as a prioritization signal, not proof.
- Sensitive to phrasing/identifiers; confidence varies with surface form.

## Retraining

```bash
python -m mlscan.train      # downloads the dataset, compares models, saves the winner
```
