# ML Vulnerability Scanner (`mlscan`)

A **standalone, offline** vulnerability scanner. No LLM, no API keys, no network
at inference time. It combines a **trained ML classifier** with a
**deterministic rule engine** and runs locally in milliseconds.

---

## 1. The headline engineering finding: the public dataset leaks

The first version of this scanner reported **macro-F1 0.581** (later 0.6543 after
a taxonomy fix). Those numbers were **not trustworthy**, and finding out why is
the most important result in this module.

**9.1% of the test split is byte-identical to rows in the training split** —
and the duplication is concentrated exactly where a macro-average is decided:

| Class | % of test rows also in train |
|-------|-----------------------------:|
| CWE-89 (SQL Injection) | **82.7%** |
| CWE-200 | 66.3% |
| MEMORY-OOB | 62.6% |
| CWE-20 | 59.8% |
| CWE-502 | 55.0% |
| CWE-79 | 52.2% |
| CWE-94 | 50.4% |
| CWE-476 | 37.4% |
| safe | 4.3% |

A char-n-gram TF-IDF model on an exact duplicate is close to a hash lookup, so
the original score was substantially a **memorization** measurement, not a
generalization one. Re-scoring the *original committed model* proves the size of
the effect:

| Original model, 9-class | macro-F1 |
|---|---:|
| all test rows | 0.6543 |
| **rows it had never seen** | **0.4474** |

**A memorization premium of 0.207.**

### What we did about it
- **De-duplicate the TRAIN split only** (`data.py`): drop training rows whose
  code hashes to a validation/test row. 3,475 rows (2.54%) removed. Evaluation
  splits are *never* altered — verified byte-for-byte, and train↔test hash
  overlap is now **0**.
- **Report two numbers, always**: all-rows *and* unseen-only.
- **Compare fairly**: a de-duplicated model is structurally denied the premium,
  so all-rows-vs-all-rows would punish it for fixing the leak. The only
  apples-to-apples bar is **unseen-only vs unseen-only**.

---

## 2. Results

Winner: **Calibrated LinearSVC** (C=2.0), chosen from 12 candidates on
validation macro-F1 — it beat LightGBM *and* fits in **83 s instead of 88 min**.

### The fair comparison (rows neither model was trained on)

| | macro-F1 |
|---|---:|
| Committed v1 baseline | 0.4474 |
| **This model (v2)** | **0.4854** |
| **Improvement** | **+0.0380** ✅ |

### Full test numbers

| Metric | All rows (n=17,121) | Unseen only (n=15,564) |
|---|---:|---:|
| macro-F1 | 0.5908 | 0.4854 |
| accuracy | 0.930 | 0.951 |
| weighted-F1 | 0.926 | 0.954 |

Per-class F1:

| Class | all rows | unseen |
|-------|---------:|-------:|
| safe | 0.967 | 0.977 |
| CWE-502 Insecure Deserialization | 0.814 | 0.667 |
| CWE-89 SQL Injection | 0.742 | 0.500 |
| CWE-94 Code Injection | 0.722 | 0.612 |
| CWE-79 XSS | 0.642 | 0.560 |
| MEMORY-OOB | 0.520 | 0.324 |
| CWE-476 NULL Deref | 0.422 | 0.364 |
| CWE-20 Input Validation | 0.310 | 0.246 |
| CWE-200 Information Exposure | 0.181 | 0.119 |

`CWE-20` and `CWE-200` are the weakest — both are broad, vaguely-defined
categories rather than concrete defect patterns.

---

## 3. Design

### Taxonomy — 9 classes
`safe` + CWE-89, CWE-79, CWE-94, CWE-502, CWE-20, CWE-200, CWE-476, and
**MEMORY-OOB**.

`MEMORY-OOB` merges CWE-119 / CWE-787 / CWE-125. Keeping them apart forced the
model to split hairs the data does not support (CWE-787 scored F1 **0.199**,
mostly confused with the other two). They describe one defect at three
granularities.

### Features
- TF-IDF **word** 1–2 grams (API names: `os.system`, `eval`)
- TF-IDF **char_wb** 3–5 grams (operators, string concatenation, format strings)
- **`SecurityFeatures`** — hand-crafted dense indicators (unsafe C functions,
  injection sinks, SQL-keyword + concatenation, deserialization calls, weak
  hashes, disabled TLS). Deliberately **not keyed on identifier names**.

40,045 features total.

### Training
- Train split de-duplicated, then `safe` down-sampled to a **4:1** ratio
  (49,030 rows). **`class_weight='balanced'` is deliberately NOT stacked on
  top** — doing both corrects the imbalance twice and measurably overshoots
  (0.4987 vs 0.5980 validation macro-F1).
- **Per-class decision offsets** tuned on **validation only**
  (`argmax_c(log P(c) − offset_c)`); the gain transferred to test (+0.022 val →
  +0.012 test).

### Evaluation integrity
- The test split is readable **only** under `--final`; every other run reports
  validation and writes `metrics_dev.json`, so the test set can't be
  multiple-tested by re-running the sweep.
- **One decision rule** lives in `mlscan/inference.py` and is used by *both* the
  tuner and the scanner, so the reported metric is literally what ships.
  (The tuner also refuses to save a model without `predict_proba`.)

### Hybrid: ML + rules
`mlscan/rules.py` is a deterministic AST-based engine for unambiguous
vulnerabilities. It exists because the classifier is **brittle to phrasing** —
the same SQL injection scored 80% with realistic identifiers and <50% with short
ones. Rules don't care what the variable is called.

Findings are tagged `source`: `ml`, `rule`, or `ml+rule`.

| Test snippet | Result |
|---|---|
| SQL injection (realistic names) | ✅ CWE-89 90% `ml+rule` |
| SQL injection (short names) | ✅ CWE-89 90% `ml+rule` |
| `eval(user_input)` | ✅ CWE-94 95% `ml+rule` |
| `os.system("ping " + host)` | ✅ CWE-94 92% `rule` *(ML alone misses this)* |
| `pickle.loads` / `yaml.load` | ✅ CWE-502 90% / 94% |
| `requests.get(..., verify=False)` | ✅ CWE-200 90% `rule` |
| parameterized SQL, `subprocess([...], shell=False)`, `sha256`, `a+b` | ✅ all clean |

**7/7 vulnerabilities caught, 0 false positives.** Note the published macro-F1
figures measure the **ML component alone**; the rule layer only adds
high-confidence hits on top.

---

## 4. Usage

```bash
pip install -r requirements-ml.txt

python -m mlscan path/to/file.py          # scan a file
python -m mlscan --code "def r(x): return eval(x)"
python -m mlscan path/to/file.py --json   # machine-readable
```
Exit code `1` when something is flagged, `0` when clean — usable as a CI gate.

```python
from mlscan.scanner import scan, model_info
scan(source_code)                 # hybrid ML + rules
scan(source_code, use_rules=False)  # ML only
model_info()                      # which artifact + decision rule is live
```

## 5. Reproducing

```bash
python -m mlscan.tune                      # dev sweep - validation only, never touches test
python -m mlscan.tune --final              # THE final run: scores test once, writes artifacts
python scripts/rescore_baseline.py         # re-score the v1 baseline on the same basis
```

## 6. Honest limitations

- **The dataset is ~92% C/C++** and its `language` column is unreliable, so the
  classifier is **language-agnostic**, not Python-specific. The *rule engine* is
  Python-specific and covers that gap for the language this project targets.
- **Unseen-only macro-F1 is 0.485** — useful for prioritization, not proof. Broad
  classes (CWE-20, CWE-200) remain weak.
- It is a **probabilistic classifier plus pattern rules, not a sound static
  analyzer**. No data-flow or taint analysis: it cannot prove user input actually
  reaches a sink.
- Rare classes have small validation support (18–99 rows), so differences below
  ~0.02 macro-F1 are noise.
- Fundamental ceiling: TF-IDF reads code as *text*, not as *code*. Fine-tuning a
  code transformer (CodeBERT/GraphCodeBERT) would be the next real step, but
  needs a GPU — out of scope for this free-tier, offline-by-design module.
