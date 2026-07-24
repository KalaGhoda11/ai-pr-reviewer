# ML Vulnerability Scanner (`mlscan`)

A **standalone, offline** vulnerability scanner: no LLM, no API keys, no network at
inference time. It ships three detectors over one 9-class taxonomy — a trained
classifier, a deterministic rule engine, and their union — and a benchmark harness
that measures all three on the same held-out rows.

This document is a measurement report, not a product page. Every number below was
read out of a committed artifact (`mlscan/model/benchmark.json`, `metrics_v2.json`,
`baseline_honest.json`, `metrics.json`), recorded in the module that produced it
(`mlscan/data.py`, `mlscan/tune.py` — cited where used), or re-measured against the
shipped model and corpus while writing this file. Where a result is unflattering it
is stated plainly, including one previously published claim that is **retracted in
§3.3**.

---

## 1. What the module is

| File | Role |
|---|---|
| `mlscan/data.py` | Loads the 9-class splits; TRAIN-only de-duplication; flags leaked eval rows |
| `mlscan/labels.py` | The 9-class taxonomy and the CWE merge map |
| `mlscan/features.py`, `security_features.py` | TF-IDF blocks + hand-crafted dense security indicators |
| `mlscan/tune.py` | Bake-off harness: selects, calibrates and saves the model |
| `mlscan/inference.py` | **The one decision rule**, shared by the tuner and the scanner |
| `mlscan/scanner.py` | Public `scan()` / `classify()` / `model_info()` |
| `mlscan/rules.py` | Deterministic AST + regex rule engine — **pure stdlib** (`re`, `ast`, `bisect`) |
| `mlscan/benchmark.py` | Head-to-head measurement of the three detectors |
| `mlscan/cli.py` | `python -m mlscan` |

Three detectors, scored on identical rows and reduced to the same primitive — *the
set of CWE ids the detector names for a row*, with "flagged" meaning that set is
non-empty:

| Detector | Definition | Dependencies |
|---|---|---|
| `ml` | `scanner.scan(..., use_rules=False)` | scikit-learn, numpy, scipy |
| `rules` | `rules.scan_rules(...)` | **none** (stdlib only) |
| `hybrid` | `scanner.scan(...)` — the union of both | scikit-learn, numpy, scipy |

Dataset: `ayshajavd/code-security-vulnerability-dataset` (HuggingFace), 171,252
rows after the taxonomy filter. Unless stated otherwise, results are on the
**TEST** split (n = 17,121; 1,466 vulnerable / 15,655 safe), seed 42, at the
shipped confidence threshold of **0.50**.

---

## 2. The leakage discovery

The first thing worth knowing about this dataset is that its published splits leak,
which makes any naive score on them partly a memorization measurement.

**9.1% of the test split (1,557 of 17,121 rows) is byte-identical to a training
row** — identical in the exact truncated string the model sees — and the
duplication is concentrated exactly where a macro-average is decided:

| Class | Test rows | Also in TRAIN, verbatim | % |
|---|---:|---:|---:|
| CWE-89 (SQL injection) | 81 | 67 | **82.7%** |
| CWE-200 (Info exposure) | 95 | 63 | 66.3% |
| MEMORY-OOB | 740 | 463 | 62.6% |
| CWE-20 (Input validation) | 246 | 147 | 59.8% |
| CWE-502 (Deserialization) | 40 | 22 | 55.0% |
| CWE-79 (XSS) | 46 | 24 | 52.2% |
| CWE-94 (Code injection) | 119 | 60 | 50.4% |
| CWE-476 (NULL deref) | 99 | 37 | 37.4% |
| safe | 15,655 | 674 | 4.3% |

A char-n-gram TF-IDF model evaluated on an exact duplicate of a training row is
close to a hash lookup. Re-scoring the *original committed v1 artifact* on the
current 9-class test split (`baseline_honest.json`, inference only, no retraining)
sizes the effect:

| v1 model, 9 classes | macro-F1 |
|---|---:|
| all test rows | 0.6543 |
| rows it had never seen | **0.4474** |
| **memorization premium** | **0.2069** |

### What was done about it

- **De-duplicate the TRAIN split only** (`mlscan/data.py`): drop every training row
  whose truncated-code hash appears in validation or test. Measured: **3,475 of
  137,036 labelled train rows (2.54%)** removed, leaving 133,561 before the safe
  down-sampling. Train↔test hash overlap afterwards is **0**.
- **Never touch the evaluation splits.** They keep their natural distribution and
  instead carry a `dup_of_train` flag, so "unseen-only" metrics can be computed
  without filtering anything away.
- **Report two numbers, always**: all-rows *and* unseen-only.
- **Compare fairly.** A de-duplicated model is structurally denied the premium, so
  an all-rows-vs-all-rows comparison against v1 would punish it for fixing the
  leak. The only apples-to-apples bar is **unseen-only vs unseen-only**.

---

## 3. Measured results

### 3.1 The classifier

Selected from **12 candidates** on *validation* macro-F1 (`metrics_v2.json`):
**`CalibratedClassifierCV(LinearSVC(C=2.0), cv=3)`**, validation macro-F1 **0.6093**
under plain argmax and **0.6317** after per-class offsets.

Cost, for context on the LightGBM it replaces:

| Model | Validation macro-F1 | Fit time | Artifact |
|---|---:|---:|---:|
| v2 winner — calibrated LinearSVC C=2.0 | 0.6093 | **83.5 s** | 6.8 MB |
| v2 bake-off LightGBM (cheap config) | 0.1125 | 637.9 s | — |
| v1 committed model — LightGBM (`metrics.json`) | 0.5743 † | **5,260.7 s (~88 min)** | 14.8 MB |

† v1's figure is an **11-class** macro-F1 and is not comparable to the 9-class
column; it is listed only to place the fit time and artifact size. The comparable
v1-vs-v2 number is the unseen-only test comparison immediately below.

**Test results** (the single sanctioned `--final` evaluation):

| Metric | All rows (n = 17,121) | Unseen only (n = 15,564) |
|---|---:|---:|
| macro-F1 | 0.5908 | **0.4854** |
| accuracy | 0.930 | 0.951 |
| weighted-F1 | 0.926 | 0.954 |

The fair comparison against the previous model — same rows, neither model trained
on any of them:

| Unseen-only macro-F1 | |
|---|---:|
| v1 baseline | 0.4474 |
| v2 (this model) | 0.4854 |
| **delta** | **+0.0380** |

Per-class F1:

| Class | Test support | All rows | Unseen only |
|---|---:|---:|---:|
| safe | 15,655 | 0.967 | 0.977 |
| CWE-502 Insecure deserialization | 40 | 0.814 | 0.667 |
| CWE-89 SQL injection | 81 | 0.742 | 0.500 |
| CWE-94 Code injection | 119 | 0.722 | 0.612 |
| CWE-79 XSS | 46 | 0.642 | 0.560 |
| MEMORY-OOB | 740 | 0.520 | 0.324 |
| CWE-476 NULL deref | 99 | 0.422 | 0.364 |
| CWE-20 Improper input validation | 246 | 0.310 | 0.246 |
| CWE-200 Information exposure | 95 | 0.181 | 0.119 |

§4 explains why the bottom two rows are what they are, and why that is a property
of the corpus rather than a tuning failure.

### 3.2 The three-detector benchmark

`python -m mlscan.benchmark` → `mlscan/model/benchmark.json`. Binary decision
("does this row contain a vulnerability"), same rows for all three detectors.

**All test rows (n = 17,121 — 1,466 vulnerable, 15,655 safe)**

| Detector | TP | FP | FN | Precision | Recall | **F1** | Fire rate on "safe" | Flag rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ml` | 734 | 369 | 732 | 0.665 | 0.501 | **0.571** | 0.024 | 0.064 |
| `rules` | 176 | 135 | 1,290 | 0.566 | 0.120 | **0.198** | 0.009 | 0.018 |
| `hybrid` | 768 | 493 | 698 | 0.609 | 0.524 | **0.563** | 0.032 | 0.074 |

**Unseen rows only (n = 15,564 — 583 vulnerable, 14,981 safe) — the generalization slice**

| Detector | TP | FP | FN | Precision | Recall | **F1** | Fire rate on "safe" |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ml` | 240 | 330 | 343 | 0.421 | 0.412 | **0.416** | 0.022 |
| `rules` | 52 | 131 | 531 | 0.284 | 0.089 | **0.136** | 0.009 |
| `hybrid` | 255 | 451 | 328 | 0.361 | 0.437 | **0.396** | 0.030 |

CWE-naming (one-vs-rest macro over the classes each detector names — see the
caveats in §10):

| Detector | Macro-F1, all rows | Macro-F1, unseen only |
|---|---:|---:|
| `ml` | 0.5293 | 0.4182 |
| `rules` | 0.1976 | 0.1270 |
| `hybrid` | 0.5031 | 0.3804 |

#### The headline honest finding: the hybrid does not beat ML alone on F1

Adding the rule layer **trades precision for recall and comes out slightly behind
on F1**, on both slices:

| Slice | Precision | Recall | F1 |
|---|---:|---:|---:|
| Unseen only: `ml` → `hybrid` | 0.421 → 0.361 (**−0.060**) | 0.412 → 0.437 (**+0.025**) | 0.416 → 0.396 (**−0.020**) |
| All rows: `ml` → `hybrid` | 0.665 → 0.609 (**−0.056**) | 0.501 → 0.524 (**+0.023**) | 0.571 → 0.563 (**−0.008**) |

Half of that is a property of the union rule: OR-ing two detectors can only flag
more rows, so recall cannot fall. Precision is **not** similarly determined — it
rises whenever the extra flags are more precise than the base detector's, and it
simply was not so here. Measured: the union adds **34 TP against 124 FP** on all
rows (768/493 vs 734/369) and **15 TP against 121 FP** on the unseen slice
(255/451 vs 240/330). If the single
number you care about is F1 on this corpus, the ML detector alone is the better
configuration and `scan(code, use_rules=False)` gives it to you. §3.4 sets out the
measured reasons the rule layer ships anyway — none of which is "it raises F1".

### 3.3 Retraction: "7/7 vulnerabilities caught, 0 false positives"

A previous version of this document reported **"7/7 vulnerabilities caught, 0 false
positives"** for the hybrid scanner. **That claim is withdrawn.** It came from seven
snippets chosen by the author of the rules, so it measured nothing but the author's
memory of what the rules cover. A test set you pick the inputs for is an anecdote.

The measured numbers on a corpus neither detector was fitted on:

| | Old claim | Measured |
|---|---|---|
| Rule-engine recall | "7/7" (1.00) | **0.089** unseen-only (0.120 all rows) |
| Rule-engine false positives | "0" | 135 FP on all rows; fires on **0.86%** of rows labelled safe (and see §3.4 — the pooled figure is mostly label noise) |

The snippet table survives below **only as an illustrative smoke test** of scanner
behaviour and provenance tagging. It is a demonstration, not evidence of accuracy,
and no accuracy claim in this document rests on it. Outputs re-verified against the
shipped artifact while writing this document:

| Illustrative snippet | Scanner output |
|---|---|
| SQL injection, realistic identifiers | CWE-89 90% `ml+rule` |
| SQL injection, one-letter identifiers | CWE-89 90% `ml+rule` |
| `eval(user_input)` | CWE-94 95% `ml+rule` |
| `os.system("ping " + host)` | CWE-94 92% `rule` *(ML alone is silent)* |
| `pickle.loads(blob)` | CWE-502 90% `ml+rule` |
| `yaml.load(s)` | CWE-502 94% `ml+rule` |
| `requests.get(url, verify=False)` | CWE-200 90% `rule` |
| parameterised SQL / `subprocess.run([...], shell=False)` / `hashlib.sha256` / `a + b` | clean |

The confidences shown are the rule engine's **hardcoded constants** for rule
findings, not calibrated probabilities (§10).

### 3.4 Why the rule engine still ships

Five measured reasons, none of them "better F1".

**(a) It almost never fires on real safe C.** Fire rate on rows the corpus labels
safe, broken out by sub-corpus — pooling these hides the whole story:

| Source | Safe rows | `rules` fire rate | `ml` fire rate |
|---|---:|---:|---:|
| `bigvul` (real C functions) | 14,197 | **0.0036** | 0.0151 |
| `enriched_bigvul_primevul` | 1,003 | 0.0120 | 0.0907 |
| `labeled_dataset` | 455 | 0.1582 | 0.1407 |

On 14,197 real safe C functions the rules fire on **0.36%** of them — roughly a
quarter of the ML detector's rate. The 15.8% on `labeled_dataset` is largely
**corpus mislabelling, not detector error**: that sub-corpus labels
`eval(user_input)` and `subprocess.run(user_input, shell=True)` as *safe*.

**(b) Its recall is concentrated where the ML corpus is thin.** The training corpus
is ~92% C; the rules were written for everything else:

| Language group (all rows) | Detector | Precision | Recall | F1 | Fire rate on safe |
|---|---|---:|---:|---:|---:|
| non-C (n = 1,043) | `ml` | 0.872 | 0.717 | 0.787 | 0.141 |
| non-C | `rules` | 0.657 | 0.231 | 0.342 | 0.161 |
| non-C | `hybrid` | 0.778 | 0.762 | 0.770 | 0.292 |
| C family (n = 16,078) | `ml` | 0.500 | 0.352 | 0.413 | 0.020 |
| C family | `rules` | 0.376 | 0.044 | 0.078 | **0.004** |
| C family | `hybrid` | 0.463 | 0.360 | 0.405 | 0.024 |

The rule engine's recall is **5× higher on non-C code (0.231) than on C (0.044)**,
which is exactly the axis on which it complements a C-dominated classifier. Note
that even on non-C the hybrid's F1 (0.770) sits just below ML alone (0.787): the
recall gain (+0.045) does not pay for the precision loss (−0.094) on this corpus.

**(c) On patched code it is three times quieter.** The corpus carries a `code_fixed`
column (the patched version of a vulnerable sample). Firing on the patch is
*mostly* a false positive — an upper bound, see §10:

| Detector | Fires on patched code (666 pairs) | Fires **only** on the patched side | Patch sensitivity |
|---|---:|---:|---:|
| `ml` | 0.230 | 12 / 666 (0.018) | 0.596 |
| `rules` | **0.069** | **2 / 666 (0.003)** | 0.477 |
| `hybrid` | 0.297 | 12 / 666 (0.018) | 0.508 |

"Fires only on the patched side" is the unambiguous case — the detector was silent
on the vulnerable version and speaks up on the fix, so nothing it could have been
reacting to was removed. The rules do that on **2 of 666 pairs**.

**(d) It helps precisely on the injection classes it was written for.** Per-CWE F1
(all rows), hybrid vs ML alone:

| CWE | `ml` | `hybrid` | Δ |
|---|---:|---:|---:|
| CWE-89 SQL injection | 0.727 | 0.737 | **+0.010** |
| CWE-79 XSS | 0.630 | 0.653 | **+0.022** |
| MEMORY-OOB | 0.509 | 0.499 | −0.011 |
| CWE-476 NULL deref | 0.416 | 0.346 | −0.070 |
| CWE-94 Code injection | 0.649 | 0.585 | −0.064 |
| CWE-502 Deserialization | 0.854 | 0.758 | −0.096 |

The two classes that improve are the two the AST rules model properly (SQL string
building, unescaped template output). The classes that regress are ones where a
lexical rule fires on a construct the model had already correctly *declined* to
flag. `rules` alone earns CWE-502 F1 0.643 and CWE-94 F1 0.521, and **0.000** on
CWE-20, CWE-200 and CWE-476 — it has no rules for those and says so.

**(e) Properties the classifier structurally cannot offer.**

- **Determinism.** Same input, same output, forever — no artifact version, no
  floating-point drift, no threshold to re-tune.
- **A line number and an evidence quote.** Every rule finding carries a `rule_id`,
  an exact line, and the trimmed source line it fired on. The classifier emits a
  probability over a whole snippet and can point at nothing.
- **Immunity to identifier renaming.** A bag-of-n-grams model is brittle to
  phrasing; the rules read structure, not names. The illustrative pair in §3.3 is
  the smallest demonstration: the same SQL injection with realistic identifiers and
  with one-letter identifiers is the same finding to the rule engine.
- **It is what the deployed reviewer actually runs.** `mlscan/rules.py` imports only
  `re`, `ast` and `bisect`, so the live GitHub PR reviewer in `app/` uses it with
  **zero new dependencies** (§8). The ML side needs scikit-learn + numpy + scipy and
  does not fit the deployment.

---

## 4. Two findings about the labels

### 4.1 CWE-200 is unlearnable from a snippet, and there is proof

The usual explanation for a weak class ("too broad", "needs more data") is
untestable. Here it is testable, because 66.3% of CWE-200's test rows were in the
original train split *verbatim*. A model that cannot recover a class even by rote
has no representation of it at all.

Recall on the **memorized** test rows (rows byte-identical to a training row),
measured against the shipped v2 artifact:

| Class | Memorized test rows | Recall on them |
|---|---:|---:|
| CWE-502 | 22 | 0.95 |
| CWE-89 | 67 | 0.87 |
| CWE-79 | 24 | 0.83 |
| CWE-94 | 60 | 0.73 |
| MEMORY-OOB | 463 | 0.59 |
| CWE-476 | 37 | 0.41 |
| CWE-20 | 147 | 0.27 |
| **CWE-200** | **63** | **0.14** |

Cause, measured on the corpus: **824 of the 870 CWE-200 rows are C/C++ from the two
BigVul-derived sources** (406 `bigvul` + 418 `enriched_bigvul_primevul`; 822 C, 2
C++) — kernel and library functions. In those samples the information leak is a
**missing `memset`/sanitization**, visible only in the commit diff, not in the
function body. You cannot detect an *absence* from a snippet that never contained
it. This is an absence of signal in the representation, not a tuning failure.

### 4.2 CWE-20 is a mislabelled catch-all

**488 of 2,473 CWE-20 rows (19.7%) have a byte-identical twin elsewhere in the
corpus carrying a different label.** Counting the rows on the other side of those
collisions:

| Partner label | Rows sharing a byte-identical code string with a CWE-20 row |
|---|---:|
| CWE-89 | 356 |
| CWE-79 | 103 |
| MEMORY-OOB | 92 |
| CWE-94 | 44 |
| safe | 17 |
| CWE-476 / CWE-502 / CWE-200 | 3 / 2 / 1 |

A concrete case (verified): the line

```csharp
string commandText = $"SELECT * FROM Users WHERE Username = '{userInput}'";
```

appears as **CWE-89** in `source=cybernative_dpo` (`language=C#`) and, byte for
byte, as **CWE-20** in `source=labeled_dataset` (`language=Java`) — the two
sub-corpora disagree about both the label *and* the language for identical bytes,
and in this case both copies sit in the **train** split, so the model is fitted on
contradictory targets for the same input. CWE-20 is where each sub-corpus put
whatever it did not classify more specifically. A share of the model's CWE-20 error
is therefore **mathematically irreducible**: no function of the code can produce two
different labels for identical input.

This is also why CWE-20 is *not* in the default weak-class list (§5): unlike
CWE-200 it is a labelling artefact rather than an unlearnable target, its rows are
useful negatives, and folding or dropping it was measured to cost more on CWE-89
than it buys.

---

## 5. The class-set decision (and why the easy +0.10 was refused)

Dropping the two weak classes raises the headline macro-F1 substantially (measured
via `--drop-weak-classes`; the reduced-taxonomy rationale is recorded in
`mlscan/tune.py`):

| Class set | macro-F1 (test, all rows) | Δ |
|---|---:|---:|
| **All 9 classes (shipped default)** | **0.5908** | — |
| minus CWE-200 | 0.6421 | +0.0513 |
| minus CWE-200 and CWE-20 | 0.6896 | +0.0988 (+16.7% relative) |

**Provenance of those two figures:** they are not separate archived runs. They are
§3.1's nine per-class F1s re-averaged over the reduced class set — pure arithmetic,
reproducible by hand: `(0.5908×9 − 0.181)/8 = 0.6420`, then `(… − 0.310)/7 =
0.6895`. The separate check that retraining without the dropped classes moves every
*surviving* class by at most **+0.0135** was run with `--drop-weak-classes` and is
**not archived** in `mlscan/model/` (the only artifact there records
`"drop_weak_classes": false`, i.e. the shipped default).

**Essentially 100% of the gain is therefore a denominator effect.** The model does not get better; the
average is simply taken over fewer, easier classes. A 7-class macro-F1 printed next
to a 9-class one is not a comparison.

So the **9-class number stays the default and the headline**. The
`--drop-weak-classes` flag exists (CWE-200 only, by default), it is **OFF unless
asked for**, and when it is on the run prints a disclosure banner, records
`weak_class_exclusion` with `comparable_to_9class: false`, measures the coverage
given up, and refuses to print the v1 baseline comparison at all. Any use of it
must be reported as a **coverage reduction**, not an improvement.

---

## 6. A negative result: the offset search grid had nothing left to give

Reported because a measured non-result is still a result, and because it closes off
an obvious "you should have searched harder" objection.

The decision rule is `argmax_c(log P(c) − offset_c)`, with offsets fitted on
validation only. Two follow-up experiments on the shipped v2 artifact
(recorded in `mlscan/tune.py`):

| Change | Validation macro-F1 | Test macro-F1 | Verdict |
|---|---|---|---|
| `--offset-max` 2.0 / 3.0 / 6.0 / 12.0 | 0.6317 (identical) | 0.5908 (identical) | **No-op.** All four return the *bit-identical* offset vector; the shipped offsets peak at magnitude 0.667, an order of magnitude inside the boundary. Range was never the binding constraint. |
| `--offset-step` 0.25 → 0.05 | **+0.005** | **−0.010** | A finer grid fits the tuning split. Rejected. |

No gain was available, so **the model was not re-tuned and the test split was not
looked at a second time**. For scale, the offset calibration that did ship
transfers as `val_gain 0.0224 → test_gain 0.0118` (`offset_transfer` in
`metrics_v2.json`): about half the in-sample gain is optimism.

---

## 7. Design

### Taxonomy — 9 classes

`safe` + CWE-89, CWE-79, CWE-94, CWE-502, CWE-20, CWE-200, CWE-476 and
**MEMORY-OOB**.

`MEMORY-OOB` merges CWE-119 / CWE-787 / CWE-125. Kept apart, the model was forced
to split hairs the data does not support — in the v1 11-class report CWE-787 scored
F1 **0.199**, mostly confused with the other two. They describe one defect at three
granularities.

### Features (40,045 total)

- TF-IDF **word** 1–2 grams (API names: `os.system`, `eval`)
- TF-IDF **char_wb** 3–5 grams (operators, concatenation, format strings)
- **`SecurityFeatures`** — dense hand-crafted indicators (unsafe C functions,
  injection sinks, SQL keyword + concatenation, deserialization calls, weak hashes,
  disabled TLS), followed by `MaxAbsScaler` so the matrix stays sparse.
  Deliberately **not keyed on identifier names**.

One `FeatureUnion` is fitted on TRAIN once and reused by every candidate.

### Training

- Train split de-duplicated, then `safe` down-sampled to a **4:1** ratio
  (49,030 rows). **`class_weight='balanced'` is deliberately not stacked on top** —
  doing both corrects the same imbalance twice and overshoots. Measured validation
  macro-F1 (recorded in `mlscan/data.py`): 0.4987 for 1:1 + balanced, 0.5661 for
  1:1 unweighted, **0.5980** for 4:1 unweighted.
- Every candidate must expose `predict_proba`; margin-only estimators are wrapped
  in `CalibratedClassifierCV(cv=3)` fitted on TRAIN only. An uncalibrated LinearSVC
  used to be selectable and would have made `scanner.scan()` raise on first use.

### Evaluation integrity

- The test split is readable **only** under `--final`. Every other run drops it from
  memory immediately after loading, reports validation, and writes
  `metrics_dev.json` — so the test set cannot be multiple-tested by re-running the
  sweep. `--final` is also rejected together with `--sample`.
- **One decision rule** lives in `mlscan/inference.py` and is used by *both* the
  tuner and the scanner. `metrics_v2.json` records the cross-check: 17,095
  validation rows re-decided through `predict_with_offsets`, **0 disagreements**.
- `mlscan/benchmark.py` batches the ML pass for speed, then re-scores 25 rows
  through the real `scanner.scan` and aborts on any disagreement.
- Sampled benchmark runs write `benchmark_smoke.json`, so a smoke test can never
  overwrite — or be mistaken for — the headline artifact.

---

## 8. Integration into the deployed PR reviewer

`mlscan/rules.py` is pure stdlib, so `app/security_scan.py` wires deterministic
findings into the live GitHub PR reviewer with **zero new dependencies** —
`requirements.txt` is unchanged, and importing `app.main` loads no `sklearn`,
`numpy`, `pandas` or `scipy` (verified: none of those appear in `sys.modules`
afterwards). The ML side is never imported by `app/`: one `scan()` measured +167 MB
RSS, which does not fit the 512 MB deployment.

- Rule findings are passed into the Gemini prompt as **already reported, do not
  repeat**, so the model spends its budget on logic and design issues instead of
  restating pattern matches.
- Findings carry provenance: `source` = `llm` | `rules` | `llm+rules`, badged in the
  posted comment.
- The rules run over each file's **reconstructed new-side image**, not the raw diff
  — a diff contains deleted lines, so scanning it directly would report a PR that
  *removes* `yaml.load` as introducing CWE-502.
- Findings on lines the PR did not add are dropped (pre-existing debt would
  otherwise be re-posted on every push).
- Severity is derived from the CWE class, never from the rule's `confidence` float.
- The static scan is isolated: if it raises, it logs and the Gemini-only review
  proceeds unchanged.

---

## 9. Usage and reproduction

```bash
pip install -r requirements-ml.txt

python -m mlscan path/to/file.py          # scan a file
python -m mlscan --code "def r(x): return eval(x)"
python -m mlscan path/to/file.py --json   # machine-readable
```

Exit code `1` when something is flagged, `0` when clean — usable as a CI gate.

```python
from mlscan.scanner import scan, model_info
scan(source_code)                    # hybrid: ML + rules (higher recall)
scan(source_code, use_rules=False)   # ML only (higher precision and F1 on this corpus)
model_info()                         # which artifact and decision rule is live
```

```bash
python -m mlscan.tune                 # dev sweep — validation only, never touches test
python -m mlscan.tune --final         # THE final run: scores test once, writes artifacts
python -m mlscan.benchmark            # three-detector benchmark -> model/benchmark.json
python -m mlscan.benchmark --sample 400   # smoke run -> benchmark_smoke.json
python scripts/rescore_baseline.py    # re-score the v1 baseline on the same basis
```

Iterate rule changes with `--split validation`; look at `--split test` once.

---

## 10. Caveats that travel with these numbers

These are recorded in `benchmark.json` itself and are not optional footnotes.

1. **`code_fixed` fire rates are an UPPER BOUND on the false-positive rate.** A
   patch fixes *one* defect; the function may still contain another, many "fixes"
   in this corpus are LLM-authored and keep the dangerous construct (an `eval` fix
   that becomes `eval(compile(validated_tree))` still contains `eval`), and a rule
   may legitimately fire on a different CWE than the one patched.
2. **Pooled `fire_rate_on_safe` measures label noise as much as detector
   precision.** `source=labeled_dataset` labels `eval(user_input)` and
   `subprocess.run(user_input, shell=True)` as *safe*. Always read `by_source`.
3. **Per-CWE metrics are one-vs-rest over the *set* of CWEs a detector names**, so a
   detector gets credit whenever the right CWE is anywhere in its output. This is
   generous by construction; at the shipped 0.50 threshold the ML set holds at most
   one class, so it loosens mainly the `rules` and `hybrid` columns.
4. **Only the `unseen_only` slice measures generalization.** `all_rows` includes
   rows byte-identical to a training row.
5. **The rule engine's confidences are hardcoded constants, not calibrated
   probabilities.** They are deliberately excluded from the benchmark and are never
   printed as a percentage in a PR comment.
6. Rare classes have small support (test: 40–246 rows outside `safe`/MEMORY-OOB), so
   differences below roughly 0.02 macro-F1 are noise.

## 11. Limitations

- **The dataset is ~92% C/C++** and its `language` column is unreliable (the same
  bytes appear under two different languages — see §4.2), so the classifier is
  **language-agnostic**, not Python-specific. The rule engine is Python-first, with
  bounded regex fallbacks for C, PHP, Java, JavaScript, Go, Ruby and C#.
- **Unseen-only macro-F1 is 0.4854.** That is useful for prioritization, not proof.
- **This is a probabilistic classifier plus pattern rules, not a sound static
  analyzer.** There is no whole-program data-flow or taint analysis; it cannot prove
  user input reaches a sink. The rules deliberately omit checks that would need that
  knowledge (bare `memcpy`, `malloc` arithmetic) because measured on this corpus
  they run near chance precision, which is worse than silence.
- **Fundamental ceiling:** TF-IDF reads code as *text*, not as *code*. Fine-tuning a
  code transformer (CodeBERT / GraphCodeBERT) is the next real step, but needs a GPU
  — out of scope for this free-tier, offline-by-design module.

## 12. Tests

**241 tests** collected and passing (up from 118), covering the rule engine and its
safe counterparts, the de-duplication invariants (eval splits unmodified, zero
train↔test overlap), the benchmark's metric arithmetic on hand-built fixtures, and
the deployed reviewer's diff reconstruction and provenance merging. The
sklearn-dependent tests `importorskip`, so the suite also runs in the ML-free
deployment environment.

```bash
python -m pytest            # full suite
```
