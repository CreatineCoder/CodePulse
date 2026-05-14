# 2026-05-14

### Status snapshot

| Stage                        | State        | Notes                                               |
|------------------------------|--------------|-----------------------------------------------------|
| Mining pipeline              | Done         | now supports multi-repo append                      |
| Mined repos                  | Done         | httpie/cli, pallets/flask                           |
| Labeling function (code)     | Done         | `src/labeling/labeler.py`                           |
| Source-file filter           | Done         | excludes `.md`, `.yml`, configs                     |
| Label threshold              | Decided      | `avg_bug_confidence > 0.23`                         |
| Temporal split (90-day)      | Done         | implemented in `feature_engineer.py`                |
| Feature engineering          | Partial      | 9 base features; ratio/velocity features pending    |
| Model training               | Not yet      | next step                                           |
| API                          | Not yet      |                                                     |

---

### Decision: Label threshold — `avg_bug_confidence > 0.23`

**What:** a file is `is_bug_prone = 1` iff its `avg_bug_confidence` across all
pre-cutoff commits exceeds 0.23.

**Why:** built a threshold-vs-percent-bug-prone plot in `explore_db.ipynb`
(cell `label-threshold`). Two curves — `avg_bug_confidence` and
`max_bug_confidence` — over thresholds 0→1.

At 0.23 on the `avg` curve, ~25% of files are flagged — a healthy class
balance for training. Higher thresholds collapse the positive class too far
(3% at 0.5); lower thresholds bloat it (70% at 0).

**Alternatives rejected:** `num_bug_commits > 0` (70% positive — too loose);
`avg > 0.5` (3% positive — too strict); `max > 0.5` (still under evaluation
but ranks files with one keyword-only commit the same as a file with three
PR-labeled bug fixes).

---

### Decision: Filter to source files only

**What:** `feature_engineer.py` drops any file whose extension is not in
`{.py, .js, .ts, .jsx, .tsx, .java, .go, .c, .cpp, .cc, .h, .hpp, .rb, .rs,
.cs, .php, .swift, .kt, .scala, .sh, .bash}`.

**Why:** `.md`, `.yml`, `.json`, `.env` files can't carry runtime bugs — a
"bug fix" commit touching `README.md` is almost always a docs change. Keeping
them adds noise both to the label (false positive bug-prone files) and the
features (config-only files dominate frequent-change patterns).

Result: 353 files → 179 source files in httpie/cli alone.

**Alternatives rejected:** allow-all (noise); path-based filter on `tests/`
and `docs/` (too aggressive — those dirs can still contain real source code
like build scripts).

---

### Decision: Multi-repo storage with `repo` column

**What:** `save_to_sqlite` adds a `repo` column to every table and uses
`if_exists="append"` instead of `replace`. Idempotent re-runs are handled by
`DELETE FROM <table> WHERE repo = ?` before each insert.

**Why:** training on one repo (179 files) is too little data. Mining
additional repos and stacking them gives the model more rows to learn from.
A `repo` column keeps lineage so we can stratify train/test splits by repo
later if needed.

Currently mined:
- `httpie/cli`: 4,281 events, 727 PRs, 907 issues
- `pallets/flask`: 9,254 events, 2,822 PRs, 2,735 issues

**Risk noted:** different repos have different label cultures and codebase
sizes. Mitigated by sticking to similar active Python repos and planning to
add normalized features like `num_bug_commits / num_commits`.

---

### Decision: Temporal split — single cutoff per repo, 90-day label window

**What:** instead of using all of history for both features and labels,
features and labels are split temporally per repo:

```
T       = max(committed_at for repo) - 90 days
features = aggregate events where committed_at < T
label    = 1 if any commit in [T, T + 90d] for this file had bug_confidence > 0.23
```

**Why:** the previous implementation computed `num_bug_commits` over the
entire history, including the bug fixes the model is supposed to predict.
That's data leakage. The new flow mimics inference: "given everything we
knew about this file up to T, will it be bug-fixed in the next 90 days?"

**Result:** label rate dropped from 24% to 2.9%. This is expected — only a
small fraction of files are touched by bug fixes in any 90-day window. The
class imbalance is real and will need handling during training (class
weights, SMOTE, or threshold tuning at inference).

**Alternatives rejected:** rolling 90-day windows (Option B — more rows per
file, but more complex; deferred to v2 once baseline model is running);
no temporal split (data leakage).

---

### Open questions

- Class imbalance strategy for training: `class_weight="balanced"` in
  scikit-learn, oversampling (SMOTE), or threshold tuning at inference?
- Stratify train/test by `repo` to ensure both splits cover both repos?
- Add normalized features (`bug_commit_ratio`, `commits_per_month`,
  `days_since_last_commit`) before or after training the baseline model?
  Plan: train baseline first, add features based on what fails.
