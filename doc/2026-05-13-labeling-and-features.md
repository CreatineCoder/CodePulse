# 2026-05-13

### Status snapshot

| Stage                        | State        | Notes                                               |
|------------------------------|--------------|-----------------------------------------------------|
| Mining pipeline              | Done         | `src/pipeline/miner.py`                             |
| Exploration notebook         | Done         | `notebooks/explore_db.ipynb`                        |
| Labeling function (code)     | Done         | `src/labeling/labeler.py` — noisy-OR, 4 signals     |
| commit_labels table          | Done         | 1690 rows, saved to SQLite                          |
| Sanity checks (notebook)     | Done         | PR label coverage, keyword sample, top labels       |
| Feature engineering (table)  | Done         | `src/features/feature_engineer.py` — file_labels   |
| Time-split scaffolding       | Not yet      |                                                     |
| Model training               | Not yet      |                                                     |
| API                          | Not yet      |                                                     |

---

### Decision: Bug signal combination — noisy-OR with continuous confidence

**What:** instead of a binary `is_bug` flag, each commit gets a continuous
`bug_confidence` score in [0, 1] computed via noisy-OR over matched signals.

```
score = 1 - prod(1 - p_i)   for each matched signal i
```

Signal probabilities (tunable):

| Signal    | p    | Reasoning                                                   |
|-----------|------|-------------------------------------------------------------|
| `keyword` | 0.30 | Low — "fix typo", "fix tests" are common false positives    |
| `issue`   | 0.60 | Medium — issue could be a feature request, not a bug        |
| `pr`      | 0.80 | High — human-applied bug label on a PR                      |
| `revert`  | 0.95 | Very high — reverts almost always undo a broken change      |

**Why:** a discrete label wastes signal strength. A commit linked to both a
bug-labeled PR and a bug-linked issue is more certain than one with only a
keyword match. noisy-OR encodes exactly that — independent signals combine
multiplicatively.

**Alternatives rejected:** max-of-signals (doesn't compound evidence);
fixed binary threshold (loses the gradient between weak and strong signal).

---

### Decision: PR/issue bug detection — labels AND title

**What:** a PR or issue is flagged as bug-related if its `labels` field OR its
`title` field matches a bug pattern (via `has_bug_keyword`).

**Why:** only 10.5% of PRs in httpie/cli have any label at all. Restricting to
labels alone would miss many real bug-fix PRs whose titles say "Fix crash when..."
or "Resolve auth error". Titles carry consistent signal even when labels are absent.

**Alternatives rejected:** labels only (too sparse — 1.4% bug-labeled PRs);
body text (too noisy and expensive to parse reliably).

---

### Decision: File-level label — avg_bug_confidence > 0.5

**What:** a file is labeled `is_bug_prone = 1` if the average `bug_confidence`
across all commits that touched it exceeds 0.5.

**Why:** two simpler alternatives were tried and rejected:
- `num_bug_commits > 0` — 70% of files flagged; too loose. Config files and
  templates get flagged because a nearby commit happened to use a bug keyword.
- `avg_bug_confidence > 0.5` — 3.4% of files flagged after first pass (very
  strict because avg is dragged down by many clean commits on the same file).

Still under active tuning. `max_bug_confidence > 0.5` is the next candidate:
a file is bug-prone if at least one commit touching it had strong bug signal.

**Alternatives rejected:** `num_bug_commits > 0` (70% positive rate — too noisy);
manual threshold on `max_bug_confidence` still being evaluated.

---

### Decision: Feature table shape — one row per file

**What:** `file_labels` table aggregates `events` joined with `commit_labels`
into one row per `file_path` with the following columns:

| Column                 | Source                        |
|------------------------|-------------------------------|
| `num_commits`          | count distinct commit_hash    |
| `num_authors`          | count distinct author_email   |
| `total_lines_added`    | sum lines_added               |
| `total_lines_deleted`  | sum lines_deleted             |
| `total_lines_changed`  | derived sum                   |
| `first_seen`           | min committed_at              |
| `last_seen`            | max committed_at              |
| `age_days`             | last_seen − first_seen        |
| `num_bug_commits`      | sum is_bug from commit_labels |
| `max_bug_confidence`   | max bug_confidence            |
| `avg_bug_confidence`   | mean bug_confidence           |
| `is_bug_prone`         | label (threshold TBD)         |

**Why:** everything links to files through the events table (one row per
commit–file pair). Aggregating there gives the natural unit for prediction:
"given this file's history, will it be involved in a bug fix?"

---

### Open questions

- Final threshold for `is_bug_prone`: `avg > 0.5` is too strict, `num > 0` is too loose. Try `max_bug_confidence > 0.5`?
- Time-split scaffolding: features must be computed at cutoff T, label = bug in [T, T+90d]. Implement before model training.
- Which files to exclude from training: auto-generated files, docs, config-only files?
