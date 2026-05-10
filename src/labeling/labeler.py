import re
import sqlite3
import pandas as pd

from config.settings import DB_PATH

# Bug-signal patterns. Each entry is a regex stem that allows common suffixes
# (fix/fixes/fixed/fixing). Case-insensitive via re.IGNORECASE.
BUG_KEYWORD_PATTERNS = [
    r"fix(?:es|ed|ing)?",
    r"bug(?:s|fix|fixes|gy)?",
    r"hotfix(?:es|ed)?",
    r"patch(?:es|ed|ing)?",
    r"broke(?:n)?",
    r"break(?:s|ing)?",
    r"crash(?:es|ed|ing)?",
    r"workaround(?:s)?",
    r"regress(?:ion|ions|ed)?",
    r"error(?:s)?",
    r"fail(?:s|ed|ing|ure|ures)?",
    r"defect(?:s)?",
    r"issue(?:s)?",
    r"typo(?:s)?",
    r"glitch(?:es|ed)?",
    r"revert(?:s|ed|ing)?",
    r"corrupt(?:s|ed|ion)?",
    r"leak(?:s|ed|ing)?",
    r"incorrect(?:ly)?",
    r"invalid",
    r"wrong(?:ly)?",
    r"mistake(?:s|n)?",
    r"problem(?:s|atic)?",
    r"resolve(?:s|d)?",
]
BUG_KEYWORD_RE = re.compile(
    r"\b(?:" + "|".join(BUG_KEYWORD_PATTERNS) + r")\b", re.IGNORECASE
)

# Bug-label patterns. Matched as substrings inside labels (case-insensitive) to
# handle the wide variation across repos: "bug", "Bug", "type: bug", "kind/bug",
# "C-bug", "🐛 bug", "area/bug", "T-bug", "bugfix", "defect", etc.
BUG_LABEL_PATTERNS = [
    "bug",
    "defect",
    "regression",
    "hotfix",
    "crash",
    "broken",
    "fault",
    "fix-needed",
    "needs-fix",
    "incident",
    "outage",
    "blocker",
]
BUG_LABEL_RE = re.compile("|".join(BUG_LABEL_PATTERNS), re.IGNORECASE)

# Per-signal probability that the signal *alone* indicates a bug.
# Combined with Noisy-OR: score = 1 - prod(1 - p_i) over matched signals.
# Revert is 0.95 (not 1.0) to leave room for false positives like reverting a
# feature merge rather than a true bug.
SIGNAL_PROBS = {
    "keyword": 0.30,
    "issue": 0.60,
    "pr": 0.80,
    "revert": 0.95,
}


def noisy_or(probs: list[float]) -> float:
    """Combine independent signal probabilities. Returns 0 if probs is empty."""
    p_not_bug = 1.0
    for p in probs:
        p_not_bug *= 1.0 - p
    return 1.0 - p_not_bug


def has_bug_keyword(message: str) -> bool:
    if not message:
        return False
    return bool(BUG_KEYWORD_RE.search(message))


def has_bug_label(label_str: str) -> bool:
    """Check if any of the comma-separated labels contains a bug-related pattern."""
    if not label_str:
        return False
    return bool(BUG_LABEL_RE.search(label_str))


def build_commit_labels(conn) -> pd.DataFrame:
    events = pd.read_sql("SELECT * FROM events", conn)
    pull_requests = pd.read_sql("SELECT * FROM pull_requests", conn)
    pr_commits = pd.read_sql("SELECT * FROM pr_commits", conn)
    issues = pd.read_sql("SELECT * FROM issues", conn)
    commit_issues = pd.read_sql("SELECT * FROM commit_issues", conn)

    # A PR/issue is bug-flagged if EITHER its label matches a bug pattern
    # OR its title contains a bug keyword (titles often carry signal even
    # when labels are missing).
    pull_requests["is_bug_pr"] = pull_requests["labels"].apply(
        has_bug_label
    ) | pull_requests["title"].apply(has_bug_keyword)
    issues["is_bug_issue"] = issues["labels"].apply(has_bug_label) | issues[
        "title"
    ].apply(has_bug_keyword)

    bug_pr_numbers = set(pull_requests.loc[pull_requests["is_bug_pr"], "pr_number"])
    bug_issue_numbers = set(issues.loc[issues["is_bug_issue"], "issue_number"])

    # Get unique commits with their messages and revert flag
    commits = events[["commit_hash", "commit_message", "is_revert"]].drop_duplicates(
        subset="commit_hash"
    )

    # Build commit→bug-PR and commit→bug-issue lookups
    pr_commits_bug = pr_commits[pr_commits["pr_number"].isin(bug_pr_numbers)]
    commits_in_bug_pr = set(pr_commits_bug["commit_hash"])

    commit_issues_bug = commit_issues[
        commit_issues["issue_number"].isin(bug_issue_numbers)
    ]
    commits_linked_to_bug_issue = set(commit_issues_bug["commit_hash"])

    rows = []
    for _, c in commits.iterrows():
        signals = []
        probs = []

        if has_bug_keyword(c["commit_message"]):
            signals.append("keyword")
            probs.append(SIGNAL_PROBS["keyword"])

        if c["commit_hash"] in commits_linked_to_bug_issue:
            signals.append("issue")
            probs.append(SIGNAL_PROBS["issue"])

        if c["commit_hash"] in commits_in_bug_pr:
            signals.append("pr")
            probs.append(SIGNAL_PROBS["pr"])

        if c["is_revert"]:
            signals.append("revert")
            probs.append(SIGNAL_PROBS["revert"])

        score = noisy_or(probs)

        rows.append(
            {
                "commit_hash": c["commit_hash"],
                "bug_confidence": round(score, 4),
                "signals_matched": ",".join(signals) if signals else None,
                "num_signals": len(signals),
                "is_bug": score > 0,
            }
        )

    return pd.DataFrame(rows)


def run():
    conn = sqlite3.connect(DB_PATH)

    print("Building commit labels...")
    commit_labels = build_commit_labels(conn)

    commit_labels.to_sql("commit_labels", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved commit_labels: {len(commit_labels)} rows")
    print("\nSignal distribution:")
    print(commit_labels["signals_matched"].value_counts(dropna=False))
    print(
        f"\nBug commits: {commit_labels['is_bug'].sum()} / {len(commit_labels)} "
        f"({commit_labels['is_bug'].mean():.1%})"
    )
    print("\nConfidence distribution (binned):")
    bins = [0, 0.001, 0.3, 0.6, 0.8, 0.95, 1.01]
    labels = ["0", "0-0.3", "0.3-0.6", "0.6-0.8", "0.8-0.95", "0.95+"]
    print(
        pd.cut(commit_labels["bug_confidence"], bins=bins, labels=labels, right=False)
        .value_counts()
        .sort_index()
    )

    return commit_labels


if __name__ == "__main__":
    run()


"""Saved commit_labels: 1690 rows

Signal distribution:
signals_matched
NaN                  1254
keyword               313
keyword,pr             53
issue                  27
keyword,issue          26
keyword,revert          8
pr                      5
keyword,issue,pr        2
issue,pr                1
keyword,pr,revert       1
Name: count, dtype: int64

Bug commits: 436 / 1690 (25.8%)

Confidence distribution (binned):
bug_confidence
0           1254
0-0.3          0
0.3-0.6      313
0.6-0.8       53
0.8-0.95      61
0.95+          9
Name: count, dtype: int64"""
