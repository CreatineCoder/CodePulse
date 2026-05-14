import os
import re
import sqlite3

import pandas as pd
from pydriller import Repository
from github import Github

from config.settings import GITHUB_TOKEN, DB_PATH, CLONE_DIR


# Match "Merge pull request #123" or similar
PR_MERGE_RE = re.compile(r"Merge pull request #(\d+)", re.IGNORECASE)
# Match "fixes #123", "closes #45", "resolves #67" — common issue references
ISSUE_REF_RE = re.compile(
    r"(?:fix(?:es|ed)?|close(?:s|d)?|resolve(?:s|d)?)\s+#(\d+)", re.IGNORECASE
)
# Match "(#123)" at end of commit message — squash-merge convention
SQUASH_PR_RE = re.compile(r"\(#(\d+)\)")


def parse_repo_name(repo_url: str) -> str:
    """Extract 'owner/repo' from a GitHub URL."""
    match = re.search(r"github\.com[:/](.+?/[^/]+?)(?:\.git)?$", repo_url)
    if not match:
        raise ValueError(f"Could not parse repo name from URL: {repo_url}")
    return match.group(1)


def extract_pr_numbers(message: str) -> list[int]:
    """Find PR numbers referenced in a commit message."""
    nums = set()
    nums.update(int(m) for m in PR_MERGE_RE.findall(message))
    nums.update(int(m) for m in SQUASH_PR_RE.findall(message))
    return sorted(nums)


def extract_issue_numbers(message: str) -> list[int]:
    """Find issue numbers fixed/closed by this commit."""
    return sorted(set(int(m) for m in ISSUE_REF_RE.findall(message)))


def mine_events(repo_url: str, clone_dir: str = None):
    """
    Walk every commit via PyDriller. Returns:
      - events_df: one row per commit-file pair
      - pr_commits_df: commit→PR links parsed from messages
      - commit_issues_df: commit→issue links parsed from messages
    """
    event_rows = []
    pr_commit_rows = []
    commit_issue_rows = []

    repo_kwargs = {}
    if clone_dir:
        os.makedirs(clone_dir, exist_ok=True)
        repo_kwargs["clone_repo_to"] = clone_dir

    for commit in Repository(repo_url, **repo_kwargs).traverse_commits():
        is_revert = "revert" in commit.msg.lower()
        is_merge = len(commit.parents) > 1

        pr_nums = extract_pr_numbers(commit.msg)
        issue_nums = extract_issue_numbers(commit.msg)
        for n in pr_nums:
            pr_commit_rows.append({"pr_number": n, "commit_hash": commit.hash})
        for n in issue_nums:
            commit_issue_rows.append({"issue_number": n, "commit_hash": commit.hash})

        for file in commit.modified_files:
            event_rows.append(
                {
                    "commit_hash": commit.hash,
                    "committed_at": commit.committer_date,
                    "author_name": commit.author.name,
                    "author_email": commit.author.email,
                    "branch": ",".join(commit.branches) if commit.branches else None,
                    "commit_message": commit.msg.strip(),
                    "file_path": file.new_path or file.old_path,
                    "old_path": file.old_path,
                    "change_type": file.change_type.name,
                    "lines_added": file.added_lines,
                    "lines_deleted": file.deleted_lines,
                    "num_files_in_commit": len(commit.modified_files),
                    "is_revert": is_revert,
                    "is_merge": is_merge,
                }
            )

    return (
        pd.DataFrame(event_rows),
        pd.DataFrame(pr_commit_rows),
        pd.DataFrame(commit_issue_rows),
    )


def mine_pull_requests(repo_name: str) -> pd.DataFrame:
    """Fetch PR metadata only — no per-PR commit calls."""
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(repo_name)

    rows = []
    for pr in repo.get_pulls(state="all"):
        rows.append(
            {
                "pr_number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "merged": pr.merged,
                "labels": ",".join(l.name for l in pr.labels),
                "created_at": pr.created_at,
                "merged_at": pr.merged_at,
                "closed_at": pr.closed_at,
                "author": pr.user.login if pr.user else None,
            }
        )
    return pd.DataFrame(rows)


def mine_issues(repo_name: str) -> pd.DataFrame:
    """Fetch issue metadata. Skip issues that are actually PRs."""
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(repo_name)

    rows = []
    for issue in repo.get_issues(state="all"):
        if issue.pull_request:
            continue
        rows.append(
            {
                "issue_number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "labels": ",".join(l.name for l in issue.labels),
                "created_at": issue.created_at,
                "closed_at": issue.closed_at,
                "author": issue.user.login if issue.user else None,
                "body": issue.body,
            }
        )
    return pd.DataFrame(rows)


def save_to_sqlite(
    events_df, pr_df, pr_commits_df, issue_df, commit_issues_df, db_path: str, repo_name: str
):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Remove existing rows for this repo before appending (idempotent re-runs)
    existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ("events", "pull_requests", "pr_commits", "issues", "commit_issues"):
        if table in existing:
            conn.execute(f"DELETE FROM {table} WHERE repo = ?", (repo_name,))
    conn.commit()

    for df in (events_df, pr_df, pr_commits_df, issue_df, commit_issues_df):
        df["repo"] = repo_name

    events_df.to_sql("events", conn, if_exists="append", index=False)
    pr_df.to_sql("pull_requests", conn, if_exists="append", index=False)
    pr_commits_df.to_sql("pr_commits", conn, if_exists="append", index=False)
    issue_df.to_sql("issues", conn, if_exists="append", index=False)
    commit_issues_df.to_sql("commit_issues", conn, if_exists="append", index=False)

    conn.close()
    print(f"Saved to {db_path}")
    print(f"  events:        {len(events_df)} rows")
    print(f"  pull_requests: {len(pr_df)} rows")
    print(f"  pr_commits:    {len(pr_commits_df)} rows")
    print(f"  issues:        {len(issue_df)} rows")
    print(f"  commit_issues: {len(commit_issues_df)} rows")


def run(repo_url: str):
    repo_name = parse_repo_name(repo_url)
    print(f"Mining {repo_name}...")

    print("  [1/4] Walking commits (local, no API calls)...")
    events_df, pr_commits_df, commit_issues_df = mine_events(
        repo_url, clone_dir=CLONE_DIR
    )

    print("  [2/4] Fetching PR metadata...")
    pr_df = mine_pull_requests(repo_name)

    print("  [3/4] Fetching issue metadata...")
    issue_df = mine_issues(repo_name)

    print("  [4/4] Saving to SQLite...")
    save_to_sqlite(
        events_df, pr_df, pr_commits_df, issue_df, commit_issues_df, DB_PATH, repo_name
    )

    return events_df, pr_df, pr_commits_df, issue_df, commit_issues_df


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/psf/requests"
    run(url)
