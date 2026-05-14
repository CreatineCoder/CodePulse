import sqlite3
import pandas as pd
from datetime import timedelta

from config.settings import DB_PATH

LABEL_WINDOW_DAYS = 90
BUG_CONFIDENCE_THRESHOLD = 0.23

SOURCE_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".c", ".cpp", ".cc", ".h", ".hpp",
    ".rb", ".rs", ".cs", ".php", ".swift", ".kt", ".scala",
    ".sh", ".bash",
}


def is_source_file(path: str) -> bool:
    suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return suffix in SOURCE_CODE_EXTENSIONS


def build_file_label_table(conn) -> pd.DataFrame:
    events = pd.read_sql("SELECT * FROM events", conn)
    commit_labels = pd.read_sql("SELECT * FROM commit_labels", conn)

    # Keep only source code files
    events = events[events["file_path"].apply(is_source_file)]
    events["committed_at"] = pd.to_datetime(events["committed_at"], utc=True)

    # Join events with commit-level bug scores
    df = events.merge(commit_labels, on="commit_hash", how="left")

    # Temporal split per repo: cutoff T = repo's last commit - 90 days
    # Features = events with committed_at < T
    # Label    = 1 if any commit in [T, T + 90 days] for that file had bug_confidence > threshold
    out_rows = []
    for repo, repo_df in df.groupby("repo"):
        T = repo_df["committed_at"].max() - timedelta(days=LABEL_WINDOW_DAYS)
        T_end = T + timedelta(days=LABEL_WINDOW_DAYS)

        before = repo_df[repo_df["committed_at"] < T]
        window = repo_df[(repo_df["committed_at"] >= T) & (repo_df["committed_at"] < T_end)]

        # Features: aggregate before-T events per file
        agg = before.groupby("file_path").agg(
            num_commits=("commit_hash", "nunique"),
            num_authors=("author_email", "nunique"),
            total_lines_added=("lines_added", "sum"),
            total_lines_deleted=("lines_deleted", "sum"),
            first_seen=("committed_at", "min"),
            last_seen=("committed_at", "max"),
            num_bug_commits=("is_bug", "sum"),
            max_bug_confidence=("bug_confidence", "max"),
            avg_bug_confidence=("bug_confidence", "mean"),
        ).reset_index()

        # Label: any commit in the 90-day window with bug_confidence > threshold
        window_bugs = window[window["bug_confidence"] > BUG_CONFIDENCE_THRESHOLD]
        bug_files = set(window_bugs["file_path"].unique())
        agg["is_bug_prone"] = agg["file_path"].isin(bug_files).astype(int)

        agg["repo"] = repo
        agg["cutoff_date"] = T
        out_rows.append(agg)

    result = pd.concat(out_rows, ignore_index=True)
    result["total_lines_changed"] = result["total_lines_added"] + result["total_lines_deleted"]
    result["age_days"] = (result["last_seen"] - result["first_seen"]).dt.days

    return result


def run():
    conn = sqlite3.connect(DB_PATH)
    print("Building file label table...")
    table = build_file_label_table(conn)
    table.to_sql("file_labels", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved file_labels: {len(table)} rows")
    print(f"\nBug-prone files: {table['is_bug_prone'].sum()} / {len(table)} ({table['is_bug_prone'].mean():.1%})")
    print("\nSample:")
    print(table[["file_path", "num_commits", "num_bug_commits", "max_bug_confidence", "is_bug_prone"]].head(10).to_string())
    return table


if __name__ == "__main__":
    run()
