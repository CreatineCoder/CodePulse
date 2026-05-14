"""
FastAPI backend for CodePulse frontend.

Endpoints:
  GET /api/repos                   List mined repos with summary stats.
  GET /api/insights?repo=<name>    Full insight payload for one repo.
"""

import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config.settings import DB_PATH

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "model_artifacts" / "lightgbm_best.joblib"

app = FastAPI(title="CodePulse API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_artifact: dict | None = None
_explainer: shap.TreeExplainer | None = None


def get_model():
    global _artifact, _explainer
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(500, "Model artifact not found. Run trainer first.")
        _artifact = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_artifact["model"])
    return _artifact, _explainer


def conn():
    return sqlite3.connect(DB_PATH)


@app.get("/api/repos")
def list_repos() -> list[dict[str, Any]]:
    with conn() as c:
        files = pd.read_sql(
            """
            SELECT repo,
                   COUNT(*) AS total_files,
                   SUM(is_bug_prone) AS bug_prone_labeled,
                   SUM(num_commits) AS total_commits,
                   SUM(total_lines_changed) AS total_lines_changed,
                   AVG(age_days) AS avg_age_days,
                   AVG(total_lines_changed) AS avg_lines_changed
            FROM file_labels
            GROUP BY repo
            ORDER BY total_files DESC
            """,
            c,
        )
        events = pd.read_sql(
            """
            SELECT repo,
                   COUNT(DISTINCT author_email) AS distinct_authors,
                   MAX(committed_at) AS last_update
            FROM events
            GROUP BY repo
            """,
            c,
        )
        try:
            prs = pd.read_sql(
                "SELECT repo, COUNT(*) AS prs FROM pull_requests GROUP BY repo", c
            )
        except Exception:
            prs = pd.DataFrame(columns=["repo", "prs"])
    df = files.merge(events, on="repo", how="left").merge(prs, on="repo", how="left")
    df = df.fillna(0)
    return df.to_dict(orient="records")


@app.get("/api/insights")
def insights(repo: str = Query(...)) -> dict[str, Any]:
    artifact, explainer = get_model()
    model = artifact["model"]
    features = artifact["features"]

    with conn() as c:
        df = pd.read_sql(
            "SELECT * FROM file_labels WHERE repo = ?",
            c,
            params=(repo,),
        )

    if df.empty:
        raise HTTPException(404, f"Repo not found: {repo}")

    X = df[features].astype(float).fillna(0)
    proba = model.predict_proba(X)[:, 1]
    df["probability"] = proba

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_df = pd.DataFrame(shap_values, columns=features)

    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        [{"feature": f, "importance": float(v)} for f, v in zip(features, mean_abs)],
        key=lambda r: -r["importance"],
    )

    top_idx = np.argsort(-proba)[:20]
    top_files = []
    for i in top_idx:
        row = df.iloc[i]
        shap_row = shap_df.iloc[i]
        contributions = sorted(
            [{"feature": f, "value": float(row[f]), "shap": float(shap_row[f])} for f in features],
            key=lambda r: -abs(r["shap"]),
        )
        top_files.append({
            "file_path": row["file_path"],
            "probability": float(row["probability"]),
            "num_commits": int(row["num_commits"]),
            "num_authors": int(row["num_authors"]),
            "age_days": int(row["age_days"]),
            "total_lines_changed": int(row["total_lines_changed"]),
            "is_bug_prone_labeled": int(row["is_bug_prone"]),
            "contributions": contributions,
        })

    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(proba, bins=bins)
    probability_distribution = [
        {"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "count": int(hist[i])} for i in range(len(hist))
    ]

    summary = {
        "repo": repo,
        "total_files": int(len(df)),
        "bug_prone_labeled": int(df["is_bug_prone"].sum()),
        "bug_prone_predicted": int((proba >= 0.5).sum()),
        "total_commits": int(df["num_commits"].sum()),
        "total_lines_changed": int(df["total_lines_changed"].sum()),
        "avg_age_days": float(df["age_days"].mean()),
        "max_probability": float(proba.max()),
        "mean_probability": float(proba.mean()),
    }

    scatter = df[["num_commits", "total_lines_changed", "age_days", "num_authors"]].copy()
    scatter["probability"] = proba
    scatter["file_path"] = df["file_path"]
    scatter_data = scatter.to_dict(orient="records")

    return {
        "summary": summary,
        "top_files": top_files,
        "feature_importance": feature_importance,
        "probability_distribution": probability_distribution,
        "scatter": scatter_data,
        "model_tag": artifact.get("tag", "unknown"),
        "features": features,
    }


@app.get("/api/health")
def health():
    return {"ok": True}
