"""
Train and evaluate bug-prone-file classifiers.

Pipeline:
  1. Load `file_labels` from SQLite.
  2. Build two feature sets — "churn-only" and "churn+bug-history".
  3. Stratified 80/20 split, mixed across repos.
  4. Fit Logistic Regression baseline + LightGBM (per feature set).
  5. Report AUC and PR-AUC.
  6. Run TreeSHAP on the LightGBM models, save summary plots.
  7. Persist best LightGBM model + feature list to disk.
"""

import sqlite3
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.settings import DB_PATH

MODEL_DIR = Path(__file__).resolve().parents[2] / "model_artifacts"
MODEL_DIR.mkdir(exist_ok=True)

# Columns that are identifiers / dates / the label — never features.
NON_FEATURE_COLS = {
    "file_path", "repo", "cutoff_date",
    "first_seen", "last_seen",
    "is_bug_prone",
}

# Features that leak past bug-fix history into the model. We train with AND
# without these to measure how much signal comes from "past bugs" vs churn.
BUG_HISTORY_COLS = {"num_bug_commits", "max_bug_confidence", "avg_bug_confidence"}


def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM file_labels", conn)
    conn.close()
    return df


def split_features_label(df: pd.DataFrame, include_bug_history: bool):
    drop = set(NON_FEATURE_COLS)
    if not include_bug_history:
        drop |= BUG_HISTORY_COLS
    feature_cols = [c for c in df.columns if c not in drop]
    X = df[feature_cols].astype(float).fillna(0)
    y = df["is_bug_prone"].astype(int)
    return X, y, feature_cols


def evaluate(name: str, y_true, y_score) -> dict:
    auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    print(f"  {name:30s}  AUC={auc:.3f}  PR-AUC={pr_auc:.3f}")
    return {"model": name, "auc": auc, "pr_auc": pr_auc}


def train_logreg(X_tr, y_tr, X_te, y_te, tag: str):
    # LogReg needs scaled features; trees do not.
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_tr_s, y_tr)
    proba = model.predict_proba(X_te_s)[:, 1]
    return evaluate(f"LogReg [{tag}]", y_te, proba), model


def train_lightgbm(X_tr, y_tr, X_te, y_te, tag: str):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        is_unbalance=True,        # handles class imbalance internally
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    return evaluate(f"LightGBM [{tag}]", y_te, proba), model


def run_shap(model, X_te, feature_cols, tag: str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)

    # LightGBM binary returns either a 2D array (positive class) or a list of
    # two arrays — handle both shapes.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, X_te, feature_names=feature_cols, show=False)
    out = MODEL_DIR / f"shap_summary_{tag}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary saved → {out.name}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_cols, mean_abs), key=lambda x: -x[1])
    print(f"  Top features by mean |SHAP| [{tag}]:")
    for feat, val in ranked[:10]:
        print(f"    {feat:30s} {val:.4f}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} files. Positive rate: {df['is_bug_prone'].mean():.1%}")
    print(f"Per repo: {df.groupby('repo')['is_bug_prone'].agg(['count', 'sum']).to_dict()}\n")

    results = []
    best = None

    for tag, include_bh in [("churn-only", False), ("churn+bug-history", True)]:
        print(f"=== Feature set: {tag} ===")
        X, y, feature_cols = split_features_label(df, include_bug_history=include_bh)
        print(f"  Features ({len(feature_cols)}): {feature_cols}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"  Train: {len(X_tr)} ({y_tr.sum()} pos)   Test: {len(X_te)} ({y_te.sum()} pos)")

        r1, _ = train_logreg(X_tr, y_tr, X_te, y_te, tag)
        r2, lgbm = train_lightgbm(X_tr, y_tr, X_te, y_te, tag)
        results.extend([r1, r2])

        run_shap(lgbm, X_te, feature_cols, tag)

        if best is None or r2["pr_auc"] > best["pr_auc"]:
            best = {**r2, "model_obj": lgbm, "features": feature_cols, "tag": tag}
        print()

    print("=== Summary ===")
    print(pd.DataFrame(results).to_string(index=False))

    artifact = {"model": best["model_obj"], "features": best["features"], "tag": best["tag"]}
    out = MODEL_DIR / "lightgbm_best.joblib"
    joblib.dump(artifact, out)
    print(f"\nBest model ({best['tag']}, PR-AUC={best['pr_auc']:.3f}) saved → {out}")


if __name__ == "__main__":
    main()


'''
Example output:
