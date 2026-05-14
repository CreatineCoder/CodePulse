# 2026-05-15

### Status snapshot

| Stage                        | State        | Notes                                               |
|------------------------------|--------------|-----------------------------------------------------|
| Mining pipeline              | Done         | multi-repo append                                   |
| Labeling function            | Done         | `src/labeling/labeler.py`                           |
| Temporal split + features    | Done         | `src/features/feature_engineer.py`                  |
| Model choice                 | Decided      | LogReg baseline + LightGBM + SHAP                   |
| Model training (code)        | Not yet      | next step                                           |
| API                          | Not yet      |                                                     |

---

### Decision: Model — LogReg baseline + LightGBM with SHAP

**What:** train two models. Logistic regression as a sanity baseline, LightGBM
as the production model. Explain LightGBM predictions with SHAP (TreeSHAP).

**Why:**
- **Interpretability is a hard requirement.** CodePulse must surface *which
  features* matter and *how features interact*, then feed those explanations
  into the LLM layer. SHAP provides per-prediction attributions plus global
  importance plus interaction values — covers all three needs.
- **LogReg first as a diagnostic.** If LogReg gets AUC ≈ 0.5, the features are
  weak and no model will save us — that's a feature-engineering signal, not a
  model signal. If LogReg gets AUC ≈ LightGBM, the relationships are linear
  and LightGBM is overkill. The gap between the two models is itself
  diagnostic information.
- **LightGBM for production.** Best-in-class on tabular data, handles class
  imbalance via `is_unbalance` / `scale_pos_weight`, captures non-linear
  interactions, and TreeSHAP computes exact attributions in polynomial time
  (versus exponential for model-agnostic SHAP).
- **Severe class imbalance (2.9% positive)** rules out plain accuracy as a
  metric. Use **AUC** (ranking quality across thresholds) and **PR-AUC**
  (focuses on the positive class) together.

**Alternatives rejected:**
- Decision Tree alone — unstable, one weird file flips the tree.
- Random Forest — tied with LightGBM on accuracy, loses on speed and SHAP
  integration.
- XGBoost — essentially tied with LightGBM; pick one. LightGBM chosen for
  lower memory and faster training on small datasets.
- Neural nets — overkill for ~hundreds of rows, terrible interpretability.

---

### Decision: Evaluation metrics — AUC + PR-AUC

**What:** report both ROC-AUC and PR-AUC on the test set. Do NOT report
plain accuracy as the headline metric.

**Why:** at 2.9% positive rate, a model that always predicts "clean" achieves
97.1% accuracy while being useless. AUC is threshold-independent and
imbalance-insensitive — it measures ranking quality (probability that a
random bug-prone file scores higher than a random clean file). PR-AUC is
sensitive to the positive class specifically, which is what we actually
care about.

**Alternatives rejected:** accuracy alone (broken under imbalance);
F1 at threshold 0.5 (arbitrary threshold).

---

### Open questions

- Class imbalance strategy: `class_weight="balanced"` for LogReg,
  `is_unbalance=True` for LightGBM, or oversampling? Plan: start with
  built-in class weights, only add SMOTE if metrics demand it.
- Train/test split: stratify by `repo` to ensure both repos appear in both
  splits, or hold out one full repo as the test set (harder generalization
  test)?
- Cross-validation: skip for v1 (single split), revisit if results are
  unstable.
