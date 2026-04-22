# Model Comparison — MS Subtype Classification
## Dataset: 525 samples (3 subtypes: RRMS, SPMS, PPMS — CIS removed)

---

## Test Set Performance Summary

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC (OvR) |
|-------|----------|----------|-------------|----------------|
| Logistic Regression | 0.8571 | 0.8270 | 0.8581 | 0.9674 |
| **Random Forest** | **0.8952** | 0.8655 | 0.8961 | 0.9697 |
| Extra Trees | 0.8857 | 0.8521 | 0.8815 | 0.9638 |
| **XGBoost** | **0.8952** | **0.8698** | **0.8964** | **0.9730** |
| CatBoost | 0.8857 | 0.8569 | 0.8872 | 0.9661 |

> **XGBoost and Random Forest tie on accuracy (0.8952)**, but XGBoost leads on Macro F1 (0.8698) and ROC-AUC (0.9730). XGBoost is selected for SHAP and counterfactual analysis.

---

## Cross-Validation Performance (5-Fold Stratified)

| Model | CV Accuracy | CV Macro F1 | CV Weighted F1 |
|-------|-------------|-------------|----------------|
| Logistic Regression | 0.8762 ± 0.0307 | 0.8484 ± 0.0430 | 0.8766 ± 0.0316 |
| Random Forest | 0.9000 ± 0.0497 | 0.8717 ± 0.0659 | 0.8983 ± 0.0514 |
| Extra Trees | 0.9024 ± 0.0254 | 0.8751 ± 0.0364 | 0.9005 ± 0.0262 |
| **XGBoost** | **0.9095 ± 0.0416** | 0.8814 ± 0.0549 | **0.9088 ± 0.0419** |
| CatBoost | 0.9024 ± 0.0364 | **0.8759 ± 0.0504** | 0.9033 ± 0.0359 |

> **XGBoost leads CV accuracy (0.9095).** Extra Trees has the lowest CV variance (σ=0.025), making it the most stable.

---

## Per-Class F1 Scores (Test Set)

| Model | PPMS | RRMS | SPMS |
|-------|------|------|------|
| Logistic Regression | 0.8000 | 0.9358 | 0.7451 |
| Random Forest | 0.8077 | **0.9725** | **0.8163** |
| Extra Trees | **0.8571** | 0.9550 | 0.7442 |
| XGBoost | 0.8302 | 0.9630 | **0.8163** |
| CatBoost | 0.8077 | 0.9630 | 0.8000 |

---

## Overfitting Diagnostics

| Model | Train Acc | Test Acc | Gap |
|-------|-----------|----------|-----|
| Logistic Regression | — | 0.8571 | — |
| Random Forest | 1.0000 | 0.8952 | 0.105 |
| Extra Trees | 0.9238 | 0.8857 | 0.038 |
| XGBoost | 1.0000 | 0.8952 | 0.105 |
| CatBoost | 1.0000 | 0.8857 | 0.114 |

> Extra Trees' low train accuracy (0.924) is due to `max_depth=2` constraint (intentional).

---

## Ranking Summary

| Metric | 🥇 Best | 🥈 Second | 🥉 Third |
|--------|---------|-----------|----------|
| **Accuracy** | XGBoost / RF (0.8952) | ET / CB (0.8857) | LR (0.8571) |
| **Macro F1** | XGBoost (0.8698) | RF (0.8655) | CatBoost (0.8569) |
| **ROC-AUC** | XGBoost (0.9730) | RF (0.9697) | LR (0.9674) |
| **CV Stability** | ET (σ=0.025) | LR (σ=0.031) | CB (σ=0.036) |
| **PPMS F1** | Extra Trees (0.8571) | XGBoost (0.8302) | RF / CB (0.8077) |
| **SPMS F1** | RF / XGBoost (0.8163) | CatBoost (0.8000) | LR (0.7451) |

---

## Recommendations

1. **Best overall: XGBoost** — Tied highest accuracy (0.8952), best Macro F1 (0.8698), and best ROC-AUC (0.9730).

2. **Selected for explainability (SHAP/DiCE): XGBoost** — Best performance combined with superior SHAP TreeExplainer and DiCE counterfactual support.

3. **Strong runner-up: Random Forest** — Matches XGBoost on accuracy (0.8952) with strong per-class balance.

4. **Most stable: Extra Trees** — Lowest CV variance (σ=0.025), but intentionally constrained (`max_depth=2`).

5. **SPMS is the hardest class** — F1 ranges 0.74–0.82, reflecting its transitional clinical nature.
