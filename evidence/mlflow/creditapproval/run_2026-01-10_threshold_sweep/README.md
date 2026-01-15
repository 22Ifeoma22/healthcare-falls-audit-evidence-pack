# MLflow Evidence Pack - Credit Approval (Baseline)

This folder contains **exported MLflow artefacts** for a single baseline run.
Purpose: provide an **audit-ready evidence snapshot** (performance , fairness signals , decision note).

## What this run covers
- **Use case:** Credit approval (demo)
- **Model:** Baseline Logistic Regression
- **Evaluation style:** Threshold-based decisioning and group metrics
- **Group / proxy attribute:** gender (female / male) for disparity checks

> NOTE: The authoritative results and decision for this run are in the assurance summary artefact below.

## Artefacts in this folder
- Assurance_summary_baseline_credit.md
  - Plain-language “audit summary” (threshold used, performance metrics, fairness deltas, and a PASS / CONDITIONAL_PASS / FAIL style outcome).
- confusion_matrix_baseline_credit.png
  - Confusion matrix for quick review of errors (FP / FN).
- group_metrics_baseline_credit.csv
  - Group metrics by gender (e.g., selection rate, FPR, FNR, accuracy, n).

## How to interpret quickly (auditor view)
1) Open **assurance_summary_baseline_credit.md** first  
2) Check the **decision** line (PASS / CONDITIONAL_PASS / FAIL)  
3) Validate supporting evidence:
   - Confusion matrix (error shape)
   - Group metrics (any disparity by group)

## How to reproduce (high level)
1) Run the notebook/script that logs this run to MLflow
2) Start MLflow UI (pointing at your local store), e.g.:
   - python -m mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri "file:./mlruns"
3) Export/copy artefacts from MLflow into this repo folder for versioned audit evidence.

## Governance note
This repo stores **exported evidence artefacts**, not the full MLflow tracking store (mlruns/).
That keeps GitHub clean and makes evidence review easy.

---
Last updated: 2026-01-11 00:41
