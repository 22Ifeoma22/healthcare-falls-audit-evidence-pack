# Credit Assurance Summary (Baseline)

## Model
- Logistic Regression
- Dataset: sample_credit.csv
- Target: approved
- Group feature (practice/audit): gender
- Threshold: 0.50

## Performance
- Accuracy: 0.8958
- ROC AUC: 0.9562

## Fairness checks (Fairlearn)
- Demographic Parity diff: 0.0345
- Equalized Odds diff: 0.1786

## Decision
CONDITIONAL_PASS

## Rationale (how to justify)
- DP measures outcome-rate differences across groups.
- EO measures error-rate differences across groups (FPR/FNR balance).
- Refer to `group_metrics_baseline_credit.csv` (includes n, FPR, FNR) to identify worst-group risk.
