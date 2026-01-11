\# MLflow Evidence — Credit Approval (Threshold Sweep)



This folder contains the exported evidence artefacts from an MLflow run used in this audit evidence pack.



\## Run metadata

\- Model: Logistic Regression (baseline)

\- Dataset: sample\_credit.csv

\- Selected threshold: 0.65

\- Intended use: demonstrate performance + group fairness checks for an approval/denial decision



\## Artefacts in this folder

\- `assurance\_summary\_baseline\_credit.md` — audit summary (performance + fairness + decision)

\- `confusion\_matrix\_baseline\_credit.png` — baseline classification performance snapshot

\- `group\_metrics\_baseline\_credit.csv` — group metrics split by protected/proxy group (e.g., gender)



\## Interpretation (how to read this pack)

\- \*\*Performance\*\* indicates how well the model predicts outcomes overall.

\- \*\*Group metrics\*\* shows whether one group experiences higher error rates than another.

\- \*\*Decision label\*\*

&nbsp; - PASS: within tolerance

&nbsp; - CONDITIONAL\_PASS: borderline disparity → proceed only with monitoring + review triggers

&nbsp; - FAIL: mitigation required before deployment



