# Model Card — Healthcare Falls Risk (Demo)

## Model overview
- Task: binary classification (fall risk: yes/no)
- Intended users: clinical/ops teams (**decision support only**)
- Output: risk score / probability + optional risk band (low/med/high)
- Human-in-the-loop: required for any action

## Intended use and decision boundary
✅ Allowed: prioritisation prompts, care planning triggers, safety review workflow  
❌ Not allowed: automated clinical decisions, discharge/denial of care, staffing decisions without review

## Data (to complete)
- Dataset type: public / synthetic (demo)
- Row unit: (e.g., patient encounter / episode)
- Label definition: fall event within [X] days of encounter (define)
- Key features: (list top 10 feature groups)
- Missingness: document patterns and handling approach

## Modelling approach (to complete)
- Baseline model: Logistic Regression (interpretable starting point)
- Alternatives tested: (optional)
- Train/validation strategy: (e.g., stratified split; temporal split if time-based)
- Threshold policy: safety-first (optimise for recall) + rationale

## Performance evidence (to populate)
- AUC:
- Recall (primary):
- Precision:
- Calibration: (plot + comment)
- Confusion matrix summary:
- Error analysis notes: who is harmed by errors?

## Explainability evidence
- Global: SHAP summary plot (top drivers)
- Local: LIME cases (3–5 example patients) with clinician interpretation prompts
- Proxy feature review: features flagged for governance review

## Fairness evidence (to populate)
- Groups tested (permitted): age bands, sex (and others if governed)
- Metrics: selection rate, TPR/FNR gaps, calibration gaps
- Findings summary:
- Mitigations / actions:

## Limitations
- Not clinically validated
- Not a deployed system
- Dataset constraints may not generalise
- Explanations are supportive, not causal

## Monitoring (to complete)
- KPIs: performance, drift, alert volume, fairness
- Review cadence: monthly (model), quarterly (governance)
- Retrain triggers:
