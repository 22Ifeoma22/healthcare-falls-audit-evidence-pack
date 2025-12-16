# AI Risk Register — Healthcare Falls Risk (Demo)

## Risk appetite (starter)
Escalate for governance review if any occur:
- sustained performance drop vs baseline
- fairness gap beyond tolerance (e.g., >5–10% absolute gap)
- material drift for 2+ monitoring cycles
- any safety incident potentially linked to model outputs

## Risk register

| ID | Risk | Harm | Likelihood | Impact | Controls / mitigations | Evidence to capture | Owner |
|---|---|---|---|---|---|---|---|
| R1 | False negatives | high-risk patient not flagged | Med | High | threshold tuned for recall; human review workflow; safety net rules | validation + threshold rationale | Model Owner + Clinical Lead |
| R2 | False positives | alert fatigue / wasted resource | High | Med | calibration; alert limits; monitor alert volume | calibration plot + KPI | Ops Lead |
| R3 | Group disparity | unequal safety outcomes | Med | High | fairness testing; mitigation plan; sign-off | fairness tables + actions | Governance Owner |
| R4 | Proxy features | hidden sensitive inference | Med | Med | feature review + SHAP/LIME review; remove/limit proxies | SHAP notes + feature list | Data Lead |
| R5 | Missingness bias | systematic exclusion | High | Med | missingness audit; imputation policy; subgroup checks | missingness report | Data Owner |
| R6 | Data leakage | inflated performance | Low–Med | High | leakage checklist; temporal split if applicable | leakage log | Validator |
| R7 | Drift | silent degradation | Med | High | monitoring plan; retraining triggers | drift metrics | Model Owner |
| R8 | Documentation gaps | audit failure | Med | Med | evidence checklist; sign-off | checklist | Governance Owner |

## Sign-off
- Clinical safety review: __________________  Date: ________
- Governance approval: _____________________ Date: ________
- Model owner: ____________________________ Date: ________
