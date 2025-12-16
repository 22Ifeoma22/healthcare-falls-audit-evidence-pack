# Monitoring & Incident Plan — Healthcare Falls Risk (Demo)

## Monitoring objectives
Ensure the model remains:
- safe (no silent harm)
- fair (no unacceptable disparity)
- reliable (no unobserved degradation)
- governable (clear escalation paths)

## Monitoring cadence
- **Weekly (operations):**
  - alert volume
  - override / clinician feedback (if available)
- **Monthly (model):**
  - performance (AUC, recall, precision)
  - calibration
  - feature drift indicators
- **Quarterly (governance):**
  - fairness review
  - documentation completeness
  - retraining decision

## Key KPIs
- Performance: AUC, recall (safety-critical), precision
- Calibration: calibration curve / Brier score
- Drift: PSI or population shift checks on key features
- Fairness: TPR / FNR gaps for permitted groups
- Operations: alerts per unit, override rate

## Escalation triggers
Escalate to governance if any occur:
- sustained performance drop vs baseline
- fairness gap beyond agreed tolerance
- drift threshold exceeded for two consecutive cycles
- any safety incident plausibly linked to model output

## Incident workflow
1. Log incident (context + timestamp)
2. Pause changes if required
3. Root cause analysis:
   - data
   - model
   - workflow
   - human factors
4. Mitigation action:
   - threshold change
   - feature review
   - retraining
   - workflow update
5. Sign-off and documentation update

## Record keeping
All incidents, reviews, and mitigations must be recorded
and retained as part of the AI governance evidence pack.
