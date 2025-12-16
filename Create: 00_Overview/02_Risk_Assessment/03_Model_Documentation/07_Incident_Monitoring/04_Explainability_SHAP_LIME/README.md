# Explainability Evidence — SHAP & LIME (Demo)

## Purpose
This section provides **explainability evidence** to support:
- transparency
- clinical safety review
- governance and audit assurance

Explainability outputs are used to:
- understand dominant risk drivers
- identify potential proxy features
- support human-in-the-loop decision review

They are **not** used to infer causality.

## Methods used
- **SHAP (global + local):** feature contribution analysis
- **LIME (local):** case-based explanation for individual predictions

## Governance framing
Explainability is reviewed to assess:
- reliance on clinically plausible features
- stability of feature importance
- potential use of proxy or sensitive attributes
- alignment with documented intended use

## Evidence included
- Global SHAP summary plot
- SHAP dependence plots (top features)
- Local explanations for selected example cases
- Interpretation notes for non-technical reviewers
