# 00 Overview — Healthcare Falls Audit Evidence Pack (Demo)

## Purpose
This repository is an **audit evidence pack** demonstrating how to document, test, and govern a falls-risk prediction use case using:
- ISO/IEC 42001-style governance evidence
- Model Risk / Responsible AI assurance practices
- Explainability and fairness audit artefacts

## Intended use (decision support only)
The model is intended to **support clinical or operational teams** by flagging **potential elevated fall risk** for review.
It must **not** be used as an automated decision-maker or as a replacement for clinical judgement.

**Decision boundary**
- ✅ Allowed: prioritisation / triage support, care-planning prompts, safety review triggers
- ❌ Not allowed: denial of care, discharge decisions, staffing decisions without human review

## Dataset statement (safety + compliance)
This audit uses a **publicly available or synthetic healthcare falls dataset** solely to demonstrate AI governance, risk assessment, explainability, and fairness auditing techniques.

This analysis does **not** represent a deployed clinical system and must **not** be used for medical or operational decision-making.

## Stakeholders
- Clinical safety lead / nursing lead
- Data protection / privacy (where applicable)
- Model owner (data science)
- Governance owner (risk & compliance)
- Operational owner (ward / care team lead)

## Key harms considered (top risks)
- **False negatives:** high-risk patient not flagged → increased likelihood of harm
- **False positives:** unnecessary alerts → alert fatigue / resource diversion
- **Bias / disparity:** uneven error rates across age/sex (and other permitted attributes)
- **Data quality:** missingness, coding drift, proxy features
- **Operational drift:** changes in patient mix, practices, or recording standards

## Non-clinical disclaimer
This repository demonstrates **governance and audit method**, not clinical efficacy.
