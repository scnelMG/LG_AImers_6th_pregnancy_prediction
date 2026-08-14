# Project Summary

## One-Line Summary

Team **우리오디가** built a binary classifier for pregnancy-success prediction in the LG Aimers 6th Phase 2 online hackathon, while keeping the public portfolio boundary non-clinical and data-safe.

## Problem

The competition task asks participants to estimate pregnancy-success probability from structured treatment-history data. The practical machine-learning challenge is to handle sparse, categorical, and imbalanced medical-adjacent tabular features without overstating what retrospective competition validation can prove.

## Role and Contribution

- Built the preprocessing flow for low-information column removal, missing-value handling, categorical encoding, and feature-name cleanup.
- Investigated the repeated missing-value pattern (6,291 rows) in relation to procedure type, then compared unified and procedure-specific modeling directions.
- Compared LightGBM, XGBoost, and CatBoost with the same 5-fold stratified ROC-AUC validation design.
- Tested imbalance strategies including model class weights, random oversampling, and SMOTE.
- Used 30 random Optuna trials followed by 100 TPE trials to tune the LightGBM candidate.
- Preserved a reusable implementation in `src/train.py` and kept the final-flow and tuning notebooks under `notebooks/`.
- Documented data-publication limits so the repo can be reviewed without raw competition files.

## Verified Competition Evidence

| Evidence | Value | Interpretation boundary |
| --- | ---: | --- |
| No-sampling LightGBM internal 5-fold mean | 0.738864 | Historical comparison evidence |
| Tuned LightGBM internal 5-fold mean | 0.740108 | Not a strict leakage-free CV estimate |
| Best recorded public submission | 0.741430139 | Public-score record; no final rank/private score claim |

The highest recorded public submission uses unified LightGBM rather than a separate DI/IVF model or an ensemble. Those alternatives are retained as comparison experiments, not presented as final results.

## Reviewer Path

1. Start with `README.md` for problem framing, role, data policy, validation, and limitations.
2. Inspect `src/train.py` for the runnable pipeline shape.
3. Review `notebooks/final_submission_pipeline.ipynb` for the competition submission flow.
4. Review `notebooks/experiments/lgbm_tuning_ensemble.ipynb` for additional tuning and ensemble code. Both notebooks intentionally have outputs removed to avoid record-level and local-environment exposure.
5. Use `docs/modeling-or-method.md` for technical decisions, internal validation evidence, and non-clinical constraints.

## Cleared Artifacts

- `src/train.py`: cleaned script version of the competition pipeline; not a newly rerun exact reproduction of the historical best submission.
- `notebooks/final_submission_pipeline.ipynb`: final submission flow with outputs removed.
- `notebooks/experiments/lgbm_tuning_ensemble.ipynb`: additional tuning and ensemble code with outputs removed.
- `requirements.txt`: dependency list for local reproduction with authorized data.
- Markdown documentation in `README.md` and `docs/`.

## Excluded Materials

- Raw competition data: `train.csv`, `test.csv`, `sample_submission.csv`.
- Generated submission files and local experiment outputs.
- Private Drive folders, raw notebook outputs, raw data folders, and any unreviewed medical-adjacent artifacts.
- Any patient-level examples, credentials, or private clinical records.

## Limitations

- The repository is not fully reproducible without authorized access to the competition dataset.
- Reported notebook validation is retrospective competition evidence only.
- No external validation, subgroup fairness review, calibration analysis, privacy threat model, or clinical safety review is included.
- This is a non-clinical portfolio repository and must not be used for patient care or treatment guidance.
