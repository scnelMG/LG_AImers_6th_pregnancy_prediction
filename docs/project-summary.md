# Project Summary

## One-Line Summary

This project builds a competition-grade binary classifier for fertility-treatment pregnancy-success prediction while keeping the public portfolio boundary non-clinical and data-safe.

## Problem

The competition task asks participants to estimate pregnancy-success probability from structured treatment-history data. The practical machine-learning challenge is to handle sparse, categorical, and imbalanced medical-adjacent tabular features without overstating what retrospective competition validation can prove.

## Role and Contribution

- Built the preprocessing flow for low-information column removal, missing-value handling, categorical encoding, and feature-name cleanup.
- Compared LightGBM, XGBoost, and CatBoost with the same 5-fold stratified ROC-AUC validation design.
- Tested imbalance strategies including model class weights, random oversampling, and SMOTE.
- Used Optuna to tune selected gradient-boosted tree candidates.
- Preserved a reusable implementation in `src/train.py` and kept exploratory notebooks under `notebooks/`.
- Documented data-publication limits so the repo can be reviewed without raw competition files.

## Reviewer Path

1. Start with `README.md` for problem framing, role, data policy, validation, and limitations.
2. Inspect `src/train.py` for the runnable pipeline shape.
3. Review `notebooks/LG_AImers_6기_우리오디가_제출코드.ipynb` for the competition submission flow.
4. Treat `notebooks/experiments/` as blocked pending user review because it contains an `_원본.ipynb` original-copy notebook; do not present that folder as public-safe evidence yet.
5. Use `docs/modeling-or-method.md` for technical decisions and non-clinical constraints.

## Cleared Artifacts

- `src/train.py`: cleaned script version of the competition pipeline.
- `notebooks/LG_AImers_6기_우리오디가_제출코드.ipynb`: final submission notebook already present in the repository.
- `requirements.txt`: dependency list for local reproduction with authorized data.
- Markdown documentation in `README.md` and `docs/`.

Not public-safe yet: `notebooks/experiments/` includes a tracked `_원본.ipynb` original-copy notebook and requires user review before repo-level publication.

## Excluded Materials

- Raw competition data: `train.csv`, `test.csv`, `sample_submission.csv`.
- Generated submission files and local experiment outputs.
- Private Drive folders, scratch/copy notebooks, raw data folders, and any unreviewed medical-adjacent artifacts.
- Any patient-level examples, credentials, or private clinical records.

## Limitations

- The repository is not fully reproducible without authorized access to the competition dataset.
- Reported notebook validation is retrospective competition evidence only.
- No external validation, subgroup fairness review, calibration analysis, privacy threat model, or clinical safety review is included.
- This is a non-clinical portfolio repository and must not be used for patient care or treatment guidance.
