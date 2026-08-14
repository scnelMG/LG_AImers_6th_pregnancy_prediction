# Modeling and Method Notes

## Data Inputs

The training pipeline expects three competition-provided CSV files under `data/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

These files are excluded from the public repository. The repository therefore supports method inspection and authorized local reruns, not public end-to-end reproduction.

## Feature Handling

The script in `src/train.py` applies several feature-handling steps before modeling:

- removes `ID`;
- drops constant columns and fully missing columns;
- removes high-missingness elapsed-day fields that were not stable enough for the submitted pipeline;
- fills selected timing fields with sentinel, mean, or mode values depending on field type;
- aligns related genetic-test flags before dropping redundant columns;
- fills treatment-specific missing groups with type-aware defaults;
- maps count and age-range categories into ordered numeric values;
- one-hot encodes remaining categorical columns with unknown-category handling;
- normalizes generated feature names for model compatibility.

These choices are competition-engineering decisions. They are not clinical feature-importance claims.

## Validation and Evaluation

The validation loop uses:

- 5-fold `StratifiedKFold` with shuffling and fixed random state;
- ROC-AUC as the model-selection metric;
- fold-level probability predictions, not hard class labels;
- model comparison under shared validation conditions.

ROC-AUC was selected because the competition target is imbalanced and the submission objective rewards probability ranking. It does not prove probability calibration, threshold safety, or medical usefulness.

## Imbalance Strategy

The workflow compares multiple imbalance treatments:

- no resampling baseline;
- negative-class undersampling ratios;
- `RandomOverSampler`;
- `SMOTE`;
- model-native class weighting such as `scale_pos_weight`, `is_unbalance`, and `auto_class_weights`.

The goal was to compare strategies under the same validation protocol rather than assume that synthetic sampling or class weights would always help.

## Model Choice

The project uses gradient-boosted tree models because they are strong baselines for heterogeneous tabular competition data:

- `XGBClassifier`
- `LGBMClassifier`
- `CatBoostClassifier`

The final script selects the best model/sampling combination from cross-validation, tunes it with Optuna, fits the selected candidate on the prepared training data, and writes a submission file.

The recorded highest-public-score submission used a unified single LightGBM rather than a procedure-specific model or an ensemble. Procedure-specific training and ensembling remain documented experiments; the public materials do not support treating either as the final selected approach.

## Postprocessing

Submission postprocessing sets probability to zero for a small set of deterministic-looking conditions observed in the competition fields, such as specific embryo-creation reasons and unknown procedure-age encoding.

This is intentionally documented as a competition heuristic. It is not a medical rule and should not be used outside the original competition context.

## Evidence and Validation Boundary

The public repository keeps the input code but removes notebook outputs that displayed record-level previews and local environment paths. The following results were cross-checked against the team final presentation and the recorded submission history before output clearing:

| Experiment | Mean internal 5-fold ROC-AUC |
| --- | ---: |
| No sampling + LightGBM default comparison | 0.738864 |
| LightGBM tuned with Optuna | 0.740108 |

The best recorded public submission score is **0.741430139 ROC-AUC**. The competition's public score uses a 50% test-data sample; the retained materials do not establish the final private score or final placement, so neither is claimed.

Evidence readers can inspect:

- `src/train.py`: cleaned implementation of the current pipeline.
- `notebooks/final_submission_pipeline.ipynb`: final submission flow with outputs removed.
- `notebooks/experiments/lgbm_tuning_ensemble.ipynb`: tuning and ensemble experiment code with outputs removed.

The original notebooks apply some data-dependent preprocessing, including mean/mode imputation and one-hot category fitting, before the cross-validation split. These transformations do not use the target, but the resulting ROC-AUC values are **not** strict leakage-free estimates. They are internal competition-validation evidence only, not external, clinical, or deployment evidence.

`src/train.py` is a cleaned, runnable reference to the preserved competition flow. It is not presented as a newly rerun, byte-for-byte reproduction of the historical best submission.

## Reproducibility Boundary

An authorized reviewer can run:

```bash
pip install -r requirements.txt
python src/train.py
```

This requires placing the competition CSVs under `data/`. Without those files, the public repository remains inspection-only.

## Ethical and Non-Clinical Limits

Pregnancy and fertility-treatment data is sensitive. This repository:

- does not include raw records or patient-level examples;
- does not claim clinical validity;
- does not provide treatment advice;
- does not include external validation, subgroup safety analysis, monitoring, consent review, or clinical-governance review.

Any real-world use would require domain expert review, external validation on appropriately governed data, privacy review, fairness analysis, calibration checks, and clear clinical accountability.
