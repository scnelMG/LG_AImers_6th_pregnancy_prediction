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

## Postprocessing

Submission postprocessing sets probability to zero for a small set of deterministic-looking conditions observed in the competition fields, such as specific embryo-creation reasons and unknown procedure-age encoding.

This is intentionally documented as a competition heuristic. It is not a medical rule and should not be used outside the original competition context.

## Evidence

Evidence boundary:

Currently inspectable public evidence in this section is limited to the cleaned script and final submission notebook. The `notebooks/experiments/` paths below, including the tracked `_원본.ipynb` original-copy notebook, are blocked pending user review and must not be treated as public-safe or inspectable evidence yet.

- `src/train.py`: cleaned implementation of the pipeline.
- `notebooks/LG_AImers_6기_우리오디가_제출코드.ipynb`: final submission notebook.
- `notebooks/experiments/전처리변경_안나누기_lgbm튜닝더.ipynb`: blocked with the experiment folder until user review clears this path.
- `notebooks/experiments/전처리변경_안나누기_lgbm튜닝더_원본.ipynb`: blocked original-copy notebook; not public-safe or inspectable evidence until user review clears or removes it.

If reviewed and cleared, notebook outputs include tuned cross-validation results around ROC-AUC 0.74. Those values remain internal competition-validation evidence, not external or clinical evidence.

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
