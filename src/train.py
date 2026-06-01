"""LG Aimers 6th pregnancy success prediction pipeline.

This script keeps the competition notebook logic in a cleaner, reproducible
form for portfolio review. Place competition CSV files under ./data before
running.
"""

from __future__ import annotations

import os
import random
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import catboost as cat
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

RANDOM_STATE = 736665
TARGET = "임신 성공 여부"
DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "최종제출본.csv"


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv").drop(columns=["ID"])
    test = pd.read_csv(data_dir / "test.csv").drop(columns=["ID"])
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample_submission


def drop_uninformative_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    constant_columns = [
        col
        for col in train.columns
        if train[col].nunique(dropna=False) == 1 and train[col].notna().all()
    ]
    all_missing_columns = [col for col in train.columns if train[col].notna().sum() == 0]
    columns_to_remove = sorted(set(constant_columns + all_missing_columns))
    return train.drop(columns=columns_to_remove), test.drop(columns=columns_to_remove)


def fill_missing_values(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    drop_columns = ["임신 시도 또는 마지막 임신 경과 연수", "난자 해동 경과일", "배아 해동 경과일"]
    minus_fill_columns = ["난자 채취 경과일"]
    mean_fill_columns = ["배아 이식 경과일", "난자 혼합 경과일"]
    mode_fill_columns = ["특정 시술 유형"]

    train = train.drop(columns=drop_columns)
    test = test.drop(columns=drop_columns)

    for col in minus_fill_columns:
        train[col] = train[col].fillna(-1)
        test[col] = test[col].fillna(-1)

    for col in mean_fill_columns:
        fill_value = train[col].mean()
        train[col] = train[col].fillna(fill_value)
        test[col] = test[col].fillna(fill_value)

    for col in mode_fill_columns:
        fill_value = train[col].mode()[0]
        train[col] = train[col].fillna(fill_value)
        test[col] = test[col].fillna(fill_value)

    train = align_genetic_test_features(train)
    test = align_genetic_test_features(test)
    train, test = fill_di_specific_missing_values(train, test)

    genetic_test_columns = ["착상 전 유전 검사 사용 여부", "PGD 시술 여부", "PGS 시술 여부"]
    return train.drop(columns=genetic_test_columns), test.drop(columns=genetic_test_columns)


def align_genetic_test_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["착상 전 유전 진단 사용 여부"] == 1, "PGD 시술 여부"] = 1
    df.loc[df["PGD 시술 여부"] == 1, "착상 전 유전 진단 사용 여부"] = 1
    df.loc[
        (df["착상 전 유전 검사 사용 여부"] == 1) & df["PGD 시술 여부"].isna(),
        "PGS 시술 여부",
    ] = 1
    return df


def fill_di_specific_missing_values(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_counts = train.isna().sum()
    di_missing_columns = missing_counts[missing_counts == 6291].index

    for col in di_missing_columns:
        if train[col].dtype == "object":
            train[col] = train[col].fillna("DI_알 수 없음")
            test[col] = test[col].fillna("DI_알 수 없음")
        elif train[col].nunique() == 2:
            fill_value = train[col].mode()[0]
            train[col] = train[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)
        else:
            fill_value = train[col].mean()
            train[col] = train[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)

    return train, test


def encode_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_map = {"0회": 0, "1회": 1, "2회": 2, "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6}
    count_columns = [
        "총 시술 횟수",
        "클리닉 내 총 시술 횟수",
        "IVF 시술 횟수",
        "DI 시술 횟수",
        "총 임신 횟수",
        "IVF 임신 횟수",
        "DI 임신 횟수",
        "총 출산 횟수",
        "IVF 출산 횟수",
        "DI 출산 횟수",
    ]
    patient_age_map = {
        "만18-34세": 26,
        "만35-37세": 36,
        "만38-39세": 39,
        "만40-42세": 41,
        "만43-44세": 44,
        "만45-50세": 48,
        "알 수 없음": -1,
    }
    donor_age_map = {
        "만20세 이하": 20,
        "만21-25세": 23,
        "만26-30세": 28,
        "만31-35세": 33,
        "만36-40세": 38,
        "만41-45세": 43,
        "알 수 없음": -1,
    }

    for columns, mapping in [
        (count_columns, count_map),
        (["시술 당시 나이"], patient_age_map),
        (["난자 기증자 나이", "정자 기증자 나이"], donor_age_map),
    ]:
        train[columns] = train[columns].replace(mapping)
        test[columns] = test[columns].replace(mapping)

    categorical_columns = train.select_dtypes(include="object").columns.tolist()
    encoder = OneHotEncoder(handle_unknown="ignore")
    train_ohe = encoder.fit_transform(train[categorical_columns])
    test_ohe = encoder.transform(test[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    train_encoded = train.drop(columns=categorical_columns).join(
        pd.DataFrame(train_ohe.toarray(), columns=encoded_columns, index=train.index)
    )
    test_encoded = test.drop(columns=categorical_columns).join(
        pd.DataFrame(test_ohe.toarray(), columns=encoded_columns, index=test.index)
    )

    train_encoded.columns = [clean_column_name(col) for col in train_encoded.columns]
    test_encoded.columns = [clean_column_name(col) for col in test_encoded.columns]
    return train_encoded, test_encoded


def clean_column_name(column: str) -> str:
    return re.sub(r"[^\w\s]", "_", column)


def apply_sampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ratio: str | int,
) -> tuple[pd.DataFrame, pd.Series]:
    if ratio == "default":
        return X_train, y_train
    if ratio == "ros":
        return RandomOverSampler(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    if ratio == "smote":
        return SMOTE(sampling_strategy=0.7, random_state=RANDOM_STATE).fit_resample(X_train, y_train)

    train_fold = pd.concat([X_train, y_train], axis=1)
    negative = train_fold[train_fold[TARGET] == 0]
    positive = train_fold[train_fold[TARGET] == 1]
    negative = negative.sample(n=int(len(positive) * int(ratio)), random_state=RANDOM_STATE)
    sampled = pd.concat([negative, positive], axis=0).reset_index(drop=True)
    return sampled.drop(columns=[TARGET]), sampled[TARGET]


def build_baseline_models(y_train: pd.Series) -> list[Any]:
    class_counts = Counter(y_train)
    imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
    return [
        xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            scale_pos_weight=imbalance_ratio,
            eval_metric="auc",
            n_jobs=-1,
        ),
        lgb.LGBMClassifier(random_state=RANDOM_STATE, is_unbalance=True, verbose=-1, n_jobs=-1),
        cat.CatBoostClassifier(
            random_state=RANDOM_STATE,
            auto_class_weights="Balanced",
            verbose=0,
        ),
    ]


def evaluate_candidates(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for ratio in ["default", 1, 2, "ros"]:
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            X_sampled, y_sampled = apply_sampling(X_train, y_train, ratio)

            for model in build_baseline_models(y_sampled):
                model.fit(X_sampled, y_sampled)
                y_proba = model.predict_proba(X_valid)[:, 1]
                rows.append(
                    {
                        "ratio": ratio,
                        "fold": fold,
                        "model": model.__class__.__name__,
                        "auc": roc_auc_score(y_valid, y_proba),
                    }
                )

    return pd.DataFrame(rows)


def build_tuned_model(trial: optuna.Trial, model_name: str) -> Any:
    if model_name == "LGBMClassifier":
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 2.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 100, 600),
            "n_estimators": trial.suggest_int("n_estimators", 500, 6000),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 5, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }
        return lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

    if model_name == "XGBClassifier":
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 5, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 5000),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5, log=True),
            "gamma": trial.suggest_float("gamma", 0, 3),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
        }
        return xgb.XGBClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)

    if model_name == "CatBoostClassifier":
        params = {
            "objective": "Logloss",
            "eval_metric": "AUC",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 5.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.5, 3),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 0.3),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 0.9),
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "max_bin": trial.suggest_int("max_bin", 100, 150),
        }
        return cat.CatBoostClassifier(
            **params,
            random_state=RANDOM_STATE,
            auto_class_weights="Balanced",
            verbose=0,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: str | int,
    model_name: str,
    random_trials: int = 30,
    tpe_trials: int = 100,
) -> tuple[dict[str, Any], float]:
    def objective(trial: optuna.Trial) -> float:
        auc_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            X_sampled, y_sampled = apply_sampling(X_train, y_train, ratio)
            model = build_tuned_model(trial, model_name)
            model.fit(X_sampled, y_sampled)
            y_proba = model.predict_proba(X_valid)[:, 1]
            auc_scores.append(roc_auc_score(y_valid, y_proba))

        return float(np.mean(auc_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=random_trials)
    study.sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study.optimize(objective, n_trials=tpe_trials)
    return study.best_params, study.best_value


def build_final_model(model_name: str, params: dict[str, Any]) -> Any:
    if model_name == "LGBMClassifier":
        return lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    if model_name == "XGBClassifier":
        return xgb.XGBClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == "CatBoostClassifier":
        return cat.CatBoostClassifier(
            **params,
            random_state=RANDOM_STATE,
            auto_class_weights="Balanced",
            verbose=0,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def apply_postprocessing(sample_submission: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    sample_submission = sample_submission.copy()
    zero_probability_reasons = ["난자 저장용", "기증용"]
    sample_submission.loc[test["배아 생성 주요 이유"].isin(zero_probability_reasons), "probability"] = 0
    sample_submission.loc[test["시술 당시 나이"] == -1, "probability"] = 0
    return sample_submission


def main() -> None:
    seed_everything()
    train, test, sample_submission = load_data()
    raw_test = test.copy()

    train, test = drop_uninformative_columns(train, test)
    train, test = fill_missing_values(train, test)
    train_encoded, test_encoded = encode_features(train, test)

    X = train_encoded.drop(columns=[TARGET])
    y = train_encoded[TARGET]

    cv_results = evaluate_candidates(X, y)
    best_case = (
        cv_results.groupby(["ratio", "model"], as_index=False)["auc"]
        .mean()
        .sort_values("auc", ascending=False)
        .iloc[0]
    )
    best_ratio = best_case["ratio"]
    best_model_name = best_case["model"]

    best_params, best_auc = tune_model(X, y, best_ratio, best_model_name)
    print({"best_ratio": best_ratio, "best_model": best_model_name, "best_auc": best_auc})

    X_sampled, y_sampled = apply_sampling(X, y, best_ratio)
    model = build_final_model(best_model_name, best_params)
    model.fit(X_sampled, y_sampled)

    sample_submission["probability"] = model.predict_proba(test_encoded)[:, 1]
    sample_submission = apply_postprocessing(sample_submission, raw_test)
    sample_submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
