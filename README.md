# LG AImers 6기 - 임신 성공 여부 예측

> 난임 시술 tabular 데이터를 바탕으로 임신 성공 확률을 예측한 의료 인접 영역의 경진대회 ML 프로젝트입니다.

[![Python](https://img.shields.io/badge/Python-ML%20Pipeline-3776AB?logo=python&logoColor=white)](src/train.py)
[![LightGBM](https://img.shields.io/badge/LightGBM-ROC--AUC-02569B)](docs/modeling-or-method.md)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-6F42C1)](docs/project-summary.md)
[![Portfolio](https://img.shields.io/badge/Portfolio-Sensitive%20Data%20Aware-2ea44f)](notebooks/README.md)

## 개요

LG AImers 6기 경진대회에서 진행한 임신 성공 여부 예측 프로젝트입니다. 난임 시술 기록의 범주형/수치형 feature, 결측, 시술 이력, 배아 관련 변수를 활용해 binary classification pipeline을 구성했습니다.

의료 인접 데이터를 다루는 프로젝트이므로, 이 저장소는 임상적 주장보다 데이터 처리, 검증 설계, 모델 비교, 공개 안전성에 초점을 둡니다. 진단, 치료 판단, 의료 조언 목적이 아닙니다.

## 빠른 검토 경로

| 먼저 볼 것 | 확인할 내용 |
| --- | --- |
| [docs/project-summary.md](docs/project-summary.md) | 문제 정의, 역할, 데이터 공개 경계 |
| [docs/modeling-or-method.md](docs/modeling-or-method.md) | 전처리, validation, 모델 비교, tuning 전략 |
| [notebooks/README.md](notebooks/README.md) | notebook별 실험 흐름과 공개 범위 |
| [src/train.py](src/train.py) | 재사용 가능한 학습 스크립트 |

## 문제 정의

난임 시술 데이터는 시술 유형, 환자 상태, 시술 이력, 배아 관련 변수, 결측값, 범주형 코드가 복합적으로 섞여 있습니다. 목표는 임신 성공 여부를 binary classification으로 예측하고, ROC-AUC 기준으로 모델을 비교하는 것입니다.

핵심은 높은 점수를 만드는 것뿐 아니라 민감한 도메인에서 과장된 의학적 해석을 피하고, 공개 가능한 자료와 비공개 자료를 명확히 분리하는 것입니다.

## 내 역할

- 상수/완전 결측/저신호 컬럼 정리
- 시술 맥락에 따른 결측값 처리와 범주형 encoding
- LightGBM, XGBoost, CatBoost 비교
- class weight, RandomOverSampler, SMOTE 등 불균형 대응 실험
- Optuna 기반 hyperparameter tuning과 submission 생성
- notebook 실험과 `src/train.py` 실행 경로 분리

## 기술적 의사결정

| 영역 | 선택 | 이유 |
| --- | --- | --- |
| 평가 지표 | ROC-AUC | 확률 ranking 품질을 비교하기 위한 대회 metric입니다. |
| 모델군 | LightGBM, XGBoost, CatBoost | tabular competition에서 강하고 결측/범주형 처리 전략을 비교하기 좋습니다. |
| 검증 | 5-fold validation | 단일 split에 대한 과적합을 줄이고 tuning 안정성을 확인했습니다. |
| 불균형 대응 | class weight, oversampling, SMOTE | 성공/실패 class 비율 차이에 따른 trade-off를 점검했습니다. |
| 공개 정책 | inspection-first | 원본 의료 인접 대회 데이터와 생성 파일을 공개하지 않습니다. |

## 파이프라인

```mermaid
flowchart LR
    A["대회 CSV"] --> B["컬럼 정리 / 결측 처리"]
    B --> C["범주형 encoding"]
    C --> D["5-fold validation"]
    D --> E["LGBM / XGB / CatBoost"]
    E --> F["Optuna tuning"]
    F --> G["Submission 생성"]
```

## 재현 가능성

공식 대회 CSV가 있어야 전체 실행이 가능합니다.

```bash
pip install -r requirements.txt
python src/train.py
```

공개 checkout에서는 `docs/`, `notebooks/README.md`, `src/train.py`를 통해 모델링 구조를 검토할 수 있습니다.

## 공개/비공개 경계

제외한 것:

- 원본 대회 CSV와 생성 submission
- 의료 인접 raw data, Drive scratch notebook, 대용량 archive
- token, credential, 개인정보 가능 자료

포함한 것:

- 공개 가능한 학습 스크립트와 notebook 안내
- 모델링 방법론과 한계 문서
- 데이터 공개 경계 설명

## 한계

- 임상적 의사결정에 사용할 수 없습니다.
- leaderboard 점수만으로 일반화 성능을 보장할 수 없습니다.
- postprocessing은 대회 feature에 맞춘 heuristic이며 의학 지식으로 해석하면 안 됩니다.
- 배포, 모니터링, privacy threat model은 포함하지 않았습니다.
