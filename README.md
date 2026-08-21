# LG Aimers 6기 | 임신 성공 여부 예측

<p align="center">결측 패턴과 class 불균형을 검증하며 난임 시술 tabular 데이터의 임신 성공 확률을 예측한 경진대회 ML 프로젝트</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-ML%20Pipeline-3776AB?logo=python&logoColor=white" alt="Python ML Pipeline">
  <img src="https://img.shields.io/badge/LightGBM-ROC--AUC-02569B" alt="LightGBM ROC-AUC">
  <img src="https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-6F42C1" alt="Optuna">
  <img src="https://img.shields.io/badge/Scope-Non--clinical-6B7280" alt="Non-clinical scope">
</p>

> LG Aimers 6기 Phase 2 온라인 해커톤에서 팀 **우리오디가**로 난임 시술 기록을 바탕으로 임신 성공 확률을 예측했습니다. 이 저장소는 의료 조언이나 임상 모델이 아니라, **범주형·결측·불균형 tabular 데이터의 처리와 검증 판단**을 검토하는 포트폴리오 공개본입니다.

## 프로젝트 한눈에 보기

| 항목 | 내용 |
| --- | --- |
| 공식 과제 | [난임 환자 대상 임신 성공 여부 예측 AI 모델 개발](https://dacon.io/competitions/official/236452/overview/description) |
| 참가 | LG Aimers 6기 Phase 2 온라인 해커톤 · 팀 우리오디가 |
| 평가 지표 | ROC-AUC — 성공 확률의 ranking 품질을 비교하는 대회 지표 |
| 데이터 규모 | ID 제거 후 Train 256,351행·68열 / Test 90,067행·67열 |
| 클래스 분포 | 실패 190,123건 / 성공 66,228건 — 약 74:26 불균형 |
| 최고 공개 제출 기록 | **0.741430139 ROC-AUC** — 제출 이력 기준, 최종 순위·private 점수는 미확인 |
| 내 역할 | 결측 패턴 분석·전처리·인코딩·불균형 실험·모델 비교·Optuna 튜닝·제출 파이프라인 정리 |
| 공개 범위 | 원본 CSV와 제출 파일은 제외, 코드·검증 근거·한계만 공개 |

## 왜 이 문제를 풀었나

난임 시술 데이터는 시술 유형, 시술 이력, 배아 관련 값, 결측과 범주형 코드가 함께 얽힌 고차원 tabular 데이터입니다. 단순한 정확도보다, 확률 예측의 순위를 ROC-AUC로 비교하고 전처리·샘플링·모델 선택의 영향을 분리해 확인하는 것이 핵심이었습니다.

다만 대회 데이터가 의료 인접 민감 영역에 속하므로, 결과를 진단·치료 판단이나 임상적 인과관계로 해석하지 않았습니다. 포트폴리오에서는 **어떤 데이터 제약을 어떻게 다뤘는지**와 **검증값이 말할 수 있는 범위**를 중심으로 남겼습니다.

## 접근 흐름

```mermaid
flowchart LR
    A["공식 대회 CSV"] --> B["상수·완전 결측 컬럼 정리"]
    B --> C["결측 처리·범주형 인코딩"]
    C --> D["Stratified 5-fold ROC-AUC"]
    D --> E["LightGBM · XGBoost · CatBoost"]
    E --> F["불균형 대응·Optuna 튜닝"]
    F --> G["확률 예측·대회용 후처리"]
```

| 단계 | 구현과 판단 |
| --- | --- |
| 결측 원인 탐색 | 동일한 결측 패턴 6,291행을 확인해 시술 유형(DI/IVF) 맥락을 검토. 분리 학습도 실험했지만 최종 제출은 통합 학습으로 선택 |
| 데이터 정리 | 상수·완전 결측 컬럼과 결측률 80% 이상 후보를 정리하고, DI 관련 21개 컬럼은 타입별 규칙으로 처리 |
| 특성 처리 | 횟수·연령 구간을 순서형 값으로 변환하고 나머지 범주형 값은 one-hot encoding |
| 검증 | 고정 seed의 Stratified 5-fold로 fold별 ROC-AUC 비교 |
| 불균형 대응 | class weight, undersampling, RandomOverSampler, SMOTE를 같은 조건에서 비교 |
| 모델 선택 | LightGBM·XGBoost·CatBoost를 비교한 뒤, 최종 후보 LightGBM을 Optuna random 30회 + TPE 100회로 탐색 |

## 최종 제출은 어떻게 선택했나

단순히 내부 CV가 높은 조합을 그대로 제출하지 않고, 실험 결과와 public 제출 기록이 함께 일관적인지를 확인했습니다.

| 후보 | 확인한 결과 | 최종 판단 |
| --- | --- | --- |
| No sampling + LightGBM | 기본 5-fold 평균 0.738864 | 튜닝의 기준선으로 채택 |
| Optuna 튜닝 LightGBM | 내부 5-fold 평균 0.740108 | 단일 모델 최종 후보로 채택 |
| DI/IVF 분리 학습 | 결측 원인 탐색을 바탕으로 실험 | 최상위 제출 구성에는 채택하지 않음 |
| 앙상블 | 내부 지표 상승 가능성을 확인 | public 제출 결과까지 고려해 최상위 구성에는 채택하지 않음 |
| 나이 기반 확률 후처리 | 대회 데이터의 특정 미상 범주를 대상으로 한 heuristic | 대회 제출 코드에만 한정 적용 |

따라서 공개 제출 이력의 최고 조합은 **시술 유형을 나누지 않은 단일 LightGBM**, DI 맥락 결측 처리, Optuna random 30회 + TPE 100회, 대회 한정 후처리로 정리됩니다. 이 판단은 의료 규칙이나 일반 서비스의 모델 선택 기준이 아닙니다.

## 실제 검증 근거와 해석 범위

실제 팀 발표자료와 제출 이력에서 교차 확인한 결과입니다.

| 실험 | ROC-AUC |
| --- | ---: |
| No sampling + LightGBM 기본 비교 · 내부 5-fold 평균 | 0.738864 |
| LightGBM Optuna 튜닝 · 내부 5-fold 평균 | 0.740108 |
| 제출 이력상 최고 public score | **0.741430139** |

최고 public score는 대회 public 평가 표본(테스트의 50%)에서 나온 제출 이력입니다. 대회 기준선 0.688을 넘었지만, 보존된 자료에는 최종 private score와 최종 순위 근거가 없어 이를 주장하지 않습니다.

내부 CV는 실험 방향을 비교하는 근거이지 일반화 성능이나 임상적 유효성을 증명하지 않습니다. 특히 당시 노트북과 `src/train.py`는 일부 결측 대치·범주형 인코딩을 fold 분리 전에 수행합니다. target을 사용하지 않은 변환이지만, 엄격한 leakage-free 검증 추정치로 해석할 수 없으므로 결과를 과장하지 않습니다.

세부 근거와 한계는 [모델링·검증 문서](docs/modeling-or-method.md)에서 확인할 수 있습니다.

## 내가 수행한 일

- 상수·완전 결측·저신호 컬럼을 정리하고 시술 맥락별 결측 처리 규칙을 구현했습니다.
- 반복 결측 6,291행의 패턴을 시술 유형 맥락에서 분석하고, DI/IVF 분리 학습의 효과와 데이터 규모 제약을 함께 비교했습니다.
- 횟수·연령 구간의 순서성을 유지하는 변환과 범주형 encoding 흐름을 구성했습니다.
- LightGBM, XGBoost, CatBoost를 동일한 5-fold ROC-AUC 기준으로 비교했습니다.
- class weight, RandomOverSampler, SMOTE, undersampling의 trade-off를 실험했습니다.
- LightGBM 후보를 Optuna random 30회와 TPE 100회로 탐색하고, 제출 이력상 최고 public ROC-AUC 0.741430139를 기록했습니다.
- 대회 제출용 확률 후처리가 원본 범주값에서도 일관되게 적용되도록 공개 스크립트의 조건을 보정했습니다.
- 원본 행 출력과 로컬 환경 로그를 제거해, 민감 데이터 공개 경계를 문서·노트북에 일관되게 적용했습니다.

## 팀과 담당 역할

팀 **우리오디가**로 참가했습니다. 이 공개본에는 확인 가능한 본인 기여만 기재합니다.

| 구성 | 담당 |
| --- | --- |
| 박민규 | 반복 결측 패턴 분석, 전처리·인코딩, 불균형 대응 실험, LightGBM·XGBoost·CatBoost 비교, Optuna 튜닝, 제출 파이프라인 및 공개본 정리 |

## 기술 구성

| 기술 | 사용 목적 |
| --- | --- |
| Python · pandas · NumPy | 대회 CSV 처리와 tabular feature 변환 |
| LightGBM · XGBoost · CatBoost | 이질적인 수치·범주형 특성의 gradient-boosting 비교 |
| scikit-learn | Stratified 5-fold, ROC-AUC, one-hot encoding |
| imbalanced-learn | undersampling·RandomOverSampler·SMOTE 비교 |
| Optuna | LightGBM 후보의 random search 30회 + TPE 100회 탐색 |
| Jupyter | 최종 제출 및 튜닝 실험 흐름 보존 |

## 구현물과 읽는 순서

1. [프로젝트 요약](docs/project-summary.md) — 문제, 역할, 공개 범위
2. [모델링·검증 문서](docs/modeling-or-method.md) — 전처리, 모델 비교, 검증 한계
3. [학습 스크립트](src/train.py) — 공개 검토용으로 정리한 재실행 파이프라인 구조
4. [최종 제출 흐름 노트북](notebooks/final_submission_pipeline.ipynb) — 데이터 로드부터 예측·후처리까지의 실제 대회 흐름을 정리한 출력 없는 공개본
5. [튜닝·앙상블 실험 노트북](notebooks/experiments/lgbm_tuning_ensemble.ipynb) — 추가 튜닝과 앙상블 탐색 코드

## 자료 공개 원칙

| 자료 | 공개 상태 | 이유 |
| --- | --- | --- |
| 코드·출력 없는 노트북·모델링 근거 | 공개 | 분석 과정과 판단을 검토할 수 있도록 보존 |
| 원본 CSV·제출 CSV | 비공개 | 대회 약관과 의료 인접 데이터의 공개 경계 준수 |
| 팀 최종 발표자료 | 수치만 문서화, 파일·이미지는 비공개 | 원본이 접근 권한이 필요한 Drive 자료이므로 외부 열람을 보장할 수 없음 |
| 최종 순위·private score | 미기재 | 보존된 근거가 없어 사실로 검증할 수 없음 |

## 실행과 공개 기준

공식 대회 CSV는 포함하지 않습니다. 권한이 있는 환경에서만 아래 파일을 `data/`에 준비하면 파이프라인을 실행할 수 있습니다.

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

```bash
pip install -r requirements.txt
python src/train.py
```

현재 `requirements.txt`는 패키지 목록이며, 당시의 모든 라이브러리 버전과 대회 실행 환경을 완전히 고정하지는 않습니다. 따라서 이 저장소는 공개 데이터만으로 동일 점수를 보장하는 재현 패키지가 아니라, **실제 분석의 구조·의사결정·검증 한계를 검토하는 포트폴리오 저장소**입니다. 새 실행에서 모델 결과를 해석할 때는 패키지 버전과 외부 데이터 권한이 결과에 영향을 준다는 점을 전제로 해야 합니다.

## 한계와 비사용 범위

- 원본 데이터 접근 없이 end-to-end 실행이나 점수 재현은 불가능합니다.
- 외부 검증, calibration, subgroup fairness, privacy threat model은 수행하지 않았습니다.
- 대회용 확률 후처리는 특정 필드에 관한 heuristic이며 의료 규칙이 아닙니다.
- 이 프로젝트는 진단, 치료 판단, 환자 선별, 의료 조언에 사용할 수 없습니다.

## 이용 안내

이 저장소는 포트폴리오 열람을 위해 공개합니다. 코드·문서·이미지의 재사용, 수정, 배포는 사전 문의가 필요합니다.
