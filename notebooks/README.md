# Notebooks

## 공개 기준

두 노트북은 실제 대회 코드 흐름을 보존하되, 원본 행 미리보기·로컬 경로·실험 로그가 포함된 출력은 제거했습니다. 원본 데이터와 제출 CSV는 저장소에 포함하지 않습니다.

## 최종 제출 흐름

- `final_submission_pipeline.ipynb`: 데이터 로드, 전처리, 모델 비교, Optuna 튜닝, 추론, 후처리를 포함합니다. 내부 5-fold 수치는 [모델링·검증 문서](../docs/modeling-or-method.md)의 해석 범위를 따릅니다.

## 실험 기록

- `experiments/lgbm_tuning_ensemble.ipynb`: LightGBM 추가 튜닝과 앙상블 실험을 포함합니다.

실행하려면 루트에 `data/` 디렉터리를 만들고 `train.csv`, `test.csv`, `sample_submission.csv`를 배치하세요. 이 노트북의 CV 전처리 범위와 결과 해석 한계는 [모델링·검증 문서](../docs/modeling-or-method.md)를 함께 확인하세요.
