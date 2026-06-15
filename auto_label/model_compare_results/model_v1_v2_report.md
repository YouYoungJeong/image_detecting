# Model v1 vs Model v2 성능 비교 보고서

## 1. 비교 목적

- dataset_v1 기반으로 학습한 기존 모델 v1과 dataset_v2 기반으로 개선 학습한 모델 v2의 성능을 비교한다.
- 동일한 test 데이터셋을 기준으로 평가하여 모델 개선 여부를 확인한다.
- 전체 성능과 클래스별 성능을 함께 확인하여 어떤 객체에서 개선 또는 하락이 발생했는지 분석한다.

## 2. 전체 성능 비교

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| model_v1 | 0.7142 | 0.7297 | 0.7471 | 0.5365 |
| model_v2 | 0.8288 | 0.7496 | 0.8063 | 0.5830 |

## 3. v2 개선 결과

- Precision 변화: +0.1146
- Recall 변화: +0.0199
- mAP50 변화: +0.0592
- mAP50-95 변화: +0.0465

## 4. 해석 기준

- Precision이 상승하면 잘못 탐지하는 비율이 줄어든 것으로 볼 수 있다.
- Recall이 상승하면 실제 객체를 놓치는 비율이 줄어든 것으로 볼 수 있다.
- mAP50은 객체 탐지 성능을 전체적으로 판단할 때 가장 직관적으로 확인하기 좋은 지표이다.
- mAP50-95는 더 엄격한 IoU 기준까지 포함하므로 박스 위치 정확도까지 함께 반영한다.

## 5. 클래스별 개선 상위 항목

| Class | v1 mAP50 | v2 mAP50 | Difference |
| --- | ---: | ---: | ---: |
| Tent1 | 0.7342 | 0.9421 | +0.2079 |
| Wall | 0.3832 | 0.5744 | +0.1913 |
| House2 | 0.5141 | 0.5844 | +0.0703 |

## 6. 클래스별 확인 필요 항목

| Class | v1 mAP50 | v2 mAP50 | Difference |
| --- | ---: | ---: | ---: |
| Human3 | 0.9199 | 0.9094 | -0.0105 |
| Human2 | 0.9054 | 0.9084 | +0.0030 |
| Human1 | 0.7116 | 0.7183 | +0.0067 |

## 7. 저장된 결과 파일

- overall_metric_comparison.png
- class_map50_comparison.png
- class_recall_comparison.png
- class_map50_improvement.png
- model_compare_summary.csv
- model_compare_class_metrics.csv
