# YOLO26n Model1 이미지 사이즈별 비교 결과

## 1. 실험 설정

- 모델: YOLO26n
- 사전학습 가중치: yolo26n.pt
- 데이터셋: C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\dataset_v1_yolo26n_aug
- data.yaml: C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\dataset_v1_yolo26n_aug\data.yaml
- 이미지 사이즈 후보: [640, 480, 320, 288]
- epochs: 200
- patience: 25
- batch: 16
- device: 0

## 2. 비교 결과

|   imgsz | model_name              | best_path                                                                                                       | run_dir                                                                                         |   val_precision |   val_recall |   val_mAP50 |   val_mAP50_95 |   val_preprocess_ms |   val_inference_ms |   val_postprocess_ms |   test_precision |   test_recall |   test_mAP50 |   test_mAP50_95 |   test_preprocess_ms |   test_inference_ms |   test_postprocess_ms |   avg_latency_ms |   median_latency_ms |   p95_latency_ms |   avg_fps |   latency_image_count | per_image_latency_csv                                                                                                           |
|--------:|:------------------------|:----------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|----------------:|-------------:|------------:|---------------:|--------------------:|-------------------:|---------------------:|-----------------:|--------------:|-------------:|----------------:|---------------------:|--------------------:|----------------------:|-----------------:|--------------------:|-----------------:|----------:|----------------------:|:--------------------------------------------------------------------------------------------------------------------------------|
|     640 | model1_yolo26n_imgsz640 | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\runs\detect\model1_yolo26n_imgsz640\weights\best.pt | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\runs\detect\model1_yolo26n_imgsz640 |        0.848165 |     0.834842 |    0.835048 |       0.635574 |            0.547376 |           1.22289  |            0.199064  |         0.875529 |      0.845963 |     0.8575   |        0.661536 |             0.519802 |            1.0574   |             0.135657  |          9.26231 |             9.69725 |          11.693  |  113.006  |                  1042 | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\model1_yolo26n_imgsz_compare_results\latency_per_image_imgsz640.csv |
|     480 | model1_yolo26n_imgsz480 | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\runs\detect\model1_yolo26n_imgsz480\weights\best.pt | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\runs\detect\model1_yolo26n_imgsz480 |        0.846588 |     0.810543 |    0.816169 |       0.608081 |            0.301446 |           0.968105 |            0.103827  |         0.883159 |      0.827758 |     0.84986  |        0.641513 |             0.373294 |            0.911946 |             0.192512  |         10.9379  |            10.9271  |          16.8278 |   99.4096 |                  1042 | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\model1_yolo26n_imgsz_compare_results\latency_per_image_imgsz480.csv |
|     320 | model1_yolo26n_imgsz320 | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\runs\detect\model1_yolo26n_imgsz320\weights\best.pt | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\runs\detect\model1_yolo26n_imgsz320 |        0.846054 |     0.766119 |    0.763516 |       0.545453 |            0.177013 |           1.04129  |            0.114889  |         0.84528  |      0.795103 |     0.783687 |        0.567358 |             0.171022 |            0.809061 |             0.134493  |          9.82569 |            10.389   |          12.7701 |  107.572  |                  1042 | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\model1_yolo26n_imgsz_compare_results\latency_per_image_imgsz320.csv |
|     288 | model1_yolo26n_imgsz288 | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\runs\detect\model1_yolo26n_imgsz288\weights\best.pt | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\runs\detect\model1_yolo26n_imgsz288 |        0.834547 |     0.752509 |    0.742442 |       0.518423 |            0.131499 |           0.884944 |            0.0957301 |         0.855659 |      0.770801 |     0.776034 |        0.543215 |             0.137238 |            0.644773 |             0.0999393 |          8.8432  |             9.4507  |          11.9919 |  120.524  |                  1042 | C:\Users\User\work\HDRT_FinalProject\image_detecting\model1\model1_yolo26n_imgsz_compare_results\latency_per_image_imgsz288.csv |

## 3. 해석 기준

- mAP50, mAP50-95, Precision, Recall은 높을수록 좋다.
- latency는 낮을수록 좋다.
- FPS는 높을수록 좋다.
- 시뮬레이션 연동에서는 정확도와 FPS를 동시에 고려해야 한다.
- 원거리 Tank 탐지가 중요하면 320 이하 크기에서 성능 저하 여부를 반드시 확인해야 한다.
