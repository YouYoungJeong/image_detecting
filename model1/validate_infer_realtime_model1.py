# ============================================================
# validate_infer_realtime_model1.py
# Model1 YOLO26n 검증 / 추론 / 실시간성 확인 코드
# ============================================================
# 목적:
# - 학습 완료된 best.pt 불러오기
# - val 데이터셋 성능 검증
# - test 데이터셋 성능 검증
# - test 이미지 추론 결과 저장
# - latency, FPS 측정
# - 이미지별 추론 시간 CSV 저장
# - 최종 검증 요약 md/json 저장
# ============================================================

from pathlib import Path
import json
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "model1_yolo26n_aug"

DATASET_AUG_DIR = BASE_DIR / "dataset_v1_yolo26n_aug"
DATA_YAML = DATASET_AUG_DIR / "data.yaml"

RUNS_DIR = BASE_DIR / "runs"
DETECT_RUNS_DIR = RUNS_DIR / "detect"
MODEL_RUN_DIR = DETECT_RUNS_DIR / MODEL_NAME

BEST_MODEL_PATH = MODEL_RUN_DIR / "weights" / "best.pt"

EVAL_DIR = BASE_DIR / "eval_results" / MODEL_NAME
VAL_RESULT_DIR = EVAL_DIR / "val_metrics"
TEST_RESULT_DIR = EVAL_DIR / "test_metrics"
PREDICT_RESULT_DIR = EVAL_DIR / "test_predictions"
REALTIME_RESULT_DIR = EVAL_DIR / "realtime_latency"

IMGSZ = 640
CONF = 0.25
IOU = 0.70
DEVICE = 0
BATCH = 16
WORKERS = 4

MAX_LATENCY_IMAGES = None
WARMUP_COUNT = 10
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# ============================================================
# 1. 유틸 함수
# ============================================================

def make_dirs():
    for d in [
        EVAL_DIR,
        VAL_RESULT_DIR,
        TEST_RESULT_DIR,
        PREDICT_RESULT_DIR,
        REALTIME_RESULT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def get_test_image_dir():
    return DATASET_AUG_DIR / "test" / "images"


def check_paths():
    if not DATASET_AUG_DIR.exists():
        raise FileNotFoundError(f"[ERROR] 증강 데이터셋 경로가 없습니다: {DATASET_AUG_DIR}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"[ERROR] data.yaml 경로가 없습니다: {DATA_YAML}")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"[ERROR] best.pt 경로가 없습니다: {BEST_MODEL_PATH}")

    test_image_dir = get_test_image_dir()
    if not test_image_dir.exists():
        raise FileNotFoundError(f"[ERROR] test/images 경로가 없습니다: {test_image_dir}")


def get_image_files(image_dir: Path):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(image_dir.glob(f"*{ext}"))
        files.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(files)


def save_json(data: dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def safe_format(value, digits=4):
    if value is None:
        return "None"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def extract_class_metrics(metrics):
    rows = []

    names = getattr(metrics, "names", None)
    box = getattr(metrics, "box", None)

    if names is None or box is None:
        return pd.DataFrame(rows)

    class_maps = getattr(box, "maps", None)
    p_values = getattr(box, "p", None)
    r_values = getattr(box, "r", None)
    ap50_values = getattr(box, "ap50", None)

    for cls_id, cls_name in names.items():
        cls_id_int = int(cls_id)

        row = {
            "class_id": cls_id_int,
            "class_name": str(cls_name),
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP50_95": None,
        }

        try:
            if p_values is not None and len(p_values) > cls_id_int:
                row["precision"] = safe_float(p_values[cls_id_int])
        except Exception:
            pass

        try:
            if r_values is not None and len(r_values) > cls_id_int:
                row["recall"] = safe_float(r_values[cls_id_int])
        except Exception:
            pass

        try:
            if ap50_values is not None and len(ap50_values) > cls_id_int:
                row["mAP50"] = safe_float(ap50_values[cls_id_int])
        except Exception:
            pass

        try:
            if class_maps is not None and len(class_maps) > cls_id_int:
                row["mAP50_95"] = safe_float(class_maps[cls_id_int])
        except Exception:
            pass

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 2. val/test 성능 검증
# ============================================================

def validate_split(model: YOLO, split: str, save_project: Path):
    print(f"\n========== {split.upper()} 성능 검증 시작 ==========")

    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=IMGSZ,
        batch=BATCH,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        workers=WORKERS,
        project=str(save_project),
        name=split,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    box = metrics.box

    summary = {
        "split": split,
        "model_path": str(BEST_MODEL_PATH),
        "data_yaml": str(DATA_YAML),
        "imgsz": IMGSZ,
        "batch": BATCH,
        "conf": CONF,
        "iou": IOU,
        "device": DEVICE,
        "precision_mean": safe_float(box.mp),
        "recall_mean": safe_float(box.mr),
        "mAP50": safe_float(box.map50),
        "mAP50_95": safe_float(box.map),
        "mAP75": safe_float(box.map75),
    }

    split_dir = save_project / split
    split_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = split_dir / f"{split}_metrics_summary.json"
    save_json(summary, summary_json_path)

    class_df = extract_class_metrics(metrics)
    class_csv_path = split_dir / f"{split}_class_metrics.csv"
    class_df.to_csv(class_csv_path, index=False, encoding="utf-8-sig")

    print(f"\n[INFO] {split} 검증 요약 저장: {summary_json_path}")
    print(f"[INFO] {split} 클래스별 지표 저장: {class_csv_path}")
    print(f"Precision mean : {summary['precision_mean']}")
    print(f"Recall mean    : {summary['recall_mean']}")
    print(f"mAP50          : {summary['mAP50']}")
    print(f"mAP50-95       : {summary['mAP50_95']}")

    return metrics, summary, class_df


def validate_model(model: YOLO):
    val_metrics, val_summary, val_class_df = validate_split(
        model=model,
        split="val",
        save_project=VAL_RESULT_DIR,
    )

    test_metrics, test_summary, test_class_df = validate_split(
        model=model,
        split="test",
        save_project=TEST_RESULT_DIR,
    )

    total_summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": MODEL_NAME,
        "model_path": str(BEST_MODEL_PATH),
        "data_yaml": str(DATA_YAML),
        "val": val_summary,
        "test": test_summary,
    }

    save_path = EVAL_DIR / "val_test_metrics_summary.json"
    save_json(total_summary, save_path)

    print(f"\n[INFO] val/test 통합 검증 요약 저장: {save_path}")

    return total_summary


# ============================================================
# 3. test 이미지 추론 결과 저장
# ============================================================

def predict_test_images(model: YOLO):
    print("\n========== Test 이미지 추론 결과 저장 시작 ==========")

    test_image_dir = get_test_image_dir()

    results = model.predict(
        source=str(test_image_dir),
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(PREDICT_RESULT_DIR),
        name="predict",
        exist_ok=True,
        verbose=False,
    )

    predict_save_dir = PREDICT_RESULT_DIR / "predict"

    rows = []

    for r in results:
        image_path = Path(r.path)

        box_count = 0
        classes = []
        confidences = []

        if r.boxes is not None and len(r.boxes) > 0:
            box_count = len(r.boxes)

            try:
                classes = [int(x) for x in r.boxes.cls.detach().cpu().numpy().tolist()]
            except Exception:
                classes = []

            try:
                confidences = [float(x) for x in r.boxes.conf.detach().cpu().numpy().tolist()]
            except Exception:
                confidences = []

        rows.append({
            "image": image_path.name,
            "detected_box_count": box_count,
            "detected_classes": ",".join(map(str, classes)),
            "confidences": ",".join([f"{x:.4f}" for x in confidences]),
        })

    pred_df = pd.DataFrame(rows)
    pred_csv_path = predict_save_dir / "test_prediction_summary.csv"
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 추론 이미지 저장 폴더: {predict_save_dir}")
    print(f"[INFO] 추론 요약 CSV 저장: {pred_csv_path}")

    return predict_save_dir, pred_csv_path


# ============================================================
# 4. 실시간성 Latency / FPS 측정
# ============================================================

def measure_realtime_latency(model: YOLO):
    print("\n========== 실시간성 확인: Latency / FPS 측정 시작 ==========")

    test_image_dir = get_test_image_dir()
    image_files = get_image_files(test_image_dir)

    if MAX_LATENCY_IMAGES is not None:
        image_files = image_files[:MAX_LATENCY_IMAGES]

    if len(image_files) == 0:
        raise RuntimeError(f"[ERROR] latency 측정용 이미지가 없습니다: {test_image_dir}")

    print(f"[INFO] 측정 이미지 수: {len(image_files)}")
    print(f"[INFO] warmup 횟수: {WARMUP_COUNT}")

    warmup_files = image_files[:min(WARMUP_COUNT, len(image_files))]

    for img_path in warmup_files:
        _ = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False,
        )

    rows = []

    for img_path in tqdm(image_files, desc="Latency 측정 중"):
        start = time.perf_counter()

        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False,
        )

        end = time.perf_counter()

        wall_time_ms = (end - start) * 1000.0

        r = results[0]
        speed = getattr(r, "speed", {}) or {}

        preprocess_ms = safe_float(speed.get("preprocess", 0.0)) or 0.0
        inference_ms = safe_float(speed.get("inference", 0.0)) or 0.0
        postprocess_ms = safe_float(speed.get("postprocess", 0.0)) or 0.0

        model_speed_total_ms = preprocess_ms + inference_ms + postprocess_ms

        detected_box_count = 0
        if r.boxes is not None:
            detected_box_count = len(r.boxes)

        fps_by_wall_time = 1000.0 / wall_time_ms if wall_time_ms > 0 else 0.0
        fps_by_model_speed = 1000.0 / model_speed_total_ms if model_speed_total_ms > 0 else 0.0

        rows.append({
            "image": img_path.name,
            "wall_time_ms": wall_time_ms,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "model_speed_total_ms": model_speed_total_ms,
            "fps_by_wall_time": fps_by_wall_time,
            "fps_by_model_speed": fps_by_model_speed,
            "detected_box_count": detected_box_count,
        })

    df = pd.DataFrame(rows)

    latency_csv_path = REALTIME_RESULT_DIR / "latency_per_image.csv"
    df.to_csv(latency_csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": MODEL_NAME,
        "model_path": str(BEST_MODEL_PATH),
        "data_yaml": str(DATA_YAML),
        "test_image_dir": str(test_image_dir),
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "device": DEVICE,
        "image_count": int(len(df)),
        "wall_time_ms_mean": safe_float(df["wall_time_ms"].mean()),
        "wall_time_ms_median": safe_float(df["wall_time_ms"].median()),
        "wall_time_ms_min": safe_float(df["wall_time_ms"].min()),
        "wall_time_ms_max": safe_float(df["wall_time_ms"].max()),
        "wall_time_ms_p95": safe_float(df["wall_time_ms"].quantile(0.95)),
        "preprocess_ms_mean": safe_float(df["preprocess_ms"].mean()),
        "inference_ms_mean": safe_float(df["inference_ms"].mean()),
        "postprocess_ms_mean": safe_float(df["postprocess_ms"].mean()),
        "model_speed_total_ms_mean": safe_float(df["model_speed_total_ms"].mean()),
        "inference_ms_median": safe_float(df["inference_ms"].median()),
        "inference_ms_min": safe_float(df["inference_ms"].min()),
        "inference_ms_max": safe_float(df["inference_ms"].max()),
        "inference_ms_p95": safe_float(df["inference_ms"].quantile(0.95)),
        "fps_by_wall_time_mean": safe_float(df["fps_by_wall_time"].mean()),
        "fps_by_wall_time_median": safe_float(df["fps_by_wall_time"].median()),
        "fps_by_model_speed_mean": safe_float(df["fps_by_model_speed"].mean()),
        "fps_by_model_speed_median": safe_float(df["fps_by_model_speed"].median()),
    }

    latency_summary_path = REALTIME_RESULT_DIR / "latency_summary.json"
    save_json(summary, latency_summary_path)

    latency_report_path = REALTIME_RESULT_DIR / "latency_report.md"
    save_latency_report(summary, latency_csv_path, latency_report_path)

    print(f"\n[INFO] 이미지별 latency CSV 저장: {latency_csv_path}")
    print(f"[INFO] latency 요약 JSON 저장: {latency_summary_path}")
    print(f"[INFO] latency 리포트 저장: {latency_report_path}")

    print("\n========== Latency / FPS 요약 ==========")
    print(f"평균 wall time      : {summary['wall_time_ms_mean']:.2f} ms")
    print(f"P95 wall time       : {summary['wall_time_ms_p95']:.2f} ms")
    print(f"평균 inference time : {summary['inference_ms_mean']:.2f} ms")
    print(f"P95 inference time  : {summary['inference_ms_p95']:.2f} ms")
    print(f"평균 FPS(wall)      : {summary['fps_by_wall_time_mean']:.2f}")
    print(f"평균 FPS(model)     : {summary['fps_by_model_speed_mean']:.2f}")

    return df, summary


def save_latency_report(summary: dict, latency_csv_path: Path, report_path: Path):
    text = f"""# {MODEL_NAME} 실시간성 측정 결과

## 1. 측정 설정

| 항목 | 값 |
| --- | --- |
| 모델 | {summary['model_path']} |
| data.yaml | {summary['data_yaml']} |
| test 이미지 경로 | {summary['test_image_dir']} |
| 이미지 크기 | {summary['imgsz']} |
| confidence | {summary['conf']} |
| IoU | {summary['iou']} |
| device | {summary['device']} |
| 측정 이미지 수 | {summary['image_count']} |

## 2. Wall Time 기준 Latency

| 항목 | 값 |
| --- | --- |
| 평균 wall time | {safe_format(summary['wall_time_ms_mean'], 2)} ms |
| 중앙값 wall time | {safe_format(summary['wall_time_ms_median'], 2)} ms |
| 최소 wall time | {safe_format(summary['wall_time_ms_min'], 2)} ms |
| 최대 wall time | {safe_format(summary['wall_time_ms_max'], 2)} ms |
| P95 wall time | {safe_format(summary['wall_time_ms_p95'], 2)} ms |

## 3. Ultralytics Speed 기준

| 항목 | 값 |
| --- | --- |
| 평균 preprocess | {safe_format(summary['preprocess_ms_mean'], 2)} ms |
| 평균 inference | {safe_format(summary['inference_ms_mean'], 2)} ms |
| 평균 postprocess | {safe_format(summary['postprocess_ms_mean'], 2)} ms |
| 평균 model speed total | {safe_format(summary['model_speed_total_ms_mean'], 2)} ms |
| P95 inference | {safe_format(summary['inference_ms_p95'], 2)} ms |

## 4. FPS

| 기준 | FPS |
| --- | --- |
| wall time 기준 평균 FPS | {safe_format(summary['fps_by_wall_time_mean'], 2)} |
| wall time 기준 중앙값 FPS | {safe_format(summary['fps_by_wall_time_median'], 2)} |
| model speed 기준 평균 FPS | {safe_format(summary['fps_by_model_speed_mean'], 2)} |
| model speed 기준 중앙값 FPS | {safe_format(summary['fps_by_model_speed_median'], 2)} |

## 5. 해석 기준

- `wall_time_ms`는 Python에서 이미지 1장을 입력하고 결과가 반환될 때까지 걸린 전체 시간이다.
- `inference_ms`는 Ultralytics가 기록한 순수 모델 추론 시간이다.
- 실시간성을 발표할 때는 평균 latency뿐 아니라 `P95 latency`를 함께 제시하는 것이 좋다.
- 실제 시뮬레이션 연동 시에는 카메라 입력, 전처리, 후처리, 시각화, 통신 시간이 추가될 수 있다.

## 6. 저장 파일

| 파일 | 설명 |
| --- | --- |
| {latency_csv_path} | 이미지별 latency, FPS 기록 |
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)


# ============================================================
# 5. 최종 평가 리포트 저장
# ============================================================

def save_final_eval_summary(metrics_summary: dict, latency_summary: dict, predict_save_dir: Path):
    final_path = EVAL_DIR / "final_eval_summary.md"

    val = metrics_summary["val"]
    test = metrics_summary["test"]

    text = f"""# {MODEL_NAME} 최종 검증 / 추론 / 실시간성 요약

## 1. 모델 정보

| 항목 | 내용 |
| --- | --- |
| 모델 | YOLO26n 기반 신규 Model1 |
| best.pt | {BEST_MODEL_PATH} |
| data.yaml | {DATA_YAML} |
| imgsz | {IMGSZ} |
| conf | {CONF} |
| IoU | {IOU} |
| device | {DEVICE} |

## 2. Validation 성능

| 지표 | 값 |
| --- | --- |
| Precision mean | {safe_format(val['precision_mean'], 4)} |
| Recall mean | {safe_format(val['recall_mean'], 4)} |
| mAP50 | {safe_format(val['mAP50'], 4)} |
| mAP50-95 | {safe_format(val['mAP50_95'], 4)} |
| mAP75 | {safe_format(val['mAP75'], 4)} |

## 3. Test 성능

| 지표 | 값 |
| --- | --- |
| Precision mean | {safe_format(test['precision_mean'], 4)} |
| Recall mean | {safe_format(test['recall_mean'], 4)} |
| mAP50 | {safe_format(test['mAP50'], 4)} |
| mAP50-95 | {safe_format(test['mAP50_95'], 4)} |
| mAP75 | {safe_format(test['mAP75'], 4)} |

## 4. Test 이미지 추론 결과

| 항목 | 경로 |
| --- | --- |
| 추론 이미지 저장 폴더 | {predict_save_dir} |
| 추론 라벨 저장 폴더 | {predict_save_dir / 'labels'} |

## 5. 실시간성 측정 결과

| 항목 | 값 |
| --- | --- |
| 평균 wall time | {safe_format(latency_summary['wall_time_ms_mean'], 2)} ms |
| P95 wall time | {safe_format(latency_summary['wall_time_ms_p95'], 2)} ms |
| 평균 inference time | {safe_format(latency_summary['inference_ms_mean'], 2)} ms |
| P95 inference time | {safe_format(latency_summary['inference_ms_p95'], 2)} ms |
| wall time 기준 평균 FPS | {safe_format(latency_summary['fps_by_wall_time_mean'], 2)} |
| model speed 기준 평균 FPS | {safe_format(latency_summary['fps_by_model_speed_mean'], 2)} |

## 6. 결과 파일 위치

| 항목 | 경로 |
| --- | --- |
| val 검증 결과 | {VAL_RESULT_DIR} |
| test 검증 결과 | {TEST_RESULT_DIR} |
| test 추론 결과 | {PREDICT_RESULT_DIR} |
| latency 결과 | {REALTIME_RESULT_DIR} |
| 이미지별 latency CSV | {REALTIME_RESULT_DIR / 'latency_per_image.csv'} |
| latency 리포트 | {REALTIME_RESULT_DIR / 'latency_report.md'} |

## 7. 발표용 정리 문장

본 검증 단계에서는 학습된 YOLO26n 기반 신규 Model1의 `best.pt`를 사용하여 validation 및 test 데이터셋에 대한 탐지 성능을 평가하였다.  
또한 test 이미지 전체에 대해 추론 결과 이미지를 저장하고, 이미지별 latency와 FPS를 측정하여 실시간 적용 가능성을 함께 확인하였다.  
성능 평가는 Precision, Recall, mAP50, mAP50-95를 기준으로 수행하였으며, 실시간성 평가는 평균 latency와 P95 latency, 평균 FPS를 함께 기록하였다.
"""

    with open(final_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[INFO] 최종 평가 요약 저장: {final_path}")

    return final_path


# ============================================================
# 6. main
# ============================================================

def main():
    print("========== Model1 YOLO26n 검증 / 추론 / 실시간성 확인 시작 ==========")

    make_dirs()
    check_paths()

    print("\n[INFO] 설정 확인")
    print(f"  MODEL_NAME      : {MODEL_NAME}")
    print(f"  BEST_MODEL_PATH : {BEST_MODEL_PATH}")
    print(f"  DATA_YAML       : {DATA_YAML}")
    print(f"  EVAL_DIR        : {EVAL_DIR}")
    print(f"  IMGSZ           : {IMGSZ}")
    print(f"  CONF            : {CONF}")
    print(f"  IOU             : {IOU}")
    print(f"  DEVICE          : {DEVICE}")

    print("\n[STEP 1] best.pt 모델 로드")
    model = YOLO(str(BEST_MODEL_PATH))

    print("\n[STEP 2] val/test 성능 검증")
    metrics_summary = validate_model(model)

    print("\n[STEP 3] test 이미지 추론 결과 저장")
    predict_save_dir, pred_csv_path = predict_test_images(model)

    print("\n[STEP 4] latency / FPS 측정")
    latency_df, latency_summary = measure_realtime_latency(model)

    print("\n[STEP 5] 최종 평가 요약 저장")
    final_summary_path = save_final_eval_summary(
        metrics_summary=metrics_summary,
        latency_summary=latency_summary,
        predict_save_dir=predict_save_dir,
    )

    print("\n========== 전체 완료 ==========")
    print(f"[FINAL SUMMARY] {final_summary_path}")
    print(f"[EVAL DIR]      {EVAL_DIR}")

    print("\n주요 결과 파일:")
    print(f"  - {EVAL_DIR / 'val_test_metrics_summary.json'}")
    print(f"  - {VAL_RESULT_DIR / 'val' / 'val_metrics_summary.json'}")
    print(f"  - {TEST_RESULT_DIR / 'test' / 'test_metrics_summary.json'}")
    print(f"  - {predict_save_dir / 'test_prediction_summary.csv'}")
    print(f"  - {REALTIME_RESULT_DIR / 'latency_per_image.csv'}")
    print(f"  - {REALTIME_RESULT_DIR / 'latency_summary.json'}")
    print(f"  - {REALTIME_RESULT_DIR / 'latency_report.md'}")
    print(f"  - {EVAL_DIR / 'final_eval_summary.md'}")


if __name__ == "__main__":
    main()
