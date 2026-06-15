# pip install ultralytics opencv-python pandas matplotlib pyyaml tqdm

# ============================================================
# train_compare_model1_yolo26n_imgsz.py
# YOLO26n Model1 이미지 사이즈별 학습/검증/속도 비교 코드
# ============================================================
# 목적:
# - 증강 데이터셋 dataset_v1_yolo26n_aug 사용
# - YOLO26n 모델을 imgsz 640, 480, 320, 288로 각각 학습
# - 각 모델의 val/test 성능 비교
# - test 이미지 기준 latency, FPS 측정
# - 모델별 이미지 추론 시간 CSV 저장
# - 전체 비교 summary CSV 저장
# - subplot 비교 그래프 저장
#
# 전제:
# - make_model1_aug_dataset_only.py 실행 완료
# - dataset_v1_yolo26n_aug/data.yaml 존재
# ============================================================

from pathlib import Path
import time
import json
from datetime import datetime
from statistics import mean, median

import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 증강 데이터셋
DATASET_AUG_DIR = BASE_DIR / "dataset_v1_yolo26n_aug"
DATA_YAML = DATASET_AUG_DIR / "data.yaml"

# YOLO26n 사전학습 가중치
PRETRAINED_MODEL = "yolo26n.pt"

# 결과 저장 위치
RUNS_DIR = BASE_DIR / "runs"
DETECT_RUNS_DIR = RUNS_DIR / "detect"

# 이미지 사이즈별 실험
IMAGE_SIZES = [640, 480, 320, 288]

# 학습 설정
EPOCHS = 200
PATIENCE = 25
BATCH = 16
DEVICE = 0          # GPU 0번 사용, CPU면 "cpu"
WORKERS = 4
SEED = 42

# 추론 속도 측정 설정
CONF_THRES = 0.25
IOU_THRES = 0.7
MAX_DET = 300
WARMUP_IMAGES = 10

# 이미 학습된 best.pt가 있으면 재학습 생략 여부
# True  : best.pt가 있으면 학습 생략 후 검증/속도 측정만 수행
# False : 항상 다시 학습
SKIP_TRAIN_IF_BEST_EXISTS = True

# 결과 파일 접두사
EXPERIMENT_PREFIX = "model1_yolo26n_imgsz_compare"


# ============================================================
# 1. 유틸 함수
# ============================================================

def get_image_files(image_dir: Path):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    files = []

    for ext in exts:
        files.extend(image_dir.glob(f"*{ext}"))
        files.extend(image_dir.glob(f"*{ext.upper()}"))

    return sorted(files)


def load_yaml(yaml_path: Path):
    if not yaml_path.exists():
        raise FileNotFoundError(f"[ERROR] data.yaml이 없습니다: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_path(data_yaml_path: Path, split_key: str):
    """
    data.yaml 내부 train/val/test 경로를 실제 Path로 변환한다.
    """
    data = load_yaml(data_yaml_path)
    base_dir = data_yaml_path.parent

    if split_key not in data:
        return None

    split_value = data[split_key]

    if split_value is None:
        return None

    split_path = Path(split_value)

    if split_path.is_absolute():
        return split_path

    return base_dir / split_path


def get_eval_image_dir(data_yaml_path: Path):
    """
    latency 측정용 이미지 경로를 반환한다.
    우선순위:
    1. test/images
    2. val/images
    """
    test_dir = resolve_dataset_path(data_yaml_path, "test")

    if test_dir is not None and test_dir.exists():
        return test_dir

    val_dir = resolve_dataset_path(data_yaml_path, "val")

    if val_dir is not None and val_dir.exists():
        return val_dir

    raise FileNotFoundError("[ERROR] test 또는 val 이미지 경로를 찾을 수 없습니다.")


def cuda_synchronize_if_needed(device):
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def extract_metrics(metrics):
    """
    Ultralytics DetMetrics 객체에서 주요 성능값 추출
    """
    box = getattr(metrics, "box", None)

    if box is None:
        return {
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP50_95": None
        }

    return {
        "precision": safe_float(getattr(box, "mp", None)),
        "recall": safe_float(getattr(box, "mr", None)),
        "mAP50": safe_float(getattr(box, "map50", None)),
        "mAP50_95": safe_float(getattr(box, "map", None))
    }


def extract_speed(metrics):
    """
    Ultralytics validation 결과의 speed 정보 추출
    단위는 보통 ms/image
    """
    speed = getattr(metrics, "speed", None)

    if not isinstance(speed, dict):
        return {
            "val_preprocess_ms": None,
            "val_inference_ms": None,
            "val_loss_ms": None,
            "val_postprocess_ms": None
        }

    return {
        "val_preprocess_ms": safe_float(speed.get("preprocess")),
        "val_inference_ms": safe_float(speed.get("inference")),
        "val_loss_ms": safe_float(speed.get("loss")),
        "val_postprocess_ms": safe_float(speed.get("postprocess"))
    }


def make_run_name(imgsz: int):
    return f"model1_yolo26n_imgsz{imgsz}"


def make_run_dir(imgsz: int):
    return DETECT_RUNS_DIR / make_run_name(imgsz)


def make_best_path(imgsz: int):
    return make_run_dir(imgsz) / "weights" / "best.pt"


# ============================================================
# 2. 모델 학습
# ============================================================

def train_model_for_imgsz(imgsz: int):
    """
    특정 imgsz로 YOLO26n 모델 학습
    """
    run_name = make_run_name(imgsz)
    run_dir = make_run_dir(imgsz)
    best_path = make_best_path(imgsz)

    if SKIP_TRAIN_IF_BEST_EXISTS and best_path.exists():
        print(f"\n[SKIP TRAIN] 이미 best.pt 존재: {best_path}")
        return best_path, run_dir, None

    print("\n" + "=" * 70)
    print(f"[TRAIN START] YOLO26n imgsz={imgsz}")
    print("=" * 70)

    model = YOLO(PRETRAINED_MODEL)

    train_start = time.time()

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=imgsz,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        workers=WORKERS,
        project=str(DETECT_RUNS_DIR),
        name=run_name,
        exist_ok=True,
        seed=SEED,

        # 외부 증강 데이터셋을 이미 만들었으므로 YOLO 내부 추가 증강은 꺼둠
        augment=False,

        optimizer="auto",
        plots=True,
        val=True
    )

    train_end = time.time()

    elapsed_min = (train_end - train_start) / 60

    print(f"[TRAIN DONE] imgsz={imgsz}")
    print(f"[BEST] {best_path}")
    print(f"[TIME] {elapsed_min:.2f} min")

    train_config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "imgsz": imgsz,
        "pretrained_model": PRETRAINED_MODEL,
        "data_yaml": str(DATA_YAML),
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "seed": SEED,
        "train_elapsed_min": elapsed_min,
        "best_path": str(best_path)
    }

    config_path = run_dir / "training_config.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=4, ensure_ascii=False)

    return best_path, run_dir, results


# ============================================================
# 3. val/test 검증
# ============================================================

def validate_model(best_path: Path, imgsz: int, split: str):
    """
    val 또는 test split 기준 성능 측정
    """
    if not best_path.exists():
        raise FileNotFoundError(f"[ERROR] best.pt가 없습니다: {best_path}")

    print(f"\n[VALIDATE] imgsz={imgsz}, split={split}")

    model = YOLO(str(best_path))

    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=imgsz,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        plots=True,
        verbose=False
    )

    metric_dict = extract_metrics(metrics)
    speed_dict = extract_speed(metrics)

    result = {
        "imgsz": imgsz,
        "split": split,
        **metric_dict,
        **speed_dict
    }

    return result


# ============================================================
# 4. latency / FPS 측정
# ============================================================

def measure_latency_and_fps(best_path: Path, imgsz: int, save_dir: Path):
    """
    test 이미지 기준 이미지별 latency와 FPS 측정
    """
    if not best_path.exists():
        raise FileNotFoundError(f"[ERROR] best.pt가 없습니다: {best_path}")

    image_dir = get_eval_image_dir(DATA_YAML)
    image_files = get_image_files(image_dir)

    if len(image_files) == 0:
        raise RuntimeError(f"[ERROR] latency 측정용 이미지가 없습니다: {image_dir}")

    print(f"\n[LATENCY] imgsz={imgsz}")
    print(f"[IMAGE DIR] {image_dir}")
    print(f"[IMAGE COUNT] {len(image_files)}")

    model = YOLO(str(best_path))

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------
    warmup_files = image_files[:min(WARMUP_IMAGES, len(image_files))]

    print(f"[WARMUP] {len(warmup_files)} images")

    for img_path in warmup_files:
        _ = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=MAX_DET,
            device=DEVICE,
            verbose=False
        )

    cuda_synchronize_if_needed(DEVICE)

    # --------------------------------------------------------
    # 실제 latency 측정
    # --------------------------------------------------------
    rows = []

    for idx, img_path in enumerate(tqdm(image_files, desc=f"Measure imgsz={imgsz}")):
        cuda_synchronize_if_needed(DEVICE)
        start = time.perf_counter()

        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=MAX_DET,
            device=DEVICE,
            verbose=False
        )

        cuda_synchronize_if_needed(DEVICE)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0

        detection_count = 0
        if len(results) > 0 and results[0].boxes is not None:
            detection_count = len(results[0].boxes)

        rows.append({
            "imgsz": imgsz,
            "image_index": idx,
            "image_path": str(img_path),
            "file_name": img_path.name,
            "latency_ms": latency_ms,
            "fps": fps,
            "detection_count": detection_count
        })

    latency_df = pd.DataFrame(rows)

    save_dir.mkdir(parents=True, exist_ok=True)

    per_image_csv = save_dir / f"latency_per_image_imgsz{imgsz}.csv"
    latency_df.to_csv(per_image_csv, index=False, encoding="utf-8-sig")

    latency_values = latency_df["latency_ms"].tolist()
    fps_values = latency_df["fps"].tolist()

    avg_latency = mean(latency_values)
    median_latency = median(latency_values)
    p95_latency = latency_df["latency_ms"].quantile(0.95)
    avg_fps = mean(fps_values)

    summary = {
        "imgsz": imgsz,
        "latency_image_count": len(latency_df),
        "avg_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "avg_fps": avg_fps,
        "per_image_latency_csv": str(per_image_csv)
    }

    print(f"[LATENCY DONE] imgsz={imgsz}")
    print(f"  avg_latency_ms    : {avg_latency:.3f}")
    print(f"  median_latency_ms : {median_latency:.3f}")
    print(f"  p95_latency_ms    : {p95_latency:.3f}")
    print(f"  avg_fps           : {avg_fps:.3f}")

    return summary, latency_df


# ============================================================
# 5. 비교 그래프 생성
# ============================================================

def plot_comparison_subplots(summary_df: pd.DataFrame, save_path: Path):
    """
    이미지 사이즈별 성능/속도 비교 subplot 저장
    """
    plot_df = summary_df.sort_values("imgsz", ascending=False).copy()

    metrics = [
        ("val_mAP50", "Validation mAP50"),
        ("val_mAP50_95", "Validation mAP50-95"),
        ("val_precision", "Validation Precision"),
        ("val_recall", "Validation Recall"),
        ("avg_latency_ms", "Average Latency (ms)"),
        ("avg_fps", "Average FPS"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    x_labels = plot_df["imgsz"].astype(str).tolist()

    for ax, (col, title) in zip(axes, metrics):
        if col not in plot_df.columns:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue

        values = plot_df[col].tolist()

        ax.bar(x_labels, values)
        ax.set_title(title)
        ax.set_xlabel("imgsz")
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.3)

        for i, v in enumerate(values):
            if pd.isna(v):
                continue

            if "latency" in col:
                label = f"{v:.1f}"
            elif "fps" in col:
                label = f"{v:.1f}"
            else:
                label = f"{v:.3f}"

            ax.text(i, v, label, ha="center", va="bottom", fontsize=9)

    fig.suptitle("YOLO26n Model1 Image Size Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[INFO] 비교 subplot 그래프 저장: {save_path}")


def plot_latency_boxplot(latency_dfs: dict, save_path: Path):
    """
    이미지 사이즈별 latency 분포 boxplot 저장
    """
    labels = []
    data = []

    for imgsz in sorted(latency_dfs.keys(), reverse=True):
        df = latency_dfs[imgsz]
        labels.append(str(imgsz))
        data.append(df["latency_ms"].tolist())

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title("Latency Distribution by Image Size")
    plt.xlabel("imgsz")
    plt.ylabel("Latency (ms)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[INFO] latency boxplot 저장: {save_path}")


def plot_metric_lines(summary_df: pd.DataFrame, save_path: Path):
    """
    이미지 사이즈별 주요 지표 line plot 저장
    """
    plot_df = summary_df.sort_values("imgsz").copy()

    x = plot_df["imgsz"].tolist()

    plt.figure(figsize=(10, 6))

    for col, label in [
        ("val_mAP50", "val mAP50"),
        ("val_mAP50_95", "val mAP50-95"),
        ("val_precision", "val Precision"),
        ("val_recall", "val Recall"),
    ]:
        if col in plot_df.columns:
            plt.plot(x, plot_df[col], marker="o", label=label)

    plt.title("Validation Metrics by Image Size")
    plt.xlabel("imgsz")
    plt.ylabel("score")
    plt.xticks(x)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[INFO] metric line plot 저장: {save_path}")


def plot_speed_lines(summary_df: pd.DataFrame, save_path: Path):
    """
    이미지 사이즈별 latency/FPS line plot 저장
    """
    plot_df = summary_df.sort_values("imgsz").copy()

    x = plot_df["imgsz"].tolist()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x, plot_df["avg_latency_ms"], marker="o", label="Average latency (ms)")
    ax1.set_xlabel("imgsz")
    ax1.set_ylabel("Average latency (ms)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, plot_df["avg_fps"], marker="o", label="Average FPS")
    ax2.set_ylabel("Average FPS")

    plt.title("Latency and FPS by Image Size")
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[INFO] speed line plot 저장: {save_path}")


# ============================================================
# 6. 결과 요약 저장
# ============================================================

def save_experiment_summary(summary_df: pd.DataFrame, save_dir: Path):
    """
    비교 결과 CSV, JSON, MD 저장
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / f"{EXPERIMENT_PREFIX}_summary.csv"
    json_path = save_dir / f"{EXPERIMENT_PREFIX}_summary.json"
    md_path = save_dir / f"{EXPERIMENT_PREFIX}_summary.md"

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=4, ensure_ascii=False)

    md_text = "# YOLO26n Model1 이미지 사이즈별 비교 결과\n\n"
    md_text += "## 1. 실험 설정\n\n"
    md_text += f"- 모델: YOLO26n\n"
    md_text += f"- 사전학습 가중치: {PRETRAINED_MODEL}\n"
    md_text += f"- 데이터셋: {DATASET_AUG_DIR}\n"
    md_text += f"- data.yaml: {DATA_YAML}\n"
    md_text += f"- 이미지 사이즈 후보: {IMAGE_SIZES}\n"
    md_text += f"- epochs: {EPOCHS}\n"
    md_text += f"- patience: {PATIENCE}\n"
    md_text += f"- batch: {BATCH}\n"
    md_text += f"- device: {DEVICE}\n\n"

    md_text += "## 2. 비교 결과\n\n"
    md_text += summary_df.to_markdown(index=False)
    md_text += "\n\n"

    md_text += "## 3. 해석 기준\n\n"
    md_text += "- mAP50, mAP50-95, Precision, Recall은 높을수록 좋다.\n"
    md_text += "- latency는 낮을수록 좋다.\n"
    md_text += "- FPS는 높을수록 좋다.\n"
    md_text += "- 시뮬레이션 연동에서는 정확도와 FPS를 동시에 고려해야 한다.\n"
    md_text += "- 원거리 Tank 탐지가 중요하면 320 이하 크기에서 성능 저하 여부를 반드시 확인해야 한다.\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"[INFO] summary CSV 저장: {csv_path}")
    print(f"[INFO] summary JSON 저장: {json_path}")
    print(f"[INFO] summary MD 저장: {md_path}")

    return csv_path, json_path, md_path


# ============================================================
# 7. main
# ============================================================

def main():
    print("========== YOLO26n Model1 이미지 사이즈별 비교 시작 ==========")

    if not DATASET_AUG_DIR.exists():
        raise FileNotFoundError(
            f"[ERROR] 증강 데이터셋이 없습니다: {DATASET_AUG_DIR}\n"
            f"먼저 make_model1_aug_dataset_only.py를 실행하세요."
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"[ERROR] data.yaml이 없습니다: {DATA_YAML}")

    compare_save_dir = BASE_DIR / f"{EXPERIMENT_PREFIX}_results"
    compare_save_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    latency_dfs = {}

    for imgsz in IMAGE_SIZES:
        print("\n" + "#" * 80)
        print(f"# IMAGE SIZE EXPERIMENT: imgsz={imgsz}")
        print("#" * 80)

        # ----------------------------------------------------
        # STEP 1. 학습
        # ----------------------------------------------------
        best_path, run_dir, _ = train_model_for_imgsz(imgsz)

        # ----------------------------------------------------
        # STEP 2. validation 검증
        # ----------------------------------------------------
        val_result = validate_model(best_path, imgsz, split="val")

        # ----------------------------------------------------
        # STEP 3. test 검증
        # ----------------------------------------------------
        try:
            test_result = validate_model(best_path, imgsz, split="test")
        except Exception as e:
            print(f"[WARN] test split 검증 실패. val만 사용합니다. 이유: {e}")
            test_result = {
                "imgsz": imgsz,
                "split": "test",
                "precision": None,
                "recall": None,
                "mAP50": None,
                "mAP50_95": None,
                "val_preprocess_ms": None,
                "val_inference_ms": None,
                "val_loss_ms": None,
                "val_postprocess_ms": None
            }

        # ----------------------------------------------------
        # STEP 4. latency / FPS 측정
        # ----------------------------------------------------
        latency_summary, latency_df = measure_latency_and_fps(
            best_path=best_path,
            imgsz=imgsz,
            save_dir=compare_save_dir
        )

        latency_dfs[imgsz] = latency_df

        # ----------------------------------------------------
        # STEP 5. 한 행으로 요약
        # ----------------------------------------------------
        row = {
            "imgsz": imgsz,
            "model_name": make_run_name(imgsz),
            "best_path": str(best_path),
            "run_dir": str(run_dir),

            "val_precision": val_result.get("precision"),
            "val_recall": val_result.get("recall"),
            "val_mAP50": val_result.get("mAP50"),
            "val_mAP50_95": val_result.get("mAP50_95"),
            "val_preprocess_ms": val_result.get("val_preprocess_ms"),
            "val_inference_ms": val_result.get("val_inference_ms"),
            "val_postprocess_ms": val_result.get("val_postprocess_ms"),

            "test_precision": test_result.get("precision"),
            "test_recall": test_result.get("recall"),
            "test_mAP50": test_result.get("mAP50"),
            "test_mAP50_95": test_result.get("mAP50_95"),
            "test_preprocess_ms": test_result.get("val_preprocess_ms"),
            "test_inference_ms": test_result.get("val_inference_ms"),
            "test_postprocess_ms": test_result.get("val_postprocess_ms"),

            "avg_latency_ms": latency_summary.get("avg_latency_ms"),
            "median_latency_ms": latency_summary.get("median_latency_ms"),
            "p95_latency_ms": latency_summary.get("p95_latency_ms"),
            "avg_fps": latency_summary.get("avg_fps"),
            "latency_image_count": latency_summary.get("latency_image_count"),
            "per_image_latency_csv": latency_summary.get("per_image_latency_csv")
        }

        all_rows.append(row)

    # --------------------------------------------------------
    # STEP 6. 전체 결과 저장
    # --------------------------------------------------------
    summary_df = pd.DataFrame(all_rows)
    summary_df = summary_df.sort_values("imgsz", ascending=False)

    csv_path, json_path, md_path = save_experiment_summary(summary_df, compare_save_dir)

    # --------------------------------------------------------
    # STEP 7. 그래프 저장
    # --------------------------------------------------------
    subplot_path = compare_save_dir / f"{EXPERIMENT_PREFIX}_subplots.png"
    latency_boxplot_path = compare_save_dir / f"{EXPERIMENT_PREFIX}_latency_boxplot.png"
    metric_line_path = compare_save_dir / f"{EXPERIMENT_PREFIX}_metric_lines.png"
    speed_line_path = compare_save_dir / f"{EXPERIMENT_PREFIX}_speed_lines.png"

    plot_comparison_subplots(summary_df, subplot_path)
    plot_latency_boxplot(latency_dfs, latency_boxplot_path)
    plot_metric_lines(summary_df, metric_line_path)
    plot_speed_lines(summary_df, speed_line_path)

    print("\n========== YOLO26n Model1 이미지 사이즈별 비교 완료 ==========")
    print(f"[SUMMARY CSV] {csv_path}")
    print(f"[SUMMARY JSON] {json_path}")
    print(f"[SUMMARY MD] {md_path}")
    print(f"[SUBPLOTS] {subplot_path}")
    print(f"[LATENCY BOXPLOT] {latency_boxplot_path}")
    print(f"[METRIC LINES] {metric_line_path}")
    print(f"[SPEED LINES] {speed_line_path}")

    print("\n결과 확인 파일:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    print(f"  - {subplot_path}")
    print(f"  - {latency_boxplot_path}")
    print(f"  - {metric_line_path}")
    print(f"  - {speed_line_path}")

    print("\n각 모델 best.pt 경로:")
    for imgsz in IMAGE_SIZES:
        print(f"  - imgsz={imgsz}: {make_best_path(imgsz)}")


if __name__ == "__main__":
    main()
'''
    model1_yolo26n_imgsz_compare_summary.csv
    model1_yolo26n_imgsz_compare_summary.md
    model1_yolo26n_imgsz_compare_subplots.png
    model1_yolo26n_imgsz_compare_latency_boxplot.png
    model1_yolo26n_imgsz_compare_metric_lines.png
    model1_yolo26n_imgsz_compare_speed_lines.png
    latency_per_image_imgsz640.csv
    latency_per_image_imgsz480.csv
    latency_per_image_imgsz320.csv
    latency_per_image_imgsz288.csv
'''