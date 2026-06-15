# ============================================================
# compare_model2_320_480_520.py
# 모델2 위험도 분류 모델 비교 코드
# imgsz=320 / imgsz=480 / imgsz=520 test 결과 비교
#
# 기능
# 1. 모델별 test metric 비교
# 2. 모델별 confusion matrix 저장
# 3. confusion matrix 행/열 합계 확인 CSV 저장
# 4. confusion matrix 숫자 글씨 가독성 개선
# 5. 모델별 오답 CSV 저장
# 6. 모델별 오답 이미지 grid 저장
# 7. true_label -> pred_label 오답 유형별 이미지 grid 저장
# 8. 세 모델 공통 오답 이미지 분석
#
# VSCode / 로컬 py 실행용
# ============================================================

from pathlib import Path
import math
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
from ultralytics import YOLO

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 0. 폴더 초기화 함수
# ============================================================

def reset_dir(target_dir: Path):
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "risk_dataset_aug"
TEST_DIR = DATASET_DIR / "test"

WEIGHTS_DIR = BASE_DIR / "weights"
RUNS_DIR = BASE_DIR / "runs"
CLASSIFY_RUNS_DIR = RUNS_DIR / "classify"

MODEL_CONFIGS = [
    {
        "model_name": "model2_imgsz320",
        "imgsz": 320,
        "weight_path": WEIGHTS_DIR / "model2_risk_from_model1_v2_imgsz320_best.pt",
        "train_result_csv": CLASSIFY_RUNS_DIR / "model2_risk_from_model1_v2_imgsz320" / "results.csv",
    },
    {
        "model_name": "model2_imgsz480",
        "imgsz": 480,
        "weight_path": WEIGHTS_DIR / "model2_risk_from_model1_v2_imgsz480_best.pt",
        "train_result_csv": CLASSIFY_RUNS_DIR / "model2_risk_from_model1_v2_imgsz480" / "results.csv",
    },
    {
        "model_name": "model2_imgsz520",
        "imgsz": 520,
        "weight_path": WEIGHTS_DIR / "model2_risk_from_model1_v2_imgsz520_best.pt",
        "train_result_csv": CLASSIFY_RUNS_DIR / "model2_risk_from_model1_v2_imgsz520" / "results.csv",
    },
]

COMPARE_DIR = RUNS_DIR / "compare_model2_320_480_520"

reset_dir(COMPARE_DIR)

GRAPH_DIR = COMPARE_DIR / "graphs"
CSV_DIR = COMPARE_DIR / "csv"
WRONG_IMAGE_DIR = COMPARE_DIR / "wrong_images"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
WRONG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1-1. 위험도 클래스 순서 고정
# ============================================================
# 혼동 행렬에서 모델별 클래스 순서가 바뀌는 문제를 막기 위해
# 라벨 순서를 고정한다.
#
# 행(row) 합계 = 실제 클래스 개수
# 열(column) 합계 = 모델이 해당 클래스로 예측한 개수
#
# risk_2_high 행 합계는 모델별로 같아야 정상이다.
# risk_2_high 열 합계는 모델별로 달라도 정상이다.
# ============================================================

RISK_LABELS = [
    "risk_0_safe",
    "risk_1_low",
    "risk_2_high",
]


# ============================================================
# 2. GPU / CPU 확인 함수
# ============================================================

def get_device():
    print("========== Device Check ==========")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        return 0
    else:
        print("GPU를 사용할 수 없습니다. CPU로 검정합니다.")
        return "cpu"


# ============================================================
# 3. 테스트 이미지 목록 생성 함수
# ============================================================

def collect_test_images(test_dir: Path):
    if not test_dir.exists():
        raise FileNotFoundError(f"test 폴더가 없습니다: {test_dir}")

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    rows = []

    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    if len(class_dirs) == 0:
        raise FileNotFoundError(f"test 폴더 안에 클래스 폴더가 없습니다: {test_dir}")

    for class_dir in class_dirs:
        true_label = class_dir.name

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in image_exts:
                rows.append({
                    "image_path": img_path,
                    "file_name": img_path.name,
                    "true_label": true_label,
                })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise FileNotFoundError(f"test 이미지가 없습니다: {test_dir}")

    print("========== Test Dataset Check ==========")
    print(df["true_label"].value_counts().sort_index())
    print("total:", len(df))

    # 예상 클래스 외 폴더가 있는지 확인
    unknown_labels = sorted(set(df["true_label"]) - set(RISK_LABELS))
    if unknown_labels:
        print("[주의] RISK_LABELS에 없는 test 폴더명이 있습니다:", unknown_labels)
        print("RISK_LABELS 또는 폴더명을 확인하세요.")

    # 클래스별 이미지 수 CSV 저장
    test_count_df = (
        df["true_label"]
        .value_counts()
        .reindex(RISK_LABELS, fill_value=0)
        .reset_index()
    )
    test_count_df.columns = ["class_name", "test_image_count"]

    test_count_csv_path = CSV_DIR / "test_dataset_class_count.csv"
    test_count_df.to_csv(test_count_csv_path, index=False, encoding="utf-8-sig")

    print("test 클래스별 이미지 개수 저장:", test_count_csv_path)

    return df


# ============================================================
# 4. 단일 모델 예측 및 지표 계산
# ============================================================

def evaluate_one_model(model_config, test_df: pd.DataFrame, device):
    model_name = model_config["model_name"]
    imgsz = model_config["imgsz"]
    weight_path = model_config["weight_path"]

    print("\n" + "=" * 70)
    print(f"Model Test Start: {model_name} | imgsz={imgsz}")
    print("=" * 70)

    if not weight_path.exists():
        raise FileNotFoundError(f"모델 가중치 파일이 없습니다: {weight_path}")

    model = YOLO(str(weight_path))

    print(f"[{model_name}] model.names:", model.names)

    pred_rows = []

    for _, row in test_df.iterrows():
        image_path = row["image_path"]
        true_label = row["true_label"]

        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            device=device,
            verbose=False
        )

        result = results[0]

        pred_idx = int(result.probs.top1)
        pred_conf = float(result.probs.top1conf)
        pred_label = result.names[pred_idx]

        pred_rows.append({
            "model_name": model_name,
            "imgsz": imgsz,
            "file_name": row["file_name"],
            "image_path": str(image_path),
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": pred_conf,
            "is_correct": true_label == pred_label,
        })

    pred_df = pd.DataFrame(pred_rows)

    pred_csv_path = CSV_DIR / f"{model_name}_test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 라벨 순서 고정
    # --------------------------------------------------------
    labels = RISK_LABELS

    # true_label / pred_label에 예상하지 못한 라벨이 있는지 확인
    unknown_true = sorted(set(pred_df["true_label"]) - set(labels))
    unknown_pred = sorted(set(pred_df["pred_label"]) - set(labels))

    if unknown_true:
        print(f"[주의] {model_name} 예상하지 못한 true_label 발견:", unknown_true)

    if unknown_pred:
        print(f"[주의] {model_name} 예상하지 못한 pred_label 발견:", unknown_pred)
        print("모델 학습 시 클래스 폴더명 또는 model.names를 확인해야 합니다.")

    y_true = pred_df["true_label"]
    y_pred = pred_df["pred_label"]

    acc = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true, y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true, y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true, y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_true, y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true, y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true, y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    summary = {
        "model_name": model_name,
        "imgsz": imgsz,
        "test_count": len(pred_df),
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "avg_confidence": pred_df["confidence"].mean(),
        "wrong_count": int((pred_df["is_correct"] == False).sum()),
        "correct_count": int((pred_df["is_correct"] == True).sum()),
    }

    # --------------------------------------------------------
    # Classification Report 저장
    # --------------------------------------------------------
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_csv_path = CSV_DIR / f"{model_name}_classification_report.csv"
    report_df.to_csv(report_csv_path, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Confusion Matrix 저장
    # --------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    cm_csv_path = CSV_DIR / f"{model_name}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Confusion Matrix 행/열 합계 확인 저장
    # --------------------------------------------------------
    cm_check_df = cm_df.copy()

    # 행 합계 = 실제 클래스 개수
    cm_check_df["row_sum_true_count"] = cm_check_df.sum(axis=1)

    # 열 합계 = 해당 클래스로 예측한 개수
    col_sum = cm_df.sum(axis=0).to_frame().T
    col_sum.index = ["col_sum_pred_count"]

    cm_check_df = pd.concat([cm_check_df, col_sum], axis=0)

    cm_check_csv_path = CSV_DIR / f"{model_name}_confusion_matrix_with_sum.csv"
    cm_check_df.to_csv(cm_check_csv_path, encoding="utf-8-sig")

    print(f"[{model_name}] Accuracy: {acc:.4f}")
    print(f"[{model_name}] Macro F1: {f1_macro:.4f}")
    print(f"[{model_name}] Wrong Count: {summary['wrong_count']}")
    print(f"[{model_name}] true risk_2_high count:", int(cm_df.loc["risk_2_high"].sum()))
    print(f"[{model_name}] pred risk_2_high count:", int(cm_df["risk_2_high"].sum()))
    print("예측 결과 저장:", pred_csv_path)
    print("Classification Report 저장:", report_csv_path)
    print("Confusion Matrix 저장:", cm_csv_path)
    print("Confusion Matrix 합계 확인 저장:", cm_check_csv_path)

    return summary, pred_df, report_df, cm_df


# ============================================================
# 5. 전체 지표 비교 그래프
# ============================================================

def plot_metric_bar(summary_df: pd.DataFrame):
    metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]

    plot_df = summary_df.set_index("model_name")[metrics]

    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(14, 6))

    for i, model_name in enumerate(plot_df.index):
        values = plot_df.loc[model_name].values
        offset = (i - 1) * width

        plt.bar(x + offset, values, width, label=model_name)

        for j, v in enumerate(values):
            plt.text(
                x[j] + offset,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.xticks(x, metrics, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model2 Test Metric Comparison: 320 vs 480 vs 520")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = GRAPH_DIR / "model2_metric_comparison_320_480_520.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("지표 비교 그래프 저장:", save_path)


# ============================================================
# 6. 오답 개수 비교 그래프
# ============================================================

def plot_wrong_count(summary_df: pd.DataFrame):
    plt.figure(figsize=(9, 5))

    x = summary_df["model_name"]
    y = summary_df["wrong_count"]

    plt.bar(x, y)

    for i, v in enumerate(y):
        plt.text(
            i,
            v + 0.3,
            str(v),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

    plt.ylabel("Wrong Count")
    plt.title("Wrong Prediction Count Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = GRAPH_DIR / "model2_wrong_count_comparison_320_480_520.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("오답 개수 비교 그래프 저장:", save_path)


# ============================================================
# 7. Confusion Matrix 그래프
# ============================================================

def plot_confusion_matrix(cm_df: pd.DataFrame, model_name: str):
    labels = list(cm_df.index)
    cm = cm_df.values

    plt.figure(figsize=(8, 7))

    im = plt.imshow(cm, cmap="Blues")

    plt.title(f"Confusion Matrix - {model_name}", fontsize=15, pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    plt.xticks(
        np.arange(len(labels)),
        labels,
        rotation=30,
        ha="right",
        fontsize=10
    )
    plt.yticks(
        np.arange(len(labels)),
        labels,
        fontsize=10
    )

    # 셀 배경이 진하면 흰색 글씨, 연하면 검은색 글씨
    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i, j]
            text_color = "white" if value > threshold else "black"

            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=text_color
            )

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    save_path = GRAPH_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Confusion Matrix 그래프 저장:", save_path)


# ============================================================
# 8. 클래스별 F1 비교 그래프
# ============================================================

def plot_class_f1_compare(report_dict_by_model):
    rows = []

    for model_name, report_df in report_dict_by_model.items():
        for idx in report_df.index:
            if str(idx).startswith("risk_"):
                rows.append({
                    "model_name": model_name,
                    "class_name": idx,
                    "precision": report_df.loc[idx, "precision"],
                    "recall": report_df.loc[idx, "recall"],
                    "f1_score": report_df.loc[idx, "f1-score"],
                    "support": report_df.loc[idx, "support"],
                })

    class_df = pd.DataFrame(rows)

    class_csv_path = CSV_DIR / "model2_class_metric_comparison_320_480_520.csv"
    class_df.to_csv(class_csv_path, index=False, encoding="utf-8-sig")

    classes = RISK_LABELS
    models = list(report_dict_by_model.keys())

    x = np.arange(len(classes))
    width = 0.22

    plt.figure(figsize=(11, 6))

    for i, model_name in enumerate(models):
        values = []

        for cls in classes:
            target = class_df[
                (class_df["model_name"] == model_name) &
                (class_df["class_name"] == cls)
            ]

            if len(target) == 0:
                v = 0
            else:
                v = target["f1_score"].values[0]

            values.append(v)

        offset = (i - 1) * width

        plt.bar(x + offset, values, width, label=model_name)

        for j, v in enumerate(values):
            plt.text(
                x[j] + offset,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.xticks(x, classes, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("F1 Score")
    plt.title("Class-wise F1 Score Comparison: 320 vs 480 vs 520")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = GRAPH_DIR / "model2_class_f1_comparison_320_480_520.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("클래스별 F1 비교 그래프 저장:", save_path)
    print("클래스별 지표 CSV 저장:", class_csv_path)


# ============================================================
# 9. 학습 accuracy curve 비교 그래프
# ============================================================

def find_column(df: pd.DataFrame, candidates):
    cols = list(df.columns)

    for candidate in candidates:
        for col in cols:
            if col.strip() == candidate:
                return col

    for candidate in candidates:
        for col in cols:
            if candidate in col:
                return col

    return None


def plot_train_curve_compare():
    candidate_top1_cols = [
        "metrics/accuracy_top1",
        "metrics/accuracy_top1(B)",
        "accuracy_top1",
    ]

    plt.figure(figsize=(10, 6))

    has_any_curve = False

    for config in MODEL_CONFIGS:
        model_name = config["model_name"]
        csv_path = config["train_result_csv"]

        if not csv_path.exists():
            print(f"[주의] 학습 results.csv 없음: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        epoch_col = find_column(df, ["epoch"])
        top1_col = find_column(df, candidate_top1_cols)

        if epoch_col is None:
            df["epoch"] = range(1, len(df) + 1)
            epoch_col = "epoch"

        if top1_col is None:
            print(f"[주의] top1 accuracy 컬럼을 찾지 못했습니다: {csv_path}")
            print("사용 가능한 컬럼:", list(df.columns))
            continue

        plt.plot(
            df[epoch_col],
            df[top1_col],
            marker="o",
            markersize=2,
            label=model_name
        )

        has_any_curve = True

    if has_any_curve:
        plt.xlabel("Epoch")
        plt.ylabel("Top1 Accuracy")
        plt.title("Training Accuracy Curve Comparison")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = GRAPH_DIR / "model2_train_accuracy_curve_comparison_320_480_520.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("학습 accuracy curve 저장:", save_path)

    plt.close()


# ============================================================
# 10. 오답 목록 CSV 저장
# ============================================================

def save_wrong_predictions(pred_df_by_model):
    for model_name, pred_df in pred_df_by_model.items():
        wrong_df = pred_df[pred_df["is_correct"] == False].copy()

        wrong_csv_path = CSV_DIR / f"{model_name}_wrong_predictions.csv"
        wrong_df.to_csv(wrong_csv_path, index=False, encoding="utf-8-sig")

        wrong_group = (
            wrong_df
            .groupby(["true_label", "pred_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        wrong_group_csv_path = CSV_DIR / f"{model_name}_wrong_group_summary.csv"
        wrong_group.to_csv(wrong_group_csv_path, index=False, encoding="utf-8-sig")

        print(f"[{model_name}] 오답 목록 저장:", wrong_csv_path)
        print(f"[{model_name}] 오답 유형 요약 저장:", wrong_group_csv_path)


# ============================================================
# 11. 오답 이미지 Grid 저장
# ============================================================

def make_wrong_image_grid(
    wrong_df: pd.DataFrame,
    save_path: Path,
    title: str,
    max_images: int = 16,
    cols: int = 4,
    thumb_size: int = 180
):
    if len(wrong_df) == 0:
        print(f"[SKIP] 오답 이미지 없음: {title}")
        return

    sample_df = wrong_df.head(max_images).copy()

    rows = math.ceil(len(sample_df) / cols)

    plt.figure(figsize=(cols * 4, rows * 4))

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img_path = Path(row["image_path"])

        plt.subplot(rows, cols, idx + 1)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((thumb_size, thumb_size))
            plt.imshow(img)

            caption = (
                f"T: {row['true_label']}\n"
                f"P: {row['pred_label']}\n"
                f"conf: {row['confidence']:.3f}"
            )

            plt.title(caption, fontsize=9)
            plt.axis("off")

        except Exception as e:
            plt.text(
                0.5,
                0.5,
                f"Image Load Error\n{img_path.name}",
                ha="center",
                va="center"
            )
            plt.axis("off")
            print(f"[이미지 로드 실패] {img_path} | {e}")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("오답 이미지 grid 저장:", save_path)


def save_wrong_image_grids(pred_df_by_model):
    for model_name, pred_df in pred_df_by_model.items():
        model_wrong_dir = WRONG_IMAGE_DIR / model_name
        model_wrong_dir.mkdir(parents=True, exist_ok=True)

        wrong_df = pred_df[pred_df["is_correct"] == False].copy()

        if len(wrong_df) == 0:
            print(f"[{model_name}] 오답 이미지 없음")
            continue

        # 모델별 전체 오답 이미지 grid
        all_wrong_save_path = model_wrong_dir / f"{model_name}_all_wrong_grid.png"

        make_wrong_image_grid(
            wrong_df=wrong_df.sort_values("confidence", ascending=False),
            save_path=all_wrong_save_path,
            title=f"{model_name} - All Wrong Predictions",
            max_images=16,
            cols=4
        )

        # 오답 유형별 이미지 grid
        wrong_types = (
            wrong_df
            .groupby(["true_label", "pred_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        for _, type_row in wrong_types.iterrows():
            true_label = type_row["true_label"]
            pred_label = type_row["pred_label"]

            type_df = wrong_df[
                (wrong_df["true_label"] == true_label) &
                (wrong_df["pred_label"] == pred_label)
            ].copy()

            safe_name = f"{true_label}_to_{pred_label}"
            save_path = model_wrong_dir / f"{model_name}_{safe_name}_grid.png"

            make_wrong_image_grid(
                wrong_df=type_df.sort_values("confidence", ascending=False),
                save_path=save_path,
                title=f"{model_name} | True: {true_label} -> Pred: {pred_label}",
                max_images=16,
                cols=4
            )


# ============================================================
# 12. 모델별 오답 공통/차이 비교 저장
# ============================================================

def save_wrong_intersection_analysis(pred_df_by_model):
    rows = []

    for model_name, pred_df in pred_df_by_model.items():
        temp_df = pred_df.copy()

        for _, row in temp_df.iterrows():
            rows.append({
                "file_name": row["file_name"],
                "image_path": row["image_path"],
                "true_label": row["true_label"],
                "model_name": model_name,
                "pred_label": row["pred_label"],
                "confidence": row["confidence"],
                "is_correct": row["is_correct"],
                "is_wrong": not row["is_correct"],
            })

    long_df = pd.DataFrame(rows)

    long_csv_path = CSV_DIR / "model2_all_model_prediction_long_format.csv"
    long_df.to_csv(long_csv_path, index=False, encoding="utf-8-sig")

    wrong_pivot = long_df.pivot_table(
        index=["file_name", "image_path", "true_label"],
        columns="model_name",
        values="is_wrong",
        aggfunc="first"
    ).reset_index()

    model_cols = [
        c for c in wrong_pivot.columns
        if c not in ["file_name", "image_path", "true_label"]
    ]

    wrong_pivot["wrong_model_count"] = wrong_pivot[model_cols].sum(axis=1)

    pivot_csv_path = CSV_DIR / "model2_wrong_intersection_summary.csv"
    wrong_pivot.to_csv(pivot_csv_path, index=False, encoding="utf-8-sig")

    print("전체 모델 예측 long format 저장:", long_csv_path)
    print("모델별 오답 교집합 요약 저장:", pivot_csv_path)

    # 세 모델 모두 틀린 이미지
    all_wrong_df = wrong_pivot[
        wrong_pivot["wrong_model_count"] == len(model_cols)
    ].copy()

    all_wrong_csv_path = CSV_DIR / "model2_all_models_wrong_images.csv"
    all_wrong_df.to_csv(all_wrong_csv_path, index=False, encoding="utf-8-sig")

    print("세 모델 모두 틀린 이미지 목록 저장:", all_wrong_csv_path)

    if len(all_wrong_df) > 0:
        grid_rows = []

        for _, row in all_wrong_df.iterrows():
            file_name = row["file_name"]

            one = long_df[
                (long_df["file_name"] == file_name) &
                (long_df["is_wrong"] == True)
            ].iloc[0]

            grid_rows.append({
                "file_name": file_name,
                "image_path": row["image_path"],
                "true_label": row["true_label"],
                "pred_label": f"wrong_count={int(row['wrong_model_count'])}",
                "confidence": float(one["confidence"]),
                "is_correct": False,
            })

        grid_df = pd.DataFrame(grid_rows)

        make_wrong_image_grid(
            wrong_df=grid_df,
            save_path=WRONG_IMAGE_DIR / "all_models_wrong_grid.png",
            title="Images Wrongly Predicted by All Models",
            max_images=16,
            cols=4
        )

    # 특정 모델만 틀린 이미지 목록 저장
    for model_name in model_cols:
        only_wrong_df = wrong_pivot[
            (wrong_pivot[model_name] == True) &
            (wrong_pivot["wrong_model_count"] == 1)
        ].copy()

        only_wrong_csv_path = CSV_DIR / f"{model_name}_only_wrong_images.csv"
        only_wrong_df.to_csv(only_wrong_csv_path, index=False, encoding="utf-8-sig")

        print(f"[{model_name}] 해당 모델만 틀린 이미지 목록 저장:", only_wrong_csv_path)


# ============================================================
# 13. Confusion Matrix 합계 전체 비교 저장
# ============================================================

def check_confusion_matrix_sums(cm_df_by_model):
    rows = []

    for model_name, cm_df in cm_df_by_model.items():
        for label in cm_df.index:
            rows.append({
                "model_name": model_name,
                "class_name": label,
                "true_count_row_sum": int(cm_df.loc[label].sum()),
                "pred_count_col_sum": int(cm_df[label].sum()),
            })

    check_df = pd.DataFrame(rows)

    save_path = CSV_DIR / "confusion_matrix_sum_check.csv"
    check_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n========== Confusion Matrix Sum Check ==========")
    print(check_df)
    print("혼동 행렬 합계 확인 CSV 저장:", save_path)

    # 클래스별 실제 개수가 모델별로 같은지 확인
    print("\n========== True Count Row Sum Consistency Check ==========")

    for label in RISK_LABELS:
        temp = check_df[check_df["class_name"] == label]
        unique_true_counts = temp["true_count_row_sum"].unique()

        if len(unique_true_counts) == 1:
            print(f"[정상] {label} 실제 개수(row sum)는 모든 모델에서 동일:", unique_true_counts[0])
        else:
            print(f"[주의] {label} 실제 개수(row sum)가 모델별로 다릅니다:", unique_true_counts)

    print("\n[해석]")
    print("- true_count_row_sum: 실제 클래스 개수입니다. 모델별로 같아야 정상입니다.")
    print("- pred_count_col_sum: 모델이 해당 클래스로 예측한 개수입니다. 모델별로 달라도 정상입니다.")


# ============================================================
# 14. 비교 보고서 저장
# ============================================================

def save_compare_report(summary_df: pd.DataFrame):
    sorted_df = summary_df.sort_values("f1_macro", ascending=False).reset_index(drop=True)

    best = sorted_df.iloc[0]

    report_path = COMPARE_DIR / "model2_compare_report_320_480_520.txt"

    lines = []

    lines.append("========== Model2 320 / 480 / 520 Test Comparison Report ==========\n")

    lines.append("[비교 기준]")
    lines.append("- 동일한 risk_dataset_aug/test 데이터셋 기준으로 비교")
    lines.append("- 비교 모델: imgsz=320, imgsz=480, imgsz=520")
    lines.append("- 주요 기준 지표: accuracy, macro precision, macro recall, macro F1, wrong count")
    lines.append("- 위험도 분류는 클래스별 균형 성능이 중요하므로 macro F1-score를 우선 기준으로 판단")
    lines.append("- 오답 이미지를 함께 저장하여 정량 평가와 정성 평가를 함께 진행")
    lines.append("- confusion matrix의 row sum은 실제 클래스 개수이며, 모델별로 같아야 정상")
    lines.append("- confusion matrix의 column sum은 모델이 해당 클래스로 예측한 개수이며, 모델별로 달라도 정상\n")

    lines.append("[전체 결과]")
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['model_name']} | imgsz={int(row['imgsz'])} | "
            f"accuracy={row['accuracy']:.4f}, "
            f"macro_precision={row['precision_macro']:.4f}, "
            f"macro_recall={row['recall_macro']:.4f}, "
            f"macro_f1={row['f1_macro']:.4f}, "
            f"wrong_count={int(row['wrong_count'])}, "
            f"avg_confidence={row['avg_confidence']:.4f}"
        )

    lines.append("\n[우수 모델 판단]")
    lines.append(
        f"- macro F1 기준 우수 모델: {best['model_name']} "
        f"(imgsz={int(best['imgsz'])})"
    )

    lines.append("\n[해석 기준]")
    lines.append("- imgsz=320 모델은 입력 크기가 작아 학습 및 추론 속도 측면에서 유리하다.")
    lines.append("- imgsz=480 모델은 32 배수 크기이며 포신 방향과 차체 방향 정보 보존에 유리하다.")
    lines.append("- imgsz=520 모델은 더 큰 입력 해상도를 사용하므로 세부 형태 보존 가능성은 높지만, 학습 및 추론 비용이 증가한다.")
    lines.append("- 단순 accuracy보다 risk_1_low, risk_2_high 클래스의 recall과 F1-score를 함께 확인해야 한다.")
    lines.append("- 오답 이미지는 confidence가 높은데 틀린 사례를 우선 확인하는 것이 좋다.")
    lines.append("- 세 모델이 모두 틀린 이미지는 데이터 라벨 오류, crop 품질 문제, 클래스 기준 모호성 여부를 우선 점검해야 한다.")
    lines.append("- 특정 모델만 틀린 이미지는 입력 이미지 크기 변화가 분류 결과에 어떤 영향을 주는지 확인하는 데 활용할 수 있다.")
    lines.append("- imgsz가 커졌는데 macro F1 또는 high risk recall 향상이 작다면, 실시간성 측면에서 320 또는 480 모델을 선택하는 것이 적절하다.")
    lines.append("- imgsz=520이 high risk 클래스 recall을 뚜렷하게 개선한다면, 위험도 판단 목적상 520 모델을 고려할 수 있다.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("비교 보고서 저장:", report_path)


# ============================================================
# 15. 실행 함수
# ============================================================

def main():
    print("========== Model2 320 / 480 / 520 Compare Start ==========")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"데이터셋 폴더가 없습니다: {DATASET_DIR}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"test 폴더가 없습니다: {TEST_DIR}")

    device = get_device()

    test_df = collect_test_images(TEST_DIR)

    summary_rows = []
    pred_df_by_model = {}
    report_df_by_model = {}
    cm_df_by_model = {}

    for config in MODEL_CONFIGS:
        summary, pred_df, report_df, cm_df = evaluate_one_model(
            model_config=config,
            test_df=test_df,
            device=device
        )

        summary_rows.append(summary)
        pred_df_by_model[config["model_name"]] = pred_df
        report_df_by_model[config["model_name"]] = report_df
        cm_df_by_model[config["model_name"]] = cm_df

    summary_df = pd.DataFrame(summary_rows)

    summary_csv_path = CSV_DIR / "model2_320_480_520_test_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    print("\n========== Summary ==========")
    print(summary_df)

    # --------------------------------------------------------
    # 그래프 저장
    # --------------------------------------------------------
    plot_metric_bar(summary_df)
    plot_wrong_count(summary_df)
    plot_class_f1_compare(report_df_by_model)
    plot_train_curve_compare()

    for model_name, cm_df in cm_df_by_model.items():
        plot_confusion_matrix(cm_df, model_name)

    # --------------------------------------------------------
    # Confusion Matrix 행/열 합계 확인
    # --------------------------------------------------------
    check_confusion_matrix_sums(cm_df_by_model)

    # --------------------------------------------------------
    # 오답 CSV 저장
    # --------------------------------------------------------
    save_wrong_predictions(pred_df_by_model)

    # --------------------------------------------------------
    # 오답 이미지 grid 저장
    # --------------------------------------------------------
    save_wrong_image_grids(pred_df_by_model)

    # --------------------------------------------------------
    # 세 모델 오답 공통/차이 분석 저장
    # --------------------------------------------------------
    save_wrong_intersection_analysis(pred_df_by_model)

    # --------------------------------------------------------
    # 비교 보고서 저장
    # --------------------------------------------------------
    save_compare_report(summary_df)

    print("\n========== Compare Done ==========")
    print("결과 저장 폴더:", COMPARE_DIR)
    print("요약 CSV:", summary_csv_path)


# ============================================================
# 16. 실행
# ============================================================

if __name__ == "__main__":
    main()