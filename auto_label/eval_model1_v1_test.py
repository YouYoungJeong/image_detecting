"""
모델1 v1 best.pt 성능 확인 코드

목적:
1. dataset_v1/test 데이터셋으로 best.pt 성능 검증
2. 전체 성능 지표 저장
3. 클래스별 성능 지표 저장
4. 클래스별 객체 수 그래프 저장
5. 클래스별 mAP 그래프 저장
6. Ultralytics YOLO 기본 검증 그래프 저장
   - confusion matrix
   - precision-recall curve
   - F1 curve
   - P curve
   - R curve

실행 전 설치:
pip install ultralytics pandas matplotlib pyyaml
"""

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import koreanize_matplotlib


# =========================================================
# 1. 경로 설정
# =========================================================

# 현재 코드가 실행되는 폴더
BASE_DIR = os.getcwd()

# 데이터셋 폴더
DATASET_DIR = os.path.join(BASE_DIR, "dataset_v1")

# data.yaml 경로
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

# split_report.txt 경로
SPLIT_REPORT_PATH = os.path.join(DATASET_DIR, "split_report.txt")

# 학습된 모델 best.pt 경로
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")

# 결과 저장 폴더
RESULTS_DIR = os.path.join(BASE_DIR, "model_eval_results")

# 이번 검증 결과 이름
RUN_NAME = "model1_v1_test_eval"

# 결과 폴더 생성
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# 2. 파일 존재 확인
# =========================================================

if not os.path.exists(DATA_YAML_PATH):
    raise FileNotFoundError(f"data.yaml 파일을 찾을 수 없습니다: {DATA_YAML_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"best.pt 파일을 찾을 수 없습니다: {MODEL_PATH}")

if not os.path.exists(SPLIT_REPORT_PATH):
    print(f"주의: split_report.txt 파일을 찾을 수 없습니다: {SPLIT_REPORT_PATH}")
    print("객체 수 그래프는 사용자가 코드 안에 입력한 test_counts 기준으로 생성됩니다.")


# =========================================================
# 3. data.yaml 클래스 이름 불러오기
# =========================================================

with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml["names"]

print("클래스 목록:")
for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")


# =========================================================
# 4. test 데이터 객체 수 입력
# =========================================================
# split_report.txt에 있는 test object count 기준
# 나중에 새 test 데이터셋을 만들면 이 값만 바꾸거나 split_report를 다시 생성하면 됩니다.

test_counts = {
    "House1": 39,
    "House2": 45,
    "Human1": 85,
    "Human2": 571,
    "Human3": 907,
    "Rock": 678,
    "Tank": 729,
    "Tent1": 42,
    "Wall": 42,
    "car": 856,
}


# =========================================================
# 5. YOLO 모델 로드
# =========================================================

model = YOLO(MODEL_PATH)


# =========================================================
# 6. test 데이터셋 기준 모델 검증
# =========================================================
# split="test"를 사용하면 data.yaml 내부 test 경로를 기준으로 평가합니다.
# plots=True를 주면 confusion matrix, PR curve 등 기본 그래프가 자동 저장됩니다.

metrics = model.val(
    data=DATA_YAML_PATH,
    split="test",
    imgsz=640,
    batch=16,
    conf=0.001,
    iou=0.6,
    plots=True,
    save_json=False,
    project=RESULTS_DIR,
    name=RUN_NAME,
    exist_ok=True,
)

SAVE_DIR = os.path.join(RESULTS_DIR, RUN_NAME)

print("\n검증 결과 저장 폴더:")
print(SAVE_DIR)


# =========================================================
# 7. 전체 성능 지표 저장
# =========================================================

summary_data = {
    "model": ["model1_v1_best.pt"],
    "dataset": ["dataset_v1_test"],
    "precision_mean": [float(metrics.box.mp)],
    "recall_mean": [float(metrics.box.mr)],
    "mAP50": [float(metrics.box.map50)],
    "mAP50_95": [float(metrics.box.map)],
}

summary_df = pd.DataFrame(summary_data)

summary_csv_path = os.path.join(SAVE_DIR, "model1_v1_test_summary.csv")
summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

print("\n전체 성능 요약:")
print(summary_df)


# =========================================================
# 8. 클래스별 성능 지표 저장
# =========================================================
# Ultralytics 버전에 따라 클래스별 precision/recall 속성이 다를 수 있어서
# 안전하게 가져오도록 처리합니다.

class_result_rows = []

# 클래스별 mAP50-95
class_maps = metrics.box.maps

# 클래스별 AP50
# ap50 배열이 있는 경우 사용
class_ap50 = getattr(metrics.box, "ap50", None)

# 클래스별 Precision, Recall
class_precision = getattr(metrics.box, "p", None)
class_recall = getattr(metrics.box, "r", None)

for class_id, class_name in enumerate(class_names):
    row = {
        "class_id": class_id,
        "class_name": class_name,
        "test_object_count": test_counts.get(class_name, 0),
        "mAP50_95": float(class_maps[class_id]) if class_id < len(class_maps) else None,
    }

    if class_ap50 is not None and class_id < len(class_ap50):
        row["mAP50"] = float(class_ap50[class_id])
    else:
        row["mAP50"] = None

    if class_precision is not None and class_id < len(class_precision):
        row["precision"] = float(class_precision[class_id])
    else:
        row["precision"] = None

    if class_recall is not None and class_id < len(class_recall):
        row["recall"] = float(class_recall[class_id])
    else:
        row["recall"] = None

    class_result_rows.append(row)

class_df = pd.DataFrame(class_result_rows)

class_csv_path = os.path.join(SAVE_DIR, "model1_v1_test_class_metrics.csv")
class_df.to_csv(class_csv_path, index=False, encoding="utf-8-sig")

print("\n클래스별 성능:")
print(class_df)


# =========================================================
# 9. 전체 성능 그래프 저장
# =========================================================

overall_metrics = {
    "Precision": float(metrics.box.mp),
    "Recall": float(metrics.box.mr),
    "mAP50": float(metrics.box.map50),
    "mAP50-95": float(metrics.box.map),
}

plt.figure(figsize=(8, 5))
plt.bar(overall_metrics.keys(), overall_metrics.values())
plt.ylim(0, 1)
plt.title("Model1 v1 - Overall Test Performance")
plt.ylabel("Score")

for idx, value in enumerate(overall_metrics.values()):
    plt.text(idx, value + 0.02, f"{value:.3f}", ha="center")

overall_graph_path = os.path.join(SAVE_DIR, "overall_test_performance.png")
plt.savefig(overall_graph_path, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 10. 클래스별 test 객체 수 그래프 저장
# =========================================================

plt.figure(figsize=(12, 6))
plt.bar(test_counts.keys(), test_counts.values())
plt.title("Test Dataset Object Count by Class")
plt.xlabel("Class")
plt.ylabel("Object Count")
plt.xticks(rotation=45, ha="right")

for idx, value in enumerate(test_counts.values()):
    plt.text(idx, value + 10, str(value), ha="center", fontsize=8)

object_count_graph_path = os.path.join(SAVE_DIR, "test_object_count_by_class.png")
plt.savefig(object_count_graph_path, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 11. 클래스별 mAP50-95 그래프 저장
# =========================================================

plt.figure(figsize=(12, 6))
plt.bar(class_df["class_name"], class_df["mAP50_95"])
plt.ylim(0, 1)
plt.title("Model1 v1 - Class-wise mAP50-95 on Test Dataset")
plt.xlabel("Class")
plt.ylabel("mAP50-95")
plt.xticks(rotation=45, ha="right")

for idx, value in enumerate(class_df["mAP50_95"]):
    if value is not None:
        plt.text(idx, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)

class_map_graph_path = os.path.join(SAVE_DIR, "class_wise_map50_95.png")
plt.savefig(class_map_graph_path, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 12. 클래스별 객체 수와 mAP 비교 그래프 저장
# =========================================================
# 객체 수가 적은 클래스가 성능이 낮은지 확인하기 위한 그래프입니다.
# 예: Wall, Tent1, House1, House2처럼 객체 수가 적은 클래스 확인 가능

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(class_df["class_name"], class_df["test_object_count"], alpha=0.6)
ax1.set_xlabel("Class")
ax1.set_ylabel("Test Object Count")
ax1.tick_params(axis="x", rotation=45)

ax2 = ax1.twinx()
ax2.plot(class_df["class_name"], class_df["mAP50_95"], marker="o")
ax2.set_ylabel("mAP50-95")
ax2.set_ylim(0, 1)

plt.title("Test Object Count vs Class-wise mAP50-95")

count_vs_map_graph_path = os.path.join(SAVE_DIR, "object_count_vs_map50_95.png")
plt.savefig(count_vs_map_graph_path, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 12-1. 1순위 핵심 그래프 subplots 대시보드 저장
# =========================================================
# 목적:
# 여러 결과 이미지를 따로 보지 않고,
# 모델1 v1의 핵심 성능을 한 장의 이미지로 확인하기 위한 그래프입니다.
#
# 포함 내용:
# 1. 전체 성능 지표
# 2. 클래스별 mAP50-95
# 3. 클래스별 test 객체 수와 mAP50-95 비교
# 4. Normalized Confusion Matrix

import matplotlib.image as mpimg

dashboard_path = os.path.join(SAVE_DIR, "model1_v1_priority_dashboard.png")

# Ultralytics가 자동 저장한 normalized confusion matrix 경로
confusion_matrix_path = os.path.join(SAVE_DIR, "confusion_matrix_normalized.png")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

fig.suptitle(
    "모델1 v1 best.pt 테스트 성능 핵심 대시보드",
    fontsize=18,
    fontweight="bold"
)


# ---------------------------------------------------------
# subplot 1. 전체 성능 지표
# ---------------------------------------------------------

metric_names = ["Precision", "Recall", "mAP50", "mAP50-95"]
metric_values = [
    float(metrics.box.mp),
    float(metrics.box.mr),
    float(metrics.box.map50),
    float(metrics.box.map),
]

axes[0, 0].bar(metric_names, metric_values)
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_title("전체 테스트 성능 지표")
axes[0, 0].set_ylabel("점수")

for idx, value in enumerate(metric_values):
    axes[0, 0].text(
        idx,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=10
    )


# ---------------------------------------------------------
# subplot 2. 클래스별 mAP50-95
# ---------------------------------------------------------

axes[0, 1].bar(class_df["class_name"], class_df["mAP50_95"])
axes[0, 1].set_ylim(0, 1)
axes[0, 1].set_title("클래스별 mAP50-95 성능")
axes[0, 1].set_xlabel("클래스")
axes[0, 1].set_ylabel("mAP50-95")
axes[0, 1].tick_params(axis="x", rotation=45)

for idx, value in enumerate(class_df["mAP50_95"]):
    if value is not None:
        axes[0, 1].text(
            idx,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            fontsize=8
        )


# ---------------------------------------------------------
# subplot 3. test 객체 수 vs mAP50-95
# ---------------------------------------------------------

ax_count = axes[1, 0]
ax_map = ax_count.twinx()

ax_count.bar(
    class_df["class_name"],
    class_df["test_object_count"],
    alpha=0.6,
    label="객체 수"
)

ax_map.plot(
    class_df["class_name"],
    class_df["mAP50_95"],
    marker="o",
    label="mAP50-95"
)

ax_count.set_title("클래스별 테스트 객체 수와 mAP50-95 비교")
ax_count.set_xlabel("클래스")
ax_count.set_ylabel("테스트 객체 수")
ax_map.set_ylabel("mAP50-95")
ax_map.set_ylim(0, 1)

ax_count.tick_params(axis="x", rotation=45)


# ---------------------------------------------------------
# subplot 4. normalized confusion matrix
# ---------------------------------------------------------

if os.path.exists(confusion_matrix_path):
    confusion_img = mpimg.imread(confusion_matrix_path)
    axes[1, 1].imshow(confusion_img)
    axes[1, 1].set_title("정규화 혼동 행렬")
    axes[1, 1].axis("off")
else:
    axes[1, 1].text(
        0.5,
        0.5,
        "confusion_matrix_normalized.png 파일 없음",
        ha="center",
        va="center",
        fontsize=12
    )
    axes[1, 1].set_title("정규화 혼동 행렬")
    axes[1, 1].axis("off")


plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"1순위 핵심 그래프 대시보드: {dashboard_path}")

# =========================================================
# 13. 결과 txt 요약 저장
# =========================================================

summary_txt_path = os.path.join(SAVE_DIR, "model1_v1_test_eval_summary.txt")

with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("[Model1 v1 best.pt Test Evaluation Summary]\n\n")
    f.write(f"Model Path: {MODEL_PATH}\n")
    f.write(f"Data YAML: {DATA_YAML_PATH}\n")
    f.write(f"Evaluation Split: test\n\n")

    f.write("[Overall Metrics]\n")
    f.write(f"Precision Mean: {metrics.box.mp:.4f}\n")
    f.write(f"Recall Mean: {metrics.box.mr:.4f}\n")
    f.write(f"mAP50: {metrics.box.map50:.4f}\n")
    f.write(f"mAP50-95: {metrics.box.map:.4f}\n\n")

    f.write("[Class-wise Metrics]\n")
    f.write(class_df.to_string(index=False))


# =========================================================
# 14. 완료 출력
# =========================================================

print("\n성능 평가 완료")
print(f"전체 성능 CSV: {summary_csv_path}")
print(f"클래스별 성능 CSV: {class_csv_path}")
print(f"전체 성능 그래프: {overall_graph_path}")
print(f"클래스별 객체 수 그래프: {object_count_graph_path}")
print(f"클래스별 mAP 그래프: {class_map_graph_path}")
print(f"객체 수 vs mAP 그래프: {count_vs_map_graph_path}")
print(f"요약 TXT: {summary_txt_path}")

print("\nUltralytics 기본 그래프도 같은 폴더에 저장됩니다.")
print("예상 저장 파일:")
print("- confusion_matrix.png")
print("- confusion_matrix_normalized.png")
print("- F1_curve.png")
print("- P_curve.png")
print("- R_curve.png")
print("- PR_curve.png")