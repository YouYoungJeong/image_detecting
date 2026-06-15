

import os

# ============================================================
# OpenMP 중복 로딩 오류 임시 해결
# ============================================================
# Windows 환경에서 torch, numpy, matplotlib, opencv 등이
# libiomp5md.dll을 중복으로 불러오면 아래 오류가 발생할 수 있다.
#
# OMP: Error #15: Initializing libiomp5md.dll,
# but found libiomp5md.dll already initialized.
#
# 이 설정은 중복 로딩을 허용해서 학습이 중단되지 않도록 하는 임시 해결 방법이다.
# 단, 공식 메시지에서도 unsafe workaround라고 안내하므로
# 최종 실험 환경에서는 패키지 환경을 정리하는 것이 더 좋다.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

import json
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# 1. 기본 경로 설정
# ============================================================

# 현재 코드가 실행되는 프로젝트 폴더
BASE_DIR = os.getcwd()

# 기존 데이터셋 v1
DATASET_V1_DIR = os.path.join(BASE_DIR, "dataset_v1")
DATASET_V1_YAML = os.path.join(DATASET_V1_DIR, "data.yaml")

# 개선 데이터셋 v2
DATASET_V2_DIR = os.path.join(BASE_DIR, "dataset_v2")
DATASET_V2_YAML = os.path.join(DATASET_V2_DIR, "data.yaml")

# 기존 v1 모델 best.pt 경로
# 본인 runs 폴더에 맞게 수정 가능
V1_BEST_PT = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "model_v1_train",
    "weights",
    "best.pt"
)

# 결과 저장 폴더
OUTPUT_DIR = os.path.join(BASE_DIR, "model_compare_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# v2 학습 결과 저장 이름
V2_TRAIN_PROJECT = os.path.join(BASE_DIR, "runs", "detect")
V2_TRAIN_NAME = "model_v2_train"


# ============================================================
# 2. 경로 확인
# ============================================================

def check_path(path, description):
    """
    필요한 파일이나 폴더가 실제로 존재하는지 확인하는 함수.
    경로가 잘못되면 학습이나 평가가 중간에 실패하기 때문에
    시작 전에 미리 확인한다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} 경로를 찾을 수 없습니다: {path}")


check_path(DATASET_V1_YAML, "dataset_v1 data.yaml")
check_path(DATASET_V2_YAML, "dataset_v2 data.yaml")
check_path(V1_BEST_PT, "기존 v1 best.pt")


# ============================================================
# 3. dataset_v2로 성능 개선 학습
# ============================================================

def train_v2_model():
    """
    기존 v1 모델의 best.pt를 불러와서 dataset_v2로 추가 학습한다.
    
    이유:
    - v1 모델은 기존 dataset_v1 기준으로 학습된 모델
    - dataset_v2에는 검수된 오토라벨링 이미지가 추가되었거나 라벨 품질이 개선됨
    - 따라서 v1 best.pt를 시작점으로 사용하면 처음부터 학습하는 것보다 효율적
    """

    model = YOLO(V1_BEST_PT)

    results = model.train(
        data=DATASET_V2_YAML,

        # YOLO26m 모델을 사용 중이라면 기존 best.pt 기반으로 이어서 학습
        # 너무 크게 잡으면 오래 걸리므로 처음에는 50 정도 추천
        # 최대 학습 epoch를 120으로 설정
        # 단, 아래 patience 조건에 따라 성능 개선이 없으면 120까지 가지 않고 자동 종료됨
        epochs=120,

        # GPU 메모리에 따라 조절
        # CUDA out of memory가 발생하면 8 또는 4로 줄이기
        batch=16,

        # 이미지 크기
        # 기존 학습과 동일하게 맞추는 것이 비교에 유리
        imgsz=640,

        # 과적합 방지를 위한 patience
        # 성능 개선이 일정 epoch 동안 없으면 조기 종료
        # 예: patience=15이면 15 epoch 동안 성능 개선이 없을 때 학습 중단
        patience=15,

        # 학습 결과 저장 위치
        project=V2_TRAIN_PROJECT,
        name=V2_TRAIN_NAME,

        # 기존 폴더가 있으면 덮어쓰기 허용
        exist_ok=True,

        # 학습 로그 출력
        verbose=True,
        device=0
    )

    return results


# ============================================================
# 4. 모델 평가 함수
# ============================================================

def evaluate_model(model_path, data_yaml, model_name):
    """
    특정 모델을 test 데이터 기준으로 평가한다.
    
    반환값:
    - 전체 성능 summary
    - 클래스별 성능 DataFrame
    """

    model = YOLO(model_path)

    # split="test"를 사용하면 data.yaml 안의 test 경로 기준으로 평가한다.
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        batch=16,
        project=OUTPUT_DIR,
        name=f"{model_name}_test_eval",
        exist_ok=True,
        verbose=True,
        device=0
    )

    # 전체 성능 요약
    summary = {
        "model": model_name,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map)
    }

    # 클래스 이름
    names = metrics.names

    # 클래스별 precision, recall, mAP50, mAP50-95
    class_rows = []

    for class_idx, class_name in names.items():
        row = {
            "model": model_name,
            "class_id": class_idx,
            "class_name": class_name,
            "precision": float(metrics.box.p[class_idx]) if len(metrics.box.p) > class_idx else None,
            "recall": float(metrics.box.r[class_idx]) if len(metrics.box.r) > class_idx else None,
            "mAP50": float(metrics.box.ap50[class_idx]) if len(metrics.box.ap50) > class_idx else None,
            "mAP50-95": float(metrics.box.ap[class_idx]) if len(metrics.box.ap) > class_idx else None
        }
        class_rows.append(row)

    class_df = pd.DataFrame(class_rows)

    return summary, class_df


# ============================================================
# 5. 그래프 저장 함수
# ============================================================

def save_overall_metric_chart(summary_df):
    """
    v1 모델과 v2 모델의 전체 성능을 막대그래프로 비교한다.
    보고서에서 가장 먼저 보여주기 좋은 그래프이다.
    """

    metrics = ["precision", "recall", "mAP50", "mAP50-95"]

    plot_df = summary_df.set_index("model")[metrics].T

    plt.figure(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=plt.gca())

    plt.title("Model v1 vs Model v2 Overall Performance")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Model")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "overall_metric_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[저장 완료] 전체 성능 비교 그래프: {save_path}")


def save_class_map50_chart(class_df):
    """
    클래스별 mAP50을 v1과 v2로 비교한다.
    
    목적:
    - 어떤 라벨에서 성능이 좋아졌는지 확인
    - 어떤 라벨에서 성능이 떨어졌는지 확인
    - 데이터 추가 후 개선 효과를 클래스 단위로 분석
    """

    pivot_df = class_df.pivot(
        index="class_name",
        columns="model",
        values="mAP50"
    )

    plt.figure(figsize=(12, 7))
    pivot_df.plot(kind="bar", ax=plt.gca())

    plt.title("Class-wise mAP50 Comparison")
    plt.xlabel("Class")
    plt.ylabel("mAP50")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Model")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "class_map50_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[저장 완료] 클래스별 mAP50 비교 그래프: {save_path}")


def save_class_recall_chart(class_df):
    """
    클래스별 recall을 비교한다.
    
    recall은 실제 객체를 얼마나 놓치지 않고 잡았는지 보는 지표이다.
    오토라벨링 모델에서는 recall이 낮으면 실제 객체를 많이 놓칠 수 있으므로 중요하다.
    """

    pivot_df = class_df.pivot(
        index="class_name",
        columns="model",
        values="recall"
    )

    plt.figure(figsize=(12, 7))
    pivot_df.plot(kind="bar", ax=plt.gca())

    plt.title("Class-wise Recall Comparison")
    plt.xlabel("Class")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Model")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "class_recall_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[저장 완료] 클래스별 Recall 비교 그래프: {save_path}")


def save_improvement_chart(class_df):
    """
    v2가 v1 대비 얼마나 개선되었는지 차이값을 계산해서 저장한다.
    
    계산 방식:
    improvement = v2 mAP50 - v1 mAP50
    
    양수면 개선, 음수면 성능 하락이다.
    """

    pivot_df = class_df.pivot(
        index="class_name",
        columns="model",
        values="mAP50"
    )

    if "model_v1" not in pivot_df.columns or "model_v2" not in pivot_df.columns:
        print("[경고] model_v1 또는 model_v2 컬럼이 없어 개선 그래프를 만들 수 없습니다.")
        return

    pivot_df["mAP50_improvement"] = pivot_df["model_v2"] - pivot_df["model_v1"]
    pivot_df = pivot_df.sort_values("mAP50_improvement", ascending=False)

    plt.figure(figsize=(12, 7))
    pivot_df["mAP50_improvement"].plot(kind="bar", ax=plt.gca())

    plt.title("mAP50 Improvement by Class")
    plt.xlabel("Class")
    plt.ylabel("mAP50 Difference: v2 - v1")
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "class_map50_improvement.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[저장 완료] 클래스별 mAP50 개선 그래프: {save_path}")


# ============================================================
# 6. 보고서용 텍스트 자동 생성
# ============================================================

def make_report(summary_df, class_df):
    """
    성능 비교 결과를 markdown 보고서 형태로 저장한다.
    GitHub README나 회의록에 붙여넣기 좋게 만든다.
    """

    v1 = summary_df[summary_df["model"] == "model_v1"].iloc[0]
    v2 = summary_df[summary_df["model"] == "model_v2"].iloc[0]

    map50_diff = v2["mAP50"] - v1["mAP50"]
    map_diff = v2["mAP50-95"] - v1["mAP50-95"]
    precision_diff = v2["precision"] - v1["precision"]
    recall_diff = v2["recall"] - v1["recall"]

    pivot_df = class_df.pivot(
        index="class_name",
        columns="model",
        values="mAP50"
    )

    if "model_v1" in pivot_df.columns and "model_v2" in pivot_df.columns:
        pivot_df["mAP50_improvement"] = pivot_df["model_v2"] - pivot_df["model_v1"]
        best_classes = pivot_df.sort_values("mAP50_improvement", ascending=False).head(3)
        weak_classes = pivot_df.sort_values("mAP50_improvement", ascending=True).head(3)
    else:
        best_classes = pd.DataFrame()
        weak_classes = pd.DataFrame()

    report_path = os.path.join(OUTPUT_DIR, "model_v1_v2_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model v1 vs Model v2 성능 비교 보고서\n\n")

        f.write("## 1. 비교 목적\n\n")
        f.write("- dataset_v1 기반으로 학습한 기존 모델 v1과 dataset_v2 기반으로 개선 학습한 모델 v2의 성능을 비교한다.\n")
        f.write("- 동일한 test 데이터셋을 기준으로 평가하여 모델 개선 여부를 확인한다.\n")
        f.write("- 전체 성능과 클래스별 성능을 함께 확인하여 어떤 객체에서 개선 또는 하락이 발생했는지 분석한다.\n\n")

        f.write("## 2. 전체 성능 비교\n\n")
        f.write("| Model | Precision | Recall | mAP50 | mAP50-95 |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")

        for _, row in summary_df.iterrows():
            f.write(
                f"| {row['model']} | "
                f"{row['precision']:.4f} | "
                f"{row['recall']:.4f} | "
                f"{row['mAP50']:.4f} | "
                f"{row['mAP50-95']:.4f} |\n"
            )

        f.write("\n## 3. v2 개선 결과\n\n")
        f.write(f"- Precision 변화: {precision_diff:+.4f}\n")
        f.write(f"- Recall 변화: {recall_diff:+.4f}\n")
        f.write(f"- mAP50 변화: {map50_diff:+.4f}\n")
        f.write(f"- mAP50-95 변화: {map_diff:+.4f}\n\n")

        f.write("## 4. 해석 기준\n\n")
        f.write("- Precision이 상승하면 잘못 탐지하는 비율이 줄어든 것으로 볼 수 있다.\n")
        f.write("- Recall이 상승하면 실제 객체를 놓치는 비율이 줄어든 것으로 볼 수 있다.\n")
        f.write("- mAP50은 객체 탐지 성능을 전체적으로 판단할 때 가장 직관적으로 확인하기 좋은 지표이다.\n")
        f.write("- mAP50-95는 더 엄격한 IoU 기준까지 포함하므로 박스 위치 정확도까지 함께 반영한다.\n\n")

        f.write("## 5. 클래스별 개선 상위 항목\n\n")

        if not best_classes.empty:
            f.write("| Class | v1 mAP50 | v2 mAP50 | Difference |\n")
            f.write("| --- | ---: | ---: | ---: |\n")

            for class_name, row in best_classes.iterrows():
                f.write(
                    f"| {class_name} | "
                    f"{row['model_v1']:.4f} | "
                    f"{row['model_v2']:.4f} | "
                    f"{row['mAP50_improvement']:+.4f} |\n"
                )
        else:
            f.write("- 클래스별 개선 항목을 계산할 수 없음\n")

        f.write("\n## 6. 클래스별 확인 필요 항목\n\n")

        if not weak_classes.empty:
            f.write("| Class | v1 mAP50 | v2 mAP50 | Difference |\n")
            f.write("| --- | ---: | ---: | ---: |\n")

            for class_name, row in weak_classes.iterrows():
                f.write(
                    f"| {class_name} | "
                    f"{row['model_v1']:.4f} | "
                    f"{row['model_v2']:.4f} | "
                    f"{row['mAP50_improvement']:+.4f} |\n"
                )
        else:
            f.write("- 클래스별 하락 항목을 계산할 수 없음\n")

        f.write("\n## 7. 저장된 결과 파일\n\n")
        f.write("- overall_metric_comparison.png\n")
        f.write("- class_map50_comparison.png\n")
        f.write("- class_recall_comparison.png\n")
        f.write("- class_map50_improvement.png\n")
        f.write("- model_compare_summary.csv\n")
        f.write("- model_compare_class_metrics.csv\n")

    print(f"[저장 완료] 보고서 파일: {report_path}")


# ============================================================
# 7. 전체 실행
# ============================================================

def main():
    print("========== 1. dataset_v2 기반 개선 학습 시작 ==========")

    train_v2_model()

    # v2 학습 완료 후 생성되는 best.pt 경로
    V2_BEST_PT = os.path.join(
        V2_TRAIN_PROJECT,
        V2_TRAIN_NAME,
        "weights",
        "best.pt"
    )

    check_path(V2_BEST_PT, "개선된 v2 best.pt")

    print("========== 2. model_v1 test 평가 시작 ==========")

    # 비교 기준은 dataset_v2의 test로 통일
    # dataset_v2의 test가 dataset_v1의 test와 동일하게 유지되어 있어야 공정한 비교 가능
    v1_summary, v1_class_df = evaluate_model(
        model_path=V1_BEST_PT,
        data_yaml=DATASET_V2_YAML,
        model_name="model_v1"
    )

    print("========== 3. model_v2 test 평가 시작 ==========")

    v2_summary, v2_class_df = evaluate_model(
        model_path=V2_BEST_PT,
        data_yaml=DATASET_V2_YAML,
        model_name="model_v2"
    )

    print("========== 4. 결과 CSV 저장 ==========")

    summary_df = pd.DataFrame([v1_summary, v2_summary])
    class_df = pd.concat([v1_class_df, v2_class_df], ignore_index=True)

    summary_csv_path = os.path.join(OUTPUT_DIR, "model_compare_summary.csv")
    class_csv_path = os.path.join(OUTPUT_DIR, "model_compare_class_metrics.csv")

    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    class_df.to_csv(class_csv_path, index=False, encoding="utf-8-sig")

    print(f"[저장 완료] 전체 성능 CSV: {summary_csv_path}")
    print(f"[저장 완료] 클래스별 성능 CSV: {class_csv_path}")

    print("========== 5. 결과 시각화 저장 ==========")

    save_overall_metric_chart(summary_df)
    save_class_map50_chart(class_df)
    save_class_recall_chart(class_df)
    save_improvement_chart(class_df)

    print("========== 6. 보고서 markdown 생성 ==========")

    make_report(summary_df, class_df)

    print("========== 완료 ==========")
    print(f"결과 저장 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()