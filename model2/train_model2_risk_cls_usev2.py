# ============================================================
# train_model2_risk_from_model1_v2.py
# 모델2 위험도 분류 YOLO Classification 학습 코드
# model1_v2_best.pt로 생성한 Tank crop 데이터셋 기반 학습
# VSCode / 로컬 py 실행용
# ============================================================

from pathlib import Path
import shutil
import torch
from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 모델1 v2 best.pt 경로
# model1_v2_best.pt는 Tank detection/crop에 사용된 모델
MODEL1_V2_BEST_PATH = BASE_DIR / "weights" / "model1_v2_best.pt"

# 모델2 위험도 분류용 증강 데이터셋
DATASET_AUG_DIR = BASE_DIR / "risk_dataset_aug"

# runs 저장 경로
RUNS_DIR = BASE_DIR / "runs"
CLASSIFY_RUNS_DIR = RUNS_DIR / "classify"

# 모델2 저장 이름
MODEL2_RUN_NAME = "model2_risk_from_model1_v2"

# Ultralytics 학습 결과 폴더
MODEL2_RUN_DIR = CLASSIFY_RUNS_DIR / MODEL2_RUN_NAME
MODEL2_BEST_PATH = MODEL2_RUN_DIR / "weights" / "best.pt"

# 최종 best.pt를 따로 저장할 경로
FINAL_WEIGHTS_DIR = BASE_DIR / "weights"
FINAL_MODEL2_BEST_PATH = FINAL_WEIGHTS_DIR / "model2_risk_from_model1_v2_best.pt"


# ============================================================
# 2. 데이터셋 구조 확인 함수
# ============================================================

def check_classification_dataset(dataset_dir: Path):
    print("========== Dataset Check ==========")

    required_splits = ["train", "val", "test"]

    for split in required_splits:
        split_dir = dataset_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(f"{split} 폴더가 없습니다: {split_dir}")

        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        if len(class_dirs) == 0:
            raise FileNotFoundError(f"{split} 폴더 안에 클래스 폴더가 없습니다: {split_dir}")

        print(f"\n[{split}]")

        total_count = 0

        for class_dir in class_dirs:
            image_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            ]

            count = len(image_files)
            total_count += count

            print(f"  {class_dir.name}: {count}")

        print(f"  total: {total_count}")

    print("\n데이터셋 구조 확인 완료")


# ============================================================
# 3. GPU / CPU 확인 함수
# ============================================================

def get_device():
    print("========== Device Check ==========")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        return 0
    else:
        print("GPU를 사용할 수 없습니다. CPU로 학습합니다.")
        return "cpu"


# ============================================================
# 4. 모델1 v2 가중치 확인 함수
# ============================================================

def check_model1_v2_weight():
    print("========== Model1 v2 Weight Check ==========")

    if not MODEL1_V2_BEST_PATH.exists():
        raise FileNotFoundError(
            f"model1_v2_best.pt 파일이 없습니다: {MODEL1_V2_BEST_PATH}"
        )

    print("model1_v2_best.pt 확인 완료:", MODEL1_V2_BEST_PATH)

    # 주의:
    # model1_v2_best.pt는 Tank detection 모델이므로
    # YOLO Classification 학습의 시작 가중치로 직접 사용하지 않음.
    # 모델2는 yolo26m-cls.pt를 기반으로 위험도 분류 학습을 진행함.


# ============================================================
# 5. 최종 best.pt 복사 함수
# ============================================================

def save_final_model2_best():
    print("========== Save Final Model2 Best ==========")

    if not MODEL2_BEST_PATH.exists():
        raise FileNotFoundError(f"학습된 best.pt가 없습니다: {MODEL2_BEST_PATH}")

    FINAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(MODEL2_BEST_PATH, FINAL_MODEL2_BEST_PATH)

    print("최종 모델2 best.pt 저장 완료")
    print("저장 위치:", FINAL_MODEL2_BEST_PATH)


# ============================================================
# 6. 모델 학습 함수
# ============================================================

def train_model2():
    print("========== Model2 Risk Classification Train Start ==========")

    # model1_v2_best.pt 존재 여부 확인
    check_model1_v2_weight()

    # 데이터셋 확인
    if not DATASET_AUG_DIR.exists():
        raise FileNotFoundError(f"증강 데이터셋 폴더가 없습니다: {DATASET_AUG_DIR}")

    check_classification_dataset(DATASET_AUG_DIR)

    # GPU / CPU 확인
    device = get_device()

    # ========================================================
    # YOLO Classification 모델 불러오기
    # --------------------------------------------------------
    # 중요:
    # model1_v2_best.pt는 Tank Detection 모델이므로
    # 위험도 Classification 모델2의 시작 가중치로 직접 사용하지 않음.
    #
    # 모델2는 Classification 구조인 yolo26m-cls.pt로 시작해야 함.
    # model1_v2_best.pt는 Tank crop 생성에 사용된 모델로 기록함.
    # ========================================================
    model = YOLO("yolo26m-cls.pt")

    results = model.train(
        data=str(DATASET_AUG_DIR),
        task="classify",

        epochs=100,
        imgsz=224,
        batch=16,
        patience=20,

        device=device,
        workers=2,

        project=str(CLASSIFY_RUNS_DIR),
        name=MODEL2_RUN_NAME,
        exist_ok=True,

        optimizer="auto",
        lr0=0.001,
        cos_lr=True,

        pretrained=True,
        seed=42,
        verbose=True,

        # 이미 augment_risk_train.py에서 train 데이터 증강을 진행했으므로
        # 학습 중 증강은 과하지 않게 설정
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=5,
        translate=0.05,
        scale=0.1,
        fliplr=0.5,
        flipud=0.0,
    )

    print("\n========== Model2 Risk Classification Train Done ==========")
    print("학습 결과 폴더:", MODEL2_RUN_DIR)
    print("Ultralytics best.pt 위치:", MODEL2_BEST_PATH)

    # 최종 best.pt를 weights 폴더에 별도 저장
    save_final_model2_best()

    print("\n========== Final Output ==========")
    print("모델1 v2 가중치:", MODEL1_V2_BEST_PATH)
    print("모델2 최종 가중치:", FINAL_MODEL2_BEST_PATH)

    return results


# ============================================================
# 7. 실행
# ============================================================

if __name__ == "__main__":
    train_model2()