# ============================================================
# train_model2_risk.py
# 모델2 위험도 분류 YOLO Classification 학습 코드
# VSCode / 로컬 py 실행용
# ============================================================

from pathlib import Path
import torch
from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_AUG_DIR = BASE_DIR / "risk_dataset_aug"

RUNS_DIR = BASE_DIR / "runs"
CLASSIFY_RUNS_DIR = RUNS_DIR / "classify"

MODEL2_RUN_NAME = "model2_risk_v1"
MODEL2_RUN_DIR = CLASSIFY_RUNS_DIR / MODEL2_RUN_NAME
MODEL2_BEST_PATH = MODEL2_RUN_DIR / "weights" / "best.pt"


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
# 4. 모델 학습 함수
# ============================================================

def train_model2():
    print("========== Model2 Risk Classification Train Start ==========")

    if not DATASET_AUG_DIR.exists():
        raise FileNotFoundError(f"증강 데이터셋 폴더가 없습니다: {DATASET_AUG_DIR}")

    check_classification_dataset(DATASET_AUG_DIR)

    device = get_device()

    # ========================================================
    # YOLO Classification 모델 불러오기
    # --------------------------------------------------------
    # 처음 학습할 때는 yolo26m-cls.pt 같은 classification pretrained 모델 사용
    # 만약 YOLOv8을 사용한다면 yolo8m-cls.pt 또는 yolov8m-cls.pt로 변경
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

        # 이미 augment_risk_train.py에서 train 데이터 증강을 했으므로
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
    print("best.pt 위치:", MODEL2_BEST_PATH)

    return results


# ============================================================
# 5. 실행
# ============================================================

if __name__ == "__main__":
    train_model2()