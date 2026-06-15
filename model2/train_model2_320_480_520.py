# ============================================================
# train_model2_320_480_520.py
# 모델2 위험도 분류 YOLO Classification 학습 코드
# imgsz=320 / 480 / 520 세 버전 비교 학습
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

MODEL1_V2_BEST_PATH = BASE_DIR / "weights" / "model1_v2_best.pt"

DATASET_AUG_DIR = BASE_DIR / "risk_dataset_aug"

RUNS_DIR = BASE_DIR / "runs"
CLASSIFY_RUNS_DIR = RUNS_DIR / "classify"

FINAL_WEIGHTS_DIR = BASE_DIR / "weights"

# 비교할 이미지 사이즈 목록
IMG_SIZE_LIST = [320, 480, 520]

BASE_MODEL2_RUN_NAME = "model2_risk_from_model1_v2"


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


# ============================================================
# 5. 최종 best.pt 복사 함수
# ============================================================

def save_final_model2_best(model2_best_path: Path, final_model2_best_path: Path):
    print("========== Save Final Model2 Best ==========")

    if not model2_best_path.exists():
        raise FileNotFoundError(f"학습된 best.pt가 없습니다: {model2_best_path}")

    FINAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model2_best_path, final_model2_best_path)

    print("최종 모델2 best.pt 저장 완료")
    print("저장 위치:", final_model2_best_path)


# ============================================================
# 6. 이미지 크기별 batch 설정 함수
# ============================================================

def get_batch_size(imgsz: int):
    if imgsz >= 512:
        return 4
    elif imgsz >= 448:
        return 8
    else:
        return 16


# ============================================================
# 7. 단일 imgsz 학습 함수
# ============================================================

def train_model2_one_size(imgsz: int, device):
    print("\n" + "=" * 70)
    print(f"Model2 Risk Classification Train Start | imgsz={imgsz}")
    print("=" * 70)

    model2_run_name = f"{BASE_MODEL2_RUN_NAME}_imgsz{imgsz}"

    model2_run_dir = CLASSIFY_RUNS_DIR / model2_run_name
    model2_best_path = model2_run_dir / "weights" / "best.pt"

    final_model2_best_path = FINAL_WEIGHTS_DIR / f"{model2_run_name}_best.pt"

    model = YOLO("yolo26m-cls.pt")

    batch_size = get_batch_size(imgsz)

    print(f"imgsz={imgsz}, batch={batch_size}")

    results = model.train(
        data=str(DATASET_AUG_DIR),
        task="classify",

        epochs=100,
        imgsz=imgsz,
        batch=batch_size,
        patience=20,

        device=device,
        workers=2,

        project=str(CLASSIFY_RUNS_DIR),
        name=model2_run_name,
        exist_ok=True,

        optimizer="auto",
        lr0=0.001,
        cos_lr=True,

        pretrained=True,
        seed=42,
        verbose=True,

        # 이미 train 데이터 증강을 별도로 진행했으므로
        # 학습 중 증강은 약하게 설정
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=3,
        translate=0.02,
        scale=0.05,
        fliplr=0.5,
        flipud=0.0,
    )

    print("\n========== Model2 Risk Classification Train Done ==========")
    print("imgsz:", imgsz)
    print("batch:", batch_size)
    print("학습 결과 폴더:", model2_run_dir)
    print("Ultralytics best.pt 위치:", model2_best_path)

    save_final_model2_best(model2_best_path, final_model2_best_path)

    print("\n========== Final Output ==========")
    print("모델1 v2 가중치:", MODEL1_V2_BEST_PATH)
    print("모델2 최종 가중치:", final_model2_best_path)

    return results


# ============================================================
# 8. 전체 학습 함수
# ============================================================

def train_model2_multi_imgsz():
    print("========== Model2 Multi Image Size Train Start ==========")

    check_model1_v2_weight()

    if not DATASET_AUG_DIR.exists():
        raise FileNotFoundError(f"증강 데이터셋 폴더가 없습니다: {DATASET_AUG_DIR}")

    check_classification_dataset(DATASET_AUG_DIR)

    device = get_device()

    all_results = {}

    for imgsz in IMG_SIZE_LIST:
        results = train_model2_one_size(imgsz=imgsz, device=device)
        all_results[imgsz] = results

    print("\n" + "=" * 70)
    print("모든 이미지 크기 학습 완료")
    print("=" * 70)

    for imgsz in IMG_SIZE_LIST:
        model2_run_name = f"{BASE_MODEL2_RUN_NAME}_imgsz{imgsz}"
        final_model2_best_path = FINAL_WEIGHTS_DIR / f"{model2_run_name}_best.pt"

        print(f"imgsz={imgsz} best.pt:")
        print(final_model2_best_path)

    return all_results


# ============================================================
# 9. 실행
# ============================================================

if __name__ == "__main__":
    train_model2_multi_imgsz()