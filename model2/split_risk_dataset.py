import os
import random
import shutil
from pathlib import Path


# ============================================================
# 1. 경로 설정
# ============================================================
# BASE_DIR = Path("") 는 현재 이 파이썬 파일이 실행되는 위치를 기준으로 함.
# 즉, split_risk_dataset.py 파일을 auto_label 폴더 안에서 실행하면 그대로 사용 가능.

BASE_DIR = Path("")

# 원본 risk 이미지 폴더
# 구조:
# risk_images/
# ├── risk_0_safe/
# ├── risk_1_low/
# └── risk_2_high/
RISK_ORIGINAL_DIR = BASE_DIR / "risk_images"

# train / val / test로 나눈 결과 저장 폴더
RISK_DATASET_V1_DIR = BASE_DIR / "risk_dataset_v1"


# ============================================================
# 2. 클래스 이름 설정
# ============================================================
# danger + critical을 합친 상태 기준
CLASS_NAMES = [
    "risk_0_safe",
    "risk_1_low",
    "risk_2_high"
]


# ============================================================
# 3. 분할 비율 설정
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 랜덤 분할 고정값
# 같은 seed를 사용하면 매번 같은 방식으로 train/val/test가 나뉨
RANDOM_SEED = 42


# ============================================================
# 4. 이미지 확장자 설정
# ============================================================
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# ============================================================
# 5. 기존 결과 폴더 삭제 여부
# ============================================================
# True  : 기존 risk_dataset_v1 폴더가 있으면 삭제 후 새로 생성
# False : 기존 폴더가 있으면 에러 방지를 위해 중단
RESET_OUTPUT_DIR = True


# ============================================================
# 6. 유틸 함수
# ============================================================
def is_image_file(file_path: Path) -> bool:
    """
    파일이 이미지 확장자인지 확인하는 함수
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def make_dir(path: Path):
    """
    폴더가 없으면 생성하는 함수
    """
    path.mkdir(parents=True, exist_ok=True)


def copy_images(image_list, target_dir: Path):
    """
    이미지 리스트를 target_dir로 복사하는 함수
    """
    make_dir(target_dir)

    for image_path in image_list:
        target_path = target_dir / image_path.name
        shutil.copy2(image_path, target_path)


def split_image_list(image_list):
    """
    이미지 리스트를 train / val / test로 나누는 함수
    """
    total_count = len(image_list)

    train_count = int(total_count * TRAIN_RATIO)
    val_count = int(total_count * VAL_RATIO)

    train_images = image_list[:train_count]
    val_images = image_list[train_count:train_count + val_count]
    test_images = image_list[train_count + val_count:]

    return train_images, val_images, test_images


# ============================================================
# 7. 메인 실행 함수
# ============================================================
def main():
    print("========== Risk Dataset Split 시작 ==========")

    # ------------------------------------------------------------
    # 1) 원본 폴더 존재 확인
    # ------------------------------------------------------------
    if not RISK_ORIGINAL_DIR.exists():
        raise FileNotFoundError(f"원본 폴더가 존재하지 않습니다: {RISK_ORIGINAL_DIR}")

    # ------------------------------------------------------------
    # 2) 기존 결과 폴더 처리
    # ------------------------------------------------------------
    if RISK_DATASET_V1_DIR.exists():
        if RESET_OUTPUT_DIR:
            print(f"[INFO] 기존 결과 폴더 삭제: {RISK_DATASET_V1_DIR}")
            shutil.rmtree(RISK_DATASET_V1_DIR)
        else:
            raise FileExistsError(
                f"결과 폴더가 이미 존재합니다: {RISK_DATASET_V1_DIR}\n"
                f"삭제 후 다시 실행하거나 RESET_OUTPUT_DIR = True 로 변경하세요."
            )

    # ------------------------------------------------------------
    # 3) train / val / test 기본 폴더 생성
    # ------------------------------------------------------------
    for split_name in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            make_dir(RISK_DATASET_V1_DIR / split_name / class_name)

    # ------------------------------------------------------------
    # 4) 랜덤 seed 고정
    # ------------------------------------------------------------
    random.seed(RANDOM_SEED)

    # ------------------------------------------------------------
    # 5) 클래스별 이미지 분할
    # ------------------------------------------------------------
    split_report = []

    for class_name in CLASS_NAMES:
        class_dir = RISK_ORIGINAL_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"클래스 폴더가 존재하지 않습니다: {class_dir}")

        # 이미지 파일만 가져오기
        image_files = [
            file_path for file_path in class_dir.iterdir()
            if file_path.is_file() and is_image_file(file_path)
        ]

        # 랜덤 셔플
        random.shuffle(image_files)

        # train / val / test 분할
        train_images, val_images, test_images = split_image_list(image_files)

        # 복사
        copy_images(train_images, RISK_DATASET_V1_DIR / "train" / class_name)
        copy_images(val_images, RISK_DATASET_V1_DIR / "val" / class_name)
        copy_images(test_images, RISK_DATASET_V1_DIR / "test" / class_name)

        # 결과 기록
        report_line = {
            "class_name": class_name,
            "total": len(image_files),
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images)
        }
        split_report.append(report_line)

        print(
            f"[{class_name}] "
            f"total={len(image_files)}, "
            f"train={len(train_images)}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}"
        )

    # ------------------------------------------------------------
    # 6) split_report.txt 저장
    # ------------------------------------------------------------
    report_path = RISK_DATASET_V1_DIR / "split_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Risk Dataset Split Report\n")
        f.write("====================================\n")
        f.write(f"Original Dir: {RISK_ORIGINAL_DIR}\n")
        f.write(f"Output Dir  : {RISK_DATASET_V1_DIR}\n")
        f.write(f"Split Ratio : train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}\n")
        f.write(f"Random Seed : {RANDOM_SEED}\n")
        f.write("====================================\n\n")

        for item in split_report:
            f.write(
                f"{item['class_name']}: "
                f"total={item['total']}, "
                f"train={item['train']}, "
                f"val={item['val']}, "
                f"test={item['test']}\n"
            )

    print("====================================")
    print(f"[완료] 데이터셋 분할 완료")
    print(f"[저장 위치] {RISK_DATASET_V1_DIR}")
    print(f"[리포트] {report_path}")
    print("========== Risk Dataset Split 종료 ==========")


# ============================================================
# 8. 실행
# ============================================================
if __name__ == "__main__":
    main()
'''
========== Risk Dataset Split 시작 ==========
[risk_0_safe] total=349, train=244, val=52, test=53
[risk_1_low] total=158, train=110, val=23, test=25
[risk_2_high] total=225, train=157, val=33, test=35
====================================
[완료] 데이터셋 분할 완료
[저장 위치] risk_dataset_v1
[리포트] risk_dataset_v1\split_report.txt
========== Risk Dataset Split 종료 ==========
'''