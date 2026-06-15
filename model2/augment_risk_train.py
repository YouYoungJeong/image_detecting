import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps


# ============================================================
# 1. 경로 설정
# ============================================================
# 현재 파일이 model2 폴더 안에서 실행된다고 가정
# 예:
# model2/
# ├── risk_dataset_v1/
# ├── risk_dataset_aug/
# └── augment_risk_train.py

BASE_DIR = Path("")

# 분할이 완료된 원본 데이터셋
RISK_DATASET_V1_DIR = BASE_DIR / "risk_dataset_v1"

# 증강 결과를 저장할 최종 학습용 데이터셋
RISK_DATASET_AUG_DIR = BASE_DIR / "risk_dataset_aug"


# ============================================================
# 2. 클래스 설정
# ============================================================
CLASS_NAMES = [
    "risk_0_safe",
    "risk_1_low",
    "risk_2_high"
]


# ============================================================
# 3. 증강 목표 개수 설정
# ============================================================
# 각 클래스별 train 이미지 수를 어느 정도까지 맞출지 설정
# 현재 train 수:
# risk_0_safe  = 244
# risk_1_low   = 110
# risk_2_high  = 157
#
# safe가 이미 244장이므로 전체 train 클래스를 300장 정도로 맞추는 것을 추천
TARGET_TRAIN_COUNT_PER_CLASS = 300


# ============================================================
# 4. 기본 설정
# ============================================================
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

RANDOM_SEED = 42

# 기존 risk_dataset_aug 폴더가 있으면 삭제 후 새로 생성
RESET_OUTPUT_DIR = True


# ============================================================
# 5. 유틸 함수
# ============================================================
def is_image_file(file_path: Path) -> bool:
    """
    이미지 파일 확장자인지 확인하는 함수
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def make_dir(path: Path):
    """
    폴더가 없으면 생성하는 함수
    """
    path.mkdir(parents=True, exist_ok=True)


def copy_split_folder(split_name: str):
    """
    risk_dataset_v1의 특정 split 폴더를 risk_dataset_aug로 복사하는 함수

    train은 복사 후 추가 증강을 적용하고,
    val/test는 원본 그대로 복사한다.
    """
    src_split_dir = RISK_DATASET_V1_DIR / split_name
    dst_split_dir = RISK_DATASET_AUG_DIR / split_name

    if not src_split_dir.exists():
        raise FileNotFoundError(f"{split_name} 폴더가 없습니다: {src_split_dir}")

    for class_name in CLASS_NAMES:
        src_class_dir = src_split_dir / class_name
        dst_class_dir = dst_split_dir / class_name

        if not src_class_dir.exists():
            raise FileNotFoundError(f"클래스 폴더가 없습니다: {src_class_dir}")

        make_dir(dst_class_dir)

        image_files = [
            file_path for file_path in src_class_dir.iterdir()
            if file_path.is_file() and is_image_file(file_path)
        ]

        for image_path in image_files:
            shutil.copy2(image_path, dst_class_dir / image_path.name)


def get_image_files(folder_path: Path):
    """
    폴더 안의 이미지 파일 목록을 가져오는 함수
    """
    return [
        file_path for file_path in folder_path.iterdir()
        if file_path.is_file() and is_image_file(file_path)
    ]


def random_augment_image(image: Image.Image) -> Image.Image:
    """
    모델2 train 데이터용 이미지 증강 함수

    적용 증강:
    1. 좌우 반전
    2. 밝기 조절
    3. 대비 조절
    4. 색감 조절
    5. 약한 회전

    주의:
    - val/test에는 적용하지 않음
    - 포신 각도 위험도 분류이므로 너무 강한 회전은 피함
    """

    # RGB 변환
    image = image.convert("RGB")

    # 1) 좌우 반전
    # 시뮬레이션 상황에서 좌우 방향이 위험도 의미를 바꾸지 않는 경우에만 사용
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    # 2) 밝기 조절
    # 야간/조명 차이를 어느 정도 반영
    brightness_factor = random.uniform(0.65, 1.35)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)

    # 3) 대비 조절
    contrast_factor = random.uniform(0.75, 1.25)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    # 4) 색감 조절
    color_factor = random.uniform(0.80, 1.20)
    image = ImageEnhance.Color(image).enhance(color_factor)

    # 5) 약한 회전
    # 포신 각도 분류이므로 회전은 너무 크게 주지 않음
    rotate_angle = random.uniform(-5, 5)
    image = image.rotate(rotate_angle, resample=Image.BICUBIC, expand=False)

    return image


def save_augmented_image(src_image_path: Path, dst_class_dir: Path, aug_index: int):
    """
    원본 이미지 1장을 증강해서 저장하는 함수
    """
    try:
        with Image.open(src_image_path) as img:
            aug_img = random_augment_image(img)

            save_name = f"aug_{aug_index:05d}_{src_image_path.stem}.jpg"
            save_path = dst_class_dir / save_name

            aug_img.save(save_path, quality=95)

    except Exception as e:
        print(f"[WARNING] 증강 실패: {src_image_path} / 이유: {e}")


def augment_train_class(class_name: str):
    """
    특정 클래스의 train 이미지 개수를 TARGET_TRAIN_COUNT_PER_CLASS까지 늘리는 함수
    """
    train_class_dir = RISK_DATASET_AUG_DIR / "train" / class_name

    image_files = get_image_files(train_class_dir)
    original_count = len(image_files)

    if original_count == 0:
        print(f"[WARNING] train 이미지가 없습니다: {class_name}")
        return

    if original_count >= TARGET_TRAIN_COUNT_PER_CLASS:
        print(
            f"[{class_name}] 현재 {original_count}장으로 목표 {TARGET_TRAIN_COUNT_PER_CLASS}장 이상입니다. 추가 증강하지 않습니다."
        )
        return

    need_count = TARGET_TRAIN_COUNT_PER_CLASS - original_count

    print(
        f"[{class_name}] 원본 train={original_count}, "
        f"추가 증강={need_count}, "
        f"최종 목표={TARGET_TRAIN_COUNT_PER_CLASS}"
    )

    for aug_index in range(1, need_count + 1):
        src_image_path = random.choice(image_files)
        save_augmented_image(src_image_path, train_class_dir, aug_index)


def count_images_by_split():
    """
    최종 risk_dataset_aug 이미지 개수를 출력하는 함수
    """
    print("\n========== 최종 risk_dataset_aug 이미지 개수 ==========")

    for split_name in ["train", "val", "test"]:
        print(f"\n[{split_name}]")

        split_total = 0

        for class_name in CLASS_NAMES:
            class_dir = RISK_DATASET_AUG_DIR / split_name / class_name
            image_count = len(get_image_files(class_dir))
            split_total += image_count

            print(f"  {class_name}: {image_count}")

        print(f"  total: {split_total}")


def save_augment_report():
    """
    증강 결과 리포트를 txt 파일로 저장하는 함수
    """
    report_path = RISK_DATASET_AUG_DIR / "augment_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Risk Dataset Augmentation Report\n")
        f.write("====================================\n")
        f.write(f"Source Dataset : {RISK_DATASET_V1_DIR}\n")
        f.write(f"Output Dataset : {RISK_DATASET_AUG_DIR}\n")
        f.write(f"Target Train Count Per Class : {TARGET_TRAIN_COUNT_PER_CLASS}\n")
        f.write(f"Random Seed : {RANDOM_SEED}\n")
        f.write("====================================\n\n")

        for split_name in ["train", "val", "test"]:
            f.write(f"[{split_name}]\n")

            split_total = 0

            for class_name in CLASS_NAMES:
                class_dir = RISK_DATASET_AUG_DIR / split_name / class_name
                image_count = len(get_image_files(class_dir))
                split_total += image_count

                f.write(f"{class_name}: {image_count}\n")

            f.write(f"total: {split_total}\n\n")

    print(f"\n[리포트 저장 완료] {report_path}")


# ============================================================
# 6. 메인 실행 함수
# ============================================================
def main():
    print("========== Risk Train Augmentation 시작 ==========")

    random.seed(RANDOM_SEED)

    # ------------------------------------------------------------
    # 1) risk_dataset_v1 존재 확인
    # ------------------------------------------------------------
    if not RISK_DATASET_V1_DIR.exists():
        raise FileNotFoundError(f"분할 데이터셋 폴더가 없습니다: {RISK_DATASET_V1_DIR}")

    # ------------------------------------------------------------
    # 2) 기존 risk_dataset_aug 처리
    # ------------------------------------------------------------
    if RISK_DATASET_AUG_DIR.exists():
        if RESET_OUTPUT_DIR:
            print(f"[INFO] 기존 증강 폴더 삭제: {RISK_DATASET_AUG_DIR}")
            shutil.rmtree(RISK_DATASET_AUG_DIR)
        else:
            raise FileExistsError(
                f"증강 결과 폴더가 이미 존재합니다: {RISK_DATASET_AUG_DIR}\n"
                f"삭제 후 다시 실행하거나 RESET_OUTPUT_DIR = True 로 변경하세요."
            )

    # ------------------------------------------------------------
    # 3) train / val / test 전체 복사
    # ------------------------------------------------------------
    # train은 복사 후 추가 증강
    # val/test는 복사만 하고 증강하지 않음
    print("[INFO] risk_dataset_v1 → risk_dataset_aug 복사 중")

    for split_name in ["train", "val", "test"]:
        copy_split_folder(split_name)

    # ------------------------------------------------------------
    # 4) train 폴더만 증강
    # ------------------------------------------------------------
    print("\n[INFO] train 데이터 증강 시작")

    for class_name in CLASS_NAMES:
        augment_train_class(class_name)

    # ------------------------------------------------------------
    # 5) 최종 개수 출력 및 리포트 저장
    # ------------------------------------------------------------
    count_images_by_split()
    save_augment_report()

    print("\n========== Risk Train Augmentation 완료 ==========")


# ============================================================
# 7. 실행
# ============================================================
if __name__ == "__main__":
    main()