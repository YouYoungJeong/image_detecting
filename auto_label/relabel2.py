# ============================================================
# rename_4class_split_dataset_to_single_folder.py
# 이미 4클래스로 변환된 YOLO 데이터셋의 train/val/test 구조를 읽어서
# 하나의 images / labels 폴더로 합치고 파일명만 변경
#
# 라벨 내용은 수정하지 않음
#
# 예:
# train/images/abc.jpg  -> images/img_4class_4148.jpg
# train/labels/abc.txt  -> labels/img_4class_4148.txt
#
# 실행:
# python rename_4class_split_dataset_to_single_folder.py
# ============================================================

from pathlib import Path
import shutil
import csv


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 이미 4클래스로 변환된 데이터셋 폴더
SOURCE_DATASET_DIR = BASE_DIR / "dataset_v3"

# 파일명 변경 후 하나로 모아 저장할 폴더
OUTPUT_DATASET_DIR = BASE_DIR / "dataset_4class_single"

OUTPUT_IMAGE_DIR = OUTPUT_DATASET_DIR / "images"
OUTPUT_LABEL_DIR = OUTPUT_DATASET_DIR / "labels"

# 기존 파일에 이어서 저장할 경우 True
# 새로 만들 경우 False
APPEND_MODE = True

# 새 파일명 시작 번호
START_INDEX = 4418

# 새 파일명 접두사
NEW_PREFIX = "img_4class"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# val / valid 둘 다 대응
SPLITS = ["train", "valid", "val", "test"]


# ============================================================
# 2. 유틸 함수
# ============================================================

def reset_dir(dir_path: Path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def make_dir(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path):
    return path.suffix.lower() in IMAGE_EXTENSIONS


def get_image_files(image_dir: Path):
    if not image_dir.exists():
        return []

    return sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and is_image_file(p)
    ])


def copy_data_yaml_if_exists():
    source_yaml = SOURCE_DATASET_DIR / "data.yaml"
    output_yaml = OUTPUT_DATASET_DIR / "data.yaml"

    if source_yaml.exists():
        shutil.copy2(source_yaml, output_yaml)


# ============================================================
# 3. 메인 함수
# ============================================================

def rename_split_dataset_to_single_folder():
    print("========== 4클래스 split 데이터셋 파일명 변경 시작 ==========")

    if not SOURCE_DATASET_DIR.exists():
        raise FileNotFoundError(f"[ERROR] 원본 데이터셋 폴더 없음: {SOURCE_DATASET_DIR}")

    # 출력 폴더 처리
    if APPEND_MODE:
        make_dir(OUTPUT_DATASET_DIR)
        make_dir(OUTPUT_IMAGE_DIR)
        make_dir(OUTPUT_LABEL_DIR)
        print("[INFO] APPEND_MODE=True: 기존 출력 폴더에 이어서 저장합니다.")
    else:
        reset_dir(OUTPUT_DATASET_DIR)
        make_dir(OUTPUT_IMAGE_DIR)
        make_dir(OUTPUT_LABEL_DIR)
        print("[INFO] APPEND_MODE=False: 기존 출력 폴더를 초기화하고 새로 저장합니다.")

    current_index = START_INDEX
    mapping_rows = []
    total_image_count = 0
    missing_label_count = 0

    for split in SPLITS:
        image_dir = SOURCE_DATASET_DIR / split / "images"
        label_dir = SOURCE_DATASET_DIR / split / "labels"

        if not image_dir.exists():
            continue

        image_files = get_image_files(image_dir)

        if len(image_files) == 0:
            continue

        print(f"\n[{split}]")
        print(f"[INFO] 이미지 수: {len(image_files)}")

        for image_path in image_files:
            old_label_path = label_dir / f"{image_path.stem}.txt"

            new_stem = f"{NEW_PREFIX}_{current_index}"
            new_image_name = f"{new_stem}{image_path.suffix.lower()}"
            new_label_name = f"{new_stem}.txt"

            new_image_path = OUTPUT_IMAGE_DIR / new_image_name
            new_label_path = OUTPUT_LABEL_DIR / new_label_name

            if new_image_path.exists() or new_label_path.exists():
                raise FileExistsError(
                    f"[ERROR] 파일명이 이미 존재합니다. START_INDEX를 확인하세요: {new_stem}"
                )

            shutil.copy2(image_path, new_image_path)

            if old_label_path.exists():
                shutil.copy2(old_label_path, new_label_path)
            else:
                with open(new_label_path, "w", encoding="utf-8") as f:
                    pass
                missing_label_count += 1

            mapping_rows.append({
                "original_split": split,
                "old_image_name": image_path.name,
                "new_image_name": new_image_name,
                "old_label_name": old_label_path.name,
                "new_label_name": new_label_name
            })

            current_index += 1
            total_image_count += 1

        print(f"[INFO] {split} 처리 완료")

    copy_data_yaml_if_exists()

    mapping_path = OUTPUT_DATASET_DIR / "filename_mapping.csv"

    # append 모드면 기존 mapping 뒤에 추가
    write_header = True
    if APPEND_MODE and mapping_path.exists():
        write_header = False

    with open(mapping_path, "a" if APPEND_MODE else "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "original_split",
            "old_image_name",
            "new_image_name",
            "old_label_name",
            "new_label_name"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerows(mapping_rows)

    print("\n========== 파일명 변경 완료 ==========")
    print(f"[INFO] 원본 데이터셋: {SOURCE_DATASET_DIR}")
    print(f"[INFO] 저장 폴더: {OUTPUT_DATASET_DIR}")
    print(f"[INFO] 전체 이미지 수: {total_image_count}")
    print(f"[INFO] 라벨 없는 이미지 수: {missing_label_count}")
    print(f"[INFO] 시작 번호: {START_INDEX}")
    print(f"[INFO] 마지막 번호: {current_index - 1}")
    print(f"[INFO] 매핑 파일: {mapping_path}")


# ============================================================
# 4. 실행
# ============================================================

if __name__ == "__main__":
    rename_split_dataset_to_single_folder()