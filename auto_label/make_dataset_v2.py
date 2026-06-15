import os
import shutil
from pathlib import Path
import yaml


# ============================================================
# 1. 경로 설정
# ============================================================
# BASE_DIR = Path("") 는 현재 이 파이썬 파일이 실행되는 위치를 기준으로 함.
# 즉, make_dataset_v2.py 파일이 project 폴더 안에 있으면 그대로 사용하면 됨.

BASE_DIR = Path("")

DATASET_V1_DIR = BASE_DIR / "dataset_v1"
AUTO_CHECK_DIR = BASE_DIR / "auto_check"
DATASET_V2_DIR = BASE_DIR / "dataset_v2"

AUTO_IMAGES_DIR = AUTO_CHECK_DIR / "images"
AUTO_LABELS_DIR = AUTO_CHECK_DIR / "labels"


# ============================================================
# 2. 클래스 이름 설정
# ============================================================
# 중요:
# dataset_v1에서 사용하던 class 순서와 반드시 같아야 함.
# YOLO 라벨 txt 파일의 첫 번째 숫자가 이 순서와 연결됨.

CLASS_NAMES = [
    "House1",
    "House2",
    "Human1",
    "Human2",
    "Human3",
    "Rock",
    "Tank",
    "Tent1",
    "Wall",
    "car"
]


# ============================================================
# 3. 기존 dataset_v2 삭제 여부 설정
# ============================================================
# True:
#   기존 dataset_v2가 있으면 삭제하고 새로 생성
#
# False:
#   기존 dataset_v2를 삭제하지 않음
#
# 보통 v2를 새로 만들 때는 True 추천.

RESET_DATASET_V2 = True


# ============================================================
# 4. 기본 함수
# ============================================================

def reset_dataset_v2():
    """
    기존 dataset_v2 폴더가 있으면 삭제하는 함수.
    이전 실행 결과와 파일이 섞이지 않게 하기 위해 사용함.
    """
    if RESET_DATASET_V2 and DATASET_V2_DIR.exists():
        shutil.rmtree(DATASET_V2_DIR)
        print(f"[초기화] 기존 dataset_v2 삭제 완료: {DATASET_V2_DIR}")


def make_v2_dirs():
    """
    dataset_v2에 필요한 폴더를 생성하는 함수.
    YOLO 학습 구조에 맞게 train, valid, test 각각 images/labels 생성.
    """
    dir_list = [
        DATASET_V2_DIR / "train" / "images",
        DATASET_V2_DIR / "train" / "labels",
        DATASET_V2_DIR / "valid" / "images",
        DATASET_V2_DIR / "valid" / "labels",
        DATASET_V2_DIR / "test" / "images",
        DATASET_V2_DIR / "test" / "labels",
    ]

    for dir_path in dir_list:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("[생성] dataset_v2 폴더 구조 생성 완료")


def copy_all_files(src_dir, dst_dir):
    """
    src_dir 안에 있는 모든 파일을 dst_dir로 복사하는 함수.

    src_dir: 원본 폴더
    dst_dir: 복사할 대상 폴더
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"원본 폴더가 존재하지 않습니다: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0

    for file_path in src_dir.iterdir():
        if file_path.is_file():
            shutil.copy2(file_path, dst_dir / file_path.name)
            copied_count += 1

    return copied_count


def copy_dataset_v1_to_v2():
    """
    dataset_v1의 train, valid, test를 dataset_v2로 복사하는 함수.

    이후 auto_check 데이터는 dataset_v2/train에만 추가됨.
    valid/test는 dataset_v1과 동일하게 유지되어야
    model1과 model2 성능 비교가 공정해짐.
    """
    copy_info = [
        ("train/images", DATASET_V1_DIR / "train" / "images", DATASET_V2_DIR / "train" / "images"),
        ("train/labels", DATASET_V1_DIR / "train" / "labels", DATASET_V2_DIR / "train" / "labels"),
        ("valid/images", DATASET_V1_DIR / "valid" / "images", DATASET_V2_DIR / "valid" / "images"),
        ("valid/labels", DATASET_V1_DIR / "valid" / "labels", DATASET_V2_DIR / "valid" / "labels"),
        ("test/images", DATASET_V1_DIR / "test" / "images", DATASET_V2_DIR / "test" / "images"),
        ("test/labels", DATASET_V1_DIR / "test" / "labels", DATASET_V2_DIR / "test" / "labels"),
    ]

    print("\n[복사] dataset_v1 → dataset_v2")

    for name, src, dst in copy_info:
        count = copy_all_files(src, dst)
        print(f"- {name}: {count}개 복사 완료")


def add_auto_check_to_v2_train():
    """
    검수 완료된 오토라벨링 데이터를 dataset_v2/train에 추가하는 함수.

    auto_check/images 안의 이미지와
    auto_check/labels 안의 txt 라벨 파일을 dataset_v2/train에 복사함.

    파일명 충돌을 방지하기 위해 auto_ 접두사를 붙여 저장함.
    예:
    image_001.jpg → auto_image_001.jpg
    image_001.txt → auto_image_001.txt
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    if not AUTO_IMAGES_DIR.exists():
        raise FileNotFoundError(f"auto_check 이미지 폴더가 없습니다: {AUTO_IMAGES_DIR}")

    if not AUTO_LABELS_DIR.exists():
        raise FileNotFoundError(f"auto_check 라벨 폴더가 없습니다: {AUTO_LABELS_DIR}")

    v2_train_images_dir = DATASET_V2_DIR / "train" / "images"
    v2_train_labels_dir = DATASET_V2_DIR / "train" / "labels"

    added_count = 0
    missing_label_count = 0

    print("\n[추가] auto_check → dataset_v2/train")

    for image_path in AUTO_IMAGES_DIR.iterdir():
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in image_extensions:
            continue

        label_path = AUTO_LABELS_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"[경고] 라벨 파일 없음, 제외됨: {image_path.name}")
            missing_label_count += 1
            continue

        new_image_name = f"auto_{image_path.name}"
        new_label_name = f"auto_{image_path.stem}.txt"

        shutil.copy2(image_path, v2_train_images_dir / new_image_name)
        shutil.copy2(label_path, v2_train_labels_dir / new_label_name)

        added_count += 1

    print(f"- 추가된 검수 이미지 수: {added_count}")
    print(f"- 라벨 누락으로 제외된 이미지 수: {missing_label_count}")


def create_data_yaml():
    """
    dataset_v2용 data.yaml을 자동 생성하는 함수.

    YOLO 학습 시 아래 경로를 사용하면 됨.
    data='dataset_v2/data.yaml'
    """
    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }

    yaml_path = DATASET_V2_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n[생성] data.yaml 생성 완료: {yaml_path}")


def count_files():
    """
    dataset_v2 생성 후 이미지 수와 라벨 수를 확인하는 함수.
    images 수와 labels 수가 다르면 학습 전에 확인 필요.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    print("\n[확인] dataset_v2 이미지/라벨 개수")

    for split in ["train", "valid", "test"]:
        images_dir = DATASET_V2_DIR / split / "images"
        labels_dir = DATASET_V2_DIR / split / "labels"

        image_count = len([
            file for file in images_dir.iterdir()
            if file.is_file() and file.suffix.lower() in image_extensions
        ])

        label_count = len([
            file for file in labels_dir.iterdir()
            if file.is_file() and file.suffix.lower() == ".txt"
        ])

        print(f"- {split}: images={image_count}, labels={label_count}")


def count_train_class_objects():
    """
    dataset_v2/train/labels 안의 YOLO 라벨 txt를 읽어서
    클래스별 객체 수를 출력하는 함수.

    v2 생성 후 클래스 불균형 상태를 확인할 수 있음.
    """
    labels_dir = DATASET_V2_DIR / "train" / "labels"

    class_counts = {class_name: 0 for class_name in CLASS_NAMES}

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line == "":
                continue

            parts = line.split()

            if len(parts) < 5:
                print(f"[경고] 라벨 형식 이상: {label_file.name} / {line}")
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                print(f"[경고] class_id 변환 실패: {label_file.name} / {line}")
                continue

            if 0 <= class_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[class_id]
                class_counts[class_name] += 1
            else:
                print(f"[경고] class_id 범위 초과: {label_file.name} / class_id={class_id}")

    total_objects = sum(class_counts.values())

    print("\n[확인] dataset_v2/train 클래스별 객체 수")

    for class_name, count in class_counts.items():
        ratio = count / total_objects * 100 if total_objects > 0 else 0
        print(f"- {class_name}: {count}개 ({ratio:.2f}%)")

    print(f"\n- 전체 객체 수: {total_objects}개")


# ============================================================
# 5. 실행
# ============================================================

if __name__ == "__main__":
    print("dataset_v2 생성 시작")

    # 1. 기존 dataset_v2 삭제
    reset_dataset_v2()

    # 2. dataset_v2 폴더 구조 생성
    make_v2_dirs()

    # 3. dataset_v1의 train, valid, test 복사
    copy_dataset_v1_to_v2()

    # 4. 검수 완료 auto_check 데이터를 dataset_v2/train에만 추가
    add_auto_check_to_v2_train()

    # 5. dataset_v2/data.yaml 자동 생성
    create_data_yaml()

    # 6. 이미지/라벨 개수 확인
    count_files()

    # 7. train 클래스별 객체 수 확인
    count_train_class_objects()

    print("\ndataset_v2 생성 완료")