# ============================================================
# convert_10class_to_4class_single_folder.py
# 10클래스 YOLO 데이터셋을 4클래스로 변환 후
# train/valid/test 구분 없이 하나의 폴더에 저장하는 코드
#
# 기존 클래스:
# 0 House1
# 1 House2
# 2 Human1
# 3 Human2
# 4 Human3
# 5 Rock
# 6 Tank
# 7 Tent1
# 8 Wall
# 9 car
#
# 변환 후 클래스:
# 0 House
# 1 Human
# 2 Tank
# 3 car
#
# 제외 클래스:
# Rock, Tent1, Wall
#
# 저장 파일명:
# images/img_4class_1.jpg
# labels/img_4class_1.txt
#
# 실행:
# python convert_10class_to_4class_single_folder.py
# ============================================================

from pathlib import Path
import shutil
import yaml


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 원본 10클래스 YOLO 데이터셋 경로
# 이 안에 train/valid/test/images, train/valid/test/labels 구조가 있다고 가정
SOURCE_DATASET_DIR = BASE_DIR / "dataset_v3"

# 변환 후 저장할 4클래스 단일 폴더 데이터셋
OUTPUT_DATASET_DIR = BASE_DIR / "dataset_4class_single"

OUTPUT_IMAGE_DIR = OUTPUT_DATASET_DIR / "images"
OUTPUT_LABEL_DIR = OUTPUT_DATASET_DIR / "labels"

# 이미지 파일명 접두사
NEW_IMAGE_PREFIX = "img_4class"

# 이미지 확장자
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# 읽어올 split 폴더
# train/valid/test를 전부 읽지만, 저장할 때는 하나의 폴더에 합침
SPLITS = ["train", "valid", "test"]


# ============================================================
# 2. 클래스 설정
# ============================================================

OLD_CLASS_NAMES = {
    0: "House1",
    1: "House2",
    2: "Human1",
    3: "Human2",
    4: "Human3",
    5: "Rock",
    6: "Tank",
    7: "Tent1",
    8: "Wall",
    9: "car",
}

NEW_CLASS_NAMES = {
    0: "House",
    1: "Human",
    2: "Tank",
    3: "car",
}

# 기존 class_id -> 신규 class_id 변환 기준
CLASS_MAPPING = {
    0: 0,  # House1 -> House
    1: 0,  # House2 -> House

    2: 1,  # Human1 -> Human
    3: 1,  # Human2 -> Human
    4: 1,  # Human3 -> Human

    6: 2,  # Tank -> Tank
    9: 3,  # car -> car

    # 5 Rock  -> 제외
    # 7 Tent1 -> 제외
    # 8 Wall  -> 제외
}

EXCLUDED_OLD_CLASS_IDS = [5, 7, 8]


# ============================================================
# 3. 유틸 함수
# ============================================================

def reset_dir(dir_path: Path):
    """
    기존 출력 폴더 삭제 후 새로 생성
    """
    if dir_path.exists():
        shutil.rmtree(dir_path)

    dir_path.mkdir(parents=True, exist_ok=True)


def make_dir(dir_path: Path):
    """
    폴더 생성
    """
    dir_path.mkdir(parents=True, exist_ok=True)


def is_image_file(file_path: Path):
    """
    이미지 파일인지 확인
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def get_image_files(image_dir: Path):
    """
    이미지 폴더에서 이미지 파일 목록 가져오기
    """
    if not image_dir.exists():
        return []

    image_files = []

    for file_path in image_dir.iterdir():
        if file_path.is_file() and is_image_file(file_path):
            image_files.append(file_path)

    return sorted(image_files)


def convert_label_file(source_label_path: Path):
    """
    기존 10클래스 YOLO 라벨을 4클래스 YOLO 라벨로 변환

    YOLO 라벨 형식:
    class_id x_center y_center width height
    """
    converted_lines = []

    old_class_count = {class_name: 0 for class_name in OLD_CLASS_NAMES.values()}
    new_class_count = {class_name: 0 for class_name in NEW_CLASS_NAMES.values()}
    excluded_count = {OLD_CLASS_NAMES[class_id]: 0 for class_id in EXCLUDED_OLD_CLASS_IDS}

    if not source_label_path.exists():
        return converted_lines, old_class_count, new_class_count, excluded_count

    with open(source_label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line == "":
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            old_class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue

        old_class_name = OLD_CLASS_NAMES.get(old_class_id, f"Unknown_{old_class_id}")

        if old_class_name in old_class_count:
            old_class_count[old_class_name] += 1

        # Rock, Tent1, Wall 제거
        if old_class_id not in CLASS_MAPPING:
            if old_class_id in EXCLUDED_OLD_CLASS_IDS:
                excluded_class_name = OLD_CLASS_NAMES[old_class_id]
                excluded_count[excluded_class_name] += 1
            continue

        # YOLO 좌표값 검증
        if not (0 <= x_center <= 1):
            continue
        if not (0 <= y_center <= 1):
            continue
        if not (0 < width <= 1):
            continue
        if not (0 < height <= 1):
            continue

        new_class_id = CLASS_MAPPING[old_class_id]
        new_class_name = NEW_CLASS_NAMES[new_class_id]

        converted_line = f"{new_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        converted_lines.append(converted_line)

        new_class_count[new_class_name] += 1

    return converted_lines, old_class_count, new_class_count, excluded_count


def add_count_dict(total_dict, add_dict):
    """
    클래스별 count 누적
    """
    for key, value in add_dict.items():
        if key not in total_dict:
            total_dict[key] = 0
        total_dict[key] += value


def save_data_yaml():
    """
    단일 폴더 구조 기준 data.yaml 생성
    """
    data = {
        "train": "../images",
        "val": "../images",
        "test": "../images",
        "nc": len(NEW_CLASS_NAMES),
        "names": [NEW_CLASS_NAMES[i] for i in range(len(NEW_CLASS_NAMES))]
    }

    yaml_path = OUTPUT_DATASET_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] data.yaml 저장 완료: {yaml_path}")


# ============================================================
# 4. 메인 변환 함수
# ============================================================

def convert_dataset_to_single_folder():
    print("========== 10클래스 -> 4클래스 단일 폴더 변환 시작 ==========")

    if not SOURCE_DATASET_DIR.exists():
        raise FileNotFoundError(f"[ERROR] 원본 데이터셋 폴더가 없습니다: {SOURCE_DATASET_DIR}")

    print(f"[INFO] 원본 데이터셋: {SOURCE_DATASET_DIR}")
    print(f"[INFO] 출력 데이터셋: {OUTPUT_DATASET_DIR}")

    # 출력 폴더 초기화
    # reset_dir(OUTPUT_DATASET_DIR)
    make_dir(OUTPUT_IMAGE_DIR)
    make_dir(OUTPUT_LABEL_DIR)

    total_image_count = 0
    total_empty_label_count = 0

    total_old_class_count = {class_name: 0 for class_name in OLD_CLASS_NAMES.values()}
    total_new_class_count = {class_name: 0 for class_name in NEW_CLASS_NAMES.values()}
    total_excluded_count = {OLD_CLASS_NAMES[class_id]: 0 for class_id in EXCLUDED_OLD_CLASS_IDS}

    file_mapping_lines = []

    new_index = 4418

    for split in SPLITS:
        source_image_dir = SOURCE_DATASET_DIR / split / "images"
        source_label_dir = SOURCE_DATASET_DIR / split / "labels"

        image_files = get_image_files(source_image_dir)

        if len(image_files) == 0:
            print(f"[WARN] 이미지가 없습니다: {source_image_dir}")
            continue

        print(f"\n[{split}]")
        print(f"[INFO] 이미지 수: {len(image_files)}")

        for idx, image_path in enumerate(image_files, start=1):
            source_label_path = source_label_dir / f"{image_path.stem}.txt"

            # 새 파일명 생성
            new_stem = f"{NEW_IMAGE_PREFIX}_{new_index}"
            new_image_name = f"{new_stem}{image_path.suffix.lower()}"
            new_label_name = f"{new_stem}.txt"

            output_image_path = OUTPUT_IMAGE_DIR / new_image_name
            output_label_path = OUTPUT_LABEL_DIR / new_label_name

            # 이미지 복사 및 이름 변경
            shutil.copy2(image_path, output_image_path)

            # 라벨 변환
            converted_lines, old_count, new_count, excluded_count = convert_label_file(source_label_path)

            # 라벨 저장
            # 변환 후 남는 객체가 없어도 빈 txt 파일 생성
            with open(output_label_path, "w", encoding="utf-8") as f:
                if len(converted_lines) > 0:
                    f.write("\n".join(converted_lines))

            if len(converted_lines) == 0:
                total_empty_label_count += 1

            # 통계 누적
            add_count_dict(total_old_class_count, old_count)
            add_count_dict(total_new_class_count, new_count)
            add_count_dict(total_excluded_count, excluded_count)

            # 파일명 변경 기록
            file_mapping_lines.append(
                f"{split},{image_path.name},{new_image_name},{source_label_path.name},{new_label_name}"
            )

            total_image_count += 1
            new_index += 1

            if idx % 500 == 0 or idx == len(image_files):
                print(f"[INFO] 진행률: {idx}/{len(image_files)}")

    # data.yaml 생성
    save_data_yaml()

    # 파일명 매핑 저장
    mapping_path = OUTPUT_DATASET_DIR / "filename_mapping.csv"

    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("original_split,original_image_name,new_image_name,original_label_name,new_label_name\n")
        f.write("\n".join(file_mapping_lines))

    # 리포트 저장
    report_path = OUTPUT_DATASET_DIR / "convert_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("========== 10-Class to 4-Class Single Folder Convert Report ==========\n\n")

        f.write("[원본 클래스]\n")
        for class_id, class_name in OLD_CLASS_NAMES.items():
            f.write(f"{class_id}: {class_name}\n")

        f.write("\n[변환 후 클래스]\n")
        for class_id, class_name in NEW_CLASS_NAMES.items():
            f.write(f"{class_id}: {class_name}\n")

        f.write("\n[변환 기준]\n")
        f.write("House1, House2 -> House\n")
        f.write("Human1, Human2, Human3 -> Human\n")
        f.write("Tank -> Tank\n")
        f.write("car -> car\n")
        f.write("Rock, Tent1, Wall -> 제거\n")

        f.write("\n[저장 구조]\n")
        f.write("train/valid/test로 나누지 않음\n")
        f.write("images/img_4class_1.확장자\n")
        f.write("labels/img_4class_1.txt\n")

        f.write("\n[경로]\n")
        f.write(f"SOURCE_DATASET_DIR: {SOURCE_DATASET_DIR}\n")
        f.write(f"OUTPUT_DATASET_DIR: {OUTPUT_DATASET_DIR}\n")

        f.write("\n[전체 통계]\n")
        f.write(f"total_image_count: {total_image_count}\n")
        f.write(f"total_empty_label_count: {total_empty_label_count}\n")

        f.write("\n[원본 클래스 객체 수]\n")
        for class_name, count in total_old_class_count.items():
            f.write(f"{class_name}: {count}\n")

        f.write("\n[변환 후 클래스 객체 수]\n")
        for class_name, count in total_new_class_count.items():
            f.write(f"{class_name}: {count}\n")

        f.write("\n[제거된 클래스 객체 수]\n")
        for class_name, count in total_excluded_count.items():
            f.write(f"{class_name}: {count}\n")

    print("\n========== 변환 완료 ==========")
    print(f"[INFO] 전체 이미지 수: {total_image_count}")
    print(f"[INFO] 빈 라벨 파일 수: {total_empty_label_count}")
    print(f"[INFO] 출력 이미지 폴더: {OUTPUT_IMAGE_DIR}")
    print(f"[INFO] 출력 라벨 폴더: {OUTPUT_LABEL_DIR}")
    print(f"[INFO] 파일명 매핑 저장: {mapping_path}")
    print(f"[INFO] 리포트 저장: {report_path}")

    print("\n========== 변환 후 클래스 객체 수 ==========")
    for class_name, count in total_new_class_count.items():
        print(f"{class_name}: {count}")

    print("\n========== 제거된 클래스 객체 수 ==========")
    for class_name, count in total_excluded_count.items():
        print(f"{class_name}: {count}")


# ============================================================
# 5. 실행
# ============================================================

if __name__ == "__main__":
    convert_dataset_to_single_folder()