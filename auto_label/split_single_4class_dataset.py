# ============================================================
# 하나로 모아둔 4클래스 YOLO 데이터셋을
# dataset_v1/train/valid/test = 80:10:10 으로 분할
#
# 최종 클래스:
# 0 House
# 1 Human
# 2 Tank
# 3 car
# ============================================================

from pathlib import Path
import shutil
import random
import yaml


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 하나로 모아둔 데이터셋 폴더
# 이 안에 images/, labels/ 가 있어야 함
SOURCE_DATASET_DIR = BASE_DIR / "dataset_4class_single"

SOURCE_IMAGE_DIR = SOURCE_DATASET_DIR / "images"
SOURCE_LABEL_DIR = SOURCE_DATASET_DIR / "labels"

# 최종 분할 데이터셋 저장 폴더
OUTPUT_DATASET_DIR = BASE_DIR / "dataset_v1"

# 분할 비율
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# 랜덤 고정
RANDOM_SEED = 42

# 이미지 확장자
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# 최종 클래스
CLASS_NAMES = ["House", "Human", "Tank", "car"]
VALID_CLASS_IDS = set(range(len(CLASS_NAMES)))


# ============================================================
# 2. 유틸 함수
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
    이미지 파일 여부 확인
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def get_image_files(image_dir: Path):
    """
    이미지 폴더 내 이미지 파일 목록 가져오기
    """
    if not image_dir.exists():
        return []

    image_files = [
        file_path for file_path in image_dir.iterdir()
        if file_path.is_file() and is_image_file(file_path)
    ]

    return sorted(image_files)


def check_label_file(label_path: Path):
    """
    라벨 파일 검증

    반환:
    - valid_lines: 정상 라벨 라인
    - invalid_line_count: 비정상 라벨 라인 수
    """
    valid_lines = []
    invalid_line_count = 0

    if not label_path.exists():
        return valid_lines, invalid_line_count

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line == "":
            continue

        parts = line.split()

        if len(parts) != 5:
            invalid_line_count += 1
            continue

        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            invalid_line_count += 1
            continue

        if class_id not in VALID_CLASS_IDS:
            invalid_line_count += 1
            continue

        if not (0 <= x_center <= 1):
            invalid_line_count += 1
            continue

        if not (0 <= y_center <= 1):
            invalid_line_count += 1
            continue

        if not (0 < width <= 1):
            invalid_line_count += 1
            continue

        if not (0 < height <= 1):
            invalid_line_count += 1
            continue

        valid_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        valid_lines.append(valid_line)

    return valid_lines, invalid_line_count


def collect_items():
    """
    images 폴더 기준으로 이미지와 라벨 쌍 수집
    """
    image_files = get_image_files(SOURCE_IMAGE_DIR)

    if len(image_files) == 0:
        raise RuntimeError(f"[ERROR] 이미지 파일이 없습니다: {SOURCE_IMAGE_DIR}")

    items = []

    missing_label_count = 0
    empty_label_count = 0
    invalid_line_total = 0

    for image_path in image_files:
        label_path = SOURCE_LABEL_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_label_count += 1
            valid_lines = []
            invalid_line_count = 0
        else:
            valid_lines, invalid_line_count = check_label_file(label_path)

        if len(valid_lines) == 0:
            empty_label_count += 1

        invalid_line_total += invalid_line_count

        items.append({
            "image_path": image_path,
            "label_path": label_path,
            "label_lines": valid_lines
        })

    print("========== 원본 데이터셋 확인 ==========")
    print(f"[INFO] 전체 이미지 수: {len(items)}")
    print(f"[INFO] 라벨 파일 없는 이미지 수: {missing_label_count}")
    print(f"[INFO] 빈 라벨 이미지 수: {empty_label_count}")
    print(f"[INFO] 비정상 라벨 라인 수: {invalid_line_total}")

    return items


def split_items(items):
    """
    전체 데이터를 train/valid/test = 80:10:10으로 분할
    """
    random.seed(RANDOM_SEED)
    random.shuffle(items)

    total_count = len(items)

    train_count = int(total_count * TRAIN_RATIO)
    valid_count = int(total_count * VALID_RATIO)

    train_items = items[:train_count]
    valid_items = items[train_count:train_count + valid_count]
    test_items = items[train_count + valid_count:]

    split_dict = {
        "train": train_items,
        "valid": valid_items,
        "test": test_items
    }

    return split_dict


def save_split_dataset(split_dict):
    """
    split별 이미지/라벨 저장
    """
    for split_name, items in split_dict.items():
        output_image_dir = OUTPUT_DATASET_DIR / split_name / "images"
        output_label_dir = OUTPUT_DATASET_DIR / split_name / "labels"

        make_dir(output_image_dir)
        make_dir(output_label_dir)

        for item in items:
            image_path = item["image_path"]
            label_lines = item["label_lines"]

            output_image_path = output_image_dir / image_path.name
            output_label_path = output_label_dir / f"{image_path.stem}.txt"

            # 이미지 복사
            shutil.copy2(image_path, output_image_path)

            # 라벨 저장
            # 기존 txt를 그대로 복사하지 않고, 검증된 라벨 라인만 다시 저장
            # 비어 있으면 빈 txt 생성
            with open(output_label_path, "w", encoding="utf-8") as f:
                if len(label_lines) > 0:
                    f.write("\n".join(label_lines))

        print(f"[INFO] {split_name} 저장 완료: {len(items)} images")


def save_data_yaml():
    """
    YOLO 학습용 data.yaml 생성
    """
    data = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }

    yaml_path = OUTPUT_DATASET_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] data.yaml 저장 완료: {yaml_path}")


def count_split_objects(split_name: str):
    """
    split별 이미지 수, 라벨 수, 객체 수, 클래스별 객체 수 계산
    """
    image_dir = OUTPUT_DATASET_DIR / split_name / "images"
    label_dir = OUTPUT_DATASET_DIR / split_name / "labels"

    image_count = 0
    label_count = 0
    empty_label_count = 0
    object_count = 0

    class_counts = {class_name: 0 for class_name in CLASS_NAMES}

    if image_dir.exists():
        image_count = len([
            p for p in image_dir.iterdir()
            if p.is_file() and is_image_file(p)
        ])

    if label_dir.exists():
        label_files = sorted(label_dir.glob("*.txt"))
        label_count = len(label_files)

        for label_path in label_files:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip() != ""]

            if len(lines) == 0:
                empty_label_count += 1

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    continue

                class_id = int(float(parts[0]))

                if class_id in VALID_CLASS_IDS:
                    class_name = CLASS_NAMES[class_id]
                    class_counts[class_name] += 1
                    object_count += 1

    return {
        "split": split_name,
        "image_count": image_count,
        "label_count": label_count,
        "empty_label_count": empty_label_count,
        "object_count": object_count,
        "class_counts": class_counts
    }


def save_report(split_dict):
    """
    분할 결과 리포트 저장
    """
    report_path = OUTPUT_DATASET_DIR / "split_report.txt"

    split_reports = [
        count_split_objects("train"),
        count_split_objects("valid"),
        count_split_objects("test")
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("========== Dataset_v1 Split Report ==========\n\n")

        f.write("[경로]\n")
        f.write(f"SOURCE_DATASET_DIR: {SOURCE_DATASET_DIR}\n")
        f.write(f"OUTPUT_DATASET_DIR: {OUTPUT_DATASET_DIR}\n\n")

        f.write("[분할 비율]\n")
        f.write(f"train: {TRAIN_RATIO}\n")
        f.write(f"valid: {VALID_RATIO}\n")
        f.write(f"test: {TEST_RATIO}\n\n")

        f.write("[클래스]\n")
        for idx, class_name in enumerate(CLASS_NAMES):
            f.write(f"{idx}: {class_name}\n")

        f.write("\n[분할 이미지 수]\n")
        for split_name, items in split_dict.items():
            f.write(f"{split_name}: {len(items)}\n")

        f.write("\n[split별 객체 통계]\n")
        for report in split_reports:
            f.write(f"\n[{report['split']}]\n")
            f.write(f"image_count: {report['image_count']}\n")
            f.write(f"label_count: {report['label_count']}\n")
            f.write(f"empty_label_count: {report['empty_label_count']}\n")
            f.write(f"object_count: {report['object_count']}\n")

            for class_name, count in report["class_counts"].items():
                f.write(f"{class_name}: {count}\n")

    print(f"[INFO] 리포트 저장 완료: {report_path}")

    print("\n========== 최종 split별 통계 ==========")
    for report in split_reports:
        print(f"\n[{report['split']}]")
        print(f"image_count: {report['image_count']}")
        print(f"label_count: {report['label_count']}")
        print(f"empty_label_count: {report['empty_label_count']}")
        print(f"object_count: {report['object_count']}")

        for class_name, count in report["class_counts"].items():
            print(f"{class_name}: {count}")


# ============================================================
# 3. 메인 실행
# ============================================================

def main():
    print("========== dataset_v1 80:10:10 분할 시작 ==========")

    if not SOURCE_DATASET_DIR.exists():
        raise FileNotFoundError(f"[ERROR] 원본 데이터셋 폴더가 없습니다: {SOURCE_DATASET_DIR}")

    if not SOURCE_IMAGE_DIR.exists():
        raise FileNotFoundError(f"[ERROR] 원본 images 폴더가 없습니다: {SOURCE_IMAGE_DIR}")

    if not SOURCE_LABEL_DIR.exists():
        raise FileNotFoundError(f"[ERROR] 원본 labels 폴더가 없습니다: {SOURCE_LABEL_DIR}")

    print(f"[INFO] 원본 데이터셋: {SOURCE_DATASET_DIR}")
    print(f"[INFO] 출력 데이터셋: {OUTPUT_DATASET_DIR}")

    # 기존 dataset_v1 삭제 후 새로 생성
    reset_dir(OUTPUT_DATASET_DIR)

    # 이미지/라벨 수집
    items = collect_items()

    # 80:10:10 분할
    split_dict = split_items(items)

    print("\n========== 분할 결과 ==========")
    print(f"train: {len(split_dict['train'])}")
    print(f"valid: {len(split_dict['valid'])}")
    print(f"test : {len(split_dict['test'])}")

    # 저장
    save_split_dataset(split_dict)

    # data.yaml 저장
    save_data_yaml()

    # 리포트 저장
    save_report(split_dict)

    print("\n========== dataset_v1 생성 완료 ==========")
    print(f"[INFO] 최종 데이터셋 경로: {OUTPUT_DATASET_DIR}")
    print(f"[INFO] data.yaml 경로: {OUTPUT_DATASET_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()