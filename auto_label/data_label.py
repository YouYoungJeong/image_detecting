# ============================================================
# auto_label_with_model_v2_merge_classes.py
# model_v2 best.pt 기반 오토라벨링 코드
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
# 실행:
# python auto_label_with_model_v2_merge_classes.py
# ============================================================

from pathlib import Path
import shutil
import yaml
import cv2
from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# model_v2 best.pt 경로
MODEL_PATH = BASE_DIR / "runs" / "detect" / "model_v2_train" / "weights" / "best.pt"

# 오토라벨링할 원본 이미지 폴더
# 여기에 라벨링 안 된 이미지들을 넣어두면 됨
SOURCE_IMAGE_DIR = BASE_DIR / "auto_label_source" / "images"

# 오토라벨링 결과 저장 폴더
OUTPUT_DATASET_DIR = BASE_DIR / "auto_labeled_remodel"

OUTPUT_IMAGE_DIR = OUTPUT_DATASET_DIR / "images"
OUTPUT_LABEL_DIR = OUTPUT_DATASET_DIR / "labels"

# 추론 결과 이미지 저장 여부
SAVE_VISUALIZED_IMAGE = True
OUTPUT_VIS_DIR = OUTPUT_DATASET_DIR / "visualized"

# confidence threshold
CONF_THRESHOLD = 0.25

# 이미지 확장자
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# ============================================================
# 2. 기존 클래스 / 신규 클래스 정의
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

# 기존 class_id -> 신규 class_id 변환
CLASS_MAPPING = {
    0: 0,  # House1 -> House
    1: 0,  # House2 -> House

    2: 1,  # Human1 -> Human
    3: 1,  # Human2 -> Human
    4: 1,  # Human3 -> Human

    6: 2,  # Tank -> Tank
    9: 3,  # car -> car

    # 5: Rock 제외
    # 7: Tent1 제외
    # 8: Wall 제외
}


# ============================================================
# 3. 유틸 함수
# ============================================================

def reset_dir(dir_path: Path):
    """
    기존 결과 폴더가 있으면 삭제 후 새로 생성
    """
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def make_dir(dir_path: Path):
    """
    폴더 생성
    """
    dir_path.mkdir(parents=True, exist_ok=True)


def get_image_files(image_dir: Path):
    """
    이미지 파일 목록 가져오기
    """
    image_files = []

    for ext in IMAGE_EXTENSIONS:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))

    image_files = sorted(list(set(image_files)))
    return image_files


def xyxy_to_yolo_normalized(x1, y1, x2, y2, img_w, img_h):
    """
    xyxy 좌표를 YOLO normalized 형식으로 변환

    입력:
        x1, y1, x2, y2: box 좌표
        img_w, img_h: 이미지 크기

    출력:
        x_center, y_center, width, height
    """
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    return x_center, y_center, width, height


def clamp(value, min_value=0.0, max_value=1.0):
    """
    YOLO 좌표값이 0~1 범위를 벗어나지 않도록 보정
    """
    return max(min_value, min(max_value, value))


def save_data_yaml(output_dir: Path):
    """
    YOLO 학습용 data.yaml 생성
    """
    data = {
        "train": "../images",
        "val": "../images",
        "test": "../images",
        "nc": len(NEW_CLASS_NAMES),
        "names": [NEW_CLASS_NAMES[i] for i in range(len(NEW_CLASS_NAMES))]
    }

    yaml_path = output_dir / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] data.yaml 저장 완료: {yaml_path}")


def draw_boxes(image, label_lines):
    """
    변환된 4개 클래스 기준으로 박스 시각화
    """
    img_h, img_w = image.shape[:2]

    for line in label_lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        cls_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x1 = int((x_center - width / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)

        class_name = NEW_CLASS_NAMES[cls_id]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            class_name,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return image


# ============================================================
# 4. 오토라벨링 메인 함수
# ============================================================

def auto_label_images():
    print("========== Model_v2 오토라벨링 시작 ==========")

    # --------------------------------------------------------
    # 경로 확인
    # --------------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"[ERROR] 모델 파일이 없습니다: {MODEL_PATH}")

    if not SOURCE_IMAGE_DIR.exists():
        raise FileNotFoundError(f"[ERROR] 원본 이미지 폴더가 없습니다: {SOURCE_IMAGE_DIR}")

    # --------------------------------------------------------
    # 출력 폴더 초기화
    # --------------------------------------------------------
    reset_dir(OUTPUT_DATASET_DIR)
    make_dir(OUTPUT_IMAGE_DIR)
    make_dir(OUTPUT_LABEL_DIR)

    if SAVE_VISUALIZED_IMAGE:
        make_dir(OUTPUT_VIS_DIR)

    # --------------------------------------------------------
    # 이미지 목록
    # --------------------------------------------------------
    image_files = get_image_files(SOURCE_IMAGE_DIR)

    if len(image_files) == 0:
        raise RuntimeError(f"[ERROR] 이미지 파일이 없습니다: {SOURCE_IMAGE_DIR}")

    print(f"[INFO] 모델 경로: {MODEL_PATH}")
    print(f"[INFO] 원본 이미지 폴더: {SOURCE_IMAGE_DIR}")
    print(f"[INFO] 출력 데이터셋 폴더: {OUTPUT_DATASET_DIR}")
    print(f"[INFO] 전체 이미지 수: {len(image_files)}")
    print(f"[INFO] confidence threshold: {CONF_THRESHOLD}")

    # --------------------------------------------------------
    # 모델 로드
    # --------------------------------------------------------
    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # 통계 변수
    # --------------------------------------------------------
    total_images = 0
    labeled_images = 0
    empty_label_images = 0

    old_class_count = {class_name: 0 for class_name in OLD_CLASS_NAMES.values()}
    new_class_count = {class_name: 0 for class_name in NEW_CLASS_NAMES.values()}
    excluded_count = {
        "Rock": 0,
        "Tent1": 0,
        "Wall": 0
    }

    # --------------------------------------------------------
    # 이미지별 오토라벨링
    # --------------------------------------------------------
    for idx, image_path in enumerate(image_files, start=1):
        total_images += 1

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARN] 이미지 로드 실패, 건너뜀: {image_path}")
            continue

        img_h, img_w = image.shape[:2]

        # 추론
        results = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        label_lines = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                old_cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                old_class_name = OLD_CLASS_NAMES.get(old_cls_id, f"Unknown_{old_cls_id}")

                if old_class_name in old_class_count:
                    old_class_count[old_class_name] += 1

                # 제외 클래스는 저장하지 않음
                if old_cls_id not in CLASS_MAPPING:
                    if old_class_name in excluded_count:
                        excluded_count[old_class_name] += 1
                    continue

                new_cls_id = CLASS_MAPPING[old_cls_id]
                new_class_name = NEW_CLASS_NAMES[new_cls_id]

                # xyxy 좌표
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # 이미지 범위 밖 좌표 보정
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))

                # 이상한 박스 제거
                if x2 <= x1 or y2 <= y1:
                    continue

                x_center, y_center, width, height = xyxy_to_yolo_normalized(
                    x1, y1, x2, y2, img_w, img_h
                )

                # 0~1 범위 보정
                x_center = clamp(x_center)
                y_center = clamp(y_center)
                width = clamp(width)
                height = clamp(height)

                label_line = f"{new_cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                label_lines.append(label_line)

                new_class_count[new_class_name] += 1

        # ----------------------------------------------------
        # 이미지 복사
        # ----------------------------------------------------
        output_image_path = OUTPUT_IMAGE_DIR / image_path.name
        shutil.copy2(image_path, output_image_path)

        # ----------------------------------------------------
        # 라벨 저장
        # 탐지 결과가 없어도 빈 txt 파일 저장
        # ----------------------------------------------------
        output_label_path = OUTPUT_LABEL_DIR / f"{image_path.stem}.txt"

        with open(output_label_path, "w", encoding="utf-8") as f:
            if len(label_lines) > 0:
                f.write("\n".join(label_lines))

        if len(label_lines) > 0:
            labeled_images += 1
        else:
            empty_label_images += 1

        # ----------------------------------------------------
        # 시각화 이미지 저장
        # ----------------------------------------------------
        if SAVE_VISUALIZED_IMAGE:
            vis_image = image.copy()
            vis_image = draw_boxes(vis_image, label_lines)

            output_vis_path = OUTPUT_VIS_DIR / image_path.name
            cv2.imwrite(str(output_vis_path), vis_image)

        # ----------------------------------------------------
        # 진행 상황 출력
        # ----------------------------------------------------
        if idx % 50 == 0 or idx == len(image_files):
            print(f"[INFO] 진행률: {idx}/{len(image_files)}")

    # --------------------------------------------------------
    # data.yaml 저장
    # --------------------------------------------------------
    save_data_yaml(OUTPUT_DATASET_DIR)

    # --------------------------------------------------------
    # 리포트 저장
    # --------------------------------------------------------
    report_path = OUTPUT_DATASET_DIR / "auto_label_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("========== Model_v2 Auto Label Report ==========\n\n")

        f.write("[경로 정보]\n")
        f.write(f"MODEL_PATH: {MODEL_PATH}\n")
        f.write(f"SOURCE_IMAGE_DIR: {SOURCE_IMAGE_DIR}\n")
        f.write(f"OUTPUT_DATASET_DIR: {OUTPUT_DATASET_DIR}\n\n")

        f.write("[설정]\n")
        f.write(f"CONF_THRESHOLD: {CONF_THRESHOLD}\n\n")

        f.write("[이미지 통계]\n")
        f.write(f"total_images: {total_images}\n")
        f.write(f"labeled_images: {labeled_images}\n")
        f.write(f"empty_label_images: {empty_label_images}\n\n")

        f.write("[기존 클래스 탐지 개수]\n")
        for class_name, count in old_class_count.items():
            f.write(f"{class_name}: {count}\n")

        f.write("\n[신규 클래스 저장 개수]\n")
        for class_name, count in new_class_count.items():
            f.write(f"{class_name}: {count}\n")

        f.write("\n[제외 클래스 개수]\n")
        for class_name, count in excluded_count.items():
            f.write(f"{class_name}: {count}\n")

        f.write("\n[클래스 변환 기준]\n")
        f.write("House1, House2 -> House\n")
        f.write("Human1, Human2, Human3 -> Human\n")
        f.write("Tank -> Tank\n")
        f.write("car -> car\n")
        f.write("Rock, Tent1, Wall -> excluded\n")

    print("\n========== 오토라벨링 완료 ==========")
    print(f"[INFO] 전체 이미지 수: {total_images}")
    print(f"[INFO] 라벨 생성 이미지 수: {labeled_images}")
    print(f"[INFO] 빈 라벨 이미지 수: {empty_label_images}")
    print(f"[INFO] 결과 이미지 폴더: {OUTPUT_IMAGE_DIR}")
    print(f"[INFO] 결과 라벨 폴더: {OUTPUT_LABEL_DIR}")
    print(f"[INFO] 리포트 저장 완료: {report_path}")

    print("\n========== 신규 클래스 저장 개수 ==========")
    for class_name, count in new_class_count.items():
        print(f"{class_name}: {count}")

    print("\n========== 제외 클래스 개수 ==========")
    for class_name, count in excluded_count.items():
        print(f"{class_name}: {count}")


# ============================================================
# 5. 실행
# ============================================================

if __name__ == "__main__":
    auto_label_images()