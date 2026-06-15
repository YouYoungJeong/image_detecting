import os

os.environ["WANDB_DISABLED"] = "true"
# ============================================================
# OpenMP 중복 로딩 오류 임시 해결
# ============================================================
# Windows 환경에서 torch, numpy, matplotlib, opencv 등이
# libiomp5md.dll을 중복으로 불러오면 학습이 중단될 수 있음.
# 임시 해결용이며, 최종적으로는 conda 환경 정리를 권장.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil
import random
from pathlib import Path

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# 1. 기본 경로 설정
# ============================================================

# 현재 파이썬 파일이 실행되는 프로젝트 폴더
BASE_DIR = Path(os.getcwd())

# 기존 데이터셋
DATASET_V2_DIR = BASE_DIR / "dataset_v2"
DATASET_V2_YAML = DATASET_V2_DIR / "data.yaml"

# 최종 개선용 데이터셋
DATASET_V3_DIR = BASE_DIR / "dataset_v3"
DATASET_V3_YAML = DATASET_V3_DIR / "data.yaml"

# 검수 완료된 오토라벨링 데이터
AUTO_CHECK_DIR = BASE_DIR / "auto_check"
AUTO_IMAGES_DIR = AUTO_CHECK_DIR / "images"
AUTO_LABELS_DIR = AUTO_CHECK_DIR / "labels"

# model1_v2 best.pt
# v2 코드에서 사용한 학습 결과 경로 기준
V2_BEST_PT = BASE_DIR / "runs" / "detect" / "model_v2_train" / "weights" / "best.pt"

# v3 final 학습 결과 저장 경로
V3_TRAIN_PROJECT = BASE_DIR / "runs" / "detect"
V3_TRAIN_NAME = "model1_v3_final_train"

# 성능 비교 결과 저장 폴더
OUTPUT_DIR = BASE_DIR / "model1_v3_final_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# dataset_v3 초기화 여부
RESET_DATASET_V3 = True

# 검수 데이터 1장당 생성할 증강 이미지 개수
# 2000장 기준:
# AUG_PER_IMAGE=4이면 증강본 약 8000장 추가
# 기존 dataset_v2/train + 증강본으로 final 학습 진행
AUG_PER_IMAGE = 4

# 랜덤 고정
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 2. 클래스 이름 설정
# ============================================================
# dataset_v1, dataset_v2와 반드시 같은 순서 유지

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

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# ============================================================
# 3. 경로 확인 함수
# ============================================================

def check_path(path, description):
    """
    필요한 파일 또는 폴더가 존재하는지 확인.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{description} 경로를 찾을 수 없습니다: {path}")


# ============================================================
# 4. dataset_v3 생성 함수
# ============================================================

def reset_dataset_v3():
    """
    기존 dataset_v3 삭제.
    이전 실행 결과와 섞이지 않도록 초기화.
    """
    if RESET_DATASET_V3 and DATASET_V3_DIR.exists():
        shutil.rmtree(DATASET_V3_DIR)
        print(f"[초기화] 기존 dataset_v3 삭제 완료: {DATASET_V3_DIR}")


def make_v3_dirs():
    """
    YOLO 학습 구조에 맞게 dataset_v3 폴더 생성.
    """
    for split in ["train", "valid", "test"]:
        (DATASET_V3_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_V3_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    print("[생성] dataset_v3 폴더 구조 생성 완료")


def copy_all_files(src_dir, dst_dir):
    """
    폴더 안의 모든 파일 복사.
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


def copy_dataset_v2_to_v3():
    """
    dataset_v2 전체를 dataset_v3로 복사.

    중요:
    - train은 기존 v2 학습 데이터를 유지
    - valid/test는 그대로 유지
    - 성능 비교 공정성을 위해 valid/test는 증강하지 않음
    """
    print("\n[복사] dataset_v2 → dataset_v3")

    copy_info = [
        ("train/images", DATASET_V2_DIR / "train" / "images", DATASET_V3_DIR / "train" / "images"),
        ("train/labels", DATASET_V2_DIR / "train" / "labels", DATASET_V3_DIR / "train" / "labels"),
        ("valid/images", DATASET_V2_DIR / "valid" / "images", DATASET_V3_DIR / "valid" / "images"),
        ("valid/labels", DATASET_V2_DIR / "valid" / "labels", DATASET_V3_DIR / "valid" / "labels"),
        ("test/images", DATASET_V2_DIR / "test" / "images", DATASET_V3_DIR / "test" / "images"),
        ("test/labels", DATASET_V2_DIR / "test" / "labels", DATASET_V3_DIR / "test" / "labels"),
    ]

    for name, src, dst in copy_info:
        count = copy_all_files(src, dst)
        print(f"- {name}: {count}개 복사 완료")


# ============================================================
# 5. YOLO 라벨 처리 함수
# ============================================================

def read_yolo_label(label_path):
    """
    YOLO txt 라벨 파일 읽기.

    라벨 형식:
    class_id x_center y_center width height
    """
    labels = []

    label_path = Path(label_path)

    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line == "":
            continue

        parts = line.split()

        if len(parts) < 5:
            print(f"[경고] 라벨 형식 이상: {label_path.name} / {line}")
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            labels.append([class_id, x_center, y_center, width, height])

        except ValueError:
            print(f"[경고] 라벨 변환 실패: {label_path.name} / {line}")

    return labels


def write_yolo_label(label_path, labels):
    """
    YOLO txt 라벨 파일 저장.
    """
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w", encoding="utf-8") as f:
        for label in labels:
            class_id, x_center, y_center, width, height = label

            # 좌표값은 0~1 범위로 제한
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            width = min(max(width, 0.0), 1.0)
            height = min(max(height, 0.0), 1.0)

            f.write(
                f"{int(class_id)} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )


def flip_labels_horizontal(labels):
    """
    좌우 반전 시 YOLO x_center 좌표 수정.

    기존 x_center가 0.2이면
    좌우 반전 후 x_center는 0.8이 됨.
    """
    flipped_labels = []

    for label in labels:
        class_id, x_center, y_center, width, height = label
        new_x_center = 1.0 - x_center
        flipped_labels.append([class_id, new_x_center, y_center, width, height])

    return flipped_labels


# ============================================================
# 6. 이미지 증강 함수
# ============================================================
# 박스 좌표 안정성을 위해 기본적으로 색상/밝기/노이즈 계열을 사용.
# 좌표 변경이 필요한 증강은 horizontal flip만 사용.

def apply_brightness_contrast(image):
    """
    일반 밝기/대비 증강.
    주간 환경의 노출 차이 대응.
    """
    alpha = random.uniform(0.85, 1.25)
    beta = random.randint(-25, 25)

    aug = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return aug


def apply_night(image):
    """
    야간전투 대응용 어두운 환경 증강.

    밝기 감소 + 대비 감소 + 약한 푸른 톤 추가.
    """
    alpha = random.uniform(0.45, 0.75)
    beta = random.randint(-45, -15)

    aug = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # BGR 기준: Blue 채널을 약간 높여 야간 느낌 추가
    b, g, r = cv2.split(aug)
    b = cv2.add(b, random.randint(5, 20))
    g = cv2.add(g, random.randint(0, 8))
    aug = cv2.merge([b, g, r])

    return aug


def apply_heavy_dark(image):
    """
    더 강한 저조도 증강.
    완전한 야간 또는 그림자 환경 대응.
    """
    alpha = random.uniform(0.30, 0.55)
    beta = random.randint(-60, -25)

    aug = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return aug


def apply_shadow(image):
    """
    부분 그림자 증강.
    산, 건물, 장애물 때문에 일부 영역이 어두워지는 상황 대응.
    """
    h, w = image.shape[:2]
    aug = image.copy()

    # 랜덤 다각형 그림자 영역 생성
    top_x = random.randint(0, w)
    bottom_x = random.randint(0, w)

    polygon = np.array([[
        (top_x, 0),
        (min(w, top_x + random.randint(w // 4, w // 2)), 0),
        (min(w, bottom_x + random.randint(w // 4, w // 2)), h),
        (bottom_x, h)
    ]], dtype=np.int32)

    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, polygon, (255, 255, 255))

    shadow_factor = random.uniform(0.45, 0.75)
    darkened = cv2.convertScaleAbs(aug, alpha=shadow_factor, beta=0)

    aug = np.where(mask == 255, darkened, aug)

    return aug


def apply_noise(image):
    """
    카메라 노이즈 증강.
    야간/저조도 환경에서 센서 노이즈가 증가하는 상황 대응.
    """
    noise_std = random.uniform(5, 18)
    noise = np.random.normal(0, noise_std, image.shape).astype(np.int16)

    aug = image.astype(np.int16) + noise
    aug = np.clip(aug, 0, 255).astype(np.uint8)

    return aug


def apply_blur(image):
    """
    약한 블러 증강.
    이동 중 화면 흔들림, 초점 흐림 대응.
    """
    kernel_size = random.choice([3, 5])
    aug = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return aug


def apply_hsv_shift(image):
    """
    색상 변화 증강.
    맵 조명, 날씨, 렌더링 톤 차이 대응.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)

    h_shift = random.randint(-8, 8)
    s_shift = random.randint(-25, 25)
    v_shift = random.randint(-20, 20)

    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + h_shift, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_shift, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v_shift, 0, 255)

    hsv = hsv.astype(np.uint8)
    aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return aug


def apply_flip_horizontal(image):
    """
    좌우 반전 증강.
    객체가 좌우 방향으로 등장하는 경우 대응.
    """
    aug = cv2.flip(image, 1)

    return aug


def make_random_augmentation(image, labels):
    """
    랜덤 증강 1개 생성.

    반환값:
    - 증강 이미지
    - 증강 라벨
    - 증강 타입 이름
    """
    aug_type = random.choice([
        "brightness",
        "night",
        "heavy_dark",
        "shadow",
        "noise",
        "blur",
        "hsv",
        "flip",
        "night_noise",
        "night_blur",
        "shadow_noise"
    ])

    aug_image = image.copy()
    aug_labels = labels.copy()

    if aug_type == "brightness":
        aug_image = apply_brightness_contrast(aug_image)

    elif aug_type == "night":
        aug_image = apply_night(aug_image)

    elif aug_type == "heavy_dark":
        aug_image = apply_heavy_dark(aug_image)

    elif aug_type == "shadow":
        aug_image = apply_shadow(aug_image)

    elif aug_type == "noise":
        aug_image = apply_noise(aug_image)

    elif aug_type == "blur":
        aug_image = apply_blur(aug_image)

    elif aug_type == "hsv":
        aug_image = apply_hsv_shift(aug_image)

    elif aug_type == "flip":
        aug_image = apply_flip_horizontal(aug_image)
        aug_labels = flip_labels_horizontal(aug_labels)

    elif aug_type == "night_noise":
        aug_image = apply_night(aug_image)
        aug_image = apply_noise(aug_image)

    elif aug_type == "night_blur":
        aug_image = apply_night(aug_image)
        aug_image = apply_blur(aug_image)

    elif aug_type == "shadow_noise":
        aug_image = apply_shadow(aug_image)
        aug_image = apply_noise(aug_image)

    return aug_image, aug_labels, aug_type


def add_augmented_auto_check_to_v3_train():
    """
    검수 완료된 auto_check 데이터를 증강해서 dataset_v3/train에 추가.

    중요:
    - 원본 auto_check 데이터는 dataset_v2 생성 때 이미 train에 auto_ 접두사로 들어간 상태라고 가정.
    - 여기서는 증강본만 v3aug_ 접두사로 추가.
    - valid/test에는 절대 추가하지 않음.
    """
    check_path(AUTO_IMAGES_DIR, "auto_check/images")
    check_path(AUTO_LABELS_DIR, "auto_check/labels")

    v3_train_images_dir = DATASET_V3_DIR / "train" / "images"
    v3_train_labels_dir = DATASET_V3_DIR / "train" / "labels"

    image_paths = [
        p for p in AUTO_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    total_aug_count = 0
    missing_label_count = 0
    read_fail_count = 0

    aug_type_counts = {}

    print("\n[증강] auto_check 검수 데이터 → dataset_v3/train 증강본 추가")
    print(f"- 원본 검수 이미지 수: {len(image_paths)}")
    print(f"- 이미지당 증강 개수: {AUG_PER_IMAGE}")

    for image_path in image_paths:
        label_path = AUTO_LABELS_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"[경고] 라벨 파일 없음, 제외됨: {image_path.name}")
            missing_label_count += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[경고] 이미지 읽기 실패, 제외됨: {image_path.name}")
            read_fail_count += 1
            continue

        labels = read_yolo_label(label_path)

        if len(labels) == 0:
            print(f"[경고] 라벨 내용 없음, 제외됨: {image_path.name}")
            missing_label_count += 1
            continue

        for aug_idx in range(AUG_PER_IMAGE):
            aug_image, aug_labels, aug_type = make_random_augmentation(image, labels)

            new_stem = f"v3aug_{aug_type}_{aug_idx}_{image_path.stem}"
            new_image_name = f"{new_stem}{image_path.suffix.lower()}"
            new_label_name = f"{new_stem}.txt"

            new_image_path = v3_train_images_dir / new_image_name
            new_label_path = v3_train_labels_dir / new_label_name

            cv2.imwrite(str(new_image_path), aug_image)
            write_yolo_label(new_label_path, aug_labels)

            total_aug_count += 1
            aug_type_counts[aug_type] = aug_type_counts.get(aug_type, 0) + 1

    print(f"- 생성된 증강 이미지 수: {total_aug_count}")
    print(f"- 라벨 누락/라벨 비어있음 제외 수: {missing_label_count}")
    print(f"- 이미지 읽기 실패 수: {read_fail_count}")

    print("\n[증강 타입별 개수]")
    for aug_type, count in sorted(aug_type_counts.items()):
        print(f"- {aug_type}: {count}")


def create_v3_data_yaml():
    """
    dataset_v3용 data.yaml 생성.
    """
    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }

    with open(DATASET_V3_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n[생성] dataset_v3/data.yaml 생성 완료: {DATASET_V3_YAML}")


# ============================================================
# 7. 데이터셋 확인 함수
# ============================================================

def count_files(dataset_dir):
    """
    이미지 수와 라벨 수 확인.
    """
    dataset_dir = Path(dataset_dir)

    print(f"\n[확인] {dataset_dir.name} 이미지/라벨 개수")

    for split in ["train", "valid", "test"]:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"

        image_count = len([
            file for file in images_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ])

        label_count = len([
            file for file in labels_dir.iterdir()
            if file.is_file() and file.suffix.lower() == ".txt"
        ])

        print(f"- {split}: images={image_count}, labels={label_count}")


def count_train_class_objects(dataset_dir):
    """
    train 라벨 기준 클래스별 객체 수 확인.
    """
    dataset_dir = Path(dataset_dir)
    labels_dir = dataset_dir / "train" / "labels"

    class_counts = {class_name: 0 for class_name in CLASS_NAMES}

    for label_file in labels_dir.glob("*.txt"):
        labels = read_yolo_label(label_file)

        for label in labels:
            class_id = int(label[0])

            if 0 <= class_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[class_id]
                class_counts[class_name] += 1
            else:
                print(f"[경고] class_id 범위 초과: {label_file.name} / class_id={class_id}")

    total_objects = sum(class_counts.values())

    print(f"\n[확인] {dataset_dir.name}/train 클래스별 객체 수")

    for class_name, count in class_counts.items():
        ratio = count / total_objects * 100 if total_objects > 0 else 0
        print(f"- {class_name}: {count}개 ({ratio:.2f}%)")

    print(f"\n- 전체 객체 수: {total_objects}개")


# ============================================================
# 8. dataset_v3 생성 전체 함수
# ============================================================

def make_dataset_v3():
    """
    final 모델 학습용 dataset_v3 생성.
    """
    print("========== 1. dataset_v3 생성 시작 ==========")

    check_path(DATASET_V2_DIR, "dataset_v2")
    check_path(DATASET_V2_YAML, "dataset_v2/data.yaml")
    check_path(AUTO_IMAGES_DIR, "auto_check/images")
    check_path(AUTO_LABELS_DIR, "auto_check/labels")

    reset_dataset_v3()
    make_v3_dirs()
    copy_dataset_v2_to_v3()
    add_augmented_auto_check_to_v3_train()
    create_v3_data_yaml()

    count_files(DATASET_V3_DIR)
    count_train_class_objects(DATASET_V3_DIR)

    print("\n========== dataset_v3 생성 완료 ==========")


# ============================================================
# 9. model1_v3 final 학습 함수
# ============================================================

def train_model1_v3_final():
    """
    model1_v2 best.pt를 시작점으로 dataset_v3 학습.

    Model1_v3을 마지막 성능 개선 모델로 판단하는 경우:
    - 너무 적은 epoch보다 충분한 epoch 설정
    - patience로 과적합 전 조기 종료
    - valid 성능 기준 best.pt 저장
    """
    print("\n========== 2. model1_v3 final 학습 시작 ==========")

    check_path(V2_BEST_PT, "model1_v2 best.pt")
    check_path(DATASET_V3_YAML, "dataset_v3/data.yaml")

    model = YOLO(str(V2_BEST_PT))

    results = model.train(
        data=str(DATASET_V3_YAML),

        # final 개선 모델 기준
        # 데이터가 늘어났으므로 150 정도까지 열어두고 patience로 조기 종료
        epochs=150,

        # CPU면 매우 오래 걸림.
        # GPU 메모리 부족 시 16 → 8 → 4 순서로 낮추기
        batch=16,

        # 기존 v1/v2와 동일하게 유지해야 비교가 공정함
        imgsz=640,

        # final 학습이므로 v2보다 조금 더 여유 있게 설정
        # 25 epoch 동안 개선이 없으면 조기 종료
        patience=25,

        # YOLO 기본 증강도 사용
        # 단, 우리가 이미 야간/노이즈/그림자 증강을 넣었기 때문에 너무 과한 값은 피함
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.35,
        degrees=3.0,
        translate=0.08,
        scale=0.35,
        shear=1.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.7,
        close_mosaic=15,

        project=str(V3_TRAIN_PROJECT),
        name=V3_TRAIN_NAME,
        exist_ok=True,
        verbose=True,

        # GPU 사용
        # GPU가 안 잡히면 device="cpu"로 변경
        device=0
    )

    print("\n========== model1_v3 final 학습 완료 ==========")

    return results


# ============================================================
# 10. 모델 평가 함수
# ============================================================

def evaluate_model(model_path, data_yaml, model_name):
    """
    test 데이터 기준 모델 평가.
    """
    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=16,
        project=str(OUTPUT_DIR),
        name=f"{model_name}_test_eval",
        exist_ok=True,
        verbose=True,
        device=0
    )

    summary = {
        "model": model_name,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map)
    }

    names = metrics.names
    class_rows = []

    for class_idx, class_name in names.items():
        row = {
            "model": model_name,
            "class_id": class_idx,
            "class_name": class_name,
            "precision": float(metrics.box.p[class_idx]) if len(metrics.box.p) > class_idx else None,
            "recall": float(metrics.box.r[class_idx]) if len(metrics.box.r) > class_idx else None,
            "mAP50": float(metrics.box.ap50[class_idx]) if len(metrics.box.ap50) > class_idx else None,
            "mAP50-95": float(metrics.box.ap[class_idx]) if len(metrics.box.ap) > class_idx else None
        }

        class_rows.append(row)

    class_df = pd.DataFrame(class_rows)

    return summary, class_df


# ============================================================
# 11. 시각화 함수
# ============================================================

def save_overall_metric_chart(summary_df):
    """
    model1_v2와 model1_v3 final 전체 성능 비교.
    """
    metrics = ["precision", "recall", "mAP50", "mAP50-95"]

    plot_df = summary_df.set_index("model")[metrics].T

    plt.figure(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=plt.gca())

    plt.title("Model1 v2 vs Model1 v3 Final Overall Performance")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Model")
    plt.tight_layout()

    save_path = OUTPUT_DIR / "model1_v2_v3_overall_metric_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[저장 완료] 전체 성능 비교 그래프: {save_path}")


def save_class_map50_chart(class_df):
    """
    클래스별 mAP50 비교.
    """
    pivot_df = class_df.pivot(
        index="class_name",
        columns="model",
        values="mAP50"
    )

    plt.figure(figsize=(12, 7))
    pivot_df.plot(kind="bar", ax=plt.gca())

    plt.title("Class-wise mAP50 Comparison")
    plt.xlabel("Class")
    plt.ylabel("mAP50")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Model")
    plt.tight_layout()

    save_path = OUTPUT_DIR / "model1_v2_v3_class_map50_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[저장 완료] 클래스별 mAP50 비교 그래프: {save_path}")


def save_class_recall_chart(class_df):
    """
    클래스별 Recall 비교.
    """
    pivot_df = class_df.pivot(
        index="class_name",
        columns="model",
        values="recall"
    )

    plt.figure(figsize=(12, 7))
    pivot_df.plot(kind="bar", ax=plt.gca())

    plt.title("Class-wise Recall Comparison")
    plt.xlabel("Class")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Model")
    plt.tight_layout()

    save_path = OUTPUT_DIR / "model1_v2_v3_class_recall_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[저장 완료] 클래스별 Recall 비교 그래프: {save_path}")


def save_improvement_chart(class_df):
    """
    v3 final이 v2 대비 얼마나 개선되었는지 계산.
    """
    pivot_df = class_df.pivot(
        index="class_name",
        columns="model",
        values="mAP50"
    )

    if "model1_v2" not in pivot_df.columns or "model1_v3_final" not in pivot_df.columns:
        print("[경고] model1_v2 또는 model1_v3_final 컬럼이 없어 개선 그래프를 만들 수 없습니다.")
        return

    pivot_df["mAP50_improvement"] = pivot_df["model1_v3_final"] - pivot_df["model1_v2"]
    pivot_df = pivot_df.sort_values("mAP50_improvement", ascending=False)

    plt.figure(figsize=(12, 7))
    pivot_df["mAP50_improvement"].plot(kind="bar", ax=plt.gca())

    plt.title("mAP50 Improvement by Class: v3 Final - v2")
    plt.xlabel("Class")
    plt.ylabel("mAP50 Difference")
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "model1_v2_v3_class_map50_improvement.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[저장 완료] 클래스별 mAP50 개선 그래프: {save_path}")


# ============================================================
# 12. 보고서 생성 함수
# ============================================================

def make_report(summary_df, class_df):
    """
    model1_v2 vs model1_v3_final 비교 보고서 생성.
    """
    v2 = summary_df[summary_df["model"] == "model1_v2"].iloc[0]
    v3 = summary_df[summary_df["model"] == "model1_v3_final"].iloc[0]

    precision_diff = v3["precision"] - v2["precision"]
    recall_diff = v3["recall"] - v2["recall"]
    map50_diff = v3["mAP50"] - v2["mAP50"]
    map_diff = v3["mAP50-95"] - v2["mAP50-95"]

    pivot_df = class_df.pivot(
        index="class_name",
        columns="model",
        values="mAP50"
    )

    if "model1_v2" in pivot_df.columns and "model1_v3_final" in pivot_df.columns:
        pivot_df["mAP50_improvement"] = pivot_df["model1_v3_final"] - pivot_df["model1_v2"]
        best_classes = pivot_df.sort_values("mAP50_improvement", ascending=False).head(3)
        weak_classes = pivot_df.sort_values("mAP50_improvement", ascending=True).head(3)
    else:
        best_classes = pd.DataFrame()
        weak_classes = pd.DataFrame()

    report_path = OUTPUT_DIR / "model1_v2_v3_final_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model1 v2 vs Model1 v3 Final 성능 비교 보고서\n\n")

        f.write("## 1. 비교 목적\n\n")
        f.write("- Model1_v3을 마지막 모델 성능 개선 단계로 설정한다.\n")
        f.write("- 검수 완료된 auto_check 데이터 2000개를 기반으로 야간전투 대응 증강을 적용한다.\n")
        f.write("- dataset_v2와 동일한 test 데이터셋을 기준으로 평가하여 성능 개선 여부를 확인한다.\n")
        f.write("- valid/test 데이터는 증강하지 않고 train 데이터에만 증강을 적용한다.\n\n")

        f.write("## 2. 데이터셋 구성\n\n")
        f.write("- 기준 데이터셋: dataset_v2\n")
        f.write("- 최종 데이터셋: dataset_v3\n")
        f.write("- 증강 대상: auto_check/images, auto_check/labels의 검수 완료 데이터\n")
        f.write("- 증강 방식: brightness, night, heavy_dark, shadow, noise, blur, hsv, flip 등\n")
        f.write("- 검증/테스트 데이터: dataset_v2의 valid/test를 그대로 유지\n\n")

        f.write("## 3. 전체 성능 비교\n\n")
        f.write("| Model | Precision | Recall | mAP50 | mAP50-95 |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")

        for _, row in summary_df.iterrows():
            f.write(
                f"| {row['model']} | "
                f"{row['precision']:.4f} | "
                f"{row['recall']:.4f} | "
                f"{row['mAP50']:.4f} | "
                f"{row['mAP50-95']:.4f} |\n"
            )

        f.write("\n## 4. v3 Final 개선 결과\n\n")
        f.write(f"- Precision 변화: {precision_diff:+.4f}\n")
        f.write(f"- Recall 변화: {recall_diff:+.4f}\n")
        f.write(f"- mAP50 변화: {map50_diff:+.4f}\n")
        f.write(f"- mAP50-95 변화: {map_diff:+.4f}\n\n")

        f.write("## 5. 해석 기준\n\n")
        f.write("- Precision 상승은 잘못 탐지하는 비율 감소로 해석 가능하다.\n")
        f.write("- Recall 상승은 실제 객체를 놓치는 비율 감소로 해석 가능하다.\n")
        f.write("- mAP50 상승은 전체 탐지 성능 개선으로 해석 가능하다.\n")
        f.write("- mAP50-95 상승은 박스 위치 정확도까지 포함한 엄격한 성능 개선으로 해석 가능하다.\n")
        f.write("- 야간전투 대응 목적에서는 전체 mAP뿐 아니라 Tank, Human 계열 Recall 변화도 중요하게 확인한다.\n\n")

        f.write("## 6. 클래스별 개선 상위 항목\n\n")

        if not best_classes.empty:
            f.write("| Class | v2 mAP50 | v3 Final mAP50 | Difference |\n")
            f.write("| --- | ---: | ---: | ---: |\n")

            for class_name, row in best_classes.iterrows():
                f.write(
                    f"| {class_name} | "
                    f"{row['model1_v2']:.4f} | "
                    f"{row['model1_v3_final']:.4f} | "
                    f"{row['mAP50_improvement']:+.4f} |\n"
                )
        else:
            f.write("- 클래스별 개선 항목을 계산할 수 없음\n")

        f.write("\n## 7. 클래스별 확인 필요 항목\n\n")

        if not weak_classes.empty:
            f.write("| Class | v2 mAP50 | v3 Final mAP50 | Difference |\n")
            f.write("| --- | ---: | ---: | ---: |\n")

            for class_name, row in weak_classes.iterrows():
                f.write(
                    f"| {class_name} | "
                    f"{row['model1_v2']:.4f} | "
                    f"{row['model1_v3_final']:.4f} | "
                    f"{row['mAP50_improvement']:+.4f} |\n"
                )
        else:
            f.write("- 클래스별 하락 항목을 계산할 수 없음\n")

        f.write("\n## 8. 저장된 결과 파일\n\n")
        f.write("- model1_v2_v3_summary.csv\n")
        f.write("- model1_v2_v3_class_metrics.csv\n")
        f.write("- model1_v2_v3_overall_metric_comparison.png\n")
        f.write("- model1_v2_v3_class_map50_comparison.png\n")
        f.write("- model1_v2_v3_class_recall_comparison.png\n")
        f.write("- model1_v2_v3_class_map50_improvement.png\n")
        f.write("- model1_v2_v3_final_report.md\n")

    print(f"[저장 완료] 보고서 파일: {report_path}")


# ============================================================
# 13. v2 vs v3 final 성능 비교 함수
# ============================================================

def compare_v2_v3_final():
    """
    model1_v2와 model1_v3_final을 동일한 dataset_v3 test 기준으로 비교.

    dataset_v3의 test는 dataset_v2 test를 그대로 복사했기 때문에
    v2와 v3의 성능 비교 기준이 동일하게 유지됨.
    """
    print("\n========== 3. model1_v2 vs model1_v3_final 평가 시작 ==========")

    v3_best_pt = V3_TRAIN_PROJECT / V3_TRAIN_NAME / "weights" / "best.pt"

    check_path(V2_BEST_PT, "model1_v2 best.pt")
    check_path(v3_best_pt, "model1_v3_final best.pt")
    check_path(DATASET_V3_YAML, "dataset_v3/data.yaml")

    print("\n[평가] model1_v2 test 평가")
    v2_summary, v2_class_df = evaluate_model(
        model_path=V2_BEST_PT,
        data_yaml=DATASET_V3_YAML,
        model_name="model1_v2"
    )

    print("\n[평가] model1_v3_final test 평가")
    v3_summary, v3_class_df = evaluate_model(
        model_path=v3_best_pt,
        data_yaml=DATASET_V3_YAML,
        model_name="model1_v3_final"
    )

    summary_df = pd.DataFrame([v2_summary, v3_summary])
    class_df = pd.concat([v2_class_df, v3_class_df], ignore_index=True)

    summary_csv_path = OUTPUT_DIR / "model1_v2_v3_summary.csv"
    class_csv_path = OUTPUT_DIR / "model1_v2_v3_class_metrics.csv"

    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    class_df.to_csv(class_csv_path, index=False, encoding="utf-8-sig")

    print(f"[저장 완료] 전체 성능 CSV: {summary_csv_path}")
    print(f"[저장 완료] 클래스별 성능 CSV: {class_csv_path}")

    save_overall_metric_chart(summary_df)
    save_class_map50_chart(class_df)
    save_class_recall_chart(class_df)
    save_improvement_chart(class_df)
    make_report(summary_df, class_df)

    print("\n========== 평가 및 결과 저장 완료 ==========")
    print(f"결과 저장 폴더: {OUTPUT_DIR}")


# ============================================================
# 14. 전체 실행
# ============================================================

def main():
    """
    전체 실행 순서.

    1. dataset_v3 생성
    2. model1_v2 best.pt 기반 model1_v3_final 학습
    3. model1_v2 vs model1_v3_final 성능 비교
    """

    make_dataset_v3()
    train_model1_v3_final()
    compare_v2_v3_final()

    print("\n========== 전체 완료 ==========")
    print(f"dataset_v3 경로: {DATASET_V3_DIR}")
    print(f"v3 학습 결과 경로: {V3_TRAIN_PROJECT / V3_TRAIN_NAME}")
    print(f"성능 비교 결과 경로: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()