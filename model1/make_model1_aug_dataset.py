# pip install ultralytics albumentations opencv-python pandas matplotlib pyyaml tqdm

# ============================================================
# make_model1_aug_dataset_only.py
# Model1 증강 데이터셋 생성 전용 코드
# ============================================================
# 목적:
# - dataset_v1을 복사하여 dataset_v1_yolo26n_aug 생성
# - data.yaml을 4개 클래스 기준 데이터셋 경로에 맞게 수정
# - train 데이터에만 야간/저조도/노이즈/연기/모션블러 증강 적용
# - valid/test 데이터는 원본 유지
# - 증강 전/후 데이터 개수 CSV 저장
# - 증강 계획 CSV 저장
# - 증강 로그 CSV 저장
#
# 중요:
# - 이 코드는 모델 학습을 하지 않음
# - 이미지 위치를 바꾸는 기하학적 증강을 사용하지 않음
# - 따라서 YOLO bbox 라벨 좌표는 원본 그대로 복사함
# ============================================================

from pathlib import Path
import shutil
import random
import json
from datetime import datetime
from collections import Counter

import cv2
import yaml
import pandas as pd
from tqdm import tqdm
import albumentations as A


# ============================================================
# 0. 기본 설정
# ============================================================

SEED = 42
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent

# 원본 데이터셋
DATASET_V1_DIR = BASE_DIR / "dataset_v1"

# 증강 데이터셋 저장 위치
DATASET_AUG_DIR = BASE_DIR / "dataset_v1_yolo26n_aug"

# data.yaml 경로
ORIGINAL_DATA_YAML = DATASET_V1_DIR / "data.yaml"
AUG_DATA_YAML = DATASET_AUG_DIR / "data.yaml"

# 결과 파일 이름 접두사
OUTPUT_PREFIX = "model1_yolo26n_aug"

# 기존 증강 데이터셋 삭제 후 새로 생성 여부
# True  : 기존 dataset_v1_yolo26n_aug 삭제 후 새로 생성
# False : 기존 폴더가 있으면 삭제하지 않음
RESET_OUTPUT_DATASET = True

# 증강 배율
# 1.0이면 train 원본 이미지 수만큼 증강 이미지 추가
# 예: train 8316장 -> 증강 이미지 8316장 추가
AUGMENT_MULTIPLIER = 1.0


# ============================================================
# 1. 유틸 함수
# ============================================================

def get_image_files(image_dir: Path):
    """
    이미지 파일 목록 반환
    """
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    files = []

    for ext in exts:
        files.extend(image_dir.glob(f"*{ext}"))
        files.extend(image_dir.glob(f"*{ext.upper()}"))

    return sorted(files)


def copy_dataset_structure(src: Path, dst: Path):
    """
    dataset_v1 전체를 dataset_v1_yolo26n_aug로 복사한다.
    이후 train/images, train/labels에 증강 이미지만 추가한다.
    """
    if dst.exists():
        if RESET_OUTPUT_DATASET:
            print(f"[INFO] 기존 증강 데이터셋 삭제: {dst}")
            shutil.rmtree(dst)
        else:
            print(f"[INFO] 기존 증강 데이터셋 유지: {dst}")
            return

    print("[INFO] 데이터셋 복사 시작")
    print(f"  FROM: {src}")
    print(f"  TO  : {dst}")

    shutil.copytree(src, dst)

    print("[INFO] 데이터셋 복사 완료")


def find_split_dirs(dataset_dir: Path, split: str):
    """
    YOLO 데이터셋 split 경로 찾기
    """
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"

    if image_dir.exists() and label_dir.exists():
        return image_dir, label_dir

    raise FileNotFoundError(
        f"[ERROR] {split} 경로를 찾을 수 없습니다.\n"
        f"images: {image_dir}\n"
        f"labels: {label_dir}"
    )


def read_yolo_label(label_path: Path):
    """
    YOLO label 읽기
    format: class_id cx cy w h
    """
    bboxes = []
    class_labels = []

    if not label_path.exists():
        return bboxes, class_labels

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:
            cls_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue

        if w <= 0 or h <= 0:
            continue

        bboxes.append([cx, cy, w, h])
        class_labels.append(cls_id)

    return bboxes, class_labels


def count_dataset_objects(dataset_dir: Path):
    """
    train/valid/val/test 이미지 수와 객체 수 기록
    """
    rows = []

    for split in ["train", "valid", "val", "test"]:
        split_dir = dataset_dir / split

        if not split_dir.exists():
            continue

        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"

        if not image_dir.exists() or not label_dir.exists():
            continue

        image_files = get_image_files(image_dir)

        object_count = 0
        class_counter = Counter()

        for img_path in image_files:
            label_path = label_dir / f"{img_path.stem}.txt"
            _, class_labels = read_yolo_label(label_path)

            object_count += len(class_labels)
            class_counter.update(class_labels)

        row = {
            "split": split,
            "image_count": len(image_files),
            "object_count": object_count
        }

        for cls_id, count in sorted(class_counter.items()):
            row[f"class_{cls_id}_objects"] = count

        rows.append(row)

    return pd.DataFrame(rows)


def save_dataset_count_report(dataset_dir: Path, save_path: Path):
    """
    데이터셋 개수 리포트 저장
    """
    df = count_dataset_objects(dataset_dir)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df


# ============================================================
# 2. Albumentations 증강 모듈 정의
# ============================================================

AUGMENT_MODULES = {
    "RandomBrightnessContrast": {
        "weight": 0.225,
        "transform": A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.45, -0.10),
                contrast_limit=(-0.25, 0.25),
                p=1.0
            )
        ])
    },

    "RandomGamma": {
        "weight": 0.175,
        "transform": A.Compose([
            A.RandomGamma(
                gamma_limit=(45, 85),
                p=1.0
            )
        ])
    },

    "GaussNoise": {
        "weight": 0.125,
        "transform": A.Compose([
            A.GaussNoise(
                std_range=(0.02, 0.12),
                mean_range=(0.0, 0.0),
                per_channel=True,
                noise_scale_factor=1.0,
                p=1.0
            )
        ])
    },

    "ISONoise": {
        "weight": 0.125,
        "transform": A.Compose([
            A.ISONoise(
                color_shift=(0.01, 0.08),
                intensity=(0.1, 0.5),
                p=1.0
            )
        ])
    },

    "RandomFog": {
        "weight": 0.125,
        "transform": A.Compose([
            A.RandomFog(
                fog_coef_range=(0.10, 0.35),
                alpha_coef=0.08,
                p=1.0
            )
        ])
    },

    "CoarseDropout": {
        "weight": 0.075,
        "transform": A.Compose([
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(16, 48),
                hole_width_range=(16, 48),
                fill=0,
                p=1.0
            )
        ])
    },

    "MotionBlur": {
        "weight": 0.125,
        "transform": A.Compose([
            A.MotionBlur(
                blur_limit=(3, 9),
                p=1.0
            )
        ])
    },

    "GaussianBlur": {
        "weight": 0.075,
        "transform": A.Compose([
            A.GaussianBlur(
                blur_limit=(3, 7),
                p=1.0
            )
        ])
    },

    "Defocus": {
        "weight": 0.05,
        "transform": A.Compose([
            A.Defocus(
                radius=(2, 5),
                alias_blur=(0.1, 0.4),
                p=1.0
            )
        ])
    },

    "RandomShadow": {
        "weight": 0.075,
        "transform": A.Compose([
            A.RandomShadow(
                shadow_roi=(0, 0.3, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=1.0
            )
        ])
    },

    "HueSaturationValue": {
        "weight": 0.05,
        "transform": A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0
            )
        ])
    },

    "RGBShift": {
        "weight": 0.05,
        "transform": A.Compose([
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=1.0
            )
        ])
    },

    "ImageCompression": {
        "weight": 0.075,
        "transform": A.Compose([
            A.ImageCompression(
                quality_range=(35, 75),
                p=1.0
            )
        ])
    },

    "CLAHE": {
        "weight": 0.05,
        "transform": A.Compose([
            A.CLAHE(
                clip_limit=(1, 4),
                tile_grid_size=(8, 8),
                p=1.0
            )
        ])
    },
}


# ============================================================
# 3. 증강 계획 생성
# ============================================================

def normalize_weights(modules: dict):
    total = sum(item["weight"] for item in modules.values())

    normalized = {}
    for name, item in modules.items():
        normalized[name] = item["weight"] / total

    return normalized


def make_augmentation_plan(train_image_count: int):
    """
    전체 증강 이미지 개수와 모듈별 생성 개수 산출
    """
    total_aug_count = int(train_image_count * AUGMENT_MULTIPLIER)
    normalized_weights = normalize_weights(AUGMENT_MODULES)

    plan = {}
    assigned = 0

    names = list(normalized_weights.keys())

    for name in names[:-1]:
        count = int(total_aug_count * normalized_weights[name])
        plan[name] = count
        assigned += count

    plan[names[-1]] = total_aug_count - assigned

    return plan


def save_augmentation_plan_csv(plan: dict, save_path: Path):
    rows = []
    total = sum(plan.values())

    for name, count in plan.items():
        rows.append({
            "augmentation_module": name,
            "target_count": count,
            "actual_ratio": round(count / total, 4) if total > 0 else 0
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df


# ============================================================
# 4. train 데이터 증강
# ============================================================

def augment_train_dataset(dataset_dir: Path, plan: dict):
    """
    dataset_dir/train/images, train/labels에 증강 이미지와 라벨 추가
    """
    train_image_dir, train_label_dir = find_split_dirs(dataset_dir, "train")
    image_files = get_image_files(train_image_dir)

    if len(image_files) == 0:
        raise RuntimeError("[ERROR] train/images에 이미지가 없습니다.")

    augmentation_log = []
    module_counter = Counter()
    fail_counter = Counter()

    print("\n========== Train Augmentation 시작 ==========")
    print(f"[INFO] 원본 train 이미지 수: {len(image_files)}")
    print(f"[INFO] 추가 증강 목표 수: {sum(plan.values())}")

    for module_name, target_count in plan.items():
        transform = AUGMENT_MODULES[module_name]["transform"]

        print(f"\n[{module_name}] target={target_count}")

        success_count = 0
        attempt_count = 0
        max_attempts = max(target_count * 5, 10)

        while success_count < target_count and attempt_count < max_attempts:
            attempt_count += 1

            img_path = random.choice(image_files)
            label_path = train_label_dir / f"{img_path.stem}.txt"

            bboxes, class_labels = read_yolo_label(label_path)

            if len(bboxes) == 0:
                fail_counter[f"{module_name}_no_label"] += 1
                continue

            image_bgr = cv2.imread(str(img_path))

            if image_bgr is None:
                fail_counter[f"{module_name}_read_fail"] += 1
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            try:
                augmented = transform(image=image_rgb)
                aug_image_rgb = augmented["image"]

                new_stem = f"{img_path.stem}_aug_{module_name}_{success_count:05d}"
                new_img_path = train_image_dir / f"{new_stem}.jpg"
                new_label_path = train_label_dir / f"{new_stem}.txt"

                aug_image_bgr = cv2.cvtColor(aug_image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(new_img_path), aug_image_bgr)

                shutil.copy2(label_path, new_label_path)

                success_count += 1
                module_counter[module_name] += 1

                augmentation_log.append({
                    "module": module_name,
                    "source_image": img_path.name,
                    "aug_image": new_img_path.name,
                    "source_label": label_path.name,
                    "aug_label": new_label_path.name,
                    "object_count": len(class_labels)
                })

            except Exception as e:
                fail_counter[f"{module_name}_error"] += 1
                print(f"[ERROR] {module_name} 증강 실패: {img_path.name} / {e}")
                continue

        print(f"[{module_name}] success={success_count}, attempts={attempt_count}")

    log_df = pd.DataFrame(augmentation_log)
    log_path = BASE_DIR / f"{OUTPUT_PREFIX}_augmentation_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    print("\n========== Train Augmentation 완료 ==========")

    print("\n[INFO] 모듈별 성공 개수")
    for k, v in module_counter.items():
        print(f"  {k}: {v}")

    print("\n[INFO] 실패/스킵 카운트")
    for k, v in fail_counter.items():
        print(f"  {k}: {v}")

    print(f"\n[INFO] 증강 로그 저장: {log_path}")

    return module_counter, fail_counter, log_path


# ============================================================
# 5. data.yaml 수정
# ============================================================

def update_data_yaml_for_aug_dataset():
    """
    dataset_v1의 data.yaml을 기반으로 증강 데이터셋 내부 경로를 바라보도록 수정
    """
    if not ORIGINAL_DATA_YAML.exists():
        raise FileNotFoundError(f"[ERROR] data.yaml이 없습니다: {ORIGINAL_DATA_YAML}")

    with open(ORIGINAL_DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["train"] = "train/images"

    if (DATASET_AUG_DIR / "valid" / "images").exists():
        data["val"] = "valid/images"
    elif (DATASET_AUG_DIR / "val" / "images").exists():
        data["val"] = "val/images"
    else:
        raise FileNotFoundError("[ERROR] valid/images 또는 val/images 경로가 없습니다.")

    if (DATASET_AUG_DIR / "test" / "images").exists():
        data["test"] = "test/images"

    with open(AUG_DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] 증강 data.yaml 저장 완료: {AUG_DATA_YAML}")

    return data


def save_augmentation_config(save_path: Path, extra_info: dict):
    config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": "Model1 YOLO26n 증강 데이터셋 생성",
        "original_dataset": str(DATASET_V1_DIR),
        "augmented_dataset": str(DATASET_AUG_DIR),
        "data_yaml": str(AUG_DATA_YAML),
        "augment_multiplier": AUGMENT_MULTIPLIER,
        "seed": SEED,
        "reset_output_dataset": RESET_OUTPUT_DATASET,
        "extra_info": extra_info
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[INFO] 증강 설정 저장: {save_path}")


# ============================================================
# 6. main
# ============================================================

def main():
    print("========== Model1 증강 데이터셋 생성 시작 ==========")

    if not DATASET_V1_DIR.exists():
        raise FileNotFoundError(f"[ERROR] 원본 데이터셋 경로가 없습니다: {DATASET_V1_DIR}")

    if not ORIGINAL_DATA_YAML.exists():
        raise FileNotFoundError(f"[ERROR] 원본 data.yaml 경로가 없습니다: {ORIGINAL_DATA_YAML}")

    print("\n[STEP 1] 데이터셋 복사")
    copy_dataset_structure(DATASET_V1_DIR, DATASET_AUG_DIR)

    print("\n[STEP 2] data.yaml 수정")
    data_yaml_info = update_data_yaml_for_aug_dataset()

    print("\n[STEP 3] 증강 전 데이터 개수 기록")
    before_count_path = BASE_DIR / f"{OUTPUT_PREFIX}_dataset_count_before_aug.csv"
    before_df = save_dataset_count_report(DATASET_AUG_DIR, before_count_path)
    print(before_df)
    print(f"[INFO] 저장: {before_count_path}")

    print("\n[STEP 4] 증강 계획 생성")
    train_image_dir, _ = find_split_dirs(DATASET_AUG_DIR, "train")
    train_images = get_image_files(train_image_dir)

    plan = make_augmentation_plan(len(train_images))

    aug_plan_path = BASE_DIR / f"{OUTPUT_PREFIX}_augmentation_plan.csv"
    plan_df = save_augmentation_plan_csv(plan, aug_plan_path)
    print(plan_df)
    print(f"[INFO] 저장: {aug_plan_path}")

    print("\n[STEP 5] train 데이터 증강")
    module_counter, fail_counter, aug_log_path = augment_train_dataset(DATASET_AUG_DIR, plan)

    print("\n[STEP 6] 증강 후 데이터 개수 기록")
    after_count_path = BASE_DIR / f"{OUTPUT_PREFIX}_dataset_count_after_aug.csv"
    after_df = save_dataset_count_report(DATASET_AUG_DIR, after_count_path)
    print(after_df)
    print(f"[INFO] 저장: {after_count_path}")

    print("\n[STEP 7] 증강 설정 저장")
    config_path = BASE_DIR / f"{OUTPUT_PREFIX}_augmentation_config.json"

    extra_info = {
        "data_yaml_info": data_yaml_info,
        "augmentation_plan": plan,
        "augmentation_success_count": dict(module_counter),
        "augmentation_fail_count": dict(fail_counter),
        "augmentation_log_path": str(aug_log_path),
        "dataset_count_before_aug": str(before_count_path),
        "dataset_count_after_aug": str(after_count_path),
        "augmentation_plan_csv": str(aug_plan_path)
    }

    save_augmentation_config(config_path, extra_info)

    print("\n========== Model1 증강 데이터셋 생성 완료 ==========")
    print(f"[AUG DATASET] {DATASET_AUG_DIR}")
    print(f"[DATA YAML]   {AUG_DATA_YAML}")

    print("\n생성 파일:")
    print(f"  - {before_count_path}")
    print(f"  - {after_count_path}")
    print(f"  - {aug_plan_path}")
    print(f"  - {aug_log_path}")
    print(f"  - {config_path}")


if __name__ == "__main__":
    main()