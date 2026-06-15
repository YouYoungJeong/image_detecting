# ============================================================
# find_common_wrong_images.py
# 모델2 이미지 사이즈별 3개 모델 공통 오답 이미지 찾기
# YOLO Classification 모델용
# VSCode / 로컬 py 실행용
# ============================================================

from pathlib import Path
import shutil
import pandas as pd
from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 테스트 데이터셋 경로
# 구조 예시:
# risk_dataset_aug/test/risk_0_safe/*.png
# risk_dataset_aug/test/risk_1_low/*.png
# risk_dataset_aug/test/risk_2_high/*.png
TEST_DIR = BASE_DIR / "risk_dataset_aug" / "test"

# 모델 경로
MODEL_320_PATH = BASE_DIR / "runs" / "classify" / "model2_risk_from_model1_v2_imgsz320" / "weights" / "best.pt"
MODEL_480_PATH = BASE_DIR / "runs" / "classify" / "model2_risk_from_model1_v2_imgsz480" / "weights" / "best.pt"
MODEL_520_PATH = BASE_DIR / "runs" / "classify" / "model2_risk_from_model1_v2_imgsz520" / "weights" / "best.pt"

# 결과 저장 폴더
OUTPUT_DIR = BASE_DIR / "common_wrong_results"
COMMON_WRONG_IMG_DIR = OUTPUT_DIR / "common_wrong_images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMMON_WRONG_IMG_DIR.mkdir(parents=True, exist_ok=True)

# 이미지 확장자
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# ============================================================
# 2. 테스트 이미지 수집
# ============================================================

def get_test_images(test_dir: Path):
    image_paths = []

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        true_label = class_dir.name

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in IMAGE_EXTS:
                image_paths.append({
                    "image_path": img_path,
                    "true_label": true_label
                })

    return image_paths


# ============================================================
# 3. 단일 이미지 예측 함수
# ============================================================

def predict_one(model: YOLO, img_path: Path):
    results = model.predict(
        source=str(img_path),
        verbose=False
    )

    result = results[0]

    pred_idx = int(result.probs.top1)
    pred_label = result.names[pred_idx]
    pred_conf = float(result.probs.top1conf)

    return pred_label, pred_conf


# ============================================================
# 4. 공통 오답 찾기
# ============================================================

def main():
    print("========== 3개 모델 공통 오답 이미지 찾기 시작 ==========")
    print(f"[TEST_DIR] {TEST_DIR}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"테스트 폴더가 없습니다: {TEST_DIR}")

    model_paths = {
        "img320": MODEL_320_PATH,
        "img480": MODEL_480_PATH,
        "img520": MODEL_520_PATH,
    }

    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            raise FileNotFoundError(f"{model_name} 모델 파일이 없습니다: {model_path}")

    print()
    print("[INFO] 모델 로드 중")
    model_320 = YOLO(str(MODEL_320_PATH))
    model_480 = YOLO(str(MODEL_480_PATH))
    model_520 = YOLO(str(MODEL_520_PATH))

    test_images = get_test_images(TEST_DIR)

    print(f"[INFO] 테스트 이미지 수: {len(test_images)}")
    print()

    all_rows = []
    common_wrong_rows = []

    for idx, item in enumerate(test_images, start=1):
        img_path = item["image_path"]
        true_label = item["true_label"]

        pred_320, conf_320 = predict_one(model_320, img_path)
        pred_480, conf_480 = predict_one(model_480, img_path)
        pred_520, conf_520 = predict_one(model_520, img_path)

        wrong_320 = pred_320 != true_label
        wrong_480 = pred_480 != true_label
        wrong_520 = pred_520 != true_label

        is_common_wrong = wrong_320 and wrong_480 and wrong_520

        row = {
            "image_path": str(img_path),
            "file_name": img_path.name,
            "true_label": true_label,

            "pred_320": pred_320,
            "conf_320": conf_320,
            "wrong_320": wrong_320,

            "pred_480": pred_480,
            "conf_480": conf_480,
            "wrong_480": wrong_480,

            "pred_520": pred_520,
            "conf_520": conf_520,
            "wrong_520": wrong_520,

            "is_common_wrong": is_common_wrong,
        }

        all_rows.append(row)

        if is_common_wrong:
            common_wrong_rows.append(row)

            # 이미지 복사
            # 파일명에 true/pred 정보를 붙여서 저장
            save_name = (
                f"TRUE_{true_label}"
                f"__320_{pred_320}"
                f"__480_{pred_480}"
                f"__520_{pred_520}"
                f"__{img_path.name}"
            )

            save_path = COMMON_WRONG_IMG_DIR / save_name
            shutil.copy2(img_path, save_path)

        if idx % 20 == 0:
            print(f"[PROGRESS] {idx}/{len(test_images)} 처리 완료")

    # 전체 예측 결과 저장
    all_df = pd.DataFrame(all_rows)
    all_csv_path = OUTPUT_DIR / "all_model_predictions.csv"
    all_df.to_csv(all_csv_path, index=False, encoding="utf-8-sig")

    # 공통 오답 결과 저장
    common_wrong_df = pd.DataFrame(common_wrong_rows)
    common_wrong_csv_path = OUTPUT_DIR / "common_wrong_images.csv"
    common_wrong_df.to_csv(common_wrong_csv_path, index=False, encoding="utf-8-sig")

    print()
    print("========== 결과 요약 ==========")
    print(f"전체 테스트 이미지 수: {len(test_images)}")
    print(f"3개 모델 공통 오답 수: {len(common_wrong_rows)}")
    print()
    print(f"[SAVE] 전체 예측 결과 CSV: {all_csv_path}")
    print(f"[SAVE] 공통 오답 CSV: {common_wrong_csv_path}")
    print(f"[SAVE] 공통 오답 이미지 폴더: {COMMON_WRONG_IMG_DIR}")
    print()
    print("========== 완료 ==========")


if __name__ == "__main__":
    main()