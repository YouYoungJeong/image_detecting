# ============================================================
# OpenMP 중복 로드 오류 임시 해결
# Windows + Anaconda + PyTorch/Ultralytics 환경에서 발생 가능
# 반드시 torch, ultralytics import보다 먼저 작성
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ============================================================
# test_model2_risk.py
# 모델2 위험도 분류 best.pt 불러와서 test 검증 코드
# VSCode / 로컬 py 실행용
# ============================================================

from pathlib import Path
import math
import textwrap

import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_AUG_DIR = BASE_DIR / "risk_dataset_aug"
TEST_DIR = DATASET_AUG_DIR / "test"

RUNS_DIR = BASE_DIR / "runs"
CLASSIFY_RUNS_DIR = RUNS_DIR / "classify"

MODEL2_RUN_NAME = "model2_risk_v1"
MODEL2_RUN_DIR = CLASSIFY_RUNS_DIR / MODEL2_RUN_NAME
MODEL2_BEST_PATH = MODEL2_RUN_DIR / "weights" / "best.pt"

TEST_RUN_NAME = f"{MODEL2_RUN_NAME}_test"
TEST_RUN_DIR = CLASSIFY_RUNS_DIR / TEST_RUN_NAME

TEST_VISUAL_DIR = CLASSIFY_RUNS_DIR / f"{MODEL2_RUN_NAME}_test_visuals"

ALL_SAVE_DIR = TEST_VISUAL_DIR / "all"
WRONG_SAVE_DIR = TEST_VISUAL_DIR / "wrong"
LOW_SAVE_DIR = TEST_VISUAL_DIR / "focus_low"
GRID_SAVE_ROOT = TEST_VISUAL_DIR / "grids_by_error_type"

CSV_PATH = TEST_VISUAL_DIR / "test_prediction_results.csv"


# ============================================================
# 2. 폴더 생성
# ============================================================

def make_dirs():
    TEST_VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    ALL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    WRONG_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOW_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    GRID_SAVE_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. GPU / CPU 확인
# ============================================================

def get_device():
    print("========== Device Check ==========")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        return 0
    else:
        print("GPU를 사용할 수 없습니다. CPU로 검증합니다.")
        return "cpu"


# ============================================================
# 4. 모델 및 데이터 경로 확인
# ============================================================

def check_paths():
    if not MODEL2_BEST_PATH.exists():
        raise FileNotFoundError(f"학습된 best.pt를 찾을 수 없습니다: {MODEL2_BEST_PATH}")

    if not DATASET_AUG_DIR.exists():
        raise FileNotFoundError(f"risk_dataset_aug 폴더가 없습니다: {DATASET_AUG_DIR}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"test 폴더가 없습니다: {TEST_DIR}")

    print("========== Path Check ==========")
    print("MODEL2_BEST_PATH:", MODEL2_BEST_PATH)
    print("DATASET_AUG_DIR:", DATASET_AUG_DIR)
    print("TEST_DIR:", TEST_DIR)


# ============================================================
# 5. best.pt 불러와서 test 검증
# ============================================================

def run_test_validation():
    print("========== Model2 Test Validation Start ==========")

    check_paths()
    device = get_device()

    model = YOLO(str(MODEL2_BEST_PATH))

    metrics = model.val(
        data=str(DATASET_AUG_DIR),
        split="test",
        imgsz=224,
        batch=32,
        device=device,

        project=str(CLASSIFY_RUNS_DIR),
        name=TEST_RUN_NAME,
        exist_ok=True,
    )

    print("\n========== Model2 Test Validation Done ==========")
    print("test 검증 결과 폴더:", TEST_RUN_DIR)

    return metrics


# ============================================================
# 6. 폰트 설정
# ============================================================

def get_font(size=22):
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for fp in candidates:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size=size)

    return ImageFont.load_default()


font_title = get_font(28)
font_text = get_font(22)
font_small = get_font(18)


# ============================================================
# 7. 예측 결과 이미지 생성 함수
# ============================================================

def make_result_image(img_path, true_label, pred_label, conf, top3_text, correct, save_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    panel_h = 170

    canvas = Image.new("RGB", (w, h + panel_h), color=(255, 255, 255))
    canvas.paste(img, (0, panel_h))

    draw = ImageDraw.Draw(canvas)

    if correct:
        bar_color = (60, 170, 90)
        status_text = "CORRECT"
    else:
        bar_color = (210, 70, 70)
        status_text = "WRONG"

    draw.rectangle([(0, 0), (w, 45)], fill=bar_color)
    draw.text((15, 8), status_text, fill=(255, 255, 255), font=font_title)

    draw.text((15, 55), f"File: {img_path.name}", fill=(0, 0, 0), font=font_text)
    draw.text((15, 85), f"True: {true_label}", fill=(0, 0, 0), font=font_text)
    draw.text((15, 112), f"Pred: {pred_label} | Conf: {conf:.4f}", fill=(0, 0, 0), font=font_text)

    wrapped_top3 = textwrap.fill(f"Top3: {top3_text}", width=90)
    draw.text((15, 140), wrapped_top3, fill=(30, 30, 30), font=font_small)

    canvas.save(save_path)


# ============================================================
# 8. 이미지별 예측 결과 저장
# ============================================================

def save_prediction_visuals():
    print("========== Save Prediction Visuals Start ==========")

    check_paths()
    make_dirs()

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    test_images = [
        p for p in TEST_DIR.rglob("*")
        if p.suffix.lower() in image_exts
    ]

    if len(test_images) == 0:
        raise FileNotFoundError(f"test 이미지가 없습니다: {TEST_DIR}")

    print("test 이미지 개수:", len(test_images))

    model = YOLO(str(MODEL2_BEST_PATH))
    names = model.names

    print("class names:", names)

    records = []

    for idx, img_path in enumerate(sorted(test_images), 1):
        true_label = img_path.parent.name

        result = model.predict(
            source=str(img_path),
            imgsz=224,
            verbose=False
        )[0]

        probs = result.probs

        pred_idx = int(probs.top1)
        pred_label = names[pred_idx]
        conf = float(probs.top1conf)

        prob_array = probs.data.cpu().numpy()
        top_indices = prob_array.argsort()[::-1][:3]

        top3_parts = [
            f"{names[int(i)]}: {prob_array[int(i)]:.4f}"
            for i in top_indices
        ]

        top3_text = " | ".join(top3_parts)

        correct = true_label == pred_label

        save_name = (
            f"{idx:04d}"
            f"__file_{img_path.stem}"
            f"__true_{true_label}"
            f"__pred_{pred_label}"
            f"__{'ok' if correct else 'wrong'}"
            f"{img_path.suffix}"
        )

        all_save_path = ALL_SAVE_DIR / save_name

        make_result_image(
            img_path=img_path,
            true_label=true_label,
            pred_label=pred_label,
            conf=conf,
            top3_text=top3_text,
            correct=correct,
            save_path=all_save_path
        )

        if not correct:
            wrong_save_path = WRONG_SAVE_DIR / save_name

            make_result_image(
                img_path=img_path,
                true_label=true_label,
                pred_label=pred_label,
                conf=conf,
                top3_text=top3_text,
                correct=correct,
                save_path=wrong_save_path
            )

        if true_label == "risk_1_low" or pred_label == "risk_1_low":
            low_save_path = LOW_SAVE_DIR / save_name

            make_result_image(
                img_path=img_path,
                true_label=true_label,
                pred_label=pred_label,
                conf=conf,
                top3_text=top3_text,
                correct=correct,
                save_path=low_save_path
            )

        records.append({
            "image_path": str(img_path),
            "file_name": img_path.name,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": conf,
            "correct": correct,
            "top1": pred_label,
            "top1_conf": conf,
            "top3": top3_text,
            "visual_file": save_name
        })

    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print("\n========== Save Prediction Visuals Done ==========")
    print("전체 결과 이미지 폴더:", ALL_SAVE_DIR)
    print("오답 이미지 폴더:", WRONG_SAVE_DIR)
    print("risk_1_low 관련 폴더:", LOW_SAVE_DIR)
    print("결과 CSV:", CSV_PATH)

    print("\n전체 test 개수:", len(df))
    print("정답 개수:", df["correct"].sum())
    print("오답 개수:", (~df["correct"]).sum())
    print("정확도:", df["correct"].mean())

    return df


# ============================================================
# 9. 오답 종류별 4x4 grid 생성
# ============================================================

def create_grid_page(image_paths, save_path, title, page_idx, total_pages):
    cols = 4
    rows = 4

    thumb_w = 360
    thumb_h = 360
    caption_h = 45
    header_h = 80
    margin = 20

    canvas_w = cols * thumb_w + (cols + 1) * margin
    canvas_h = header_h + rows * (thumb_h + caption_h) + (rows + 1) * margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    full_title = f"{title} | Page {page_idx}/{total_pages}"
    draw.text((20, 22), full_title, fill=(0, 0, 0), font=font_title)

    for i, img_path in enumerate(image_paths):
        row = i // cols
        col = i % cols

        x = margin + col * (thumb_w + margin)
        y = header_h + margin + row * (thumb_h + caption_h + margin)

        img = Image.open(img_path).convert("RGB")
        img.thumbnail((thumb_w, thumb_h))

        paste_x = x + (thumb_w - img.size[0]) // 2
        paste_y = y + (thumb_h - img.size[1]) // 2

        canvas.paste(img, (paste_x, paste_y))

        draw.rectangle(
            [x, y, x + thumb_w, y + thumb_h],
            outline=(180, 180, 180),
            width=2
        )

        caption = img_path.name[:45]
        draw.text((x + 5, y + thumb_h + 8), caption, fill=(0, 0, 0), font=font_small)

    canvas.save(save_path, quality=95)
    print("저장 완료:", save_path)


def create_error_type_grids():
    print("========== Create Error Type Grids Start ==========")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV 파일이 없습니다. 먼저 save_prediction_visuals()를 실행하세요: {CSV_PATH}")

    if not ALL_SAVE_DIR.exists():
        raise FileNotFoundError(f"all 이미지 폴더가 없습니다: {ALL_SAVE_DIR}")

    df = pd.read_csv(CSV_PATH)

    wrong_df = df[df["correct"] == False].copy()

    print("전체 test 개수:", len(df))
    print("전체 오답 개수:", len(wrong_df))

    if len(wrong_df) == 0:
        print("오답 데이터가 없습니다. grid를 생성하지 않습니다.")
        return None

    summary_records = []

    grouped = wrong_df.groupby(["true_label", "pred_label"])

    for (true_label, pred_label), group_df in grouped:
        error_type_name = f"true_{true_label}__pred_{pred_label}"
        save_dir = GRID_SAVE_ROOT / error_type_name
        save_dir.mkdir(parents=True, exist_ok=True)

        visual_files = []

        for _, row in group_df.iterrows():
            visual_path = ALL_SAVE_DIR / row["visual_file"]

            if visual_path.exists():
                visual_files.append(visual_path)

        print("\n========================================")
        print("오답 종류:", f"{true_label} -> {pred_label}")
        print("CSV 기준 개수:", len(group_df))
        print("이미지 파일 개수:", len(visual_files))

        if len(visual_files) == 0:
            print("[WARN] 해당 오답 종류의 시각화 이미지를 찾지 못했습니다.")
            continue

        images_per_page = 16
        total_pages = math.ceil(len(visual_files) / images_per_page)

        for page_idx in range(total_pages):
            start = page_idx * images_per_page
            end = start + images_per_page

            page_files = visual_files[start:end]
            save_path = save_dir / f"grid_{page_idx + 1:02d}.jpg"

            title = f"{true_label} -> {pred_label} count={len(visual_files)}"

            create_grid_page(
                image_paths=page_files,
                save_path=save_path,
                title=title,
                page_idx=page_idx + 1,
                total_pages=total_pages
            )

        summary_records.append({
            "true_label": true_label,
            "pred_label": pred_label,
            "count": len(group_df),
            "grid_folder": str(save_dir)
        })

    summary_df = pd.DataFrame(summary_records)

    summary_csv_path = GRID_SAVE_ROOT / "error_type_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    print("\n========== Create Error Type Grids Done ==========")
    print("저장 위치:", GRID_SAVE_ROOT)
    print("요약 CSV:", summary_csv_path)

    print(summary_df)

    return summary_df


# ============================================================
# 10. 전체 테스트 실행
# ============================================================

def test_model2():
    run_test_validation()
    save_prediction_visuals()
    create_error_type_grids()


# ============================================================
# 11. 실행
# ============================================================

if __name__ == "__main__":
    test_model2()