# ============================================================
# remove_selected_test_images.py
# [X] 표시된 테스트 이미지 제거용 이동 코드
# 원본 삭제가 아니라 removed_test_images 폴더로 이동
# VSCode / 로컬 py 실행용
# ============================================================

from pathlib import Path
import csv
import shutil


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 네가 수정해서 [O], [X] 표시한 CSV 파일 경로
CSV_PATH = BASE_DIR / "common_wrong_results" / "common_wrong_images.csv"

# 제거할 이미지를 이동시킬 폴더
REMOVED_DIR = BASE_DIR / "removed_test_images"
REMOVED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. image_path 앞의 [O], [X] 표시 분리 함수
# ============================================================

def parse_marked_path(raw_path: str):
    """
    예시:
    [O] C:\\...\\image.jpg
    [O - crop check] C:\\...\\image.jpg
    [X] C:\\...\\image.jpg

    반환:
    marker = O / O - crop check / X
    clean_path = 실제 이미지 경로
    """

    raw_path = raw_path.strip()

    if raw_path.startswith("[") and "]" in raw_path:
        marker = raw_path[1:raw_path.index("]")]
        clean_path = raw_path[raw_path.index("]") + 1:].strip()
        return marker, Path(clean_path)

    return "", Path(raw_path)


# ============================================================
# 3. [X] 이미지 이동
# ============================================================

def main():
    print("========== [X] 테스트 이미지 제거 시작 ==========")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {CSV_PATH}")

    moved_count = 0
    missing_count = 0
    keep_count = 0

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw_image_path = row["image_path"]
            file_name = row["file_name"]
            true_label = row["true_label"]

            marker, image_path = parse_marked_path(raw_image_path)

            # [X]가 아니면 유지
            if marker != "X":
                keep_count += 1
                continue

            if not image_path.exists():
                print(f"[MISSING] 파일 없음: {image_path}")
                missing_count += 1
                continue

            # 클래스별로 제거 폴더 구성
            target_dir = REMOVED_DIR / true_label
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path = target_dir / file_name

            # 같은 이름이 이미 있으면 덮어쓰기 방지
            if target_path.exists():
                stem = target_path.stem
                suffix = target_path.suffix
                target_path = target_dir / f"{stem}_removed{suffix}"

            shutil.move(str(image_path), str(target_path))

            print(f"[MOVE] {image_path} -> {target_path}")
            moved_count += 1

    print()
    print("========== 결과 요약 ==========")
    print(f"유지한 이미지 수: {keep_count}")
    print(f"이동한 [X] 이미지 수: {moved_count}")
    print(f"파일 없음 수: {missing_count}")
    print(f"이동 폴더: {REMOVED_DIR}")
    print("========== 완료 ==========")


if __name__ == "__main__":
    main()