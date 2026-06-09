import os
from ultralytics import YOLO


# =========================================================
# 1. 기본 프로젝트 경로 설정
# =========================================================

# ├── weights/
# │   └── best.pt
# ├── unlabeled_images/
# ├── auto_labels/
# └── auto_label_results/
project_dir = "."


# =========================================================
# 2. 사용할 폴더 경로 설정
# =========================================================

# 학습 완료된 best.pt 모델이 들어갈 폴더
weights_dir = os.path.join(project_dir, "weights")

# 오토 라벨링할 원본 이미지가 들어갈 폴더
unlabeled_image_dir = os.path.join(project_dir, "unlabeled_images")

# YOLO 형식 txt 라벨 파일이 저장될 폴더
auto_label_dir = os.path.join(project_dir, "auto_labels")

# 박스가 그려진 결과 이미지가 저장될 상위 폴더
auto_result_dir = os.path.join(project_dir, "auto_label_results")

# YOLO가 실제 결과 이미지를 저장할 하위 폴더 이름
auto_result_name = "auto_label_result"


# =========================================================
# 3. 파일 경로 설정
# =========================================================

# best.pt 모델 경로
model_path = os.path.join(weights_dir, "best.pt")


# =========================================================
# 4. 필요한 폴더 자동 생성
# =========================================================

# weights 폴더 생성
# 단, best.pt 파일은 직접 넣어야 합니다.
os.makedirs(weights_dir, exist_ok=True)

# 오토 라벨링할 이미지 폴더 생성
# 이 폴더 안에 image_001.jpg 같은 이미지들을 넣으면 됩니다.
os.makedirs(unlabeled_image_dir, exist_ok=True)

# 자동 생성된 txt 라벨이 저장될 폴더 생성
os.makedirs(auto_label_dir, exist_ok=True)

# 박스가 그려진 결과 이미지가 저장될 폴더 생성
os.makedirs(auto_result_dir, exist_ok=True)


# =========================================================
# 5. best.pt 파일 존재 여부 확인
# =========================================================

# best.pt가 없으면 모델을 불러올 수 없으므로 실행을 중단합니다.
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"best.pt 파일을 찾을 수 없습니다.\n"
        f"아래 경로에 best.pt를 넣어주세요:\n"
        f"{model_path}"
    )


# =========================================================
# 6. 오토 라벨링할 이미지 목록 가져오기
# =========================================================

# YOLO에서 처리할 이미지 확장자 목록
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# unlabeled_images 폴더 안에서 이미지 파일만 가져오기
image_files = [
    file_name for file_name in os.listdir(unlabeled_image_dir)
    if os.path.splitext(file_name)[1].lower() in image_extensions
]

# 이미지가 하나도 없으면 실행을 중단합니다.
if len(image_files) == 0:
    raise FileNotFoundError(
        f"오토 라벨링할 이미지가 없습니다.\n"
        f"아래 폴더에 jpg, png, webp 등의 이미지를 넣어주세요:\n"
        f"{unlabeled_image_dir}"
    )


# =========================================================
# 7. YOLO 모델 불러오기
# =========================================================

# weights/best.pt 모델을 불러옵니다.
model = YOLO(model_path)


# =========================================================
# 8. 오토 라벨링 설정
# =========================================================

# confidence_threshold는 모델이 객체라고 판단하는 최소 신뢰도입니다.
#
# 처음 오토 라벨링할 때 0.3을 추천하는 이유:
# - 너무 높으면 실제 객체를 많이 놓칠 수 있음
# - 너무 낮으면 잘못된 박스가 많이 생길 수 있음
# - 0.25~0.4 사이가 사람이 검수하기 좋은 초기 범위
#
# 결과가 너무 많이 잡히면 0.4로 올리고,
# 객체가 많이 빠지면 0.25로 낮추면 됩니다.
confidence_threshold = 0.3


# =========================================================
# 9. 오토 라벨링 실행
# =========================================================

for image_file in image_files:
    # 원본 이미지 전체 경로
    image_path = os.path.join(unlabeled_image_dir, image_file)

    # 모델 예측 실행
    # save=True:
    #   박스가 그려진 결과 이미지를 저장합니다.
    #
    # project=auto_result_dir:
    #   결과 이미지 저장 상위 폴더입니다.
    #
    # name=auto_result_name:
    #   결과 이미지 저장 하위 폴더 이름입니다.
    #
    # 최종 저장 위치:
    # auto_label_results/auto_label_result/
    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        save=True,
        project=auto_result_dir,
        name=auto_result_name,
        exist_ok=True
    )

    # 이미지 1장 기준 결과 가져오기
    result = results[0]

    # 이미지 파일명에서 확장자 제거
    # 예: image_001.jpg -> image_001
    base_name = os.path.splitext(image_file)[0]

    # 저장할 txt 라벨 파일 경로
    # 예: auto_labels/image_001.txt
    label_path = os.path.join(auto_label_dir, base_name + ".txt")

    # =====================================================
    # 10. YOLO 형식 txt 라벨 저장
    # =====================================================
    #
    # YOLO 라벨 형식:
    # class_id x_center y_center width height
    #
    # 모든 좌표값은 0~1 사이로 정규화된 값입니다.
    #
    # 예:
    # 0 0.512300 0.481200 0.234500 0.198700

    with open(label_path, "w", encoding="utf-8") as label_file:
        for box in result.boxes:
            # 클래스 번호
            class_id = int(box.cls[0])

            # 신뢰도 값
            confidence = float(box.conf[0])

            # 정규화된 YOLO 좌표
            # xywhn:
            # x_center, y_center, width, height 순서
            x_center, y_center, width, height = box.xywhn[0].tolist()

            # txt 파일에 저장
            label_file.write(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )

    print(f"오토 라벨링 완료: {image_file} -> {label_path}")


# =========================================================
# 11. 최종 결과 안내
# =========================================================

print()
print("전체 오토 라벨링 완료")
print()
print("[폴더 구조]")
print(f"best.pt 위치: {model_path}")
print(f"원본 이미지 폴더: {unlabeled_image_dir}")
print(f"자동 라벨 저장 폴더: {auto_label_dir}")
print(f"결과 이미지 저장 폴더: {os.path.join(auto_result_dir, auto_result_name)}")