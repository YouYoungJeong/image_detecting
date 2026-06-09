from ultralytics import YOLO

# =========================
# 1. 모델 경로 설정
# =========================

# 오토라벨링에 사용할 학습된 모델 경로
# 예시: project/weights/best.pt
model_path = "weights/best.pt"


# =========================
# 2. YOLO 모델 불러오기
# =========================

model = YOLO(model_path)


# =========================
# 3. 라벨 번호와 라벨 이름 출력
# =========================

print("모델에 저장된 라벨 번호와 라벨 이름")
print("-" * 40)

for class_id, class_name in model.names.items():
    print(f"{class_id}번 라벨: {class_name}")

print("-" * 40)
print(f"전체 클래스 개수: {len(model.names)}개")