import torch

print("CUDA 사용 가능:", torch.cuda.is_available())
print("CUDA GPU 개수:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("사용 GPU 이름:", torch.cuda.get_device_name(0))