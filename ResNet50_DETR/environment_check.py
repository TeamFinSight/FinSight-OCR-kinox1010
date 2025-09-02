import torch
import transformers

print("--- PyTorch 및 CUDA 확인 ---")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"현재 PyTorch가 사용하는 CUDA 버전: {torch.version.cuda}")
    print(f"연결된 GPU 이름: {torch.cuda.get_device_name(0)}")

print("\n--- Transformers 확인 ---")
print(f"Transformers 버전: {transformers.__version__}")

print("\n🎉 모든 라이브러리가 성공적으로 설치되었습니다!")