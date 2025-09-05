# visualize_only.py - 모델의 객체 탐지 결과를 순수하게 시각화하는 스크립트

import torch
import torchvision
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as transforms

# --- [핵심] 사용자가 직접 설정할 부분 ---
MODEL_TO_EVALUATE = '/mnt/d/Dataset/프로젝트 모델 임시 저장/ResNet18_Retina_1650/best_retina_model_map_0.7136.pth' # <-- 여기에 시각화할 모델 .pth 파일 경로를 입력하세요.
IMAGE_SOURCE_DIR  = '/mnt/d/Dataset/OCR 데이터(금융)/박스 시각화/원천데이터' # <-- 여기에 시각화할 이미지들이 있는 폴더 경로를 입력하세요.
OUTPUT_DIR        = '/mnt/d/Dataset/OCR 데이터(금융)/박스 시각화/바운딩 아웃풋' # <-- 결과 이미지가 저장될 폴더 이름입니다.
SCORE_THRESHOLD   = 0.5 # <-- 이 신뢰도 점수 이상의 예측만 표시합니다.

# --- 모델 및 클래스 설정 (학습 때와 동일하게 유지) ---
SEMANTIC_CLASSES = ['background', 'korean', 'english', 'number', 'mixed_special']
# 클래스 이름에 'background'를 0번 인덱스로 추가하여 모델 출력(1부터 시작)과 인덱스를 맞춥니다.

# --- 시각화용 색상 설정 ---
COLORS = [
    (255, 6, 255),    # Magenta (korean)
    (6, 255, 128),    # Green (english)
    (6, 128, 255),    # Blue (number)
    (255, 128, 6),    # Orange (mixed_special)
]

def visualize_model_predictions():
    """모델을 로드하고 지정된 폴더의 이미지에 대한 예측을 시각화합니다."""
    
    print("=== 모델 예측 결과 시각화 시작 ===")

    # --- 1. 필수 경로 및 파일 확인 ---
    if not os.path.exists(MODEL_TO_EVALUATE):
        print(f"오류: 모델 파일을 찾을 수 없습니다 -> {MODEL_TO_EVALUATE}")
        return
    if not os.path.exists(IMAGE_SOURCE_DIR):
        print(f"오류: 이미지 폴더를 찾을 수 없습니다 -> {IMAGE_SOURCE_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"결과 이미지는 '{OUTPUT_DIR}' 폴더에 저장됩니다.")

    # --- 2. 장치 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # --- 3. 모델 구조 생성 및 학습된 가중치 로드 ---
    num_classes = len(SEMANTIC_CLASSES)
    backbone = resnet_fpn_backbone('resnet18', weights=None, trainable_layers=3)
    model = RetinaNet(backbone, num_classes=num_classes).to(device)

    print(f"모델 가중치 로딩 중: {MODEL_TO_EVALUATE}")
    checkpoint = torch.load(MODEL_TO_EVALUATE, map_location=device)
    
    # 저장된 state_dict 키 확인 후 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval() # 모델을 평가 모드로 전환

    # --- 4. 이미지 변환 설정 ---
    transform = transforms.Compose([transforms.ToTensor()])

    # --- 5. 시각화 폰트 설정 ---
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        print("경고: 'arial.ttf' 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

    # --- 6. 지정된 폴더의 모든 이미지에 대해 예측 및 시각화 수행 ---
    image_files = [f for f in os.listdir(IMAGE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"총 {len(image_files)}개의 이미지에 대해 예측을 시작합니다...")
    
    with torch.no_grad():
        for filename in tqdm(image_files, desc="이미지 처리 중"):
            image_path = os.path.join(IMAGE_SOURCE_DIR, filename)
            
            # 이미지 열고 텐서로 변환
            original_image = Image.open(image_path).convert("RGB")
            image_tensor = transform(original_image).unsqueeze(0).to(device)

            # 모델 예측 실행
            predictions = model(image_tensor)[0]

            # 시각화를 위해 PIL 이미지에 그리기 준비
            draw = ImageDraw.Draw(original_image)

            # 신뢰도(score)가 임계값(SCORE_THRESHOLD)을 넘는 예측 결과만 필터링
            for i in range(len(predictions['scores'])):
                score = predictions['scores'][i].item()
                if score > SCORE_THRESHOLD:
                    box = predictions['boxes'][i].cpu().numpy()
                    label_idx = predictions['labels'][i].item()
                    
                    class_name = SEMANTIC_CLASSES[label_idx]
                    color = COLORS[(label_idx - 1) % len(COLORS)] # 배경 제외하고 색상 매핑
                    
                    # 바운딩 박스 그리기
                    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
                    
                    # 라벨 텍스트 (클래스 이름 + 신뢰도 점수) 그리기
                    text = f"{class_name}: {score:.2f}"
                    text_bbox = draw.textbbox((box[0], box[1] - 22), text, font=font)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((box[0], box[1] - 22), text, fill="black", font=font)

            # 결과 이미지 저장
            output_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
            original_image.save(output_path)

    print(f"\n시각화 완료! 모든 결과가 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    visualize_model_predictions()