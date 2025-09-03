# preprocess_dataset.py
import os
import json
from PIL import Image
from tqdm import tqdm

# --- 1. 설정 부분 ---
# "train" 또는 "valid"를 지정하여 어떤 데이터셋을 전처리할지 선택
DATASET_TYPE = 'Validation'  # 'valid'로 바꿔서 한번 더 실행해야 합니다.

# 원본 고해상도 이미지와 JSON 라벨이 있는 폴더
ORIGINAL_IMAGE_DIR = f'/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/{DATASET_TYPE}/01.원천데이터/VS_금융_2.보험_2-1.신청서'
ORIGINAL_JSON_DIR  = f'/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/{DATASET_TYPE}/02.라벨링데이터/VL_금융_2.보험_2-1.신청서'

# 전처리된 저해상도 이미지와 JSON 라벨을 저장할 새로운 폴더
PREPROCESSED_IMAGE_DIR = f'/mnt/d/Dataset/OCR 데이터(금융)/전처리된데이터/{DATASET_TYPE}/01.원천데이터/VS_금융_2.보험_2-1.신청서'
PREPROCESSED_JSON_DIR  = f'/mnt/d/Dataset/OCR 데이터(금융)/전처리된데이터/{DATASET_TYPE}/02.라벨링데이터/VL_금융_2.보험_2-1.신청서'

# 목표 이미지 크기 (긴 쪽을 기준으로 리사이즈)
# 8GB VRAM에서는 800 ~ 1280 사이의 값을 추천합니다. 1024로 시작하겠습니다.
TARGET_LONG_EDGE = 640
# --------------------

def preprocess_dataset():
    # 출력 폴더 생성
    os.makedirs(PREPROCESSED_IMAGE_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_JSON_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(ORIGINAL_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg'))]
    
    print(f"'{DATASET_TYPE}' 데이터셋 전처리를 시작합니다. 총 {len(image_files)}개의 이미지를 처리합니다.")

    for filename in tqdm(image_files, desc=f"Preprocessing {DATASET_TYPE}"):
        # 경로 설정
        original_img_path = os.path.join(ORIGINAL_IMAGE_DIR, filename)
        original_json_path = os.path.join(ORIGINAL_JSON_DIR, os.path.splitext(filename)[0] + '.json')
        
        new_img_path = os.path.join(PREPROCESSED_IMAGE_DIR, filename)
        new_json_path = os.path.join(PREPROCESSED_JSON_DIR, os.path.splitext(filename)[0] + '.json')

        # --- 1. 이미지 리사이즈 ---
        with Image.open(original_img_path) as img:
            original_width, original_height = img.size
            
            # 긴 쪽을 기준으로 리사이즈 비율 계산
            if original_width > original_height:
                scale_ratio = TARGET_LONG_EDGE / original_width
                new_width = TARGET_LONG_EDGE
                new_height = int(original_height * scale_ratio)
            else:
                scale_ratio = TARGET_LONG_EDGE / original_height
                new_height = TARGET_LONG_EDGE
                new_width = int(original_width * scale_ratio)
            
            # LANCZOS 필터를 사용하여 고품질로 리사이즈
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            resized_img.save(new_img_path)

        # --- 2. JSON 라벨 좌표 변환 ---
        if os.path.exists(original_json_path):
            with open(original_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            new_bboxes = []
            for bbox in data.get('bbox', []):
                # 기존 좌표에 리사이즈 비율을 곱하여 새로운 좌표 계산
                new_x = [int(coord * scale_ratio) for coord in bbox['x']]
                new_y = [int(coord * scale_ratio) for coord in bbox['y']]
                
                new_bbox = bbox.copy()
                new_bbox['x'] = new_x
                new_bbox['y'] = new_y
                new_bboxes.append(new_bbox)
            
            # 이미지 크기 정보도 새로운 크기로 업데이트
            data['Images']['width'] = new_width
            data['Images']['height'] = new_height
            data['bbox'] = new_bboxes
            
            with open(new_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n전처리 완료! 결과가 아래 폴더에 저장되었습니다:\n- 이미지: {PREPROCESSED_IMAGE_DIR}\n- 라벨: {PREPROCESSED_JSON_DIR}")

if __name__ == '__main__':
    preprocess_dataset()