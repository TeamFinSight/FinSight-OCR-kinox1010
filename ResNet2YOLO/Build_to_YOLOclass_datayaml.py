# Build_YOLO_Dataset.py (이전 스크립트 3개를 모두 대체)
import json
import os
from tqdm import tqdm
import yaml # PyYAML이 설치되어 있어야 합니다 (pip install PyYAML)

# --- 1. 설정 부분 (사용자님의 환경에 맞게 수정) ---
# 입력 경로
TRAIN_JSON_DIR = "/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_금융_1.은행_1-1.신고서"
VALID_JSON_DIR = "/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_금융_1.은행_1-1.신고서"

TRAIN_IMAGE_DIR = "/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Training/01.원천데이터/TS_금융_1.은행_1-1.신고서"
VALID_IMAGE_DIR = "/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Validation/01.원천데이터/VS_금융_1.은행_1-1.신고서"

# 출력 경로
OUTPUT_ROOT_DIR = "/mnt/d/Dataset/OCR 데이터(금융)"

# -----------------------------------------------------------

def build_yolo_dataset():
    # --- 1단계: 전체 데이터셋을 스캔하여 통합 클래스 목록 생성 ---
    print("1단계: 전체 데이터셋을 스캔하여 통합 클래스 목록을 생성합니다...")
    all_class_names = set()
    for json_dir in [TRAIN_JSON_DIR, VALID_JSON_DIR]:
        for filename in tqdm(os.listdir(json_dir), desc=f"Scanning {os.path.basename(json_dir)}"):
            if filename.endswith('.json'):
                with open(os.path.join(json_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for bbox in data.get('bbox', []):
                        all_class_names.add(bbox['data'])
    
    sorted_class_names = sorted(list(all_class_names))
    class_map = {name: i for i, name in enumerate(sorted_class_names)}
    print(f"통합 완료! 총 {len(class_map)}개의 고유한 클래스를 찾았습니다.\n")

    # --- 2단계: 통합 class_map을 사용하여 각 데이터셋 변환 ---
    for dataset_type in ['train', 'valid']:
        print(f"2단계: '{dataset_type}' 데이터셋을 YOLO 형식으로 변환합니다...")
        if dataset_type == 'train':
            json_dir = TRAIN_JSON_DIR
            output_label_dir = os.path.join(OUTPUT_ROOT_DIR, 'train', 'labels')
        else:
            json_dir = VALID_JSON_DIR
            output_label_dir = os.path.join(OUTPUT_ROOT_DIR, 'valid', 'labels')
        
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        for filename in tqdm(os.listdir(json_dir), desc=f"Converting {dataset_type} set"):
            if filename.endswith('.json'):
                with open(os.path.join(json_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                img_info = data["Images"]
                img_width = int(img_info["width"])
                img_height = int(img_info["height"])
                
                yolo_labels = []
                for bbox in data.get('bbox', []):
                    class_id = class_map[bbox['data']] # 통합 class_map 사용!
                    
                    x_coords = bbox['x']
                    y_coords = bbox['y']
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    x_center = x_min + box_width / 2
                    y_center = y_min + box_height / 2

                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = box_width / img_width
                    height_norm = box_height / img_height
                    
                    yolo_labels.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
                
                output_filename = f"{data['Images']['identifier']}.txt"
                with open(os.path.join(output_label_dir, output_filename), 'w', encoding='utf-8') as f_out:
                    f_out.write("\n".join(yolo_labels))
        print("변환 완료!\n")

    # --- 3단계: data.yaml 파일 생성 ---
    print("3단계: data.yaml 파일을 생성합니다...")
    yaml_path = os.path.join(OUTPUT_ROOT_DIR, 'data.yaml')
    
    # yaml.dump는 리스트를 Block 스타일로 예쁘게 저장해줍니다.
    yaml_data = {
        'train': TRAIN_IMAGE_DIR,
        'val': VALID_IMAGE_DIR,
        'nc': len(class_map),
        'names': sorted_class_names
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
    print(f"'{yaml_path}' 파일 생성 완료!")
    print("\n--- 모든 데이터셋 준비가 완료되었습니다. ---")


if __name__ == '__main__':
    build_yolo_dataset()