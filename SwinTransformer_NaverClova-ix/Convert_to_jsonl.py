# create_donut_dataset.py (경로 조합 버그 최종 수정 버전)
import json
import os
from tqdm import tqdm

# --- 설정 부분 (본인의 환경에 맞게 6개의 경로를 모두 수정해주세요!) ---
TRAIN_IMAGE_DIR      = '/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Training/01.원천데이터/TS_금융_1.은행_1-1.신고서'
TRAIN_ANNOTATION_DIR = '/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_금융_1.은행_1-1.신고서'

VALID_IMAGE_DIR      = '/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Validation/01.원천데이터/VS_금융_1.은행_1-1.신고서'
VALID_ANNOTATION_DIR = '/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_금융_1.은행_1-1.신고서'

TRAIN_OUTPUT_JSONL_PATH = '/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/train_metadata.jsonl'
VALID_OUTPUT_JSONL_PATH = '/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/valid_metadata.jsonl'
# ---------------------------------------------------------------------

def process_one_set(image_dir, annotation_dir, output_path, dataset_type):
    if not os.path.isdir(image_dir) or not os.path.isdir(annotation_dir):
        print(f"오류: '{dataset_type}' 데이터셋의 폴더를 찾을 수 없습니다.")
        print(f"이미지 폴더: '{image_dir}'")
        print(f"라벨 폴더: '{annotation_dir}'")
        print("경로를 올바르게 설정했는지 확인해주세요.")
        return False

    dataset = []
    print(f"'{dataset_type}' 데이터셋 변환을 시작합니다...")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files, desc=f"Processing {dataset_type} set"):
        json_filename = os.path.splitext(filename)[0] + '.json'
        annotation_path = os.path.join(annotation_dir, json_filename)
        
        if not os.path.exists(annotation_path):
            continue

        with open(annotation_path, 'r', encoding='utf-8') as f:
            try:
                annotation_data = json.load(f)
            except json.JSONDecodeError:
                print(f"경고: {json_filename} 파일이 손상되었습니다. 건너뜁니다.")
                continue

        all_texts = [bbox['data'] for bbox in annotation_data.get('bbox', [])]
        ground_truth_dict = {"text_sequence": all_texts}
        ground_truth_str = json.dumps(ground_truth_dict, ensure_ascii=False)
        
        # [수정된 부분] 불필요한 경로 조합을 제거하고, 순수한 파일명만 저장합니다.
        dataset.append({
            "file_name": filename,
            "ground_truth": ground_truth_str
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"\n변환 완료! '{output_path}' 파일이 생성되었습니다.")
    print(f"총 {len(dataset)}개의 이미지-라벨 쌍이 처리되었습니다.")
    return True

if __name__ == '__main__':
    print("--- 훈련 데이터셋 처리 시작 ---")
    train_success = process_one_set(TRAIN_IMAGE_DIR, TRAIN_ANNOTATION_DIR, TRAIN_OUTPUT_JSONL_PATH, "train")
    
    if train_success:
        print("\n--- 검증 데이터셋 처리 시작 ---")
        process_one_set(VALID_IMAGE_DIR, VALID_ANNOTATION_DIR, VALID_OUTPUT_JSONL_PATH, "valid")