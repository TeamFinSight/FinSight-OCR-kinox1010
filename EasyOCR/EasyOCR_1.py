# run_pipeline.py (테이블 처리 기능 추가 버전)

import easyocr
import numpy as np
import json
import os
import glob
from tqdm import tqdm
import torch

# --- 1. 설정 (사용자 환경에 맞게 수정) ---

INPUT_DIR = '/mnt/d/Dataset/OCR 데이터(금융)/박스 시각화/원천데이터'
OUTPUT_DIR = '/mnt/d/Dataset/OCR 데이터(금융)/박스 시각화/JSON 아웃풋'

# --- [수정] --- Key-Value와 Table Header를 분리하여 정의
# 방법 1: 폼(Form) 스타일의 Key-Value 항목
FORM_KEY_WORDS = [
    '성명' , '성 명' , '주민등록번호' , '주 소' , '성  명' , '주  소' , '위임인과의' , '관계' , '위임인과의 관계' , '주민등록번호' , '(사업자등록번호)' , '(법인명)' , '(법 인 명)'
]
# 방법 2: 테이블(Table) 스타일의 열(Column) 제목
TABLE_HEADERS = [
    '예금종류' , '계좌번호' , '신규일자' , '위임일현재 예금잔액' , '(또는 신규예입액)' , '비고' , '예 금 종 류'
]

Y_AXIS_TOLERANCE = 25 # y축 높이 허용치 (픽셀)

# --- 기존 함수들은 수정 없이 그대로 사용 ---
def process_ocr_results(raw_results):
    processed = []
    for (bbox, text, prob) in raw_results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_min, y_min = int(min(x_coords)), int(min(y_coords))
        x_max, y_max = int(max(x_coords)), int(max(y_coords))
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        processed.append({
            'text': text,
            'box': [x_min, y_min, x_max, y_max],
            'center': (center_x, center_y)
        })
    return processed

def link_keys_and_values(ocr_data, key_words, y_tolerance):
    keys = []
    potential_values = []
    # Key가 아닌 단어들만 potential_values 후보로 남김
    key_texts_set = set(key_words)
    for item in ocr_data:
        is_key = any(key in item['text'] for key in key_texts_set)
        if is_key:
            keys.append(item)
        else:
            potential_values.append(item)

    classified_data = {}
    for key_item in keys:
        key_text = key_item['text']
        key_box = key_item['box']
        key_center = key_item['center']
        best_match = None
        min_horizontal_dist = float('inf')
        for value_item in potential_values:
            value_box = value_item['box']
            value_center = value_item['center']
            if value_box[0] > key_box[2] and abs(key_center[1] - value_center[1]) < y_tolerance:
                dist = value_box[0] - key_box[2]
                if dist < min_horizontal_dist:
                    min_horizontal_dist = dist
                    best_match = value_item
        if best_match:
            clean_key = ''.join(key_text.split())
            classified_data[clean_key] = best_match['text']
            potential_values.remove(best_match)
    return classified_data

# --- [신규 추가] --- 테이블 데이터 처리 함수
def link_table_data(ocr_data, table_headers, y_tolerance):
    """
    테이블 구조의 데이터를 분석하여 행과 열로 구조화하는 함수
    """
    headers = []
    cells = []
    header_texts_set = set(table_headers)
    
    # 1. 헤더와 나머지 셀(데이터) 분리
    for item in ocr_data:
        is_header = any(h in item['text'] for h in header_texts_set)
        if is_header:
            headers.append(item)
        else:
            cells.append(item)
            
    if not headers:
        return []

    # 2. 헤더를 x축 기준으로 정렬 (왼쪽 -> 오른쪽)
    headers.sort(key=lambda h: h['center'][0])
    
    # 3. 각 셀이 어떤 열(column)에 속하는지 판단
    table_rows = {} # y축 위치를 key로 사용하여 행(row)을 그룹화
    
    for cell in cells:
        cell_center = cell['center']
        
        # 헤더보다 아래에 있는 셀만 대상으로 함
        if cell_center[1] < headers[0]['center'][1]: 
            continue

        assigned_header = None
        for i, header in enumerate(headers):
            # i) 마지막 헤더인 경우
            if i == len(headers) - 1:
                if cell_center[0] >= header['center'][0] - (header['box'][2] - header['box'][0])/2:
                    assigned_header = header
            # ii) 나머지 헤더인 경우
            else:
                next_header = headers[i+1]
                # 현재 헤더의 중심과 다음 헤더의 중심 사이에 셀의 중심이 위치하는지 확인
                if header['center'][0] <= cell_center[0] < next_header['center'][0]:
                    assigned_header = header
                    break
        
        if assigned_header:
            # 4. y축 기준으로 행(Row) 그룹화
            # 비슷한 y 위치를 가진 셀들을 하나의 행으로 묶기 위해 y좌표를 key로 사용
            # y_tolerance를 이용해 비슷한 높이는 같은 키를 갖도록 반올림 효과를 줌
            row_key = round(cell_center[1] / y_tolerance)
            
            if row_key not in table_rows:
                table_rows[row_key] = {}
            
            clean_header = ''.join(assigned_header['text'].split())
            table_rows[row_key][clean_header] = cell['text']
            
    # 5. 최종 테이블 데이터 구조화 (딕셔너리 리스트로 변환)
    # y축 위치(row_key) 순서대로 정렬
    sorted_rows = sorted(table_rows.items(), key=lambda item: item[0])
    
    final_table = []
    for _, row_data in sorted_rows:
        final_table.append(row_data)

    return final_table


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading EasyOCR model...")
    reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
    print("EasyOCR model loaded.")

    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_path_list = []
    for ext in image_extensions:
        image_path_list.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    if not image_path_list:
        print(f"Error: No images found in '{INPUT_DIR}' directory.")
        return

    print(f"\nFound {len(image_path_list)} images to process.")
    all_final_data = {}

    for image_path in tqdm(image_path_list, desc="Processing Images"):
        image_filename = os.path.basename(image_path)
        
        raw_ocr_results = reader.readtext(image_path)
        processed_ocr_data = process_ocr_results(raw_ocr_results)
        
        # --- [수정] --- 두 가지 분석을 모두 수행
        
        # 1. 폼(Form) 스타일 Key-Value 분석
        form_data = link_keys_and_values(processed_ocr_data, FORM_KEY_WORDS, Y_AXIS_TOLERANCE)
        
        # 2. 테이블(Table) 스타일 분석
        table_data = link_table_data(processed_ocr_data, TABLE_HEADERS, Y_AXIS_TOLERANCE)
        
        # 3. 결과 통합
        all_final_data[image_filename] = {
            "form_data": form_data,
            "table_data": table_data
        }

    final_json_path = os.path.join(OUTPUT_DIR, "classified_results_v2.json")
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_final_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}\nAll tasks completed.")
    print(f"Results have been saved to: {final_json_path}")
    print("\n--- Result Preview (first file) ---")
    if image_path_list:
        first_file = os.path.basename(image_path_list[0])
        print(json.dumps({first_file: all_final_data.get(first_file)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()