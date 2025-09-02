# 기존 ResNet 전용 JSON 데이터를 DETR 용 COCO 라벨링 데이터로 변환

import json , os
from tqdm import tqdm
from PIL  import Image

# COCO JSON 포맷으로 변경하는 함수
def convertCOCOformat(data_type , img_dir , lab_dir , coco_dir) :
    all_classes = set()
    for filename in os.listdir(lab_dir) :
        if filename.endswith(".json") :
            with open(os.path.join(lab_dir , filename) , "r" , encoding = "utf-8") as f :
                data = json.load(f)
                for bbox in data.get("bbox" , []) :
                    all_classes.add(bbox["data"])
    
    sorted_classes = sorted(list(all_classes))
    categories     = [{"id" : i , "name" : name} for i , name in enumerate(sorted_classes)]
    classes_to_id  = {name : i for i , name in enumerate(sorted_classes)}
    
    images = []
    labels = []
    labels_id = 1
    
    print(f"{data_type} 데이터셋 변환")
    for img_id , filename in enumerate(tqdm(os.listdir(img_dir))) :
        if filename.lower().endswith((".png" , ".jpg" , ".jpeg")) :
            img_path      = os.path.join(img_dir , filename)
            json_filename = os.path.splitext(filename)[0] + ".json"
            lab_path      = os.path.join(lab_dir , json_filename)
            
            # 이미지 정보 추가
            with Image.open(img_path) as img :
                width , height = img.size
            images.append({
                "id"        : img_id ,
                "file_name" : filename ,
                "width"     : width ,
                "height"    : height
            })
            
            # 라벨링 정보 추가
            if os.path.exists(lab_path) :
                with open(lab_path , "r" , encoding = "utf-8") as f :
                    data = json.load(f)
                    for bbox in data.get("bbox" , []) :
                        x_min = min(bbox["x"])
                        y_min = min(bbox["y"])
                        box_w = max(bbox["x"]) - x_min
                        box_h = max(bbox["y"]) - y_min
                    
                    labels.append({
                        "id"          : labels_id ,
                        "image_id"    : img_id ,
                        "category_id" : classes_to_id[bbox["data"]] ,
                        "bbox"        : [x_min , y_min , box_w , box_h] ,
                        "area"        : float(box_w * box_h) ,
                        "iscrowd"     : 0
                    })
                    
                    labels_id += 1
                    
    coco_format_json = {
        "images"      : images ,
        "annotations" : labels ,
        "categories"  : categories
    }
    
    with open(coco_dir , "w" , encoding = "utf-8") as f :
        json.dump(coco_format_json , f , ensure_ascii = False , indent = 4)
    
    print("변환완료")
    print(f"총 이미지 : {len(images)} , 총 어노테이션 : {len(labels)} , 총 클래스 : {len(categories)}")

if __name__ == "__main__" :
    # 데이터셋 종류 설정
    DATASET_TYPE = "Training" # Training , Validation
    
    # 원본 이미지 및 라벨링 데이터 위치 지정
    IMAGE_DIR = f"/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/{DATASET_TYPE}/01.원천데이터/TS_금융_1.은행_1-1.신고서"
    LABEL_DIR = f"/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/{DATASET_TYPE}/02.라벨링데이터/TL_금융_1.은행_1-1.신고서"
    
    # COCO 형식의 JSON 파일을 저장할 경로 설정
    COCO_LABEL_SAVE_DIR = f"/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터{DATASET_TYPE}_coco.json"
    
    convertCOCOformat(DATASET_TYPE , IMAGE_DIR , LABEL_DIR , COCO_LABEL_SAVE_DIR)