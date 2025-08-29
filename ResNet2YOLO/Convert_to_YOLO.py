import json , os

def convert_to_yolo(json_data , class_map) :
    # 단일의 JSON 데이터파일을 YOLO 형식의 문자열 리스트로 변환 진행
    
    # 기존의 JSON 데이터 파일에서 이미지 크기 정보를 추출
    image_info = json_data["Images"]
    img_width  = int(image_info["width"])
    img_height = int(image_info["height"])
    
    # YOLO 용으로 라벨링한 데이터
    yolo_labels = []
    
    # 바운딩 박스 정보 순회
    for bbox in json_data["bbox"] :
        class_name = bbox["data"]
        
        # 클래스 이름을 숫자 ID 로 변환 진행
        # 클래스 맵에 없는 경우에는 새로 클래스를 추가함
        if class_name not in class_map :
            new_id = len(class_map)
            class_map[class_name] = new_id
            
        class_id = class_map[class_name]
        
        # 바운딩 박스의 x , y 좌표 추출
        x_coords = bbox["x"]
        y_coords = bbox["y"]
        
        # x 의 왼쪽 꼭짓점(min) 과 x 의 오른쪽 꼭짓점(max) 위치를 추출
        x_min = min(x_coords)
        x_max = max(x_coords)
        
        # y 의 아래쪽 꼭짓점(mix) 과 y 의 위쫏 꼭짓점(max) 위치를 추출
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # YOLO 의 형식에 맞도록 x 와 y 의 값을 width 와 height 로 변환함
        box_width  = x_max - x_min
        box_height = y_max - y_min
        
        # Box 의 센터점을 계산하여 center 변수에 저장함
        x_center   = x_min + box_width  / 2
        y_center   = y_min + box_height / 2

        # 정규화 진행
        # 0 ~ 1 사이의 값으로 변환함
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        
        width_norm  = box_width  / img_width
        height_norm = box_height / img_height
        
        # YOLO 라벨 문자열 생성
        yolo_labels.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
        
    return yolo_labels , class_map
    
def process_directory(json_dir , output_dir) :
    # 지정된 디렉토리의 모든 JSON 파일을 YOLO 형식으로 변환
        
    # 저장될 디렉토리가 없는 경우 디렉토리 생성
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
        
    class_map = {}  # 클래스명과 ID 를 매핑할 딕셔너리 생성
    
    # JSON 디렉토리의 모든 파일을 순회함
    for filename in os.listdir(json_dir) :
        if filename.endswith(".json") :
            json_path = os.path.join(json_dir , filename)
            # 파일의 확장자가 .json 인 경우 해당 파일의 파일 위치(주소)를 읽어들임
            
            # JSON 파일 읽기
            with open(json_path , "r" , encoding = "utf-8") as f :
                data = json.load(f)
            
            # JSON 타입을 YOLO 라벨링 데이터인 txt 파일로 변환
            yolo_labels , updated_class_map = convert_to_yolo(data , class_map)
            class_map = updated_class_map   # class_map 업데이트
            
            # 이미지 식별자(이름) 을 기반으로 텍스트 파일 생성
            image_identifier = data["Images"]["identifier"]
            output_filename  = f"{image_identifier}.txt"
            output_path      = os.path.join(output_dir , output_filename)
            
            # 텍스트 파일로 저장
            with open(output_path , "w" , encoding = "utf-8") as f :
                f.write("\n".join(yolo_labels))
            
            print(f"{filename} >> {output_filename} 변환 완료")
            
    # 모든 변환 종료 후, 클래스 맵을 파일로 저장
    class_map_filename = os.path.join(output_dir , "classes.txt")
    sorted_class_map   = sorted(class_map.items() , key = lambda item : item[1])    # ID 순으로 정렬 진행
    
    with open(class_map_filename , "w" , encoding = "utf-8") as f :
        for class_name , class_id in sorted_class_map :
            f.write(f"{class_name}\n")
    
    print(f"\n클래스 목록 파일 생성 완료 : {class_map_filename}")

# 실행
if __name__ == "__main__" :
    # 원본 ResNet JSON 라벨링 데이터가 있는 폴더 주소
    
    # Train 데이터에 대한 라벨링데이터 주소 및 저장 주소
    # json_origin_directory  = "D:/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Training/02.라벨링데이터/TL_금융_1.은행_1-1.신고서"
    # yolo_convert_directory = "D:/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Training/03.YOLO용 라벨링데이터"
    
    # Valid 데이터에 대한 라벨링데이터 주소 및 저장 주소
    json_origin_directory  = "D:/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_금융_1.은행_1-1.신고서"
    yolo_convert_directory = "D:/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Validation/03.YOLO용 라벨링데이터"
    
    process_directory(json_origin_directory , yolo_convert_directory)