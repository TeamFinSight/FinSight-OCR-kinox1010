# yaml 파일 생성
import os

# 통합된 클래스 파일에서 텍스트(라벨) 을 추출
def exportClass(path) :
    export_classes = set()  # 중복값이 생기지 않도록 set() 을 미리 사용
    
    try :
        with open(path , "r" , encoding = "utf-8") as f :
            export_classes = [line.strip() for line in f if line.strip()]   # 좌우 공백제거 후 한 줄씩(라벨 하나씩) 읽어옴
            
            return export_classes , len(export_classes) # 추출한 클래스와 전체 클래스의 개수를 반환
        
    except FileNotFoundError :
        print("지정된 경로에서 파일을 찾을 수 없습니다.")
        exit()

# yaml 파일 생성 함수
def createYaml(class_path , train_imgs_path , valid_imgs_path , yaml_path) :
    classes_count = 0
    all_classes   = set()
    
    # exportClass 함수를 사용해서 클래스(라벨) 의 개수와 라벨들을 가져옴
    all_classes , classes_count = exportClass(class_path)
    
    # 라벨들을 YOLO 형식에 맞게 변경
    labels_to_YOLO = "\n".join([f"  - '{name}'" for name in all_classes])
    
    yaml_content = f"""
    train: {train_imgs_path}
    val: {valid_imgs_path}
    
    nc: {classes_count}
    
    names:
    {labels_to_YOLO}
    """
    
    try :
        with open(yaml_path , "w" , encoding = "utf-8") as f :
            f.write(yaml_content)

        print("yaml 파일 생성 성공")
    except Exception as err :
        print("yaml 파일을 생성하지 못 했습니다.")
    

if __name__ == "__main__" :
    # 통합 클래스가 적혀있는 txt 파일 , 훈련 이미지가 들어있는 폴더 , 검증 이미지가 들어있는 폴더 , yaml 파일을 생성할 위치를 변수에 저장
    # wsl 리눅스 환경에서 CUDA 를 돌릴 예정이기에 경로는 wsl 리눅스 경로에 맞추어 저장됨
    all_classes_path = "/mnt/d/Dataset/OCR 데이터(금융)/YOLO 훈련 데이터/classes.txt"
    train_imgs_path  = "/mnt/d/Dataset/OCR 데이터(금융)/YOLO 훈련 데이터/train/images"
    valid_imgs_path  = "/mnt/d/Dataset/OCR 데이터(금융)/YOLO 훈련 데이터/valid/images"
    yaml_path        = "/mnt/d/Dataset/OCR 데이터(금융)/YOLO 훈련 데이터/data.yaml"
    
    createYaml(all_classes_path , train_imgs_path , valid_imgs_path , yaml_path)