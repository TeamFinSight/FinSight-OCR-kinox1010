# Class 합치기

import os

# 클래스 텍스트 합치는 함수
def mergeClass(train_path , valid_path , int_path) :
    all_classes = set()
    
    # 각 클래스 파일의 텍스트를 추출
    train_classes = exportClass(train_path)
    valid_classes = exportClass(valid_path)
    
    # 추출된 클래스(라벨) 등을 하나로 합침
    all_classes.update(train_classes)
    all_classes.update(valid_classes)

    # 통합된 클래스 파일 저장
    try :
        with open(int_path , "w" , encoding = "utf-8") as f :
            f.write("\n".join(all_classes))
        
        print("파일 저장 성공")
        print(len(all_classes))
        
        return len(all_classes)
        
    except Exception as err :
        print(f"파일 저장 실패 | 오류 코드 : {err}")

# 클래스 파일 내 텍스트를 추출하는 함수
def exportClass(path) :
    export_class = set()
    
    try :
        with open(path , "r" , encoding = "utf-8") as f :
            classes = {line.strip() for line in f if line.strip()}  # 지정된 경로의 파일에서 한 줄씩 추출한 텍스트를 가져와서 컴프리헨션 진행
            export_class.update(classes)    # export_class 라는 딕셔너리에 추출한 텍스트 Update
            
            return export_class
        
    except FileNotFoundError : 
        print("지정된 경로에서 파일을 찾을 수 없습니다.")
        exit()

if __name__ == "__main__" :
    # wsl 리눅스 환경에서 CUDA 를 돌릴 예정이기에 경로는 wsl 리눅스 경로에 맞추어 저장됨
    # 훈련셋에서 생성된 txt 파일 경로 설정
    train_classes_path = "/mnt/d/Dataset/OCR 데이터(금융)/YOLO 훈련 데이터/train/labels/classes.txt"

    # 검증셋에서 생성된 txt 파일 경로 설정
    valid_classes_path = "/mnt/d/Dataset/OCR 데이터(금융)/YOLO 훈련 데이터/valid/labels/classes.txt"

    # 새로 생성될 통합 txt 파일이 저장될 경로
    integrated_classes_path = "/mnt/d/Dataset/OCR 데이터(금융)/YOLO 훈련 데이터/classes.txt"
    
    # 클래스 통합 시작
    # 길이(클래스 개수)도 같이 확인 진행
    class_count = mergeClass(train_classes_path , valid_classes_path , integrated_classes_path)