# YOLOv8 을 사용한 bbox 확인
from ultralytics import YOLO

# WanDB 에 연결 진행하기 위한 라이브러리 import
import wandb

# YOLOv8 모델을 가져옴
model = YOLO("yolov8n.pt")

# WanDB 에 저장할 config 값 설정
config_setting = {
    "architecture" : "yolov8n.pt" ,
    "epochs"       : 10 ,
    "batch_size"   : 8 ,
    "image_size"   : 640
}

# WanDB 초기화
run = wandb.init(
    project = "FinSight-OCR" ,
    name    = "junoh(4060)_yolov8n_0902_1" ,
    config  = config_setting
)

# 모델 학습 진행
# 학습 설정은 앞서 설정한 config_setting 값을 가져옴
results = model.train(
    data    = "/home/junoh/datasets/YOLOv8n/data.yaml" ,
    epochs  = config_setting["epochs"] ,
    imgsz   = config_setting["image_size"] ,
    batch   = config_setting["batch_size"] ,
    half    = True ,
    workers = 8 ,
    cache   = True
)

run.finish()

print("학습 완료")