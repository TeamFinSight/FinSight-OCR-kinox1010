# OCR 모델 성능 향상 프로젝트
영등포 새싹 7차 데이터 AI 개발자 과정 파이널 팀 프로젝트

## 프로젝트 목표
- 기존의 모델명 및 수치 : ResNet50 + ViTSTR | F1-Score@loU 0.8 | 0.7134
- 상기 모델보다 더 금융에 적합된 객체 탐지와 OCR 모델을 물색하고 금융 문서에서 고객의 손글씨만을 탐색하여 폼 양식에 자동 기입해주는 최소 기능 서비스 제작
- **주된 목표는 금융 문서에 특히 적합한 모델을 물색하고 학습시키는 것**이며 해당 모델의 성능을 보여주기 위한 최소한의 서비스만 제공될 예정

## 프로젝트 기한
시작일 : 2025년 8월 26일
종료일 : 2025년 9월 19일

## 백본 또는 객체 탐지 모델 변천사
- YOLOv8n , YOLOv8s                  :: ResNet50 + ViTSTR 에서 사용된 라벨링 데이터를 YOLO 의 txt 라벨링 데이터(0~1) 으로 바꾸는 과정에서 문제가 생길 수 있음
- ResNet50 + DETR(Facebook)          :: DETR(facebook) 은 모델명 그대로 손글씨 탐색 보다는 얼굴 탐지에 특화된 모델임
- Swin Transformer + Nevar Clova-ix  :: Naver Clova-ix 가 비교적 높은 최신의 무거운 모델로서 프로젝트에 적합하지 않음
- ResNet34 + Retina                  :: **현재 학습 진행 중**

### 백본(Backborn)
> 백본은 딥러닝 모델의 핵심 특성 추출기로, 원시 데이터를 입력값으로 받아서 고수준의 특성 맵으로 변환하는 네트워크 부분 말함
>
> **주요 역할**
> > [특성 추출] : 이미지에서 엣지, 텍스처, 모양, 객체 등의 패턴을 학습하고 추출함
> > [차원 변환] : 고차원 픽셀 데이터를 의미있는 저차원 특성 벡터로 변환함
> > [계측점 학습] : 낮은 수준(엣지, 색상) 에서 높은 수준(객체, 장면) 까지 점진적으로 복잡한 특성을 학습함

## 모델 학습 환경
**1. 메인 컴퓨터**
   - CPU : i7-13620H
   - RAM : 64 GB
   - GPU : RTX 4060 Laptop 8 GB
   - WSL Ubuntu 22.04 LTS | CUDA 12.9

**2. 학원 컴퓨터**
   - CPU : i7-9700
   - RAM : 16 GB
   - GPU : GTX 1650 4 GB
   - WSL Ubuntu 22.04 LTS | CUDA 12.1

## 9월 1일
![YOLO 학습값](https://kinox0924.notion.site/image/attachment%3A8e7f424a-d3d8-4731-a31d-72473db9dcf1%3Aimage.png?table=block&id=26192c5a-6f62-80c6-9459-d54cd66c8568&spaceId=89642cca-5ede-4074-9b26-ecde57fbb0d3&width=2000&userId=&cache=v2)
- 주로 사용되는 ResNet 이 아닌 Ultralytics 의 YOLO 모델을 백본으로 사용하기 위한 라벨링데이터 Convert 스크립트 작성 완료
- YOLO 용 라벨링 데이터는 x , y 의 좌표값이 아닌 0 ~ 1 사이의 값을 사용하며 솔직히 이 라벨링데이터로 변환하면서 정확한 바운딩박스 위치를 잡을 수 있을 지 고민이었음
- 역시 라벨링 데이터 쪽에서 오류가 발생했지만 클래스 순번의 문제로 바운딩 박스의 문제는 아니었음
- 다만 학습 속도가 느려서 하루에 한 번 ~ 두 번 정도밖에 돌릴 수 없다는 문제가 있음 => 다양한 모델을 탐색해야하는 관점에서 보면 단점으로 다른 모델 탐색을 시작함

## 9월 2일 1회차
![ResNet50+DETR(Facebook)](https://kinox0924.notion.site/image/attachment%3A5f24ae3b-b17e-41a7-9e2e-cf68861839f5%3Aimage.png?table=block&id=26292c5a-6f62-807b-ac1c-d20e700c141e&spaceId=89642cca-5ede-4074-9b26-ecde57fbb0d3&width=2000&userId=&cache=v2)
- YOLOv8n 과 YOLOv8s 가 아닌 ResNet50 을 백본으로 사용하는 허깅페이스의 DETR(Facebook) 객체 탐지 모델을 사용함
- 기존 ResNET50 + ViTSTR 의 JSON 라벨링데이터는 사용할 수 없기에 DETR(Facebook) 모델에 맞는 COCO JSON 라벨링데이터로 변환시켜서 모델 학습 진행함
- 모델의 학습 시간은 준수하나 애시당초 이 모델은 모델명(Facebook) 에 걸맞게 얼굴을 탐지하는데 특화된 모델로서 손글씨 탐색 모델로는 적합하지 않음

## 9월 2일 2회차
![SwinTransformer+NaverClova-ix](https://kinox0924.notion.site/image/attachment%3Abf86c1c5-18b2-4b66-a6dc-310843719e4b%3Aimage.png?table=block&id=26292c5a-6f62-80a7-8242-c59ba18af86c&spaceId=89642cca-5ede-4074-9b26-ecde57fbb0d3&width=2000&userId=&cache=v2)
- Swin Transformer + Naver Clova-ix OCR 모델로 변경함
- Naver Clova-ix 모델의 경우 탐색한 객체를 텍스트로 변환하고 의미 확인까지 가능한 모델로 알고 있음
- 다만 최신에 가까운 모델이기에 모델이 무겁고 학습 속도가 많이 느림
- 추후 서비스까지 생각한다면 해당 모델로 빠른 시간 내에 서비스 제공이 가능할 지 의문이 듬

## 9월 3일
![ResNet34+Retina](https://kinox0924.notion.site/image/attachment%3A9ce3e55f-8f13-44ca-9b58-5a830e1f06aa%3Aimage.png?table=block&id=26392c5a-6f62-8093-b1f9-ed08f848bead&spaceId=89642cca-5ede-4074-9b26-ecde57fbb0d3&width=1460&userId=&cache=v2)
- ResNet34 + Retina 객체 탐지 모델로 변경함
- 원천 데이터의 전처리 진행 => 지금까지는 거의 날것 이미지 그대로 사용하였음(2480*3508) 따라서 이미지 전처리를 통해 세로 길이 기준 640 까지 비율유지하면서 축소
- 데이터와 라벨링데이터, 스크립트 파일까지 모두 WSL 우분투 안으로 옮겨서 최대한 I/O 간 병목 현상 축소
- 데이터를 픽클화시키고 미리 RAM 에 캐싱을 진행하며 빠른 학습 진행
- 메인 컴퓨터 기준 1 Step 당 약 45 ~ 120 초였던 학습 속도를 10 ~ 17 초로 줄임, 1 Epochs 당 약 2200 초가 나옴
  > Epochs = 10 | Batch_size = 12 | Learning_rate = 2e-4 | Num_workers = 4
