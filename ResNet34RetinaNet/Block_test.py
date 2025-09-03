# debug_bottleneck.py - 성능 병목 진단 스크립트

import torch
import torchvision
import json
import os
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as transforms

# 기존 경로 설정
TRAIN_IMAGE_DIR      = '/mnt/d/Dataset/OCR 데이터(금융)/OCR 전처리 데이터/Training/01.원천데이터/TS_금융_1.은행_1-1.신고서'
TRAIN_ANNOTATION_DIR = '/mnt/d/Dataset/OCR 데이터(금융)/OCR 전처리 데이터/Training/02.라벨링데이터/TL_금융_1.은행_1-1.신고서'

DATA_TYPE_MAP = {0: 'korean', 1: 'english', 2: 'number', 3: 'mixed_special'}
SEMANTIC_CLASSES = ['korean', 'english', 'number', 'mixed_special']

class SimpleDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, class_names, type_map):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:100]  # 처음 100개만
        self.class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}
        self.type_map = type_map
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, filename)
        json_path = os.path.join(self.annotation_dir, os.path.splitext(filename)[0] + '.json')
        
        # 시간 측정 시작
        load_start = time.time()
        image = Image.open(image_path).convert("RGB")
        load_time = time.time() - load_start
        
        parse_start = time.time()
        boxes = []
        labels = []

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for bbox in data.get('bbox', []):
                    data_type = bbox.get('data_type')
                    if data_type in self.type_map:
                        semantic_class_name = self.type_map[data_type]
                        class_id = self.class_to_idx.get(semantic_class_name)
                        
                        if class_id is not None:
                            x_min = min(bbox['x'])
                            y_min = min(bbox['y'])
                            x_max = max(bbox['x'])
                            y_max = max(bbox['y'])
                            
                            if x_max > x_min and y_max > y_min:
                                boxes.append([x_min, y_min, x_max, y_max])
                                labels.append(class_id)
        parse_time = time.time() - parse_start
        
        transform_start = time.time()
        image_tensor = self.transform(image)
        transform_time = time.time() - transform_start
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        # 디버깅 정보 저장 (첫 10개만)
        if idx < 10:
            print(f"Sample {idx}: Load={load_time:.3f}s, Parse={parse_time:.3f}s, Transform={transform_time:.3f}s")
        
        return image_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def diagnose_performance():
    print("=== 성능 진단 시작 ===")
    
    # GPU 정보
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA 사용 불가!")
        return
    
    device = torch.device("cuda")
    
    # 1. 데이터로더 속도 테스트
    print("\n=== 데이터로더 속도 테스트 ===")
    dataset = SimpleDataset(TRAIN_IMAGE_DIR, TRAIN_ANNOTATION_DIR, SEMANTIC_CLASSES, DATA_TYPE_MAP)
    
    for num_workers in [0, 4, 8, 12]:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, 
                              collate_fn=collate_fn, num_workers=num_workers)
        
        start_time = time.time()
        for i, (images, targets) in enumerate(dataloader):
            if i >= 5:  # 5배치만 테스트
                break
        end_time = time.time()
        
        print(f"Workers={num_workers}: {(end_time-start_time)/5:.3f}s per batch")
    
    # 2. 모델 순전파 속도 테스트
    print("\n=== 모델 순전파 속도 테스트 ===")
    num_classes = len(SEMANTIC_CLASSES) + 1
    backbone = resnet_fpn_backbone('resnet34', weights='DEFAULT', trainable_layers=3)
    model = RetinaNet(backbone, num_classes=num_classes).to(device)
    
    # 더미 데이터로 순전파 테스트
    dummy_images = [torch.randn(3, 454, 640).to(device) for _ in range(4)]
    dummy_targets = [{
        'boxes': torch.tensor([[10, 10, 100, 100]], dtype=torch.float32).to(device),
        'labels': torch.tensor([1], dtype=torch.int64).to(device)
    } for _ in range(4)]
    
    model.train()
    
    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_images[:2])
    
    # 실제 측정
    times = []
    for i in range(10):
        start = time.time()
        loss_dict = model(dummy_images, dummy_targets)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
        print(f"Forward pass {i+1}: {times[-1]:.3f}s")
    
    print(f"평균 순전파 시간: {sum(times)/len(times):.3f}s")
    
    # 3. 역전파 속도 테스트
    print("\n=== 역전파 속도 테스트 ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    times = []
    for i in range(5):
        start = time.time()
        
        loss_dict = model(dummy_images, dummy_targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
        print(f"Full step {i+1}: {times[-1]:.3f}s")
    
    print(f"평균 전체 스텝 시간: {sum(times)/len(times):.3f}s")
    
    # 4. 메모리 사용량
    print(f"\n=== 메모리 사용량 ===")
    print(f"할당된 GPU 메모리: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"캐시된 GPU 메모리: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    # 5. 실제 데이터로더와 모델 조합 테스트
    print(f"\n=== 실제 데이터로더+모델 테스트 ===")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, 
                          collate_fn=collate_fn, num_workers=4)
    
    times = []
    for i, (images, targets) in enumerate(dataloader):
        if i >= 3:
            break
            
        start = time.time()
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 빈 타겟 체크
        if not any(t['labels'].numel() > 0 for t in targets):
            print(f"배치 {i}: 빈 타겟 건너뛰기")
            continue
            
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
        
        print(f"실제 배치 {i+1}: {times[-1]:.3f}s, 손실: {losses.item():.4f}")
        
        # 메모리 상태
        print(f"  메모리: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    if times:
        print(f"실제 평균 스텝 시간: {sum(times)/len(times):.3f}s")
    
    # 6. 시스템 정보
    print(f"\n=== 시스템 정보 ===")
    import psutil
    print(f"CPU 사용률: {psutil.cpu_percent()}%")
    print(f"RAM 사용률: {psutil.virtual_memory().percent}%")
    print(f"사용 가능 RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")

if __name__ == "__main__":
    diagnose_performance()