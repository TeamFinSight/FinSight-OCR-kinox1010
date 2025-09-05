# fast_retina_train.py - WSL I/O 최적화 및 성능 지표 로깅 버전

import torch
import torchvision
import json
import os
import time
import shutil
import pickle
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as transforms
from torchmetrics.detection import MeanAveragePrecision # --- [추가] --- torchmetrics 임포트

# --- 1. 데이터를 로컬로 복사하는 함수 ---
def copy_dataset_to_local():
    """WSL 성능 향상을 위해 데이터셋을 /tmp로 복사"""
    
    # 원본 경로 (Windows 마운트)
    original_paths = {
        'train_img' : '/mnt/d/Dataset/OCR 데이터(금융)/전처리된데이터/Training/01.원천데이터/TS_금융_3.증권_3-2.신청서',
        'train_ann' : '/mnt/d/Dataset/OCR 데이터(금융)/전처리된데이터/Training/02.라벨링데이터/TL_금융_3.증권_3-2.신청서',
        'valid_img' : '/mnt/d/Dataset/OCR 데이터(금융)/전처리된데이터/Validation/01.원천데이터/VS_금융_3.증권_3-2.신청서',
        'valid_ann' : '/mnt/d/Dataset/OCR 데이터(금융)/전처리된데이터/Validation/02.라벨링데이터/VL_금융_3.증권_3-2.신청서'
    }
    
    # 로컬 경로 (/tmp는 메모리 파일시스템)
    local_paths = {
        'train_img' : '/tmp/dataset/train/images',
        'train_ann' : '/tmp/dataset/train/annotations', 
        'valid_img' : '/tmp/dataset/valid/images',
        'valid_ann' : '/tmp/dataset/valid/annotations'
    }
    
    print("데이터셋을 고속 로컬 스토리지로 복사 중...")
    
    for key, local_path in local_paths.items():
        original_path = original_paths[key]
        
        if os.path.exists(local_path):
            print(f"{local_path} 이미 존재함 - 건너뛰기")
            continue
            
        if not os.path.exists(original_path):
            print(f"경고: {original_path} 찾을 수 없음")
            continue
            
        os.makedirs(local_path, exist_ok=True)
        
        files = [f for f in os.listdir(original_path) if not f.startswith('.')]
        print(f"복사 중: {key} ({len(files)} 파일)")
        
        for filename in tqdm(files, desc=f"Copying {key}"):
            src = os.path.join(original_path, filename)
            dst = os.path.join(local_path, filename)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    
    return local_paths

# --- 2. 초고속 데이터셋 클래스 ---
class UltraFastDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, class_names, type_map, cache_file=None):
        super().__init__()
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}
        self.idx_to_class = {i + 1: name for i, name in enumerate(class_names)} # --- [추가] --- 클래스 이름 확인용
        self.type_map = type_map
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.cache_file = cache_file or f'/tmp/dataset_cache_{len(self.image_files)}.pkl'
        self.load_or_create_cache()
    
    def load_or_create_cache(self):
        """전체 데이터셋을 메모리에 로드"""
        if os.path.exists(self.cache_file):
            print(f"캐시 로딩 중: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.cached_data = pickle.load(f)
            print(f"캐시에서 {len(self.cached_data)} 샘플 로드됨")
            self.image_files = list(self.cached_data.keys()) # --- [수정] --- 캐시 로드 시 파일 목록 동기화
            return
        
        print("데이터셋을 메모리에 캐싱 중...")
        self.cached_data = {}
        
        for filename in tqdm(self.image_files, desc="Caching dataset"):
            image_path = os.path.join(self.image_dir, filename)
            json_path = os.path.join(self.annotation_dir, os.path.splitext(filename)[0] + '.json')
            
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.transform(image)
            except Exception as e:
                print(f"이미지 로드 실패: {filename}, 오류: {e}")
                continue
            
            boxes = []
            labels = []
            
            if os.path.exists(json_path):
                try:
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
                except Exception as e:
                    print(f"JSON 파싱 실패: {filename}, 오류: {e}")
            
            if boxes:
                self.cached_data[filename] = {
                    'image': image_tensor,
                    'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                    'labels': torch.as_tensor(labels, dtype=torch.int64)
                }
        
        self.image_files = list(self.cached_data.keys())
        
        print(f"캐시 저장 중: {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cached_data, f)
        
        print(f"총 {len(self.cached_data)} 샘플이 캐시됨")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        data = self.cached_data[filename]
        
        target = {
            'boxes': data['boxes'],
            'labels': data['labels']
        }
        
        return data['image'], target

# --- 3. 설정 및 학습 ---
DATA_TYPE_MAP    = {0: 'korean', 1: 'english', 2: 'number', 3: 'mixed_special'}
SEMANTIC_CLASSES = ['korean', 'english', 'number', 'mixed_special']

WANDB_PROJECT  = "FinSight-OCR"
WANDB_RUN_NAME = "junoh-ResNet18+Retina-0905-4060-1" # --- [수정] ---

EPOCHS              = 20
BATCH_SIZE          = 12
LEARNING_RATE       = 2e-4
NUM_WORKERS         = 1
LOG_EVERY_N_STEPS   = 10
VALIDATION_INTERVAL = 1 # --- [추가] --- 검증 주기 (매 에폭마다)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    print("=== 초고속 RetinaNet 학습 시작 (성능 지표 포함) ===")
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    local_paths = copy_dataset_to_local()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    import wandb
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "optimization": "ultra_fast_local_cache_with_perf_metrics"
    })
    
    num_classes = len(SEMANTIC_CLASSES) + 1
    backbone = resnet_fpn_backbone('resnet18', weights='DEFAULT', trainable_layers=3)
    model = RetinaNet(backbone, num_classes=num_classes).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    print("초고속 데이터셋 준비 중...")
    train_dataset = UltraFastDataset(
        local_paths['train_img'], local_paths['train_ann'], 
        SEMANTIC_CLASSES, DATA_TYPE_MAP, 
        cache_file='/tmp/train_cache.pkl'
    )
    
    valid_dataset = UltraFastDataset(
        local_paths['valid_img'], local_paths['valid_ann'], 
        SEMANTIC_CLASSES, DATA_TYPE_MAP,
        cache_file='/tmp/valid_cache.pkl'
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=NUM_WORKERS, 
        pin_memory=True, persistent_workers=True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True
    )

    # --- [추가] --- 성능 지표 계산을 위한 metric 객체 생성
    # val_metric 객체 생성 부분
    val_metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(device)
    
    print(f"학습 시작 - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
    
    global_step = 0
    best_map = 0.0 # --- [수정] --- 최고 성능 지표를 mAP로 변경
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_losses = []
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            step_start = time.time()
            
            images = list(image.to(device, non_blocking=True) for image in images)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            if torch.isfinite(losses):
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_losses.append(losses.item())
                global_step += 1
                
                step_time = time.time() - step_start
                
                if global_step % LOG_EVERY_N_STEPS == 0:
                    wandb.log({
                        "train_step_loss": losses.item(),
                        "step_time": step_time,
                        "learning_rate": optimizer.param_groups[0]['lr']
                    }, step=global_step)
                
                progress_bar.set_postfix({
                    'loss': f"{losses.item():.4f}",
                    'time': f"{step_time:.2f}s"
                })
            else:
                print(f"비정상 손실값 건너뛰기: {losses.item()}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        
        print(f"Epoch {epoch+1} 완료 - 시간: {epoch_time:.1f}s, 평균 손실: {avg_loss:.4f}")
        
        # --- [수정] --- 검증 로직 전체 수정
        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            model.eval()
            val_metric.reset() # 매 검증마다 metric 초기화
            
            with torch.no_grad():
                for images, targets in tqdm(valid_dataloader, desc="Validation"):
                    images = list(image.to(device, non_blocking=True) for image in images)
                    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                    
                    with torch.cuda.amp.autocast():
                        # 1. 성능 측정을 위해 예측 결과 가져오기
                        predictions = model(images)
                        
                        # 2. metric 업데이트
                        val_metric.update(predictions, targets)

            try:
                # 전체 검증 데이터에 대한 최종 성능 계산
                metrics = val_metric.compute()
                
                # WandB 로깅을 위한 데이터 가공
                log_data = {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "epoch_time": epoch_time,
                    "val/mAP": metrics['map'].item(),
                    "val/mAP_50": metrics['map_50'].item(),
                    "val/mAP_75": metrics['map_75'].item(),
                    "val/mAR_1": metrics['mar_1'].item(),
                    "val/mAR_10": metrics['mar_10'].item(),
                }
                
                # --- [수정] 클래스별 AP 로깅 (더 안전한 방법) ---
                # 1. torchmetrics가 반환한 실제 클래스 인덱스와 AP 점수를 가져옵니다.
                ap_per_class = metrics['map_per_class']
                present_classes_indices = metrics['classes']

                # 2. 클래스 인덱스를 클래스 이름으로 변환하여 딕셔너리로 매핑합니다.
                class_ap_map = {
                    valid_dataset.idx_to_class[c.item()]: ap.item() 
                    for c, ap in zip(present_classes_indices, ap_per_class)
                }

                # 3. 전체 클래스 목록을 순회하며, 점수가 있으면 로깅하고 없으면 0.0으로 로깅합니다.
                for class_name in SEMANTIC_CLASSES:
                    ap_score = class_ap_map.get(class_name, 0.0)
                    log_data[f'val/AP_{class_name}'] = ap_score

                wandb.log(log_data)
                
                current_map = metrics['map'].item()
                print(f"Validation mAP: {current_map:.4f}, mAP@50: {metrics['map_50'].item():.4f}")
                
                if current_map > best_map:
                    best_map = current_map
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'map': best_map,
                    }, f'/tmp/best_retina_model_map_{best_map:.4f}.pth')
                    print(f"최고 모델 저장됨 (mAP: {best_map:.4f})")
            
            except Exception as e:
                print(f"성능 지표 계산 중 오류 발생: {e}")

        # --- [수정] 매 에폭 종료 시 모델 저장 ---
        epoch_save_path = f'/mnt/d/Dataset/프로젝트 모델 임시 저장/ResNet18_Retina_4060/ResNet18_Retina_0905_1-{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, epoch_save_path)
        print(f"Epoch {epoch+1} 모델 저장됨: {epoch_save_path}")
        # --- [수정] 끝 ---

        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    wandb.finish()
    print("학습 완료!")
    print(f"최고 검증 mAP: {best_map:.4f}")
    print(f"모델 저장 위치: /tmp/best_retina_model_*.pth 및 /tmp/retina_model_epoch_*.pth")

if __name__ == "__main__":
    main()