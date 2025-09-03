# fast_retina_train_with_metrics.py - 평가 지표 추가 버전

import torch
import torchvision
import json
import os
import time
import shutil
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as transforms
from collections import defaultdict

# --- 평가 지표 계산 함수 ---
def calculate_iou(box1, box2):
    """IoU 계산"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 교집합 영역 계산
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 합집합 영역 계산
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def evaluate_model(model, dataloader, device, iou_threshold=0.5, conf_threshold=0.5):
    """모델 평가 및 지표 계산"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = list(image.to(device, non_blocking=True) for image in images)
            
            # 예측 수행
            predictions = model(images)
            
            # CPU로 이동 및 저장
            for pred, target in zip(predictions, targets):
                # 신뢰도 필터링
                keep = pred['scores'] > conf_threshold
                pred_filtered = {
                    'boxes': pred['boxes'][keep].cpu().numpy(),
                    'labels': pred['labels'][keep].cpu().numpy(),
                    'scores': pred['scores'][keep].cpu().numpy()
                }
                
                target_cpu = {
                    'boxes': target['boxes'].cpu().numpy(),
                    'labels': target['labels'].cpu().numpy()
                }
                
                all_predictions.append(pred_filtered)
                all_targets.append(target_cpu)
    
    # 클래스별 정밀도, 재현율, F1 계산
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        # 각 클래스별로 처리
        for class_id in np.unique(np.concatenate([pred_labels, target_labels])):
            pred_mask = pred_labels == class_id
            target_mask = target_labels == class_id
            
            pred_boxes_class = pred_boxes[pred_mask]
            target_boxes_class = target_boxes[target_mask]
            
            matched_targets = set()
            
            # 예측된 박스들에 대해 매칭 확인
            for pred_box in pred_boxes_class:
                best_iou = 0
                best_idx = -1
                
                for i, target_box in enumerate(target_boxes_class):
                    if i in matched_targets:
                        continue
                    
                    iou = calculate_iou(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                if best_iou >= iou_threshold:
                    class_metrics[class_id]['tp'] += 1
                    matched_targets.add(best_idx)
                else:
                    class_metrics[class_id]['fp'] += 1
            
            # 매칭되지 않은 타겟들은 FN
            class_metrics[class_id]['fn'] += len(target_boxes_class) - len(matched_targets)
    
    # 전체 지표 계산
    total_tp = sum(metrics['tp'] for metrics in class_metrics.values())
    total_fp = sum(metrics['fp'] for metrics in class_metrics.values())
    total_fn = sum(metrics['fn'] for metrics in class_metrics.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # 클래스별 지표 계산
    class_results = {}
    for class_id, metrics in class_metrics.items():
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_results[f'class_{int(class_id)}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'class_metrics': class_results,
        'total_detections': total_tp + total_fp,
        'total_ground_truth': total_tp + total_fn
    }

# --- 나머지 코드는 동일 (데이터셋 클래스 등) ---

def copy_dataset_to_local():
    """WSL 성능 향상을 위해 데이터셋을 /tmp로 복사"""
    
    # 원본 경로 (Windows 마운트)
    original_paths = {
        'train_img': '/mnt/d/Dataset/OCR 데이터(금융)/OCR 전처리데이터/Training/01.원천데이터/TS_금융_1.은행_1-1.신고서',
        'train_ann': '/mnt/d/Dataset/OCR 데이터(금융)/OCR 전처리데이터/Training/02.라벨링데이터/TL_금융_1.은행_1-1.신고서',
        'valid_img': '/mnt/d/Dataset/OCR 데이터(금융)/OCR 전처리데이터/Validation/01.원천데이터/VS_금융_1.은행_1-1.신고서',
        'valid_ann': '/mnt/d/Dataset/OCR 데이터(금융)/OCR 전처리데이터/Validation/02.라벨링데이터/VL_금융_1.은행_1-1.신고서'
    }
    
    # 로컬 경로 (/tmp는 메모리 파일시스템)
    local_paths = {
        'train_img': '/tmp/dataset/train/images',
        'train_ann': '/tmp/dataset/train/annotations', 
        'valid_img': '/tmp/dataset/valid/images',
        'valid_ann': '/tmp/dataset/valid/annotations'
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
        
        # 파일 개수 확인
        files = [f for f in os.listdir(original_path) if not f.startswith('.')]
        print(f"복사 중: {key} ({len(files)} 파일)")
        
        # 파일 복사 (tqdm으로 진행률 표시)
        for filename in tqdm(files, desc=f"Copying {key}"):
            src = os.path.join(original_path, filename)
            dst = os.path.join(local_path, filename)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    
    return local_paths

class UltraFastDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, class_names, type_map, cache_file=None):
        super().__init__()
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}
        self.type_map = type_map
        
        # 간단한 transform (속도 우선)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # 전체 데이터셋을 메모리에 캐시
        self.cache_file = cache_file or f'/tmp/dataset_cache_{len(self.image_files)}.pkl'
        self.load_or_create_cache()
    
    def load_or_create_cache(self):
        """전체 데이터셋을 메모리에 로드"""
        if os.path.exists(self.cache_file):
            print(f"캐시 로딩 중: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.cached_data = pickle.load(f)
            print(f"캐시에서 {len(self.cached_data)} 샘플 로드됨")
            return
        
        print("데이터셋을 메모리에 캐싱 중...")
        self.cached_data = {}
        
        for filename in tqdm(self.image_files, desc="Caching dataset"):
            image_path = os.path.join(self.image_dir, filename)
            json_path = os.path.join(self.annotation_dir, os.path.splitext(filename)[0] + '.json')
            
            # 이미지 로드 및 전처리
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.transform(image)
            except Exception as e:
                print(f"이미지 로드 실패: {filename}, 오류: {e}")
                continue
            
            # 어노테이션 파싱
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
            
            # 캐시에 저장
            if boxes:  # 빈 어노테이션은 제외
                self.cached_data[filename] = {
                    'image': image_tensor,
                    'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                    'labels': torch.as_tensor(labels, dtype=torch.int64)
                }
        
        # 유효한 파일만 유지
        self.image_files = list(self.cached_data.keys())
        
        # 캐시 파일로 저장
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

# --- 메인 학습 루프 (평가 지표 추가) ---
DATA_TYPE_MAP = {0: 'korean', 1: 'english', 2: 'number', 3: 'mixed_special'}
SEMANTIC_CLASSES = ['korean', 'english', 'number', 'mixed_special']

# WandB 설정
WANDB_PROJECT = "FinSight-OCR"
WANDB_RUN_NAME = "junoh-ResNet34-Retina-WithMetrics-0903"

# 최적화된 하이퍼파라미터
EPOCHS = 50
BATCH_SIZE = 12
LEARNING_RATE = 2e-4
NUM_WORKERS = 4
LOG_EVERY_N_STEPS = 10

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    print("=== 평가 지표 포함 RetinaNet 학습 시작 ===")
    
    # 1. 데이터셋 로컬 복사 (첫 실행 시만)
    local_paths = copy_dataset_to_local()
    
    # 2. 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    # 3. WandB 초기화
    import wandb
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "optimization": "ultra_fast_with_metrics"
    })
    
    # 4. 모델 준비
    num_classes = len(SEMANTIC_CLASSES) + 1
    backbone = resnet_fpn_backbone('resnet34', weights='DEFAULT', trainable_layers=3)
    model = RetinaNet(backbone, num_classes=num_classes).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler()
    
    # 5. 데이터셋 및 데이터로더
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
    
    # 6. 학습 루프
    print(f"학습 시작 - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
    
    global_step = 0
    best_f1 = 0.0
    best_loss = float('inf')
    
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
        
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        
        # === 검증 및 평가 지표 계산 (매 에폭) ===
        print(f"\nEpoch {epoch+1} 검증 중...")
        
        # 검증 손실 계산
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for images, targets in tqdm(valid_dataloader, desc="Validation Loss"):
                images = list(image.to(device, non_blocking=True) for image in images)
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    val_loss = sum(loss for loss in loss_dict.values())
                    if torch.isfinite(val_loss):
                        val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        
        # 평가 지표 계산 (5 에폭마다 또는 마지막 10 에폭은 매번)
        metrics = None
        if (epoch + 1) % 5 == 0 or epoch >= EPOCHS - 10:
            print("평가 지표 계산 중...")
            metrics = evaluate_model(model, valid_dataloader, device)
            
            # WandB에 모든 지표 로깅
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "epoch_time": epoch_time,
                "val_precision": metrics['overall_precision'],
                "val_recall": metrics['overall_recall'],
                "val_f1": metrics['overall_f1'],
                "total_detections": metrics['total_detections'],
                "total_ground_truth": metrics['total_ground_truth']
            }
            
            # 클래스별 지표도 추가
            for class_name, class_metrics in metrics['class_metrics'].items():
                log_dict[f"{class_name}_precision"] = class_metrics['precision']
                log_dict[f"{class_name}_recall"] = class_metrics['recall']
                log_dict[f"{class_name}_f1"] = class_metrics['f1']
            
            wandb.log(log_dict)
            
            print(f"검증 결과 - 손실: {avg_val_loss:.4f}, "
                  f"정밀도: {metrics['overall_precision']:.4f}, "
                  f"재현율: {metrics['overall_recall']:.4f}, "
                  f"F1: {metrics['overall_f1']:.4f}")
            
            # 최고 모델 저장 (F1 기준)
            if metrics['overall_f1'] > best_f1:
                best_f1 = metrics['overall_f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                    'f1': best_f1,
                    'metrics': metrics
                }, f'/tmp/best_retina_model_f1_{best_f1:.4f}_epoch_{epoch+1}.pth')
                print(f"최고 F1 모델 저장됨 (F1: {best_f1:.4f})")
        
        else:
            # 평가 지표 계산하지 않는 경우에도 기본 로깅
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "epoch_time": epoch_time
            })
        
        # 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # 최고 손실 기준 모델도 저장
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f'/tmp/best_retina_model_loss_{best_loss:.4f}_epoch_{epoch+1}.pth')
        
        print(f"Epoch {epoch+1} 완료 - 시간: {epoch_time:.1f}s, 훈련 손실: {avg_loss:.4f}, 검증 손실: {avg_val_loss:.4f}")
        
        # 메모리 정리
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    wandb.finish()
    print("학습 완료!")
    print(f"최고 F1 스코어: {best_f1:.4f}")
    print(f"최고 검증 손실: {best_loss:.4f}")
    print(f"모델 저장 위치: /tmp/best_retina_model_*.pth")

if __name__ == "__main__":
    main()