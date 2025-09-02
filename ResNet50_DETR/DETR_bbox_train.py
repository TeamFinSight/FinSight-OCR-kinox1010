# detr_training_complete.py

import torch , json , os , wandb
from torch.utils.data import DataLoader, Dataset
from transformers     import DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm
from PIL  import Image

# WaNDB init 초기화 설정
WANDB_PROJECT  = "FinSight-OCR"
WANDB_RUN_NAME = "junoh(4060)_ResNet50 + DETR_0902_1"

# 데이터 경로 설정
TRAIN_IMG_DIR  = '/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Training/01.원천데이터/TS_금융_1.은행_1-1.신고서'
TRAIN_ANN_FILE = '/mnt/d/Dataset/OCR 데이터(금융)/Training_coco.json'
VALID_IMG_DIR  = '/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Validation/01.원천데이터/VS_금융_1.은행_1-1.신고서'
VALID_ANN_FILE = '/mnt/d/Dataset/OCR 데이터(금융)/Validation_coco.json'

# 하이퍼파라미터 설정
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
EPOCHS        = 10
BATCH_SIZE    = 4

# PyTorch Dataset 클래스
class CocoDetection(Dataset):
    def __init__(self, img_folder, ann_file, processor):
        self.img_folder = img_folder
        self.processor  = processor
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        
        self.image_ids   = [img['id'] for img in self.coco['images']]
        self.annotations = {ann['image_id'] : [] for ann in self.coco['annotations']}
        for ann in self.coco['annotations'] :
            self.annotations[ann['image_id']].append(ann)
        self.img_info = {img['id']: img for img in self.coco['images']}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id       = self.image_ids[idx]
        img_info     = self.img_info[img_id]
        image        = Image.open(os.path.join(self.img_folder, img_info['file_name'])).convert('RGB')
        target       = {'image_id': img_id, 'annotations': self.annotations.get(img_id, [])}
        encoding     = self.processor(
            images         = image ,
            annotations    = target ,
            return_tensors = "pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()
        labels       = encoding["labels"][0]
        
        return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding     = processor.pad(pixel_values, return_tensors = "pt")
    labels       = [item["labels"] for item in batch]
    return {"pixel_values" : encoding["pixel_values"] , "pixel_mask" : encoding["pixel_mask"] , "labels" : labels}

# WandB 
run = wandb.init(
    project = WANDB_PROJECT ,
    name    = WANDB_RUN_NAME ,
    config  = {"learning_rate": LEARNING_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE}
)

# 모델 및 프로세서 로드
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 및 검증 데이터셋/데이터로더 생성
train_dataset = CocoDetection(img_folder=TRAIN_IMG_DIR, ann_file=TRAIN_ANN_FILE, processor=processor)
valid_dataset = CocoDetection(img_folder=VALID_IMG_DIR, ann_file=VALID_ANN_FILE, processor=processor)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)

# 데이터셋에서 클래스 정보를 먼저 로드
num_classes = len(train_dataset.coco['categories'])
id2label    = {cat['id']: cat['name'] for cat in train_dataset.coco['categories']}
label2id    = {v: k for k, v in id2label.items()}

# 모델을 로드할 때 클래스 정보를 직접 전달
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels = num_classes,      # 여기에 새로운 클래스 개수를 전달
    id2label   = id2label,         # 클래스 ID -> 이름 매핑 정보 전달
    label2id   = label2id,         # 클래스 이름 -> ID 매핑 정보 전달
    ignore_mismatched_sizes = True # 기존 모델과 크기가 다른 출력층을 새로 만들도록 허용
)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 학습 및 검증 루프
for epoch in range(EPOCHS):
    # 학습 루프
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask   = batch["pixel_mask"].to(device)
        labels       = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss    = outputs.loss
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)

    # 검증 루프
    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc=f"Validation Epoch {epoch + 1}/{EPOCHS}"):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask   = batch["pixel_mask"].to(device)
            labels       = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss    = outputs.loss
            total_valid_loss += loss.item()
            
    avg_valid_loss = total_valid_loss / len(valid_dataloader)

    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
    
    # WandB loss 기록
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "valid_loss": avg_valid_loss
    })

# WandB 실행 종료
run.finish()
print("모델 학습 완료")