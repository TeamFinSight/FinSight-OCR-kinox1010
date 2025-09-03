# Donut_train.py (최종 완전판)

import torch , json , os , wandb , re
from PIL  import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers     import DonutProcessor, VisionEncoderDecoderModel

# --- 1. 설정 부분 ---
WANDB_PROJECT  = "FinSight-OCR"
WANDB_RUN_NAME = "junoh_NaverClover-ix + Swin Transformer_0902_3"

# 훈련/검증 데이터 경로
TRAIN_IMAGE_DIR     = '/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Training/01.원천데이터/TS_금융_1.은행_1-1.신고서'
TRAIN_METADATA_PATH = '/mnt/d/Dataset/OCR 데이터(금융)/Training_metadata.jsonl'

VALID_IMAGE_DIR     = '/mnt/d/Dataset/OCR 데이터(금융)/01-1.정식개방데이터/Validation/01.원천데이터/VS_금융_1.은행_1-1.신고서'
VALID_METADATA_PATH = '/mnt/d/Dataset/OCR 데이터(금융)/Validation_metadata.jsonl'

# 하이퍼파라미터
EPOCHS = 50
LEARNING_RATE = 1e-5

# 메모리 최적화 설정 (8GB VRAM 환경 추천)
PHYSICAL_BATCH_SIZE  = 1  # GPU가 한 번에 처리할 실제 배치 크기 (메모리에 맞춰 1 또는 2로 설정)
EFFECTIVE_BATCH_SIZE = 8 # 우리가 원하는 논리적인 배치 크기
GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // PHYSICAL_BATCH_SIZE
USE_AMP = True # 자동 혼합 정밀도(FP16) 사용 여부

# --- 2. F1 Score 계산을 위한 헬퍼 함수 ---
def cal_f1(pred_dict, gt_dict):
    """ 예측과 정답 딕셔너리를 비교하여 TP, FP, FN을 계산합니다. """
    tp = 0
    fp = 0
    fn = 0
    
    # 예측값을 기준으로 순회
    for key, pred_value in pred_dict.items():
        gt_value = gt_dict.get(key)
        if gt_value is not None:
            if str(pred_value) == str(gt_value):
                tp += 1
            else:
                fp += 1
                fn += 1
        else:
            fp += 1
            
    # 정답값을 기준으로 순회 (예측에 누락된 값 찾기)
    for key, gt_value in gt_dict.items():
        if key not in pred_dict:
            fn += 1
            
    return tp, fp, fn

# --- 3. PyTorch Dataset 클래스 ---
class DonutDataset(Dataset):
    def __init__(self, dataset_path, image_dir, processor, max_length=512):
        super().__init__()
        self.dataset_path = dataset_path
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.dataset = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                self.dataset.append(json.loads(line))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(os.path.join(self.image_dir, item['file_name'])).convert("RGB")
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        target_sequence = item['ground_truth']
        decoder_input_ids = self.processor.tokenizer(
            target_sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()
        
        return {"pixel_values": pixel_values, "labels": decoder_input_ids, "ground_truth": target_sequence}

# --- 4. 학습 준비 ---
run = wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
    "epochs": EPOCHS, "learning_rate": LEARNING_RATE, 
    "physical_batch_size": PHYSICAL_BATCH_SIZE, "effective_batch_size": EFFECTIVE_BATCH_SIZE
})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
model.to(device)

train_dataset = DonutDataset(dataset_path=TRAIN_METADATA_PATH, image_dir=TRAIN_IMAGE_DIR, processor=processor)
valid_dataset = DonutDataset(dataset_path=VALID_METADATA_PATH, image_dir=VALID_IMAGE_DIR, processor=processor)

train_dataloader = DataLoader(train_dataset, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=PHYSICAL_BATCH_SIZE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# --- 5. 학습 및 검증 루프 ---
print("최적화된 Donut 모델 학습을 시작합니다...")
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}")):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    avg_train_loss = total_train_loss / len(train_dataloader)

    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc=f"Validation Epoch {epoch + 1}/{EPOCHS}"):
            pixel_values = batch["pixel_values"].to(device)
            task_prompt = "<s>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            
            for i, seq in enumerate(outputs.sequences):
                pred_sequence = processor.decode(seq)
                pred_sequence = pred_sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                pred_sequence = re.sub(r"<.*?>", "", pred_sequence, count=1).strip()
                gt_sequence = batch["ground_truth"][i]
                
                try:
                    pred_dict = json.loads(pred_sequence)
                    gt_dict = json.loads(gt_sequence)
                    tp, fp, fn = cal_f1(pred_dict.get('text_sequence', {}), gt_dict.get('text_sequence', {}))
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                except json.JSONDecodeError:
                    total_fp += 1 

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
    
    wandb.log({
        "epoch": epoch + 1, "train_loss": avg_train_loss,
        "precision": precision, "recall": recall, "f1_score": f1_score
    })

run.finish()
print("학습이 완료되었습니다.")