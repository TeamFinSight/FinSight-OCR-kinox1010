# 파이프라인 구성
import torch , cv2 , json , os , glob , torchvision
import numpy as np

from PIL  import Image
from tqdm import tqdm
from transformers import DonutProcessor , VisionEncoderDecoderModel
from Retina_Model import create_custom_retinanet

# 바운딩 이미지 및 결과 저장 경로 설정
INPUT_DIR  = "/mnt/d/Dataset/OCR 데이터(금융)/박스 시각화/바운딩 아웃풋"
OUTPUT_DIR = "/mnt/d/Dataset/OCR 데이터(금융)/박스 시각화/JSON 아웃풋"

# RetinaNet 모델 설정
RETINANET_MODEL_PATH = "/mnt/d/Dataset/프로젝트 모델 임시 저장/ResNet18_Retina_1650/best_retina_model_map_0.7136.pth"
CLASS_NAMES          = {
    0 : "background" ,
    1 : "korean" ,
    2 : "english" ,
    3 : "number" ,
    4 : "mixed_special"
}
RETINANET_NUM_CLASSES = 5
TARGET_CLASS_IDS      = [1 , 2 , 3 , 4]
CONFIDENCE_THRESHOLD  = 0.7

# 허깅페이스 모델 설정
DONUT_MODEL_NAME = "naver-clova-ix/donut-base"

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_retinanet_model(model_path, num_classes):
    """사용자가 훈련시킨 RetinaNet 모델을 로드합니다."""
    print(f"Loading custom RetinaNet model from: {model_path}")
    model = create_custom_retinanet(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    print("Custom RetinaNet model loaded successfully.")
    return model

def load_donut_model_and_processor(model_name):
    """Hugging Face에서 Donut 모델과 프로세서를 로드합니다."""
    print(f"Loading Donut model and processor: {model_name}")
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    print("Donut model and processor loaded successfully.")
    return model, processor

def detect_documents(image_path, model, target_ids, threshold):
    """RetinaNet으로 이미지에서 여러 문서/텍스트 영역을 탐지합니다."""
    try:
        image = Image.open(image_path).convert("RGB")
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Error loading or transforming image {image_path}: {e}")
        return []

    detected_boxes = []
    with torch.no_grad():
        predictions = model(image_tensor)

    for i in range(len(predictions[0]['boxes'])):
        score = predictions[0]['scores'][i].item()
        label_id = predictions[0]['labels'][i].item()
        
        if label_id in target_ids and score > threshold:
            box = predictions[0]['boxes'][i].cpu().numpy().astype(int)
            detected_boxes.append({
                "box": box,
                "label": CLASS_NAMES.get(label_id, "unknown"),
                "score": score
            })
    return detected_boxes

def crop_image_from_box(image_path, box):
    """탐지된 바운딩 박스를 이용해 이미지를 자릅니다."""
    image = cv2.imread(image_path)
    x1, y1, x2, y2 = box
    cropped_image = image[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

def parse_document_with_donut(image, model, processor):
    """Donut 모델로 잘라낸 이미지를 분석합니다."""
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    return processor.token2json(sequence)

# --- [수정] --- 메인 실행 블록 전체 수정
if __name__ == "__main__":
    # --- [추가] --- 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 모델 로드 (한 번만 실행)
    retinanet = load_retinanet_model(RETINANET_MODEL_PATH, RETINANET_NUM_CLASSES)
    donut_model, donut_processor = load_donut_model_and_processor(DONUT_MODEL_NAME)

    # --- [추가] --- 입력 폴더에서 이미지 파일 목록 가져오기
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_path_list = []
    for ext in image_extensions:
        image_path_list.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    print(f"\n총 {len(image_path_list)}개의 이미지를 처리합니다.")

    # --- [추가] --- 모든 결과를 저장할 딕셔너리
    all_results = {}

    # --- [추가] --- 각 이미지 파일을 순회하며 파이프라인 실행
    for image_path in tqdm(image_path_list, desc="Processing Images"):
        image_filename = os.path.basename(image_path)
        print(f"\n{'='*50}\nProcessing: {image_filename}\n{'='*50}")

        # 2. RetinaNet으로 문서/텍스트 영역 위치 모두 탐지
        detected_regions = detect_documents(image_path, retinanet, TARGET_CLASS_IDS, CONFIDENCE_THRESHOLD)
        
        image_results = []
        if detected_regions:
            # 3. 탐지된 각 영역에 대해 Donut 분석 수행
            for i, region in enumerate(detected_regions):
                print(f"  > Analyzing Region #{i+1} (Label: {region['label']}, Score: {region['score']:.2f})")
                
                # 3-1. 이미지 자르기
                cropped_image = crop_image_from_box(image_path, region['box'])
                
                # --- [수정] --- 잘라낸 이미지 저장 (고유한 이름으로)
                base_filename, _ = os.path.splitext(image_filename)
                cropped_save_path = os.path.join(OUTPUT_DIR, f"cropped_{base_filename}_region_{i+1}.png")
                cropped_image.save(cropped_save_path)
                
                # 3-2. Donut으로 내용 분석
                structured_data = parse_document_with_donut(cropped_image, donut_model, donut_processor)
                
                # 3-3. 현재 이미지의 결과 목록에 추가
                region_result = {
                    "region_id": i + 1,
                    "detection_details": {
                        **region , "box": region["box"].tolist()
                    } ,
                    "parsed_data": structured_data,
                    "cropped_image_path": cropped_save_path
                }
                image_results.append(region_result)
        else:
            print("  > No target regions detected in this image.")
        
        # 4. 전체 결과 딕셔너리에 현재 이미지 결과 추가
        all_results[image_filename] = image_results

    # --- [추가] --- 모든 작업 완료 후, 전체 결과를 JSON 파일로 저장
    final_json_path = os.path.join(OUTPUT_DIR, "pipeline_results.json")
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}\n모든 작업이 완료되었습니다.")
    print(f"종합 결과가 다음 파일에 저장되었습니다: {final_json_path}")