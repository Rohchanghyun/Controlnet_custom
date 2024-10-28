import os
import json
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import torch  # GPU 사용을 위해 추가

# LLaVA 모델 및 프로세서를 로드하는 함수
def load_llava_model():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, pad_token_id=processor.tokenizer.eos_token_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor, device


def generate_caption(image_path, model, processor, device, prompt):
    # 이미지를 로드하고, 흑백 이미지가 있을 경우 RGB로 강제 변환
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert("RGB")  # 이미지를 강제로 RGB로 변환

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    # 모델을 통해 caption 생성
    try:
        with torch.cuda.amp.autocast():
            out = model.generate(**inputs, max_new_tokens=25)
        if out is None or len(out) == 0:
            return "캡션을 생성할 수 없습니다."
        caption = processor.decode(out[0], skip_special_tokens=True)
        # 프롬프트 제거 및 'ASSISTANT:' 이후의 텍스트만 반환
        caption = caption.split('ASSISTANT:')[-1].strip()
        return caption
    except Exception as e:
        print(f"캡션 생성 중 오류 발생: {str(e)}")
        return "캡션 생성 중 오류 발생"
    finally:
        torch.cuda.empty_cache()


# 이미지 경로들을 모아주고 caption 생성
def create_caption_json_for_split(split_path, split_name, output_dir):
    data = []

    # 모델과 프로세서 로드 (GPU 사용 설정 포함)
    model, processor, device = load_llava_model()
    #prompt = "USER: <image>\nPlease provide a detailed description of the character's appearance and actions in this image.\nASSISTANT:"
    #prompt = "USER: <image>\nPlease provide a detailed description of the character's appearance and actions in this image. Include specific details about their physical features, facial expression, posture, and any movements or gestures they are making.\nASSISTANT:"
    prompt = "USER: <image>\nBrieflydescribe the character directly, focusing on appearance and actions. Omit phrases like 'The character in the image'. And don't describe character's name.\nASSISTANT:"
    # 각 ID 폴더에 접근
    id_folders = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
    
    # ID 폴더 순회 - tqdm을 사용하여 진행 상황 표시
    for id_folder in tqdm(id_folders, desc=f"Processing {split_name}", unit="folder"):
        id_path = os.path.join(split_path, id_folder)

        # ID 폴더 내 모든 이미지 파일에 대해 caption 생성
        image_files = [f for f in os.listdir(id_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue

        # 각 이미지에 대해 캡션 생성 - tqdm을 사용하여 이미지 처리 상황 표시
        for image_file in tqdm(image_files, desc=f"Captions for {id_folder}", unit="image", leave=False):
            image_path = os.path.join(id_path, image_file)
            try:
                caption = generate_caption(image_path, model, processor, device, prompt)
                
                # 해당 이미지에 대한 정보 저장
                data.append({
                    "image_path": image_path,
                    "ID": id_folder,
                    "caption": caption
                })
            except Exception as e:
                # 에러가 발생한 우 파일 경로와 에러 메시지 출력
                print(f"Error occurred with file: {image_path}, Error: {str(e)}")
                data.append({
                    "image_path": image_path,
                    "ID": id_folder,
                    "caption": "오류로 인해 캡션을 생성할 수 없습니다."
                })
            finally:
                torch.cuda.empty_cache()

    # Split에 대한 JSON 파일로 저장
    output_json_path = os.path.join(output_dir, f"{split_name}_captions.json")
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON 파일이 {output_json_path}에 저장되었습니다.")

# 사용 예시
def create_caption_jsons(root_dir, output_dir):
    # train, test, query, train_limit 각각에 대해 처리
    for split in ['abc','train','query','test']:
        split_path = os.path.join(root_dir, split)
        print(split_path)
        if os.path.exists(split_path):
            create_caption_json_for_split(split_path, split, output_dir)

# 실행
root_dir = "/workspace/data/changhyun/dataset/emoji_data/"  # top15character 폴더 경로
output_dir = "/workspace/data/changhyun/dataset/emoji_data/captions_llava/"  # 저장할 폴더
os.makedirs(output_dir, exist_ok=True)  # output 폴더 생성
create_caption_jsons(root_dir, output_dir)
