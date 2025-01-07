from langchain.llms import Ollama
from langchain.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from PIL import Image
import base64
from io import BytesIO
from deep_translator import GoogleTranslator
import os
import json
from pathlib import Path
import requests
import time
from tqdm import tqdm

def convert_to_base64(pil_image: Image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def load_image(image_path: str):
    pil_image = Image.open(image_path)
    image_b64 = convert_to_base64(pil_image)
    print("Loaded image successfully!")
    return image_b64

def translate(text, target_lang='en'):
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

# 프롬프트 수정
structured_prompt = """
Describe the character in a single sentence following this exact format:
"A [emotion] character [face emotion expression]"

Keep it concise and direct. Don't use phrases like 'the character is' or 'we can see'.
Example: "A round-faced character wearing a red dress doing a happy dance"
"""

# Ollama 클라이언트 설정 수정
llm = Ollama(
    base_url="http://127.0.0.1:11434",  # localhost IP 사용
    model="llava:13b",
    temperature=0.01,
    top_p=0.9,
    timeout=30
)

# 연결 테스트 함수 수정
def test_ollama_connection():
    try:
        # 먼저 서버 상태 확인
        response = requests.get("http://127.0.0.1:11434/api/version")
        print(f"Ollama 서버 상태: {response.json()}")
        
        # 모델 테스트
        response = llm.invoke("Hello")
        print("Ollama 서버 연결 성공!")
        return True
    except Exception as e:
        print(f"Ollama 서버 연결 실패: {str(e)}")
        print("호스트 IP와 포트가 올바른지 확인해주세요.")
        return False

# 메인 코드 실행 전에 연결 테스트
if not test_ollama_connection():
    print("Ollama 서버 연결을 확인해주세요.")
    exit(1)

def get_image_files(directory):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif')
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

# 경로 확인을 위한 코드 추가
base_path = '/workspace/data/changhyun/dataset/test/'
print(f"기본 경로 존재 여부: {os.path.exists(base_path)}")

# 처리할 폴더 범위 지정
dataset_dirs = []
for i in range(149,151):  # 0000 ~ 0050
    folder_name = f"{i:04d}"
    folder_path = os.path.join(base_path, folder_name)
    if os.path.exists(folder_path):
        dataset_dirs.append(folder_path)
    else:
        print(f"폴더를 찾을 수 없음: {folder_path}")

# 결과 확인
print(f"\n처리할 폴더 수: {len(dataset_dirs)}")
if dataset_dirs:  # 리스트가 비어있지 않은 경우에만 출력
    print("첫 번째 폴더:", dataset_dirs[0])
    print("마지막 폴더:", dataset_dirs[-1])
else:
    print("처리할 폴더를 찾을 수 없습니다!")

# 각 디렉토리별로 결과를 저장할 딕셔너리 생성
results = {}

for dir_path in dataset_dirs:
    start_time = time.time()
    print(f"\n디렉토리 처리 시작: {Path(dir_path).name}")
    results[dir_path] = []
    image_files = get_image_files(dir_path)
    total_images = len(image_files)
    
    for idx, img_path in enumerate(tqdm(image_files, desc="이미지 처리 중")):
        try:
            image_b64 = load_image(img_path)
            resp = llm.invoke(structured_prompt, images=[image_b64])
            caption = resp.strip().replace('\n', ' ')
            character_id = Path(img_path).parent.name
            
            caption_data = {
                "image_path": img_path,
                "ID": character_id,
                "caption": caption
            }
            
            results[dir_path].append(caption_data)
            
        except Exception as e:
            print(f"\n오류 발생 ({img_path}): {str(e)}")
            continue
    
    # 각 디렉토리별로 JSON 파일 저장
    output_dir = "/workspace/data/changhyun/projects/emoji_generation/Controlnet_custom/caption_generation/test"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{Path(dir_path).name}_captions.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results[dir_path], f, ensure_ascii=False, indent=2)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n{Path(dir_path).name} 처리 완료:")
    print(f"- 처리된 이미지: {total_images}개")
    print(f"- 소요 시간: {elapsed_time:.2f}초 (이미지당 평균 {elapsed_time/total_images:.2f}초)")
    print("-" * 50)


