from langchain_community.llms import Ollama
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
"Generate an image caption strictly following this format: 'A sticker image of [appearance] character'. Focus on distinctive visual traits that make the character instantly recognizable, whether humanoid, animal, or abstract. Describe ONLY:
Texture/patterns (e.g., scaly, holographic, knitted)
Unique anatomical features (antennae, mechanical joints, floating limbs)
Shape language (geometric, asymmetrical, exaggerated proportions)
Surface details (transparent, matte, glossy)
Proportion relationships (oversized head, miniature limbs)

Examples:
1. 'A sticker image of a textured-furred fox character with hexagonal-patterned ears and segmented tail'
2. 'A sticker image of a cloud-shaped character with lightning-bolt eyes and patterned edges'

Avoid:
- Any emotional descriptors (smile, angry, etc)
- Color-related attributes (ombre, iridescent, etc)
- Subjective qualifiers (cute, beautiful, etc)
- Any non-visual characteristics"
"""

# Ollama 클라이언트 설정 수정
llm = Ollama(
    base_url="http://127.0.0.1:11434",
    model="llava:13b",
    temperature=0.01,
    top_p=0.9,
    timeout=60  # 시간 좀 더 늘림
)

# 연결 테스트 함수 수정
def test_ollama_connection():
    try:
        # GPU 설정 확인
        print("GPU 설정: CUDA_VISIBLE_DEVICES =", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))
        
        # 서버 상태 확인
        response = requests.get("http://127.0.0.1:11434/api/version", timeout=10)
        print(f"Ollama 서버 상태: {response.json()}")
        
        # 모델 테스트
        response = llm.invoke("Hello")
        print("Ollama 서버 연결 성공!")
        return True
    except requests.exceptions.Timeout:
        print("서버 연결 시간 초과. GPU 초기화 중일 수 있습니다. 잠시 후 다시 시도해주세요.")
        return False
    except Exception as e:
        print(f"Ollama 서버 연결 실패: {str(e)}")
        print("1. CUDA_VISIBLE_DEVICES=0 ollama serve 명령어로 서버를 실행했는지 확인해주세요.")
        print("2. 서버 실행 후 1-2분 정도 기다려주세요.")
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

# 각 디렉토리별로 결과를 저장할 딕셔너리 생성
results = {}
dataset_dirs = ['/workspace/data/changhyun/dataset/character_emoticon_data/text_removed_sticker']

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
    output_dir = "captions"
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


