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
import shutil  # 파일 복사를 위해 추가
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
# 프롬프트 수정 (영어 버전)
structured_prompt = """
Generate a concise and clear description for the image.
Please format the description as follows: "image type, character type, emotion, action, art style, background"
Examples:
"animation, bear-like character, happy, laughing with arms raised, front view close-up, sticker style, white background"
"animation, cat character, surprised, jumping backward, side view full body, pixel art style, gradient blue background"
"animation, robot mascot, excited, waving hand, three-quarter view medium shot, 3D rendered, transparent background"
"real photo, smiling woman, content, sitting on chair, front view portrait, portrait style, blurry background"
"real photo, golden retriever dog, playful, running with toy, side view full body, natural lighting, park background"
Important rules:
- Begin with specifying whether it's an "animation" or a "real photo"
- Separate all descriptions with commas (,)
- Use concise and clear terms
- Describe the most prominent features of the character first
"""
# Ollama 클라이언트 설정 수정
llm = Ollama(
    base_url="http://127.0.0.1:11435",  # localhost IP 사용
    model="llava:13b",
    temperature=0.01,
    top_p=0.9,
    timeout=30
)
# 연결 테스트 함수 수정
def test_ollama_connection():
    try:
        # 먼저 서버 상태 확인
        response = requests.get("http://127.0.0.1:11435/api/version")
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
# 각 디렉토리별로 결과를 저장할 딕셔너리 생성
results = {}
dataset_dirs = ['/workspace/data/changhyun/dataset/emo_sketch/working/images_for_tagging_2']
# animation 이미지를 저장할 디렉토리 생성
classified_dir = '/workspace/data/changhyun/dataset/emo_sketch/classified_tagged_images'
os.makedirs(classified_dir, exist_ok=True)
# animation 및 real photo 카운터 추가
animation_count = 0
real_photo_count = 0
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
            # 따옴표를 제거하고 앞뒤 공백 제거
            caption = resp.strip().replace('"', '').strip()
            character_id = Path(img_path).parent.name
            # animation인지 real photo인지 확인
            if caption.lower().startswith("animation"):
                # animation인 경우에만 원본 이미지 저장 및 캡션 데이터 추가
                animation_count += 1
                # 원본 이미지를 classified_dir에 복사
                img_filename = Path(img_path).name
                dest_path = os.path.join(classified_dir, img_filename)
                shutil.copy2(img_path, dest_path)
                caption_data = {
                    "image_path": dest_path,
                    "caption": caption
                }
                results[dir_path].append(caption_data)
                print(f"\nAnimation 이미지 저장: {dest_path}")
            else:
                # real photo인 경우 건너뛰기
                real_photo_count += 1
                print(f"\nReal photo 이미지 건너뛰기: {img_path}")
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
    print(f"- Animation 이미지: {animation_count}개 (저장됨)")
    print(f"- Real Photo 이미지: {real_photo_count}개 (건너뜀)")
    print(f"- 소요 시간: {elapsed_time:.2f}초 (이미지당 평균 {elapsed_time/total_images:.2f}초)")
    print("-" * 50)