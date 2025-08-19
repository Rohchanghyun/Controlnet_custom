import json
import os
import glob
import random

# 설정
target_directory = "/workspace/data/changhyun/dataset/emo_sketch/working/sr_sticker_2x/"  # 실제 이미지가 있는 디렉토리
json_path = "./captions/filtered_result_2x.json"  # 원본 병합 JSON 파일 (모든 캡션 포함)

# 디렉토리의 이미지 파일 목록 가져오기
image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
image_files = []
for ext in image_extensions:
    pattern = os.path.join(target_directory, f"*{ext}")
    image_files.extend(glob.glob(pattern))
    # 대소문자 구분없이 확장자 검색
    pattern = os.path.join(target_directory, f"*{ext.upper()}")
    image_files.extend(glob.glob(pattern))

print(f"디렉토리 내 이미지 파일 수: {len(image_files)}개")

# 이미지 파일명에서 _2x 제거하기
normalized_image_filenames = {}
for path in image_files:
    filename = os.path.basename(path)
    filename_base, filename_ext = os.path.splitext(filename)
    # _2x 제거
    if filename_base.endswith("_2x"):
        normalized_name = filename_base[:-3] + filename_ext
    else:
        normalized_name = filename
    normalized_image_filenames[normalized_name] = path

print(f"정규화된 이미지 파일명 수: {len(normalized_image_filenames)}개")

# JSON 파일 읽기
try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"JSON 파일 항목 수: {len(data)}개")
    
    # JSON의 이미지 파일명 추출
    json_filenames = {}
    for item in data:
        if "image_path" in item:
            json_filename = os.path.basename(item["image_path"])
            json_filenames[json_filename] = item
    
    # 매칭되는 항목 찾기
    matched_files = 0
    missing_in_json = []
    
    for norm_name, img_path in normalized_image_filenames.items():
        if norm_name in json_filenames:
            matched_files += 1
        else:
            # 이미지는 있지만 JSON에 없는 경우
            missing_in_json.append(os.path.basename(img_path))
    
    # 결과 출력
    print(f"\n=== 비교 결과 ===")
    print(f"매칭된 파일: {matched_files}개")
    print(f"JSON에 없는 이미지: {len(missing_in_json)}개")
    
    if missing_in_json:
        print("\n=== JSON에 없는 이미지 파일 예시 (최대 10개) ===")
        for idx, filename in enumerate(missing_in_json[:10]):
            print(f"{idx+1}. {filename}")
            # 정규화 전 이름도 출력
            orig_name = filename
            normalized = os.path.splitext(orig_name)[0]
            if normalized.endswith("_2x"):
                normalized = normalized[:-3] + os.path.splitext(orig_name)[1]
            print(f"   정규화 전: {orig_name}")
            print(f"   정규화 후: {normalized}")
        
        if len(missing_in_json) > 10:
            print(f"... 외 {len(missing_in_json) - 10}개")
    
    # JSON에는 있지만 이미지 파일이 없는 항목
    missing_images = []
    for json_name, item in json_filenames.items():
        if json_name not in normalized_image_filenames:
            # 이름에 _2x를 붙여서도 확인
            filename_base, filename_ext = os.path.splitext(json_name)
            modified_name = filename_base + "_2x" + filename_ext
            
            # 수정된 이름으로도 찾아보기
            if modified_name not in normalized_image_filenames.values():
                missing_images.append(json_name)
    
    print(f"\nJSON에는 있지만 이미지 파일이 없는 항목: {len(missing_images)}개")
    
    if missing_images:
        print("\n=== 이미지 파일이 없는 JSON 항목 예시 (최대 10개) ===")
        for idx, filename in enumerate(missing_images[:10]):
            print(f"{idx+1}. {filename}")
        if len(missing_images) > 10:
            print(f"... 외 {len(missing_images) - 10}개")
    
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
except json.JSONDecodeError:
    print("JSON 파일 형식이 잘못되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}") 