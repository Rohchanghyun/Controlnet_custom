import json
import os
import glob
from collections import defaultdict

# 설정
json_files = [
    "./captions/images_for_tagging_captions.json",
    "./captions/images_for_tagging_2_captions.json",
    "./captions/images_for_tagging_3_captions.json"
]
target_image_dir = "/workspace/data/changhyun/dataset/emo_sketch/working/sr_sticker_2x/"
new_image_dir = "/workspace/data/changhyun/dataset/emo_sketch/condition_sketch/"
output_json_path = "./captions/merged_filtered_captions.json"
missing_images_json = "./captions/missing_images.json"
duplicate_captions_json = "./captions/duplicate_captions.json"

# 타겟 디렉토리의 이미지 파일 목록 가져오기
image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
image_files = []
for ext in image_extensions:
    pattern = os.path.join(target_image_dir, f"*{ext}")
    image_files.extend(glob.glob(pattern))
    # 대소문자 구분없이 확장자 검색
    pattern = os.path.join(target_image_dir, f"*{ext.upper()}")
    image_files.extend(glob.glob(pattern))

print(f"타겟 디렉토리 내 이미지 파일 수: {len(image_files)}개")

# 이미지 파일명 추출 (파일 전체 경로를 키로 사용)
available_images = set()
image_basenames = {}  # 원본 파일명 -> _2x 확장된 파일명

for path in image_files:
    basename = os.path.basename(path)
    available_images.add(basename)
    
    # 원본 파일명 역추적 (_2x 제거한 버전)
    filename_base, filename_ext = os.path.splitext(basename)
    if filename_base.endswith("_2x"):
        orig_basename = filename_base[:-3] + filename_ext
        image_basenames[orig_basename] = basename

print(f"이미지 파일 수: {len(available_images)}개")

# 모든 JSON 파일에서 데이터 로드하여 통합
all_captions = []
for json_file in json_files:
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"{json_file}에서 {len(data)}개 항목 로드됨")
            all_captions.extend(data)
    except Exception as e:
        print(f"{json_file} 파일 처리 중 오류 발생: {e}")

print(f"통합 전 전체 캡션 수: {len(all_captions)}개")

# 중복 캡션 확인을 위한 딕셔너리
filename_to_captions = defaultdict(list)
for item in all_captions:
    if "image_path" in item:
        filename = os.path.basename(item["image_path"])
        filename_to_captions[filename].append(item)

# 중복 캡션이 있는 파일 확인
duplicates = {filename: captions for filename, captions in filename_to_captions.items() if len(captions) > 1}
print(f"중복 캡션이 있는 파일 수: {len(duplicates)}개")

# 최종 결과를 저장할 리스트
filtered_captions = []
missing_images = []
used_images = set()

# 각 캡션 항목 처리
for item in all_captions:
    if "image_path" in item:
        # 원본 이미지 경로에서 파일명만 추출
        orig_filename = os.path.basename(item["image_path"])
        
        # 파일명 뒤에 _2x를 붙인 형태로 변환
        filename_base, filename_ext = os.path.splitext(orig_filename)
        new_filename = f"{filename_base}_2x{filename_ext}"
        
        # 해당 이미지가 target_image_dir에 존재하는지 확인
        if new_filename in available_images:
            # 이미지 경로 업데이트
            item["image_path"] = os.path.join(new_image_dir, new_filename)
            filtered_captions.append(item)
            used_images.add(new_filename)
        else:
            # 혹시 원본 파일명으로 매칭이 되는지 확인
            if orig_filename in image_basenames:
                actual_filename = image_basenames[orig_filename]
                item["image_path"] = os.path.join(new_image_dir, actual_filename)
                filtered_captions.append(item)
                used_images.add(actual_filename)
            else:
                missing_images.append(item)

# 사용되지 않은 이미지 파일 찾기
unused_images = available_images - used_images
print(f"필터링 후 최종 캡션 수: {len(filtered_captions)}개")
print(f"제외된 항목 수: {len(all_captions) - len(filtered_captions)}개")
print(f"매칭되지 않은 이미지 수: {len(missing_images)}개")
print(f"사용되지 않은 이미지 수: {len(unused_images)}개")

# 결과를 JSON 파일로 저장
try:
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_captions, f, ensure_ascii=False, indent=2)
    print(f"결과가 {output_json_path}에 저장되었습니다.")
    
    # 매칭되지 않은 이미지 목록 저장
    with open(missing_images_json, "w", encoding="utf-8") as f:
        json.dump(missing_images, f, ensure_ascii=False, indent=2)
    print(f"매칭되지 않은 이미지 목록이 {missing_images_json}에 저장되었습니다.")
    
    # 중복 캡션 정보 저장
    with open(duplicate_captions_json, "w", encoding="utf-8") as f:
        dup_info = {k: [item["caption"] for item in v] for k, v in duplicates.items()}
        json.dump(dup_info, f, ensure_ascii=False, indent=2)
    print(f"중복 캡션 정보가 {duplicate_captions_json}에 저장되었습니다.")
    
    # 사용되지 않은 이미지 목록 저장 (최대 100개만)
    unused_images_list = list(unused_images)[:100]
    print("\n사용되지 않은 이미지 예시 (최대 100개):")
    for i, img in enumerate(unused_images_list):
        print(f"{i+1}. {img}")
        
except Exception as e:
    print(f"결과 저장 중 오류 발생: {e}") 