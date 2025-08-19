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

# 이미지 파일명과 경로 매핑
image_paths = {}
for path in image_files:
    basename = os.path.basename(path)
    image_paths[basename] = path

print(f"이미지 파일 수: {len(image_paths)}개")

# 원본 파일명 -> _2x 파일명 매핑
image_basenames = {}
for basename, path in image_paths.items():
    filename_base, filename_ext = os.path.splitext(basename)
    if filename_base.endswith("_2x"):
        orig_basename = filename_base[:-3] + filename_ext
        image_basenames[orig_basename] = basename

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

# 사용된 이미지 파일 추적
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
        if new_filename in image_paths:
            used_images.add(new_filename)
        elif orig_filename in image_basenames:
            # 원본 파일명을 통해 매칭
            actual_filename = image_basenames[orig_filename]
            used_images.add(actual_filename)

# 사용되지 않은 이미지 파일 찾기
unused_images = set(image_paths.keys()) - used_images
print(f"캡션이 있는 이미지 수: {len(used_images)}개")
print(f"캡션이 없는 이미지 수: {len(unused_images)}개")

# 삭제할 이미지 개수 확인 및 사용자 확인
print(f"\n총 {len(unused_images)}개의 이미지 파일을 삭제할 예정입니다.")
confirm = input("정말로 삭제하시겠습니까? (y/n): ")

if confirm.lower() == 'y':
    # 삭제 카운터
    deleted_count = 0
    error_count = 0
    
    # 캡션이 없는 이미지 파일 삭제
    for basename in unused_images:
        file_path = image_paths[basename]
        try:
            os.remove(file_path)
            deleted_count += 1
            if deleted_count % 1000 == 0:
                print(f"{deleted_count}개 파일 삭제 완료...")
        except Exception as e:
            print(f"파일 삭제 오류: {file_path} - {e}")
            error_count += 1
    
    print(f"\n삭제 완료: {deleted_count}개 파일 삭제됨, {error_count}개 오류 발생")
else:
    print("삭제가 취소되었습니다.")

# 삭제되지 않은 이미지 예시 출력
if confirm.lower() != 'y':
    unused_images_list = list(unused_images)[:20]
    print("\n삭제 예정이었던 파일 예시 (최대 20개):")
    for i, img in enumerate(unused_images_list):
        print(f"{i+1}. {img} => {image_paths[img]}") 