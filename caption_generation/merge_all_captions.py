import json
import os
from collections import defaultdict

# 설정
merged_filtered_json = "./captions/merged_filtered_captions.json"
caption_json = "/workspace/data/changhyun/dataset/emo_sketch/caption_other_dir.json"
output_json_path = "./captions/final_merged_captions.json"

# 두 JSON 파일 읽기
all_captions = []

# 첫 번째 파일 (merged_filtered_captions.json) 읽기
try:
    with open(merged_filtered_json, "r", encoding="utf-8") as f:
        data1 = json.load(f)
        print(f"{merged_filtered_json}에서 {len(data1)}개 항목 로드됨")
        all_captions.extend(data1)
except Exception as e:
    print(f"{merged_filtered_json} 파일 처리 중 오류 발생: {e}")

# 두 번째 파일 (caption.json) 읽기
try:
    with open(caption_json, "r", encoding="utf-8") as f:
        data2 = json.load(f)
        print(f"{caption_json}에서 {len(data2)}개 항목 로드됨")
        all_captions.extend(data2)
except Exception as e:
    print(f"{caption_json} 파일 처리 중 오류 발생: {e}")

print(f"병합 전 전체 캡션 수: {len(all_captions)}개")

# 중복 제거 (이미지 경로 기준)
unique_captions = {}
for item in all_captions:
    if "image_path" in item:
        image_path = item["image_path"]
        # 이미 존재하는 경우 덮어쓰기 (최신 데이터 우선)
        unique_captions[image_path] = item

final_captions = list(unique_captions.values())
print(f"중복 제거 후 최종 캡션 수: {len(final_captions)}개")
print(f"제거된 중복 항목 수: {len(all_captions) - len(final_captions)}개")

# 결과를 JSON 파일로 저장
try:
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_captions, f, ensure_ascii=False, indent=2)
    print(f"최종 결과가 {output_json_path}에 저장되었습니다.")
except Exception as e:
    print(f"결과 저장 중 오류 발생: {e}")

# 이미지 경로 확인
image_dirs = defaultdict(int)
for item in final_captions:
    if "image_path" in item:
        dir_path = os.path.dirname(item["image_path"])
        image_dirs[dir_path] += 1

print("\n이미지 디렉토리 분포:")
for dir_path, count in sorted(image_dirs.items(), key=lambda x: x[1], reverse=True):
    print(f"{dir_path}: {count}개") 