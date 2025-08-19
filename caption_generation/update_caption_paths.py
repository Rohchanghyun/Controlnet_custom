import json
import os

# 설정
caption_json = "/workspace/data/changhyun/dataset/emo_sketch/caption_other_dir.json"
output_json_path = "/workspace/data/changhyun/dataset/emo_sketch/caption.json"

# JSON 파일 읽기
try:
    with open(caption_json, "r", encoding="utf-8") as f:
        captions = json.load(f)
    print(f"{caption_json}에서 {len(captions)}개 항목 로드됨")
except Exception as e:
    print(f"{caption_json} 파일 처리 중 오류 발생: {e}")
    exit(1)

# 경로 변경 카운터
changed_count = 0
unchanged_count = 0

# 이미지 경로 업데이트
for item in captions:
    if "image_path" in item:
        orig_path = item["image_path"]
        
        # conditioning 디렉토리를 condition_sketch로 변경
        if "/conditioning/" in orig_path:
            new_path = orig_path.replace("/conditioning/", "/condition_sketch/")
            item["image_path"] = new_path
            changed_count += 1
        elif "/workspace/data/changhyun/dataset/emo_sketch/" in orig_path:
            # 이미 다른 경로인 경우, condition_sketch 디렉토리로 업데이트
            basename = os.path.basename(orig_path)
            new_path = f"/workspace/data/changhyun/dataset/emo_sketch/condition_sketch/{basename}"
            item["image_path"] = new_path
            changed_count += 1
        else:
            unchanged_count += 1

print(f"총 {len(captions)}개 항목 중 {changed_count}개 경로 변경됨, {unchanged_count}개 변경되지 않음")

# 결과를 JSON 파일로 저장
try:
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    print(f"업데이트된 결과가 {output_json_path}에 저장되었습니다.")
    
    # 원본 파일 백업 및 새 파일로 교체
    backup_file = f"{caption_json}.bak"
    os.rename(caption_json, backup_file)
    os.rename(output_json_path, caption_json)
    print(f"원본 파일이 {backup_file}로 백업되고, 업데이트된 파일이 {caption_json}로 교체되었습니다.")
except Exception as e:
    print(f"결과 저장 중 오류 발생: {e}") 