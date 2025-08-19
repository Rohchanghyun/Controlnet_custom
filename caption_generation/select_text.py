import json
import os
import pathlib

# 설정
merged_json_path = "./captions/merged_result.json"  # 병합된 JSON 파일 경로
target_directory = "/workspace/data/changhyun/dataset/emo_sketch/working/sr_sticker_2x/"  # 실제 이미지 확인할 디렉토리
output_json_path = "./captions/filtered_result_2x.json"  # 결과 파일 경로

# 병합된 JSON 파일 읽기
try:
    with open(merged_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 필터링된 데이터를 저장할 리스트
    filtered_data = []
    excluded_count = 0
    
    # 각 항목 확인
    for item in data:
        if "image_path" in item:
            # 원본 경로에서 파일명만 추출
            original_path = item["image_path"]
            original_filename = os.path.basename(original_path)
            
            # 파일명에 _2x가 없으면 추가
            filename_base, filename_ext = os.path.splitext(original_filename)
            if not filename_base.endswith("_2x"):
                modified_filename = filename_base + "_2x" + filename_ext
            else:
                modified_filename = original_filename
            
            # 대상 디렉토리에 파일이 존재하는지 확인
            target_path = os.path.join(target_directory, modified_filename)
            if os.path.isfile(target_path):
                # 파일이 존재하면 새 경로로 수정하여 포함
                new_item = item.copy()
                # 원본 경로의 디렉토리 부분 추출
                dir_name = os.path.dirname(original_path)
                # 새 파일명으로 경로 생성 (대상 디렉토리를 기준으로)
                new_item["image_path"] = os.path.join(target_directory, modified_filename)
                filtered_data.append(new_item)
            else:
                excluded_count += 1
        else:
            # image_path가 없는 항목은 포함 (혹시 다른 형식이 있을 경우)
            filtered_data.append(item)
    
    # 결과 저장
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    
    print(f"필터링 완료:")
    print(f"- 원본 항목 수: {len(data)}")
    print(f"- 필터링된 항목 수: {len(filtered_data)}")
    print(f"- 제외된 항목 수: {excluded_count}")
    print(f"결과가 '{output_json_path}' 파일에 저장되었습니다.")
    
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
except json.JSONDecodeError:
    print("JSON 파일 형식이 잘못되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}")