import json

# 파일 경로 설정
file1 = "./captions/images_for_tagging_3_captions.json"  # 첫 번째 파일 이름을 여기에 입력하세요
file2 = "./captions/images_for_tagging_3_captions.json"
file3 = "./captions/images_for_tagging_captions.json"
output_file = "./captions/merged_result.json"  # 결과 파일 이름

# 모든 파일 데이터 읽기
try:
    with open(file1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    
    with open(file2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    
    with open(file3, "r", encoding="utf-8") as f:
        data3 = json.load(f)
        
    # 데이터 병합하기
    if isinstance(data1, list) and isinstance(data2, list) and isinstance(data3, list):
        merged_data = data1 + data2 + data3
        print(f"리스트 형식의 데이터 {len(merged_data)}개 항목 병합됨")
    elif isinstance(data1, dict) and isinstance(data2, dict) and isinstance(data3, dict):
        merged_data = {**data1, **data2, **data3}
        print(f"딕셔너리 형식의 데이터 {len(merged_data)}개 키 병합됨")
    else:
        print("파일 형식이 다릅니다. 수동으로 병합이 필요할 수 있습니다.")
        exit(1)
    
    # 결과 저장하기 (원본 파일은 변경되지 않음)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"파일 병합이 완료되었습니다. '{output_file}' 파일을 확인하세요.")
    
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
except json.JSONDecodeError:
    print("JSON 파일 형식이 잘못되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}")
