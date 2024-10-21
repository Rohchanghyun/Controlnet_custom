
import os
import json

# train 폴더 내 모든 이미지에 대해 캡션을 생성하고 JSON 파일로 저장하는 함수
def create_temp_caption_json(train_dir, output_json_path):
    data = []
    caption_text = "a photo of a person"  # 모든 이미지에 할당할 캡션

    # train 폴더 내의 각 ID 폴더에 접근
    id_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    
    # 각 ID 폴더에 있는 모든 이미지 파일에 대해 처리
    for id_folder in id_folders:
        id_path = os.path.join(train_dir, id_folder)

        # ID 폴더 내의 모든 이미지 파일에 대해 경로와 캡션 추가
        image_files = [f for f in os.listdir(id_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(id_path, image_file)
            
            # 각 이미지에 대해 정보 저장
            data.append({
                "image_path": image_path,
                "ID": id_folder,
                "caption": caption_text
            })
    
    # 결과를 JSON 파일로 저장
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON 파일이 {output_json_path}에 저장되었습니다.")

# 사용 예시
train_dir = "/workspace/mnt/sda/changhyun/dataset/top15character/query"  # train 폴더 경로
output_json_path = "temp_text.json"  # 저장할 json 파일 이름
create_temp_caption_json(train_dir, output_json_path)