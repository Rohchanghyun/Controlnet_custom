import os
import sys
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import shutil

# 상위 디렉토리의 모듈을 import 하기 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_adapter.data import DanbooruDataset

def test_dataset():
    # 기본 transform 설정
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # 데이터 경로 설정 (실제 경로에 맞게 수정 필요)
    data_path = "/workspace/data/changhyun/dataset/danbooru_safe/train"
    caption_dir = "/workspace/data/changhyun/dataset/danbooru_safe/captions"

    # 데이터셋 생성
    dataset = DanbooruDataset(
        transform=transform,
        dtype='train',
        data_path=data_path,
        caption_dir=caption_dir
    )

    # 캡션이 없는 이미지 찾기
    missing_captions = []
    for img_path in dataset.imgs:
        if img_path not in dataset.captions:
            missing_captions.append(img_path)
    
    # rest 폴더 경로 설정
    rest_dir = "/workspace/data/changhyun/dataset/danbooru_safe/rest"
    os.makedirs(rest_dir, exist_ok=True)
    
    print("\n캡션이 없는 이미지를 rest 폴더로 이동합니다:")
    for path in sorted(missing_captions):
        filename = os.path.basename(path)
        dest_path = os.path.join(rest_dir, filename)
        print(f"이동: {path} -> {dest_path}")
        shutil.move(path, dest_path)
    
    print(f"\n총 {len(missing_captions)}개의 이미지를 이동했습니다.")

    # DataLoader 생성 및 테스트
    batch_size = 10
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for i, batch in enumerate(dataloader):
        if i == 0:  # 첫 번째 배치만 확인
            images, labels, captions = batch
            
            print(f"배치 크기: {images.shape}")
            print(f"레이블: {labels}")
            
            # 현재 배치의 이미지 경로 출력
            print("\n현재 배치의 이미지 경로:")
            for j in range(len(images)):
                print(f"배치 내 {j}번째 이미지: {dataset.imgs[labels[j]]}")
            
            print("\n캡션:")
            for caption in captions:
                print(caption)
            print("\n")
            break

    # 전체 데이터셋 크기 출력
    print(f"데이터셋 전체 크기: {len(dataset)}")
    print(f"로드된 캡션 수: {len(dataset.captions)}")

if __name__ == "__main__":
    test_dataset()
