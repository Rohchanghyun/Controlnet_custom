import torch
from data import Data
from tqdm import tqdm
import pdb
# Data 클래스 인스턴스 생성
data = Data()

# 데이터로더 가져오기
train_loader = data.train_loader
test_loader = data.test_loader
query_loader = data.query_loader

def check_dataloader(loader, name):
    print(f"\n{name} 데이터로더 확인:")
    for batch_idx, (images, labels, captions) in enumerate(tqdm(loader)):
        print(f"\n배치 {batch_idx + 1}:")
        print(f"이미지 shape: {images.shape}")
        print(f"레이블: {labels}")
        print(f"캡션 예시: {captions}")
        
        if batch_idx == 2:  # 3개의 배치만 확인
            break

# 각 데이터로더 확인
check_dataloader(train_loader, "Train")
check_dataloader(test_loader, "Test")
check_dataloader(query_loader, "Query")

pdb.set_trace()
print("\n데이터로더 확인이 완료되었습니다.")
