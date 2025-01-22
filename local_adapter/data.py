import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re
from pathlib import Path
import random

class StickerPoseDataset(Dataset):
    def __init__(self, sticker_dir, pose_dir, transform=None):
        self.sticker_dir = Path(sticker_dir)
        self.pose_dir = Path(pose_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # 이미지 파일 리스트 생성
        self.sticker_files = sorted([f for f in self.sticker_dir.glob('*.png')])
        all_pose_files = sorted([f for f in self.pose_dir.glob('*.png')])
        
        # sticker 갯수만큼 pose 이미지를 랜덤하게 선택
        self.pose_files = random.sample(all_pose_files, len(self.sticker_files))
        
        # 고정된 캡션
        self.caption = "a photo of character sticker"
        
        # 데이터셋 길이 검증
        assert len(self.sticker_files) > 0, f"스티커 이미지가 없습니다: {sticker_dir}"
        print(f"Dataset size: {len(self.sticker_files)} pairs")

    def __len__(self):
        return len(self.sticker_files)

    def __getitem__(self, idx):
        # 스티커 이미지 로드
        sticker_path = self.sticker_files[idx]
        sticker_img = Image.open(sticker_path).convert('RGB')
        
        # 포즈 이미지 로드
        pose_path = self.pose_files[idx]
        pose_img = Image.open(pose_path).convert('RGB')
        
        # 변환 적용
        if self.transform:
            sticker_img = self.transform(sticker_img)
            pose_img = self.transform(pose_img)
        
        return {
            "target_images": sticker_img,
            "pose_images": pose_img,
            "prompt": self.caption
        }

class TokenStickerDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = Path(base_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # 각 디렉토리 경로 설정 및 존재 여부 확인
        self.visual_tokens_dir = self.base_dir / 'visual_tokens'
        self.sticker_dir = self.base_dir / 'text_removed_sticker'
        self.sketch_dir = self.base_dir / 'sketch'
        
        # 디버깅을 위한 경로 출력
        print(f"Checking directories:")
        print(f"Visual tokens dir: {self.visual_tokens_dir} (exists: {self.visual_tokens_dir.exists()})")
        print(f"Sticker dir: {self.sticker_dir} (exists: {self.sticker_dir.exists()})")
        print(f"Sketch dir: {self.sketch_dir} (exists: {self.sketch_dir.exists()})")
        
        # 파일 리스트 생성 전에 디렉토리 존재 확인
        if not self.visual_tokens_dir.exists():
            raise FileNotFoundError(f"Visual tokens directory not found: {self.visual_tokens_dir}")
        if not self.sticker_dir.exists():
            raise FileNotFoundError(f"Sticker directory not found: {self.sticker_dir}")
        if not self.sketch_dir.exists():
            raise FileNotFoundError(f"Sketch directory not found: {self.sketch_dir}")
        
        # 파일 리스트 생성 및 개수 출력
        self.token_files = sorted([f for f in self.visual_tokens_dir.glob('*.pt')])
        self.sticker_files = sorted([f for f in self.sticker_dir.glob('*.png')])
        print(f"Found {len(self.token_files)} token files")
        print(f"Found {len(self.sticker_files)} sticker files")
        
        # sketch 폴더 내의 모든 이미지 파일을 재귀적으로 수집 (여러 확장자 지원)
        self.sketch_files = []
        image_extensions = ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG')
        
        for ext in image_extensions:
            self.sketch_files.extend(list(self.sketch_dir.rglob(ext)))
        
        print(f"Found {len(self.sketch_files)} sketch files")
        print(f"Example sketch paths:")
        for path in list(self.sketch_files)[:3]:  # 처음 3개의 경로 출력
            print(f" - {path}")
        
        # 데이터셋 길이 검증
        assert len(self.token_files) > 0, f"토큰 텐서가 없습니다: {self.visual_tokens_dir}"
        assert len(self.sticker_files) > 0, f"스티커 이미지가 없습니다: {self.sticker_dir}"
        assert len(self.sketch_files) > 0, f"스케치 이미지가 없습니다: {self.sketch_dir}"
        
        # 가장 작은 데이터셋 크기에 맞춰서 인덱스 생성
        min_size = min(len(self.token_files), len(self.sticker_files))
        self.paired_indices = list(range(min_size))
        random.shuffle(self.paired_indices)
        
        print(f"Dataset size: {len(self.paired_indices)} pairs")
        print(f"Total sketch images available: {len(self.sketch_files)}")

    def __len__(self):
        return len(self.paired_indices)

    def __getitem__(self, idx):
        # 랜덤 인덱스 사용
        rand_idx = self.paired_indices[idx]
        
        # 비주얼 토큰 텐서 로드
        token_path = self.token_files[rand_idx]
        visual_tokens = torch.load(token_path, weights_only=True)
        
        # 스티커 이미지 로드
        sticker_path = self.sticker_files[rand_idx]
        sticker_img = Image.open(sticker_path).convert('RGB')
        sticker_img = self.transform(sticker_img)  # 이미지를 텐서로 변환
        
        # 스케치 이미지 랜덤 선택 및 로드
        sketch_path = random.choice(self.sketch_files)
        sketch_img = Image.open(sketch_path).convert('RGB')
        sketch_img = self.transform(sketch_img)
        
        return {
            "visual_tokens": visual_tokens,      # 텐서
            "sticker_image": sticker_img,        # 이미지->텐서
            "sketch_image": sketch_img          # 이미지->텐서
        }

class Data:
    def __init__(self, opt):
        # # opt에서 데이터 경로 가져오기
        # sticker_dir = os.path.join(opt.data_path, 'text_removed_sticker')
        # pose_dir = os.path.join(opt.data_path, 'skeleton_images')
        
        # print(f"Loading data from: \nSticker path: {sticker_dir}\nPose path: {pose_dir}")
        
        # # 데이터셋 생성
        # self.sticker_pose_dataset = StickerPoseDataset(
        #     sticker_dir=sticker_dir,
        #     pose_dir=pose_dir
        # )
        
        # # DataLoader 생성
        # self.sticker_pose_loader = DataLoader(
        #     self.sticker_pose_dataset,
        #     batch_size=opt.batchsize,  # opt에서 배치 사이즈도 가져옴
        #     shuffle=True,
        #     num_workers=4,
        #     pin_memory=True
        # )

        # opt에서 데이터 경로 가져오기
        base_dir = os.path.join(opt.data_path, 'pose_attention')
        
        print(f"Loading data from: {base_dir}")
        
        # 데이터셋 생성
        self.token_sticker_dataset = TokenStickerDataset(
            base_dir=base_dir
        )
        
        # DataLoader 생성
        self.token_sticker_loader = DataLoader(
            self.token_sticker_dataset,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
