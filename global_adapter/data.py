from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from opt import opt
import os
import json
import re
import random
import torch
import glob


class Data():
    def __init__(self):
        train_transform = transforms.Compose([
            # 이미지를 RGB 모드로 변환하여 투명도 채널 제거
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((768, 768), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((768, 768), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.vis_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor()
        ])
        
        # 기존 경로 코드 주석 처리
        """
        train_dir = os.path.join(opt.data_path, 'train')
        test_dir = os.path.join(opt.data_path, 'test')
        caption_dir = os.path.join(opt.data_path, 'captions')
        
        # Train IDs 가져오기
        train_ids = [
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
        train_ids.sort()  # 일관된 순서를 위해 정렬

        # Test IDs 가져오기 (test와 query는 동일한 ID를 가집니다)
        test_ids = [
            d for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d))
        ]
        test_ids.sort()

        # 테스트 및 쿼리 세트를 위한 id2label 매핑 생성
        id2label = {id_name: idx for idx, id_name in enumerate(test_ids)}
        
        self.trainset = DanbooruDataset(train_transform, 'train', train_dir, caption_dir=caption_dir)
        self.testset = DanbooruDataset(test_transform, 'test', test_dir, caption_dir=caption_dir, id2label=id2label)
        """
        
        # EmoSketch 데이터셋 경로 설정
        image_dir = os.path.join(opt.data_path, 'filtered_images')
        #image_dir = os.path.join(opt.data_path, 'image_low')
        caption_file = os.path.join(opt.data_path, 'caption.json')
        
        # EmoSketchDataset 사용
        self.trainset = EmoSketchDataset(train_transform, image_dir, caption_file)
        self.testset = EmoSketchDataset(test_transform, image_dir, caption_file)
        
        # 데이터로더 생성
        self.train_loader = DataLoader(self.trainset, batch_size=opt.batchid * opt.batchimage,
                                      shuffle=True, num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(self.testset, batch_size=opt.batchtest,
                                     shuffle=True, num_workers=8, pin_memory=True)

        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))

class Top15Dataset(Dataset):
    def __init__(self, transform, dtype, data_path, id2label=None, caption_file=None):
        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path
        self.dtype = dtype

        # 데이터 경로에서 모든 ID 디렉토리 가져오기
        self.id_dirs = [
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ]
        self.id_dirs.sort()  # 일관된 순서를 위해 정렬

        # id2label 매핑 생성 또는 기존 매핑 사용
        if id2label is None:
            self.id2label = {id_name: idx for idx, id_name in enumerate(self.id_dirs)}
        else:
            self.id2label = id2label

        # 이미지와 라벨 수집
        self.imgs = []
        self.labels = []
        for id_name in self.id_dirs:
            id_dir = os.path.join(self.data_path, id_name)
            label = self.id2label[id_name]
            image_files = self.list_pictures(id_dir)
            for img_path in image_files:
                self.imgs.append(img_path)
                self.labels.append(label)

        # IDs와 카메라 IDs 할당
        self._ids = [self.id(path) for path in self.imgs]
        if dtype == 'query':
            self._cameras = [0] * len(self.imgs)  # 쿼리 이미지에 카메라 ID 0 할당
        else:
            self._cameras = [1] * len(self.imgs)  # 테스트/갤러리 이미지에 카메라 ID 1 할당

        self._unique_ids = sorted(set(self._ids))

        # 캡션 로드 및 매칭 개선
        self.captions = {}
        if caption_file is not None:
            with open(caption_file, 'r') as f:
                caption_data = json.load(f)
                for item in caption_data:
                    # 전체 상대 경로를 사용하여 매칭
                    image_path = os.path.normpath(item['image_path'])
                    # data_path를 기준으로 한 상대 경로 생성
                    relative_path = os.path.relpath(image_path, self.data_path)
                    full_path = os.path.join(self.data_path, relative_path)
                    self.captions[full_path] = item['caption']

        # 디버그용 코드
        print("Caption file paths:", list(self.captions.keys())[:5])  # 캡션 파일의 경로들
        print("Dataset image paths:", self.imgs[:5])  # 데이터셋의 이미지 경로들
        print("Matched captions count:", sum(1 for img in self.imgs if img in self.captions))  # 매칭된 캡션 수

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]

        img = self.loader(path)
        if img.mode == 'P':
            img = img.convert('RGBA')
        if self.transform is not None:
            img = self.transform(img)

        # 전체 경로를 사용하여 캡션 매칭
        caption = self.captions.get(path, "No caption available")

        return img, target, caption


    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: 파일 경로 (유닉스 스타일)
        :return: 사람 ID
        """
        # 파일 경로에서 ID 폴더 이름을 추출합니다.
        return os.path.basename(os.path.dirname(file_path))

    @staticmethod
    def camera(file_path):
        """
        :param file_path: 파일 경로 (유닉스 스타일)
        :return: 카메라 ID
        """
        # 파일 이름에서 카메라 ID를 추출합니다.
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) > 1:
            # 예: 파일명이 'img_c1_001.jpg'인 경우
            for part in parts:
                if part.startswith('c'):
                    return int(part[1:])
        return 0  # 기본값

    @property
    def ids(self):
        """
        :return: 데이터셋 이미지 경로에 해당하는 사람 ID 리스트
        """
        return self._ids

    @property
    def unique_ids(self):
        """
        :return: 오름차순으로 정렬된 고유한 사람 ID 리스트
        """
        return self._unique_ids

    @property
    def cameras(self):
        """
        :return: 데이터셋 이미지 경로에 해당하는 카메라 ID 리스트
        """
        return self._cameras

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), '데이터셋이 존재하지 않습니다: {}'.format(directory)
        return sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f, flags=re.IGNORECASE)
        ])
        
        
        
        
class DanbooruDataset(Dataset):
    def __init__(self, transform, dtype, data_path, caption_dir=None, id2label=None):
        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path
        self.dtype = dtype

        # 데이터 경로에서 모든 ID 디렉토리 가져오기
        self.id_dirs = [
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ]
        self.id_dirs.sort()  # 일관된 순서를 위해 정렬

        # id2label 매핑 생성 또는 기존 매핑 사용
        if id2label is None:
            self.id2label = {id_name: idx for idx, id_name in enumerate(self.id_dirs)}
        else:
            self.id2label = id2label

        # 이미지와 라벨 수집
        self.imgs = []
        self.labels = []
        for id_name in self.id_dirs:
            id_dir = os.path.join(self.data_path, id_name)
            label = self.id2label[id_name]
            image_files = self.list_pictures(id_dir)
            for img_path in image_files:
                self.imgs.append(img_path)
                self.labels.append(label)

        # IDs와 카메라 IDs 할당
        self._ids = [self.id(path) for path in self.imgs]
        if dtype == 'query':
            self._cameras = [0] * len(self.imgs)  # 쿼리 이미지에 카메라 ID 0 할당
        else:
            self._cameras = [1] * len(self.imgs)  # 테스트/갤러리 이미지에 카메라 ID 1 할당

        self._unique_ids = sorted(set(self._ids))

        # 캡션 로드 및 매칭
        self.captions = {}
        if caption_dir is not None:
            for id_name in self.id_dirs:
                caption_file_path = os.path.join(caption_dir, f"{id_name}_captions.json")
                if os.path.exists(caption_file_path):
                    with open(caption_file_path, 'r') as f:
                        caption_data = json.load(f)
                        for item in caption_data:
                            image_path = os.path.normpath(item['image_path'])
                            relative_path = os.path.relpath(image_path, self.data_path)
                            full_path = os.path.join(self.data_path, relative_path)
                            self.captions[full_path] = item['caption']

        # 디버그용 코드
        print("Caption file paths:", list(self.captions.keys())[:5])  # 캡션 파일의 경로들
        print("Dataset image paths:", self.imgs[:5])  # 데이터셋의 이미지 경로들
        print("Matched captions count:", sum(1 for img in self.imgs if img in self.captions))  # 매칭된 캡션 수

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]

        img = self.loader(path)
        if img.mode == 'P':
            img = img.convert('RGBA')
        if self.transform is not None:
            img = self.transform(img)

        # 전체 경로를 사용하여 캡션 매칭
        caption = self.captions.get(path, "No caption available")

        return img, target, caption


    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: 파일 경로 (유닉스 스타일)
        :return: 사람 ID
        """
        # 파일 경로에서 ID 폴더 이름을 추출합니다.
        return os.path.basename(os.path.dirname(file_path))

    @staticmethod
    def camera(file_path):
        """
        :param file_path: 파일 경로 (유닉스 스타일)
        :return: 카메라 ID
        """
        # 파일 이름에서 카메라 ID를 추출합니다.
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if len(parts) > 1:
            # 예: 파일명이 'img_c1_001.jpg'인 경우
            for part in parts:
                if part.startswith('c'):
                    return int(part[1:])
        return 0  # 기본값

    @property
    def ids(self):
        """
        :return: 데이터셋 이미지 경로에 해당하는 사람 ID 리스트
        """
        return self._ids

    @property
    def unique_ids(self):
        """
        :return: 오름차순으로 정렬된 고유한 사람 ID 리스트
        """
        return self._unique_ids

    @property
    def cameras(self):
        """
        :return: 데이터셋 이미지 경로에 해당하는 카메라 ID 리스트
        """
        return self._cameras

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), '데이터셋이 존재하지 않습니다: {}'.format(directory)
        return sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f, flags=re.IGNORECASE)
        ])
    

class EmoSketchDataset(Dataset):
    def __init__(self, transform, image_dir, caption_file):
        self.transform = transform
        self.loader = default_loader
        self.image_dir = image_dir
        
        # 이미지 폴더에서 모든 이미지 파일 경로 가져오기
        print(f"이미지 디렉토리: {image_dir}")
        self.imgs = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.imgs.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
            self.imgs.extend(glob.glob(os.path.join(image_dir, '**', ext.upper()), recursive=True))
        
        self.imgs.sort()
        print(f"총 이미지 수: {len(self.imgs)}")
        
        # 파일명 정규화 함수 - 확장자 제거, 소문자 변환, _2x 제거
        def normalize_filename(filename):
            # 확장자 제거
            base = os.path.splitext(filename)[0]
            # 소문자 변환
            base = base.lower()
            # _2x 제거 (끝에 있는 경우)
            if base.endswith("_2x"):
                base = base[:-3]
            return base
        
        # 정규화된 파일명으로 이미지 매핑 생성
        self.normalized_to_path = {}
        for img_path in self.imgs:
            filename = os.path.basename(img_path)
            normalized = normalize_filename(filename)
            self.normalized_to_path[normalized] = img_path
        
        # 임의 라벨 할당
        self.labels = [0] * len(self.imgs)
        
        # 캡션 로드 및 매핑
        self.captions = {}
        if os.path.exists(caption_file):
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
                print(f"총 캡션 수: {len(caption_data)}")
                
                for item in caption_data:
                    if 'image_path' in item and 'caption' in item:
                        # 원본 캡션 파일의 경로에서 파일명만 추출
                        old_filename = os.path.basename(item['image_path'])
                        
                        # 파일명 정규화
                        normalized = normalize_filename(old_filename)
                        
                        # 정규화된 파일명으로 실제 이미지 경로 찾기
                        if normalized in self.normalized_to_path:
                            img_path = self.normalized_to_path[normalized]
                            self.captions[img_path] = item['caption']
                
                print(f"매칭된 캡션 수: {len(self.captions)}")
                
                # 매칭 실패 원인 디버깅
                if len(self.captions) < len(caption_data):
                    not_matched = 0
                    sample_captions = []
                    
                    for item in caption_data:
                        if 'image_path' in item:
                            caption_filename = os.path.basename(item['image_path'])
                            normalized = normalize_filename(caption_filename)
                            
                            if normalized not in self.normalized_to_path:
                                not_matched += 1
                                if len(sample_captions) < 5:
                                    sample_captions.append(caption_filename)
                    
                    print(f"매칭 실패한 캡션 수: {not_matched}")
                    print(f"매칭 실패한 캡션 파일명 샘플: {sample_captions}")
    
    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]
        
        img = self.loader(path)
        if img.mode == 'P':
            img = img.convert('RGBA')
        if self.transform is not None:
            img = self.transform(img)
        
        # 경로에 해당하는 캡션 가져오기
        caption = self.captions.get(path, "No caption available")
        
        return img, target, caption
    
    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), f'데이터셋 디렉토리가 존재하지 않습니다: {directory}'
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if re.match(r'([\w]+\.(?:' + ext + '))', f, flags=re.IGNORECASE)
        ]
    

