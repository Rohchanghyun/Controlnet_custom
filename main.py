import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib
from PIL import Image
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

import torch
from torch.optim import lr_scheduler

import wandb
from transformers import CLIPProcessor, CLIPModel

from opt import opt
from data import Data
from network import MGN, Image_adapter
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from sklearn.manifold import TSNE

import pdb
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"



class Main():
    def __init__(self, extractor, image_adapter):
        data = Data()
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.image_adapter = image_adapter
        self.extractor = extractor
        self.loss = Loss()
        self.optimizer = get_optimizer(self.extractor, self.image_adapter)

        self.text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # load pretrained weight
        if hasattr(opt, 'weight') and opt.weight:
            print(f"Loading pretrained weights from {opt.weight}")
            self.extractor.load_state_dict(torch.load(opt.weight))

        # result 폴더 생성
        self.result_dir = '../../../results/tsne'
        os.makedirs(self.result_dir, exist_ok=True)

    def train(self, epoch):
        
        # wandb 초기화
        wandb.init(
            project="Controlnet",
            config=vars(opt),  # opt의 모든 속성을 config로 추가
            name=f"experiment_with_extractor_{opt.mode}_lr_{opt.lr}_epoch_{opt.epoch}_{opt.batchid} * {opt.batchimage}"
        )
        
        self.extractor.train()  # freeze ID extractor
        self.image_adapter.train()  # train image adapter
        self.text_encoder.eval()

        total_loss = 0

        for batch, (inputs, labels, captions) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()

            # Image embedding
            outputs = self.extractor(inputs)
            embedding_feature = outputs[0]  # (batch, 2048) embedding 추출

            projected_image_embedding = self.image_adapter(embedding_feature)  # (batch, 512) embedding

            # Text embedding
            with torch.no_grad():
                # 캡션 텍스트를 전처리
                text_inputs = self.processor(text=captions, return_tensors="pt", padding=True).to('cuda')
                text_embedding = self.text_encoder.get_text_features(**text_inputs)

            # 손실 계산 및 역전파
            loss, Triplet_Loss, CrossEntropy_Loss, Contrastive_Loss = self.loss(outputs, labels, projected_image_embedding, text_embedding)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # wandb에 배치별 손실 기록
            wandb.log({"batch_loss": loss.item(),
                       "Triplet_Loss": Triplet_Loss.item(),
                       "CrossEntropy_Loss": CrossEntropy_Loss.item(),
                       "Contrastive_Loss": Contrastive_Loss.item()})

        # 에포크별 평균 손실을 기록
        avg_loss = total_loss / len(self.train_loader)
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch})

    def evaluate(self):
        self.extractor.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.extractor, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.extractor, tqdm(self.test_loader)).numpy()

    def sample_data(self, n_classes=30, samples_per_class=10):
        all_data = list(zip(self.train_loader.dataset.imgs, self.train_loader.dataset.ids, self.train_loader.dataset.captions))
        
        # 클래스별로 데이터 그룹화
        class_data = {}
        for img, label, caption in all_data:
            if label not in class_data:
                class_data[label] = []
            class_data[label].append((img, label, caption))
        
        # 30개의 클래스 무작위 선택
        selected_classes = random.sample(list(class_data.keys()), min(n_classes, len(class_data)))
        
        # 선택된 클래스에서 각각 10개의 샘플 선택
        sampled_data = []
        for cls in selected_classes:
            sampled_data.extend(random.sample(class_data[cls], min(samples_per_class, len(class_data[cls]))))
        
        return zip(*sampled_data)

    def tsne_visualization(self, image_embeddings, text_embeddings, labels, epoch):
        # 고유한 색상 생성
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        label_to_color = dict(zip(unique_labels, colors))

        # 1. 이미지와 텍스트 임베딩 t-SNE
        combined_embeddings = np.vstack((image_embeddings, text_embeddings))
        tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, random_state=42)
        tsne_results = tsne.fit_transform(combined_embeddings)
        
        image_tsne = tsne_results[:len(image_embeddings)]
        text_tsne = tsne_results[len(image_embeddings):]
        
        plt.figure(figsize=(15, 10))
        for label in unique_labels:
            img_mask = labels == label
            txt_mask = labels == label
            plt.scatter(image_tsne[img_mask, 0], image_tsne[img_mask, 1], c=[label_to_color[label]], marker='o', label=f'Image (Class {label})')
            plt.scatter(text_tsne[txt_mask, 0], text_tsne[txt_mask, 1], c=[label_to_color[label]], marker='^', label=f'Text (Class {label})')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('t-SNE visualization of Image and Text Embeddings')
        plt.tight_layout()
        
        # wandb에 이미지와 텍스트 임베딩 t-SNE 결과 로깅
        wandb.log({"tsne_plot_image_text": wandb.Image(plt)})
        
        # result 폴더에 저장
        plt.savefig(os.path.join(self.result_dir, f'tsne_plot_image_text_epoch_{epoch}.png'), bbox_inches='tight')
        plt.close()

        # 페어 간 평균 거리 및 유사도 계산
        pair_distances = np.linalg.norm(image_tsne - text_tsne, axis=1)
        avg_pair_distance = np.mean(pair_distances)
        similarities = np.sum(image_embeddings * text_embeddings, axis=1) / (np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(text_embeddings, axis=1))
        avg_similarity = np.mean(similarities)

        # wandb에 평균 거리와 유사도 로깅
        wandb.log({
            "avg_pair_distance": avg_pair_distance,
            "avg_similarity": avg_similarity
        })

        # 2. 클래스별 t-SNE
        tsne_image_adapter = TSNE(n_components=2, perplexity=30, n_iter=2000, random_state=42)
        tsne_results_image_adapter = tsne_image_adapter.fit_transform(image_embeddings)

        plt.figure(figsize=(15, 10))
        for label in unique_labels:
            mask = labels == label
            plt.scatter(tsne_results_image_adapter[mask, 0], tsne_results_image_adapter[mask, 1], 
                        c=[label_to_color[label]], label=f'Class {label}')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('t-SNE visualization of Image Adapter Features by Class')
        plt.tight_layout()
        
        # wandb에 클래스별 t-SNE 결과 로깅
        wandb.log({"tsne_plot_class": wandb.Image(plt)})
        
        # result 폴더에 저장
        plt.savefig(os.path.join(self.result_dir, f'tsne_plot_class_epoch_{epoch}.png'), bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    extractor = MGN().to('cuda')
    image_adapter = Image_adapter().to('cuda')
    main = Main(extractor, image_adapter)

    if opt.mode == 'train':
        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            
            main.train(epoch)
            if epoch % 500 == 0:
                print('\n모델 저장 시작')
                os.makedirs('../../../weights/id_extractor/256_resume_extractor_adapter', exist_ok=True)
                os.makedirs('../../../weights/image_adapter/256_resume_extractor_adapter', exist_ok=True)
                
                # extractor 가중치 저장
                extractor_path = f'../../../weights/id_extractor/256_resume_extractor_adapter/extractor_{epoch}.pt'
                torch.save(extractor.state_dict(), extractor_path)
                print(f'Extractor 가중치가 {extractor_path}에 저장되었습니다.')
                
                # image_adapter 가중치 저장
                image_adapter_path = f'../../../weights/image_adapter/256_resume_extractor_adapter/image_adapter_{epoch}.pt'
                torch.save(image_adapter.state_dict(), image_adapter_path)
                print(f'Image Adapter 가중치가 {image_adapter_path}에 저장되었습니다.')
                
                # 저장된 가중치 확인
                if os.path.exists(extractor_path) and os.path.exists(image_adapter_path):
                    print('두 모델의 가중치가 성공적으로 저장되었습니다.')
                else:
                    print('가중치 저장 중 오류가 발생했습니다. 파일 경로를 확인해주세요.')
                
                # t-SNE 시각화 수행
                print('\nPerforming t-SNE visualization')
                with torch.no_grad():
                    imgs, labels, captions = main.sample_data()
                    inputs = torch.stack([main.train_loader.dataset.transform(Image.open(img).convert('RGB')) for img in imgs]).to('cuda')
                    outputs = extractor(inputs)
                    embedding_feature = outputs[0]
                    projected_image_embedding = image_adapter(embedding_feature)
                    text_inputs = main.processor(text=captions, return_tensors="pt", padding=True).to('cuda')
                    text_embedding = main.text_encoder.get_text_features(**text_inputs)
                
                main.tsne_visualization(projected_image_embedding.cpu().numpy(), 
                                        text_embedding.cpu().numpy(), 
                                        np.array(labels),
                                        epoch)

    if opt.mode == 'evaluate':
        print('start evaluate')
        extractor.load_state_dict(torch.load(opt.weight))
        main.evaluate()

