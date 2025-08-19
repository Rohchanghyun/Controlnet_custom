import os
import numpy as np
from scipy.spatial.distance import cdist, cosine, euclidean
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
from network import MGN, Image_adapter, Image_adapter_77
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pdb
import random

import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"



class Main():
    def __init__(self, extractor, image_adapter):
        self.data = Data()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader
        self.query_loader = self.data.query_loader
        self.testset = self.data.testset
        self.queryset = self.data.queryset

        self.image_adapter = image_adapter
        self.extractor = extractor
        self.loss = Loss()
        self.optimizer = get_optimizer(self.extractor, self.image_adapter)

        self.text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # load pretrained weight
        if hasattr(opt, 'extractor_weight') and opt.extractor_weight:
            print(f"Loading pretrained weights from {opt.extractor_weight}")
            self.extractor.load_state_dict(torch.load(opt.extractor_weight))
            
        # if hasattr(opt, 'image_adapter_weight') and opt.image_adapter_weight:
        #     print(f"Loading pretrained weights from {opt.image_adapter_weight}")
        #     self.image_adapter.load_state_dict(torch.load(opt.image_adapter_weight))

        # result 폴더 생성
        self.result_dir = opt.output_dir
        os.makedirs(self.result_dir, exist_ok=True)

    def train(self, epoch):
    
        
        self.extractor.eval()  # train ID extractor
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
                # padding='max_length'를 명시적으로 설정하고 max_length 강제 적용
                text_inputs = self.processor(
                    text=captions, 
                    return_tensors="pt", 
                    padding='max_length',  # 'longest'가 아닌 'max_length'로 설정
                    max_length=77,
                    truncation=True
                ).to('cuda')
                text_outputs = self.text_encoder.text_model(**text_inputs)
                text_embedding = text_outputs.last_hidden_state  # shape: (batch_size, 77, 512)

            # 손실 계산 및 역전파
            loss, Triplet_Loss, CrossEntropy_Loss, Contrastive_Loss_77 = self.loss(outputs, labels, projected_image_embedding, text_embedding)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # wandb에 배치별 손실 기록
            wandb.log({"batch_loss": loss.item(),
                       "Triplet_Loss": Triplet_Loss.item(),
                       "CrossEntropy_Loss": CrossEntropy_Loss.item(),
                       "Contrastive_Loss": Contrastive_Loss_77.item()})

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

    def tsne_visualization(self):
        self.extractor.eval()
        self.image_adapter.eval()
        self.text_encoder.eval()
        
        #gallery_path = self.data.testset.imgs
        gallery_label = torch.FloatTensor()
        gallery_label = self.data.testset.ids
        #gallery_captions = self.data.testset.captions

        # print("input_query_label: ", query_label)
        
        print('extract features, this may take a few minutes')
        predict_feature_gallery, captions = extract_feature(self.extractor, tqdm(self.data.test_loader))
        
        unique_ids = list(set(gallery_label))
        selected_ids = random.sample(unique_ids, 30)
        
        gallery_label_array = np.array(gallery_label)
        
        selected_features = []
        selected_labels = []
        selected_captions = []

        max_samples_per_id = 10
        
        for id in selected_ids:
            indices = np.where(gallery_label_array == id)[0]
            if len(indices) >= max_samples_per_id:
                selected_indices = np.random.choice(indices, max_samples_per_id, replace=False)
            else:
                # Skip this ID if there are fewer than 10 samples
                print(f"Skipping ID {id} as it has fewer than 10 samples")
                continue
            selected_features.extend(predict_feature_gallery[selected_indices])
            selected_labels.extend([id] * max_samples_per_id)
            selected_captions.extend([captions[i] for i in selected_indices])
            
        selected_features = np.array(selected_features)
        selected_labels = np.array(selected_labels)

        # 1. 먼저 adapter features 생성
        adapter_features = self.image_adapter(torch.from_numpy(selected_features).float().to('cuda'))
        # adapter_features가 3차원이라면 mean pooling으로 2차원으로 변환
        if len(adapter_features.shape) == 3:
            adapter_features = adapter_features.mean(dim=1)  # 3D -> 2D로 변환
        adapter_features_cpu = adapter_features.cpu().detach().numpy()  # shape: (N, 512)
        
        # 2. 텍스트 임베딩 처리
        batch_size = 32
        selected_text_embeddings = []
        
        for i in range(0, len(selected_captions), batch_size):
            batch_captions = selected_captions[i:i + batch_size]
            with torch.no_grad():
                text_inputs = self.processor(
                    text=batch_captions,
                    return_tensors="pt",
                    padding='max_length',
                    max_length=77,
                    truncation=True
                ).to('cuda')
                text_outputs = self.text_encoder.text_model(**text_inputs)
                # 여기서 mean pooling을 사용하여 77 토큰을 하나의 벡터로 변환
                batch_embeddings = text_outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # shape: (batch_size, 512)
                selected_text_embeddings.append(batch_embeddings)
                
            # 4. GPU 캐시 정리
            torch.cuda.empty_cache()
        
        selected_text_embeddings = np.concatenate(selected_text_embeddings, axis=0)  # shape: (N, 512)
        
        print("adapter_features_cpu shape:", adapter_features_cpu.shape)
        print("selected_text_embeddings shape:", selected_text_embeddings.shape)
        
        # 3. 이제 combined_embeddings 생성
        combined_embeddings = np.concatenate([adapter_features_cpu, selected_text_embeddings], axis=0)  # shape: (2N, 512)
        combined_labels = np.concatenate([selected_labels, selected_labels])
        
        # 데이터 샘플 수에 따라 perplexity 동적 조정
        n_samples = len(combined_labels)
        perplexity = min(10, n_samples - 1)  # perplexity가 샘플 수보다 작도록 보장
        
        tsne = TSNE(n_components=2, 
                    perplexity=perplexity,  # 동적으로 조정된 perplexity 사용
                    n_iter=2000, 
                    metric='cosine')
        combined_embedded_features = tsne.fit_transform(combined_embeddings)
        
        # 색상 맵 생성
        unique_labels = set(selected_labels)
        color_map = dict(zip(unique_labels, plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))))

        # Plot
        fig = plt.figure(figsize=(20, 20))
        n = len(selected_labels)
        for label in set(selected_labels):
            x = combined_embedded_features[:n][combined_labels[:n] == label, 0]
            y = combined_embedded_features[:n][combined_labels[:n] == label, 1]
            plt.scatter(x, y, label=label)
            # Label each point with its corresponding ID
            # for i, (x, y) in enumerate(zip(x, y)):
            #     plt.text(x, y, f"{label}", fontsize=8, ha='center', va='center')
        plt.legend()
        plt.title('t-SNE Visualization of Predicted Features of extractor')
        plt.savefig(os.path.join(opt.output_dir, 'extractor_tsne.png'))
        plt.close()

        if opt.mode == 'train':
            # wandb에 extractor tsne 이미지 로깅
            wandb.log({"Extractor t-SNE": wandb.Image(os.path.join(opt.output_dir, 'extractor_tsne.png'))})
        
        
        # 데이터 샘플 수에 따라 perplexity 동적 조정
        n_samples = len(adapter_features_cpu)
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, 
                    perplexity=perplexity,
                    n_iter=2000, 
                    metric='cosine')
        adapter_embedded_features = tsne.fit_transform(adapter_features_cpu)
        
        # Plot
        fig = plt.figure(figsize=(20, 20))
        for label in set(selected_labels):
            x = adapter_embedded_features[selected_labels == label, 0]
            y = adapter_embedded_features[selected_labels == label, 1]
            plt.scatter(x, y, label=label)
            # Label each point with its corresponding ID
            # for i, (x, y) in enumerate(zip(x, y)):
            #     plt.text(x, y, f"{label}", fontsize=8, ha='center', va='center')
        plt.legend()
        plt.title('t-SNE Visualization of Predicted Features of image adapter')
        plt.savefig(os.path.join(opt.output_dir, 'image_adapter_tsne.png'))
        plt.close()

        if opt.mode == 'train':
            # wandb에 image adapter tsne 이미지 로깅
            wandb.log({"Image Adapter t-SNE": wandb.Image(os.path.join(opt.output_dir, 'image_adapter_tsne.png'))})
        
        
        # 텍스트와 이미지 임베딩 t-SNE
        combined_labels = np.concatenate([selected_labels, selected_labels])
        
        n_samples = len(combined_labels)
        perplexity = min(10, n_samples - 1)  # perplexity가 샘플 수보다 작도록 보장
        tsne = TSNE(n_components=2, 
                    perplexity=perplexity,
                    n_iter=2000, 
                    metric='cosine')
        combined_embedded_features = tsne.fit_transform(combined_embeddings)
        
        # 색상 맵 생성
        unique_labels = set(selected_labels)
        color_map = dict(zip(unique_labels, plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))))

        # 플롯
        fig, ax = plt.subplots(figsize=(20, 20))
        n = len(selected_labels)

        cosine_similarities = []
        euclidean_distances = []

        # CLIP 스타일의 유사도 계산을 위한 함수 정의
        def compute_clip_similarity(image_embeddings, text_embeddings, temperature=0.07):
            # torch 텐서로 변환
            image_embeddings = torch.from_numpy(image_embeddings).float().to('cuda')
            text_embeddings = torch.from_numpy(text_embeddings).float().to('cuda')
            
            # L2 정규화
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
            
            # 코사인 유사도 계산
            similarity = torch.matmul(image_embeddings, text_embeddings.T)
            
            # 온도 스케일링 적용
            similarity = similarity / temperature
            
            return similarity.cpu().numpy()

        for label in unique_labels:
            x_img = combined_embedded_features[:n][combined_labels[:n] == label, 0]
            y_img = combined_embedded_features[:n][combined_labels[:n] == label, 1]
            x_txt = combined_embedded_features[n:][combined_labels[n:] == label, 0]
            y_txt = combined_embedded_features[n:][combined_labels[n:] == label, 1]
            
            color = color_map[label]
            ax.scatter(x_img, y_img, color=color, label=f'Image {label}')
            ax.scatter(x_txt, y_txt, color=color, marker='x', label=f'Text {label}')
            
            # CLIP 스타일의 유사도 계산
            img_embs = adapter_features_cpu[combined_labels[:n] == label]
            txt_embs = selected_text_embeddings[combined_labels[n:] == label]
            
            if len(img_embs) > 0 and len(txt_embs) > 0:
                similarities = compute_clip_similarity(img_embs, txt_embs)
                # 대각 요소만 추출 (같은 클래스 내의 이미지-텍스트 쌍의 유사도)
                cosine_similarities.extend(np.diag(similarities))
                
                # 유클리드 거리는 기존대로 유지
                for img_emb, txt_emb in zip(img_embs, txt_embs):
                    euclidean_distances.append(euclidean(img_emb, txt_emb))

        # 평균 유사도 계산
        avg_cosine_similarity = np.mean(cosine_similarities)
        avg_euclidean_distance = np.mean(euclidean_distances)

        # 클래스 내부 및 클래스 간 거리 계산
        intra_class_img_dist = []
        intra_class_txt_dist = []
        inter_class_img_dist = []
        inter_class_txt_dist = []

        for label in unique_labels:
            # 같은 클래스 내의 이미지 임베딩
            img_embs = adapter_features_cpu[combined_labels[:n] == label]
            # 같은 클래스 내의 텍스트 임베딩
            txt_embs = selected_text_embeddings[combined_labels[n:] == label]
            
            # 클래스 내부 유사도 계산
            if len(img_embs) > 1:
                img_similarities = compute_clip_similarity(img_embs, img_embs)
                # 대각 요소를 제외한 상삼각 행렬의 값들만 사용
                intra_class_img_dist.extend(img_similarities[np.triu_indices(len(img_embs), k=1)])
            
            if len(txt_embs) > 1:
                txt_similarities = compute_clip_similarity(txt_embs, txt_embs)
                intra_class_txt_dist.extend(txt_similarities[np.triu_indices(len(txt_embs), k=1)])
            
            # 클래스 간 유사도 계산
            other_img_embs = adapter_features_cpu[combined_labels[:n] != label]
            other_txt_embs = selected_text_embeddings[combined_labels[n:] != label]
            
            if len(img_embs) > 0 and len(other_img_embs) > 0:
                inter_similarities = compute_clip_similarity(img_embs, other_img_embs)
                inter_class_img_dist.extend(inter_similarities.flatten())
            
            if len(txt_embs) > 0 and len(other_txt_embs) > 0:
                inter_similarities = compute_clip_similarity(txt_embs, other_txt_embs)
                inter_class_txt_dist.extend(inter_similarities.flatten())

        # 평균값 계산
        avg_intra_img = np.mean(intra_class_img_dist) if intra_class_img_dist else 0
        avg_intra_txt = np.mean(intra_class_txt_dist) if intra_class_txt_dist else 0
        avg_inter_img = np.mean(inter_class_img_dist) if inter_class_img_dist else 0
        avg_inter_txt = np.mean(inter_class_txt_dist) if inter_class_txt_dist else 0

        # 결과 텍스트 업데이트
        result_text = (
            f"average CLIP similarity: {avg_cosine_similarity:.4f}\n"
            f"average euclidean distance: {avg_euclidean_distance:.4f}\n"
            f"intra-class image similarity: {avg_intra_img:.4f}\n"
            f"intra-class text similarity: {avg_intra_txt:.4f}\n"
            f"inter-class image similarity: {avg_inter_img:.4f}\n"
            f"inter-class text similarity: {avg_inter_txt:.4f}"
        )

        ax.text(0.02, 0.98, result_text, transform=ax.transAxes, verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title('t-SNE Visualization of Image and Text Embeddings')
        plt.tight_layout()
        plt.savefig(os.path.join(opt.output_dir, 'combined_tsne.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"결과가 combined_tsne.png 파일에 저장되었습니다.")

        if opt.mode == 'train':
            # wandb에 combined tsne 이미지 및 메트릭 로깅
            wandb.log({
                "Combined t-SNE": wandb.Image(os.path.join(opt.output_dir, 'combined_tsne.png')),
                "Average CLIP Similarity": avg_cosine_similarity,
                "Average Euclidean Distance": avg_euclidean_distance,
                "intra-class image similarity": avg_intra_img,
                "intra-class text similarity": avg_intra_txt,
                "inter-class image similarity": avg_inter_img,
                "inter-class text similarity": avg_inter_txt
            })
        
        

if __name__ == '__main__':
    extractor = MGN().to('cuda')
    image_adapter = Image_adapter_77().to('cuda')
    main = Main(extractor, image_adapter)

    if opt.mode == 'train':
        # wandb 초기화
        wandb.init(
            project="Controlnet",
            config=vars(opt),  # opt의 모든 속성을 config로 추가
            name=f"only_adapter_blip_0.5_0.5_2.0_margin2_temperature0.20_77_512"
        )
        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            
            main.train(epoch)
            if epoch % 100 == 0:
                print('\n모델 저장 시작')
                os.makedirs(f'{opt.output_dir}/id_extractor', exist_ok=True)
                os.makedirs(f'{opt.output_dir}/image_adapter', exist_ok=True)
                
                # extractor 가중치 저장
                extractor_path = f'{opt.output_dir}/id_extractor/extractor_{epoch}.pt'
                torch.save(extractor.state_dict(), extractor_path)
                print(f'Extractor 가중치가 {extractor_path}에 저장되었습니다.')
                
                # image_adapter 가중치 저장
                image_adapter_path = f'{opt.output_dir}/image_adapter/image_adapter_{epoch}.pt'
                torch.save(image_adapter.state_dict(), image_adapter_path)
                print(f'Image Adapter 가중치가 {image_adapter_path}에 저장되었습니다.')
                
                # 저장된 가중치 확인
                if os.path.exists(extractor_path) and os.path.exists(image_adapter_path):
                    print('두 모델의 가중치가 성공적으로 저장되었습니다.')
                else:
                    print('가중치 저장 중 오류가 발생했습니다. 파일 경로를 확인해주세요.')
                    
                main.tsne_visualization()

    if opt.mode == 'evaluate':
        print('start evaluate')
        extractor.load_state_dict(torch.load(opt.weight))
        main.evaluate()
        
    if opt.mode == 'tsne':
        print('start tsne')
        main.tsne_visualization()