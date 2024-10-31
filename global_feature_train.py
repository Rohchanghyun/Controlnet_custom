import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from scipy.spatial.distance import cdist, cosine, euclidean
from tqdm import tqdm
from tqdm.auto import tqdm
import matplotlib
from PIL import Image
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL

from opt import opt
from data import Data
from network import MGN, Image_adapter, Base_adapter
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from sklearn.manifold import TSNE

import pdb
import random

import wandb

class Main():
    def __init__(self, extractor, image_adapter, projector):
        self.data = Data()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader
        self.query_loader = self.data.query_loader
        self.testset = self.data.testset
        self.queryset = self.data.queryset

        self.image_adapter = image_adapter
        self.extractor = extractor
        self.projector = projector

        self.text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # load pretrained weight
        if hasattr(opt, 'extractor_weight') and opt.extractor_weight:
            print(f"Loading pretrained weights from {opt.extractor_weight}")
            self.extractor.load_state_dict(torch.load(opt.extractor_weight))
            
        if hasattr(opt, 'image_adapter_weight') and opt.image_adapter_weight:
            print(f"Loading pretrained weights from {opt.image_adapter_weight}")
            self.image_adapter.load_state_dict(torch.load(opt.image_adapter_weight))

        # Stable Diffusion 컴포넌트 초기화
        self.model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to('cuda')
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        
        # optimizer 초기화
        self.optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=opt.lr
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae"
        ).to('cuda')
        self.vae.eval()  # VAE는 학습하지 않음
        # result 폴더 생성
        self.result_dir = opt.output_dir
        os.makedirs(self.result_dir, exist_ok=True)
        
    def train(self, epoch):
        self.unet.eval()
        self.projector.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', ncols=100)
        
        for batch, (inputs, labels, captions) in enumerate(pbar):
            # 배치 데이터 처리 - inputs를 GPU로 먼저 이동
            inputs = inputs.to('cuda')  # 여기서 GPU로 이동
            images = inputs
            labels = labels.to('cuda')
            
            with torch.no_grad():
                latents = self.vae.encode(images).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
            
            outputs = self.extractor(inputs)  # 이제 GPU에 있는 inputs를 사용
            embedding_feature = outputs[0]  # (batch, 2048) embedding 추출

            projected_image_embedding = self.image_adapter(embedding_feature)  # (batch, 512) embedding

            # Text embedding
            with torch.no_grad():
                # 캡션 텍스트를 전처리
                text_inputs = self.processor(text=captions, return_tensors="pt", padding=True).to('cuda')
                text_outputs = self.text_encoder.text_model(**text_inputs)
                text_embedding = text_outputs.last_hidden_state

            
            conditions = self.projector(projected_image_embedding, text_embedding)
            
            # 노이즈 추가
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (images.shape[0],), device='cuda'
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # UNet으로 노이즈 예측
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=conditions
            ).sample
            
            # 손실 계산
            loss = F.mse_loss(noise_pred, noise)
            
            # 역전파 및 옵티마이저 스텝
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # wandb에 step별 loss 로깅
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch
            })
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 에포크 종료 후 평균 손실 로깅
        avg_loss = total_loss / len(self.train_loader)
        wandb.log({
            "epoch_avg_loss": avg_loss,
            "epoch": epoch
        })
        print(f"\nEpoch {epoch}/{opt.epoch}, Average Loss: {avg_loss:.4f}")
        
        # # 모델 저장 (필요한 경우)
        # if (epoch + 1) % 10 == 0:
        #     self.unet.save_pretrained(f"checkpoint-{epoch+1}")


if __name__ == '__main__':
    extractor = MGN().to('cuda')
    image_adapter = Image_adapter().to('cuda')
    projector = Base_adapter().to('cuda')
    main = Main(extractor, image_adapter, projector)

    if opt.mode == 'train':
        #wandb 초기화
        wandb.init(
            project="Controlnet",
            config=vars(opt),  # opt의 모든 속성을 config로 추가
            name=f"projector_{opt.epoch}_{opt.lr}"
        )
        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            
            main.train(epoch)
            if epoch % 1000 == 0:
                
                main.extractor.eval()  # 추가
                main.image_adapter.eval()  # 추가
                main.projector.eval()  # 추가
                print('\n모델 저장 시작')
                os.makedirs(f'{opt.output_dir}/projector', exist_ok=True)
                os.makedirs(f'{opt.output_dir}/results', exist_ok=True)  # 결과 이미지 저장 디렉토리
                
                # 임의의 쿼리 이미지 선택
                random_idx = random.randint(0, len(main.query_loader.dataset) - 1)
                sample_image, _, sample_caption = main.query_loader.dataset[random_idx]
                sample_image = sample_image.unsqueeze(0).to('cuda')
                
                # 이미지 임베딩 추출 및 프로젝션
                with torch.no_grad():
                    outputs = main.extractor(sample_image)
                    embedding_feature = outputs[0]
                    projected_image_embedding = main.image_adapter(embedding_feature)
                    
                    # 텍스트 임베딩
                    text_inputs = main.processor(text=[sample_caption], return_tensors="pt", padding=True).to('cuda')
                    text_outputs = main.text_encoder.text_model(**text_inputs)
                    text_embedding = text_outputs.last_hidden_state
                    
                    # 컨디션 생성
                    conditions = main.projector(projected_image_embedding, text_embedding)
                    
                    # 노이즈에서 이미지 생성
                    latents = torch.randn((1, 4, 64, 64), device='cuda')
                    timesteps = main.noise_scheduler.timesteps
                    
                    for t in timesteps:
                        latent_model_input = latents
                        noise_pred = main.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=conditions
                        ).sample
                        latents = main.noise_scheduler.step(noise_pred, t, latents).prev_sample
                
                # VAE를 통한 디코딩 추가
                with torch.no_grad():
                    latents = 1 / main.vae.config.scaling_factor * latents
                    image = main.vae.decode(latents).sample

                # 이미지 후처리
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = (image * 255).round().astype(np.uint8)[0]

                # 생성된 이미지 저장
                result_image = Image.fromarray(image)
                result_image.save(f'{opt.output_dir}/results/generated_image_epoch_{epoch}.png')
                
                # extractor 가중치 저장
                projector_path = f'{opt.output_dir}/projector/projector_{epoch}.pt'
                torch.save(projector.state_dict(), projector_path)
                print(f'Extractor 가중치가 {projector_path}에 저장되었습니다.')
                
                # 저장된 가중치 확인
                if os.path.exists(projector_path):
                    print('모델의 가중치가 성공적으로 저장되었습니다.')
                else:
                    print('가중치 저장 중 오류가 발생했습니다. 파일 경로를 확인해주세요.')

                # 추론 후 다시 train 모드로 되돌리기
                main.projector.train()  # 추가


    if opt.mode == 'evaluate':
        print('start evaluate')
        extractor.load_state_dict(torch.load(opt.weight))
        main.evaluate()
        
    if opt.mode == 'tsne':
        print('start tsne')
        main.tsne_visualization()