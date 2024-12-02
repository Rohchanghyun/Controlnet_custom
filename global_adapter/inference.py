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

from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL, EulerAncestralDiscreteScheduler

from opt import opt
from data import Data
from network import MGN, Image_adapter_4_768, Base_adapter
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from sklearn.manifold import TSNE

import pdb
import random

import wandb

class Main():
    def __init__(self, extractor, image_adapter):
        self.data = Data()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader
        self.query_loader = self.data.query_loader
        self.testset = self.data.testset
        self.queryset = self.data.queryset

        # 기존 image adapter와 extractor 유지
        self.image_adapter = image_adapter
        self.extractor = extractor

        # load pretrained weight
        if hasattr(opt, 'extractor_weight') and opt.extractor_weight:
            print(f"Loading pretrained weights from {opt.extractor_weight}")
            self.extractor.load_state_dict(torch.load(opt.extractor_weight))
            
        if hasattr(opt, 'image_adapter_weight') and opt.image_adapter_weight:
            print(f"Loading pretrained weights from {opt.image_adapter_weight}")
            self.image_adapter.load_state_dict(torch.load(opt.image_adapter_weight))

        # Stable Diffusion 파이프라인 초기화
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.train_scheduler = DDPMScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler"
        )
        self.inference_scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000
        )
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=self.train_scheduler,
            torch_dtype=torch.float32
        ).to('cuda')
        
        # 파이프라인에서 각 컴포넌트 추출
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.noise_scheduler = self.train_scheduler
        
        # 메모리 절약을 위해 파이프라인 객체 제거 (선택사항)
        del self.pipeline
        torch.cuda.empty_cache()
        
        # VAE는 학습하지 않음
        self.vae.eval()
        
        # optimizer 초기화
        self.optimizer = torch.optim.AdamW(
            self.image_adapter.parameters(),
            lr=opt.lr
        )
        
        # result 폴더 생성
        self.result_dir = opt.output_dir
        os.makedirs(self.result_dir, exist_ok=True)
        
    def train(self, epoch):
        self.unet.eval()
        self.image_adapter.train()
        total_loss = 0
        
        total_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f'Batch 0/{total_batches}', ncols=100)
        
        for batch, (inputs, labels, captions) in enumerate(pbar):
            pbar.set_description(f'Batch {batch+1}/{total_batches}')
            # 배치 데이터 처리 - inputs를 GPU로 먼저 이동
            inputs = inputs.to('cuda')  # 여기서 GPU로 이동
            images = inputs
            labels = labels.to('cuda')
            
            with torch.no_grad():
                latents = self.vae.encode(images).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
            
            outputs = self.extractor(inputs)  # 이제 GPU에 있는 inputs를 사
            embedding_feature = outputs[0]  # (batch, 2048) embedding 추출

            projected_image_embedding = self.image_adapter(embedding_feature)  # (batch, 4, 768) embedding

            # Text embedding using SD pipeline's text_encoder
            with torch.no_grad():
                # tokenize captions
                text_input_ids = self.tokenizer(
                    captions,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to('cuda')
                
                # get text embeddings
                text_embeddings = self.text_encoder(text_input_ids)[0]
                # text_embeddings shape: [batch_size, sequence_length, hidden_size]
            
            conditions = torch.cat([projected_image_embedding, text_embeddings], dim=1)
            
            uncond_input = self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to("cuda")
            
            with torch.no_grad():
                uncond_embeddings_large = self.text_encoder(uncond_input.input_ids)[0]
                # uncond_embeddings_large를 배치 크기에 맞게 확장
                uncond_embeddings_large = uncond_embeddings_large.repeat(projected_image_embedding.shape[0], 1, 1)
                uncond_fused = torch.cat([projected_image_embedding, uncond_embeddings_large], dim=1)
                uncond_fused = uncond_fused.to(dtype=torch.float16)
            
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
    image_adapter = Image_adapter_4_768().to('cuda')
    main = Main(extractor, image_adapter)
    
    if opt.mode == 'evaluate':
        main.extractor.eval()
        main.image_adapter.eval()
        
        # 생성할 이미지 개수 설정
        num_images = 20  # 원하는 개수로 수정 가능
        num_variations = 1  # 각 이미지당 생성할 변형 수
        
        # 전체 진행상황을 보여주는 tqdm
        pbar_main = tqdm(total=num_images, desc='전체 이미지 생성 진행률', position=0)
        
        for i in range(num_images):
            # 임의의 쿼리 이미지 선택
            random_idx = random.randint(0, len(main.query_loader.dataset) - 1)
            sample_image, _, sample_caption = main.query_loader.dataset[random_idx]
            sample_image = sample_image.unsqueeze(0).to('cuda')
            
            tqdm.write(f"\n캡션: {sample_caption}")
            
            # 이미지 임베딩 추출 및 프로젝션
            with torch.no_grad():
                outputs = main.extractor(sample_image)
                embedding_feature = outputs[0]
                projected_image_embedding = main.image_adapter(embedding_feature)
                
                # 텍스트 임베딩
                text_input_ids = main.tokenizer(
                    sample_caption,
                    padding="max_length",
                    max_length=main.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to('cuda')
                
                # text embeddings 얻기
                text_embeddings = main.text_encoder(text_input_ids)[0]
                
                # conditions 생성
                conditions = torch.cat([projected_image_embedding, text_embeddings], dim=1)
                
                # unconditioned embeddings 생성
                uncond_input = main.tokenizer(
                    "",
                    padding="max_length",
                    max_length=main.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to("cuda")
                
                # uncond_image_embedding 생성
                uncond_zero_embedding = torch.zeros_like(projected_image_embedding)
                
                uncond_embeddings = main.text_encoder(uncond_input.input_ids)[0]
                
                #uncond_fused = torch.cat([uncond_zero_embedding, uncond_embeddings], dim=1)
                uncond_fused = torch.cat([uncond_embeddings, uncond_embeddings], dim=1)
                
                # 변형 생성 진행상황을 보여주는 tqdm
                pbar_variations = tqdm(total=num_variations, 
                                     desc=f'이미지 {i+1} 변형 생성 중', 
                                     position=1, 
                                     leave=False)
                
                for var in range(num_variations):
                    # 새로운 랜덤 시드 설정
                    seed = 42
                    torch.manual_seed(seed)
                    
                    # 노이즈에서 이미지 생성
                    latents = torch.randn((1, 4, 64, 64), device='cuda')
                    
                    # 노이즈 스케줄러 타임스텝 설정 수정
                    main.noise_scheduler = main.inference_scheduler
                    main.noise_scheduler.set_timesteps(30)
                    timesteps = main.noise_scheduler.timesteps
                    guidance_scale = 7.5  # CFG 스케일 추가
                    
                    # 디노이징 단계 진행상황을 보여주는 tqdm
                    pbar_steps = tqdm(timesteps, 
                                    desc=f'디노이징 스텝', 
                                    position=2, 
                                    leave=False)
                    
                    for t in pbar_steps:
                        # 입력 스케일링 추가
                        latent_model_input = main.noise_scheduler.scale_model_input(latents, t)
                        
                        # unconditional 예측
                        noise_pred_uncond = main.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=uncond_fused
                        ).sample
                        
                        # conditional 예측
                        noise_pred_text = main.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=conditions
                        ).sample
                        
                        # CFG를 사용하여 최종 노이즈 예측
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                        # 스케줄러 스텝
                        latents = main.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=t,
                            sample=latents
                        ).prev_sample
                    
                    pbar_steps.close()
                    
                    # VAE를 통한 디코딩 추가
                    latents = 1 / main.vae.config.scaling_factor * latents
                    image = main.vae.decode(latents).sample

                    # 이미지 후처리
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = (image * 255).clamp(0, 255).round()
                    image = image.cpu().permute(0, 2, 3, 1).numpy()
                    image = image.astype(np.uint8)[0]

                    # 저장 디렉토리 설정
                    base_save_dir = os.path.join(opt.output_dir, f"sample_{i+1}")
                    os.makedirs(base_save_dir, exist_ok=True)
                    
                    # 생성된 이미지 저장
                    result_image = Image.fromarray(image)
                    result_image.save(os.path.join(base_save_dir, f'generated_image_var_{var+1}_seed_{seed}.png'))
                    
                    # 캡션 정보 저장
                    with open(os.path.join(base_save_dir, 'caption.txt'), 'w', encoding='utf-8') as f:
                        f.write(f"Caption: {sample_caption}\nSeed: {seed}")
                    
                    pbar_variations.update(1)
                
                pbar_variations.close()
                
                # 원본 이미지 저장
                original_image = sample_image.cpu().squeeze(0).permute(1, 2, 0)
                original_image = ((original_image * 0.5 + 0.5) * 255).clamp(0, 255).round().numpy().astype(np.uint8)
                original_image = Image.fromarray(original_image)
                original_image.save(os.path.join(base_save_dir, 'original_image.png'))
            
            pbar_main.update(1)
        
        pbar_main.close()
    if opt.mode == 'tsne':
        print('start tsne')
        main.tsne_visualization()