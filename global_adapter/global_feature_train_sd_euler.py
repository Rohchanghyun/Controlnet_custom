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
    def __init__(self, extractor, image_adapter):
        self.data = Data()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader
        #self.query_loader = self.data.query_loader
        self.testset = self.data.testset
        #self.queryset = self.data.queryset

        # 기존 image adapter와 extractor 유지
        self.image_adapter = image_adapter
        self.extractor = extractor

        # load pretrained weight
        if hasattr(opt, 'extractor_weight') and opt.extractor_weight:
            print(f"Loading pretrained weights from {opt.extractor_weight}")
            self.extractor.load_state_dict(torch.load(opt.extractor_weight))

        # Stable Diffusion 파이프라인 초기화
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler"
        )
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=self.noise_scheduler,
            torch_dtype=torch.float32
        ).to('cuda')
        
        # 파이프라인에서 각 컴포넌트 추출
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.noise_scheduler = self.pipeline.scheduler
        
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
            
            uncond_text_input = self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to("cuda")
            
            #uncond_image_input = torch.zeros_like(projected_image_embedding)
            uncond_feature = torch.zeros((projected_image_embedding.shape[0], 2048)).to('cuda')  # (batch, 2048)
            uncond_image_input = self.image_adapter(uncond_feature)  # (batch, 4, 768)
            
            with torch.no_grad():
                uncond_embeddings_large = self.text_encoder(uncond_text_input.input_ids)[0] #(77,768)'input_ids': [49406, 49407]에서 BOS 토큰을 가져온다
                # uncond_embeddings_large를 배치 크기에 맞게 확장
                uncond_embeddings_large = uncond_embeddings_large.repeat(projected_image_embedding.shape[0], 1, 1) #(batch, 77, 768 )
                uncond_fused = torch.cat([uncond_image_input, uncond_embeddings_large], dim=1)
                uncond_fused = uncond_fused.to(dtype=torch.float32)
            
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
                "batch": batch
            })
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if batch % 1000 == 0 and batch != 0:
                
                self.extractor.eval()  # 추가
                self.image_adapter.eval()  # 추가
                print('\n모델 저장 시작')
                os.makedirs(f'{opt.output_dir}/image_adapter', exist_ok=True)
                os.makedirs(f'{opt.output_dir}/results', exist_ok=True)  # 결과 이미지 저장 디렉토리
                
                # 임의의 쿼리 이미지 선택
                random_idx = random.randint(0, len(self.test_loader.dataset) - 1)
                sample_image, _, sample_caption = self.test_loader.dataset[random_idx]
                sample_image = sample_image.unsqueeze(0).to('cuda')
                
                # 이미지 임베딩 추출 및 프로젝션
                with torch.no_grad():
                    outputs = self.extractor(sample_image)
                    embedding_feature = outputs[0]
                    projected_image_embedding = self.image_adapter(embedding_feature)
                    
                    # 텍스트 임베딩
                    text_input_ids = self.tokenizer(
                        sample_caption,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to('cuda')
                    
                    # text embeddings 얻기
                    text_embeddings = self.text_encoder(text_input_ids)[0]
                    
                    # conditions 생성
                    conditions = torch.cat([projected_image_embedding, text_embeddings], dim=1)
                    
                    uncond_text_input = self.tokenizer(
                        "",
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to("cuda")

                    #uncond_image_input = torch.zeros_like(projected_image_embedding)
                    uncond_feature = torch.zeros((projected_image_embedding.shape[0], 2048)).to('cuda')  # (batch, 2048)
                    uncond_image_input = self.image_adapter(uncond_feature)  # (batch, 4, 768)
                    
                    with torch.no_grad():
                        uncond_embeddings_large = self.text_encoder(uncond_text_input.input_ids)[0] #(77,768)'input_ids': [49406, 49407]에서 BOS 토큰을 가져온다
                        # uncond_embeddings_large를 배치 크기에 맞게 확장
                        uncond_embeddings_large = uncond_embeddings_large.repeat(projected_image_embedding.shape[0], 1, 1) #(batch, 77, 768 )
                        uncond_fused = torch.cat([uncond_image_input, uncond_embeddings_large], dim=1)
                        uncond_fused = uncond_fused.to(dtype=torch.float32)
                    
                    # 노이즈에서 이미지 생성
                    latents = torch.randn((1, 4, 96, 96), device='cuda')
                    
                    # EulerAncestral 스케줄러로 변경
                    euler_scheduler = EulerAncestralDiscreteScheduler.from_config(self.noise_scheduler.config)
                    euler_scheduler.set_timesteps(30)
                    timesteps = euler_scheduler.timesteps
                    guidance_scale = 7.5  # CFG 스케일
                    
                    for t in timesteps:
                        # 스케일링된 입력 얻기
                        latent_model_input = euler_scheduler.scale_model_input(latents, t)
                        
                        # 노이즈 예측 - CFG 적용
                        # unconditional 예측
                        noise_pred_uncond = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=uncond_fused
                        ).sample
                        
                        # conditional 예측
                        noise_pred_text = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=conditions
                        ).sample
                        
                        # CFG를 사용하여 최종 노이즈 예측
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                        # EulerAncestral 스케줄러 스텝
                        latents = euler_scheduler.step(
                            model_output=noise_pred,
                            timestep=t,
                            sample=latents
                        ).prev_sample
                
                # VAE를 통한 디코딩 추가
                with torch.no_grad():
                    latents = 1 / self.vae.config.scaling_factor * latents
                    image = self.vae.decode(latents).sample

                # 이미지 후처리
                image = (image / 2 + 0.5).clamp(0, 1)
                image = (image * 255).clamp(0, 255).round()
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = image.astype(np.uint8)[0]

                # 생성된 이미지 저장
                result_image = Image.fromarray(image)
                result_image.save(f'{opt.output_dir}/results/generated_image_epoch_{batch}.png')
                
                # 원본 이미지를 PIL Image로 변환
                original_image = sample_image.cpu().squeeze(0).permute(1, 2, 0)
                original_image = ((original_image * 0.5 + 0.5) * 255).clamp(0, 255).round().numpy().astype(np.uint8)
                original_image = Image.fromarray(original_image)

                # wandb에 원본 이미지와 생성된 이미지 함께 로깅
                wandb.log({
                    "original_image": wandb.Image(original_image, caption=f"Original - {sample_caption}"),
                    "generated_image": wandb.Image(result_image, caption=f"Generated - {sample_caption}"),
                    "caption": sample_caption,  # 캡션도 따로 텍스트로 로깅
                    "batch": batch
                })
                
                # extractor 가중치 저장
                image_adapter_path = f'{opt.output_dir}/image_adapter/image_adapter_{batch}.pt'
                torch.save(self.image_adapter.state_dict(), image_adapter_path)
                print(f'Extractor 가중치가 {image_adapter_path}에 저장되었습니다.')
                
                # 저장된 가중치 확인
                if os.path.exists(image_adapter_path):
                    print('모델의 가중치가 성공적으로 저장되었습니다.')
                else:
                    print('가중치 저장 중 오류가 발생했습니다. 파일 경로를 확인해주세요.')

                # 추론 후 다시 train 모드로 되돌리기
                self.image_adapter.train()  # 추가



                # 매칭 실패한 경우 확인
                def check_unmatched_images(dataset):
                    unmatched = [img_path for img_path in dataset.imgs if img_path not in dataset.captions]
                    return unmatched[:5] if unmatched else "모든 이미지가 매칭됨"

                print("\n=== 매칭 실패한 이미지 경로 ===")
                print("Train unmatched:", check_unmatched_images(self.train_loader.dataset))
                print("Query unmatched:", check_unmatched_images(self.test_loader.dataset))
        
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
    image_adapter = Image_adapter(extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=opt.clip_embeddings_dim).to('cuda')
    main = Main(extractor, image_adapter)

    if opt.mode == 'train':
        #wandb 초기화
        wandb.init(
            project="Controlnet",
            config=vars(opt),  # opt의 모든 속성을 config로 추가
            name=f"sd_image_adapter_cfg_euler_adapter_zero_{opt.clip_embeddings_dim}_{opt.epoch}_{opt.lr}"
        )
        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            
            main.train(epoch)



    if opt.mode == 'evaluate':
        print('start evaluate')
        extractor.load_state_dict(torch.load(opt.weight))
        main.evaluate()
        
    if opt.mode == 'tsne':
        print('start tsne')
        main.tsne_visualization()