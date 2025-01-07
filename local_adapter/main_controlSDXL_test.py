import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
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

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler
)
from opt import opt
from data import Data
from network import MGN, Image_adapter, Base_adapter
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from sklearn.manifold import TSNE

import random
import wandb
import torch.nn as nn
import itertools
from torchvision import transforms

# LoRA를 적용한 Custom Attention Processor
class LoRAAttnProcessor(nn.Module):
    """LoRA attention processor for self-attention"""
    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, scale=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.scale = scale
        
        # SDXL은 두 개의 projection paths를 가짐
        self.to_q_lora = nn.ModuleDict({
            'down': nn.Linear(hidden_size, rank, bias=False),
            'up': nn.Linear(rank, hidden_size, bias=False)
        })
        self.to_k_lora = nn.ModuleDict({
            'down': nn.Linear(hidden_size, rank, bias=False),
            'up': nn.Linear(rank, hidden_size, bias=False)
        })
        self.to_v_lora = nn.ModuleDict({
            'down': nn.Linear(hidden_size, rank, bias=False),
            'up': nn.Linear(rank, hidden_size, bias=False)
        })
            
    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        # LoRA 적용 (scaling 포함)
        lora_scale = scale * self.scale
        query = query + lora_scale * self.to_q_lora['up'](self.to_q_lora['down'](query))
        key = key + lora_scale * self.to_k_lora['up'](self.to_k_lora['down'](key))
        value = value + lora_scale * self.to_v_lora['up'](self.to_v_lora['down'](value))

        # Attention 연산
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        query = attn.head_to_batch_dim(query)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class LoRAIPAttnProcessor(nn.Module):
    """LoRA attention processor for cross-attention"""
    def __init__(self, hidden_size, cross_attention_dim, rank=4, scale=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.scale = scale
        
        # SDXL cross-attention용 LoRA layers
        self.to_q_lora = nn.ModuleDict({
            'down': nn.Linear(hidden_size, rank, bias=False),
            'up': nn.Linear(rank, hidden_size, bias=False)
        })
        self.to_k_lora = nn.ModuleDict({
            'down': nn.Linear(cross_attention_dim, rank, bias=False),
            'up': nn.Linear(rank, hidden_size, bias=False)
        })
        self.to_v_lora = nn.ModuleDict({
            'down': nn.Linear(cross_attention_dim, rank, bias=False),
            'up': nn.Linear(rank, hidden_size, bias=False)
        })
        
    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        
        # Cross attention projection
        key = self.to_k_lora['up'](self.to_k_lora['down'](encoder_hidden_states))
        value = self.to_v_lora['up'](self.to_v_lora['down'](encoder_hidden_states))
        
        # LoRA 적용 (scaling 포함)
        lora_scale = scale * self.scale
        query = query + lora_scale * self.to_q_lora['up'](self.to_q_lora['down'](query))
        key = key + lora_scale * self.to_k_lora['up'](self.to_k_lora['down'](key))
        value = value + lora_scale * self.to_v_lora['up'](self.to_v_lora['down'](value))

        # Attention 연산
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        query = attn.head_to_batch_dim(query)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class Main():
    def __init__(self):
        #self.data = Data()
        #self.train_loader = self.data.train_loader
        #self.test_loader = self.data.test_loader
        #self.testset = self.data.testset

        # SDXL 모델 ID로 변경
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.controlnet_model_id = "thibaud/controlnet-openpose-sdxl-1.0"
        self.lora_model_path = "./checkpoints/sticker_lora/StickersRedmond.safetensors"  # LoRA 가중치 디렉토리
        
        # SDXL용 스케줄러 설정
        self.noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler"
        )
        
        torch.cuda.set_device(0)
        
        # ControlNet SDXL 파이프라인 로드
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=ControlNetModel.from_pretrained(
                self.controlnet_model_id,
                torch_dtype=torch.float32,
                use_safetensors=False  # .bin 파일 사용 허용
            ),
            scheduler=self.noise_scheduler,
            torch_dtype=torch.float32
        ).to('cuda')
        
        # LoRA 가중치 로드 시도
        if os.path.exists(self.lora_model_path):
            try:
                self.pipeline.load_lora_weights(self.lora_model_path)
                self.pipeline.fuse_lora(lora_scale=0.7)
                print(f"Successfully loaded and fused LoRA weights from {self.lora_model_path}")
            except Exception as e:
                print(f"Error loading LoRA weights: {e}")
                print("Continuing without LoRA weights...")
        else:
            print(f"LoRA weights directory {self.lora_model_path} not found. Continuing without LoRA...")
        
        # 컴포넌트 추출
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2
        self.unet = self.pipeline.unet
        self.controlnet = self.pipeline.controlnet
        self.vae = self.pipeline.vae
        
        del self.pipeline
        torch.cuda.empty_cache()
        
        self.vae.eval()
        self.controlnet.eval()
        
        self.result_dir = opt.output_dir
        os.makedirs(self.result_dir, exist_ok=True)

    def train(self, epoch):
        # 학습 모드로 설정
        self.controlnet.train()
        self.unet.requires_grad_(False)  # UNet은 고정
        self.vae.requires_grad_(False)   # VAE도 고정
        
        # 데이터로더 초기화
        self.data = Data()
        train_dataloader = self.data.sticker_pose_loader
        
        # Optimizer 설정 (ControlNet의 파라미터만 학습)
        optimizer = torch.optim.AdamW(self.controlnet.parameters(), lr=1e-5)
        
        # 학습 루프
        for step, batch in enumerate(tqdm(train_dataloader)):
            # 배치 데이터 추출 (새로운 데이터로더 형식에 맞춤)
            pose_images = batch["pose_images"].to("cuda")
            target_images = batch["target_images"].to("cuda")
            prompt = batch["caption"]  # 이미 "a photo of character sticker"로 고정됨
            
            # 잠재 공간으로 인코딩
            latents = self.vae.encode(target_images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
            # 노이즈 추가
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 프롬프트 인코딩
            prompt_embeds, _ = self.encode_prompt(prompt, "", "cuda")
            
            # ControlNet을 통한 조건부 특징 추출
            down_block_res_samples, mid_block_res_sample = self.controlnet( 
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=pose_images,
                return_dict=False,
            )
            # shape 예시:
            # down_block_res_samples: [
            #     torch.Size([B, 320, 128, 128]),
            #     torch.Size([B, 640, 64, 64]),
            #     torch.Size([B, 1280, 32, 32]),
            #     ...
            # ]
            # mid_block_res_sample: torch.Size([B, 1280, 32, 32])
            
            # UNet을 통한 노이즈 예측
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
            
            # 손실 계산 (Simple L2 Loss)
            loss = F.mse_loss(noise_pred, noise)
            
            # 역전파 및 옵티마이저 스텝
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 로깅
            if step % 100 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")
        
        # 에포크마다 체크포인트 저장
        checkpoint_path = os.path.join(self.result_dir, f"controlnet_epoch_{epoch}.pth")
        torch.save(self.controlnet.state_dict(), checkpoint_path)
    
    def test(self, pose_image, prompt="a photo of character sticker", use_down_only=True):
        self.controlnet.eval()
        self.unet.eval()
        
        with torch.no_grad():
            # pose_image 전처리 (1024x1024 유지)
            if isinstance(pose_image, Image.Image):
                pose_image = transforms.ToTensor()(pose_image).unsqueeze(0)
            pose_image = transforms.Resize((1024, 1024))(pose_image)
            if pose_image.shape[1] == 4:
                pose_image = pose_image[:, :3, :, :]
            pose_image = pose_image.to(device="cuda")

            # 프롬프트 인코딩
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, "", "cuda")
            
            added_cond_kwargs = {
                "text_embeds": torch.zeros((2, 1280), device="cuda"),
                "time_ids": torch.zeros((2, 6), device="cuda")
            }
            
            # latents 크기를 64x64로 변경
            latents = torch.randn(
                (1, self.unet.config.in_channels, 64, 64),  # 128x128 -> 64x64
                device="cuda",
            )
            
            # 노이즈 스케줄러 설정
            self.noise_scheduler.set_timesteps(30)
            latents = latents * self.noise_scheduler.init_noise_sigma
            
            # 디노이징 루프
            for t in self.noise_scheduler.timesteps:
                # ControlNet으로부터 residual 얻기
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=pose_image,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                
                # 선택적으로 down_block 또는 mid_block residual만 사용
                if use_down_only:
                    # down_block residual만 사용
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=None,  # mid block 무시
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                else:
                    # mid_block residual만 사용
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        down_block_additional_residuals=None,  # down block 무시
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                
                # 노이즈 제거 스텝
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            
            # 이미지 디코딩
            image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
            image = Image.fromarray((image * 255).astype(np.uint8))
            
            return image

    def encode_prompt(self, prompt, negative_prompt, device):
        # Text encoders 처리
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        negative_text_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_text_inputs_2 = self.tokenizer_2(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_inputs = text_inputs.to(device)
        text_inputs_2 = text_inputs_2.to(device)
        negative_text_inputs = negative_text_inputs.to(device)
        negative_text_inputs_2 = negative_text_inputs_2.to(device)
        
        # 첫 번째 텍스트 인코더
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids,
            output_hidden_states=True,
        )
        negative_prompt_embeds = self.text_encoder(
            negative_text_inputs.input_ids,
            output_hidden_states=True,
        )
        
        # 두 번째 텍스트 인코더
        prompt_embeds_2 = self.text_encoder_2(
            text_inputs_2.input_ids,
            output_hidden_states=True,
        )
        negative_prompt_embeds_2 = self.text_encoder_2(
            negative_text_inputs_2.input_ids,
            output_hidden_states=True,
        )
        
        # pooled 출력과 hidden states 사용
        prompt_embeds = prompt_embeds.hidden_states[-2]
        negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
        negative_prompt_embeds_2 = negative_prompt_embeds_2.hidden_states[-2]
        
        # 임베딩 결합 (hidden states)
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds_2], dim=-1)
        
        # 배치 차원 확장
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        
        # pooled_prompt_embeds는 필요 없으므로 None 반환
        return prompt_embeds, None

if __name__ == '__main__':
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Currently allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    main = Main()
    
    # 테스트용 포즈 이미지 로드
    pose_image = Image.open("./openpose_sample_input.png")  # 실제 포즈 이미지 경로로 수정
    pose_image = transforms.ToTensor()(pose_image).unsqueeze(0).to("cuda")
    
    # test 함수 호출 시 pose_image 전달
    main.test(pose_image=pose_image)

