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
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler
)
from opt import opt
from data import Data
from network import MGN, Image_adapter, Base_adapter, ImageTokenAdapter, VisualTokenProjector
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from sklearn.manifold import TSNE

import random
import wandb
import torch.nn as nn
import itertools
import datetime
import traceback
import warnings
import math

warnings.filterwarnings("ignore", category=UserWarning)

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

class CrossAttention(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        
    def forward(self, query, key_value):
        # query: tensor of shape (batch_size, query_len, embed_dim)
        # key, value: tensors of shape (batch_size, key_len, embed_dim)
        #print(f"query shape: {query.shape}")
        #print(f"key_value shape: {key_value.shape}")
        scores = torch.matmul(query, key_value.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        #print(f"scores shape: {scores.shape}")
        attn_weights = torch.softmax(scores, dim=-1)
        #print(f"attn_weights shape: {attn_weights.shape}")
        return torch.matmul(attn_weights, key_value)

class ControlNetTrainer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        # SDXL pipeline 초기화
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
        )
        
        # controlnet을 새로 초기화
        self.controlnet = ControlNetModel(
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
            block_out_channels=(320, 640, 1280, 1280, 1280, 1280),
            layers_per_block=2,
            cross_attention_dim=2048,
            attention_head_dim=8,
        ).to("cuda", dtype=torch.float32)
        
        self.opt = opt
        self.device = torch.device('cuda')
        
        # pipeline의 noise_scheduler 사용
        self.noise_scheduler = self.pipeline.scheduler

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # pipeline의 tokenizer 사용
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        # pipeline의 text_encoder 사용
        text_embeddings = self.pipeline.text_encoder(text_input_ids.to(device))[0]

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size if negative_prompt is None else negative_prompt
            max_length = text_input_ids.shape[-1]
            uncond_input = self.pipeline.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.pipeline.text_encoder(uncond_input.input_ids.to(device))[0]
            
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

class Main():
    def __init__(self, opt):
        super().__init__()
        
        # result_dir 초기화 추가
        self.result_dir = os.path.join(opt.output_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.result_dir, exist_ok=True)
        
        print("\n=== 모델 초기화 시작 ===")
        
        # model_id 추가
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        print("\n1. SDXL pipeline 초기화 중...")
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
        )
        
        print("\n2. ControlNet 초기화 중...")
        # Scribble ControlNet 로드
        try:
            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-scribble-sdxl-1.0",
                torch_dtype=torch.float32
            )
            print("Scribble ControlNet 로드 성공")
        except Exception as e:
            print(f"Scribble ControlNet 로드 실패, 새로운 ControlNet 초기화: {str(e)}")
            self.controlnet = ControlNetModel(
                in_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
                block_out_channels=(320, 640, 1280, 1280, 1280, 1280),
                layers_per_block=2,
                cross_attention_dim=2048,
                attention_head_dim=8,
            )
        self.controlnet = self.controlnet.to("cuda", dtype=torch.float32)
        
        print("\n4. 모델 컴포넌트 GPU 이동 중...")
        print("- Text Encoder를 GPU로 이동")
        self.text_encoder = self.pipeline.text_encoder.to("cuda", dtype=torch.float32)
        print(f"  현재 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("- Text Encoder 2를 GPU로 이동")
        self.text_encoder_2 = self.pipeline.text_encoder_2.to("cuda", dtype=torch.float32)
        print(f"  현재 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # tokenizer 참조
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2
        
        print("\n3. VAE 초기화 중...")
        # VAE를 별도로 로드하고 설정
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",  # 수정된 VAE 사용
            torch_dtype=torch.float32
        ).to("cuda")
        
        # pipeline에 새로운 VAE 설정
        self.pipeline.vae = self.vae
        
        # 스케줄러 설정
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler"
        )
        
        # ControlNet 파이프라인 설정
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=self.controlnet,
            vae=self.vae,
            scheduler=self.noise_scheduler,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to('cuda')
        
        self.opt = opt
        self.opt.batchsize = 1  # 배치 사이즈를 1로 줄임
        
        print("1. CUDA 메모리 초기화 중...")
        torch.cuda.empty_cache()
        print(f"- 사용 가능한 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"- 현재 할당된 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("\n4. 메모리 최적화 설정 적용 중...")
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()
        print("- 메모리 최적화 설정 완료")
        print(f"- 최종 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"- 남은 GPU 메모리: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
        
        print("\n=== 모델 초기화 완료 ===\n")
        
        self.device = torch.device('cuda')
        
        # Image Token Adapter와 Visual Token Projector 초기화
        self.image_adapter = ImageTokenAdapter().to("cuda", dtype=torch.float32)
        self.visual_projector = VisualTokenProjector().to("cuda", dtype=torch.float32)

        # __init__ 메소드에서
        self.pipeline = self.pipeline.to(dtype=torch.float32)
        self.controlnet = self.controlnet.to(dtype=torch.float32)
        self.text_encoder = self.text_encoder.to(dtype=torch.float32)
        self.text_encoder_2 = self.text_encoder_2.to(dtype=torch.float32)

    def train(self, epoch):
        print("\n=== 학습 초기화 시작 ===")
        print(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("\n1. wandb 초기화 중...")
        wandb.init(
            project="controlnet-sticker",
            name=f"training_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "learning_rate": 1e-5,
                "architecture": "ControlNet-SDXL1.0-base",
                "dataset": "token-sticker",
                "epochs": epoch,
            }
        )

        print("\n2. 데이터 로더 초기화 중...")
        self.data = Data(self.opt)
        print(f"- 데이터 로더 초기화 후 GPU 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("\n3. CUDA 메모리 정리 중...")
        torch.cuda.empty_cache()
        
        print("\n4. 모델 컴포넌트 GPU 이동 중...")
        print("- Text Encoder를 GPU로 이동")
        self.pipeline.text_encoder.to("cuda")
        print(f"  현재 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("- Text Encoder 2를 GPU로 이동")
        self.pipeline.text_encoder_2.to("cuda")
        print(f"  현재 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("- VAE를 GPU로 이동")
        self.pipeline.vae.to("cuda")
        print(f"  현재 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("- UNet을 GPU로 이동")
        self.pipeline.unet.to("cuda")
        print(f"  현재 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("- ControlNet을 GPU로 이동")
        self.pipeline.controlnet.to("cuda")
        print(f"  현재 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("\n5. 모델 설정 중...")
        self.pipeline.controlnet.enable_gradient_checkpointing()
        self.pipeline.controlnet.train()
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        
        print("\n6. Adapter 초기화 중...")
        print("- Image Token Adapter 초기화")
        self.image_adapter = ImageTokenAdapter().to("cuda", dtype=torch.float32)
        print(f"  현재 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("- Visual Token Projector 초기화")
        self.visual_projector = VisualTokenProjector().to("cuda", dtype=torch.float32)
        print(f"  현재 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("- Cross Attention 모듈 초기화")
        self.cross_attn = CrossAttention(dim=768).to("cuda", dtype=torch.float32)
        self.cross_attn_laion = CrossAttention(dim=1280).to("cuda", dtype=torch.float32)
        print(f"  현재 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("\n=== 학습 초기화 완료 ===")
        print(f"최종 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"남은 GPU 메모리: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB\n")
        
        optimizer = torch.optim.AdamW([
            {'params': self.pipeline.controlnet.parameters()},
            {'params': self.image_adapter.parameters()},
            {'params': self.visual_projector.parameters()}
        ], lr=5e-6, weight_decay=1e-2)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=50,
            min_lr=1e-6
        )
        
        # 노이즈 스케줄러 설정
        self.noise_scheduler.set_timesteps(1000)
        
        # Image 저장 디렉토리
        image_save_dir = os.path.join(self.result_dir, 'generated_images')
        os.makedirs(image_save_dir, exist_ok=True)
        
        # Gradient scaling 추가
        scaler = torch.cuda.amp.GradScaler()

        for step, batch in enumerate(tqdm(self.data.token_sticker_loader)):
            optimizer.zero_grad()
            
            # 데이터를 float16으로 변환하고 visual_tokens를 768로 슬라이싱
            visual_tokens = batch["visual_tokens"][:, :, :768].to("cuda", dtype=torch.float32)  # [B, 192, 768]로 슬라이싱
            sticker_image = batch["sticker_image"].to("cuda", dtype=torch.float32)
            sketch_image = batch["sketch_image"].to("cuda", dtype=torch.float32)
            
            with torch.cuda.amp.autocast():
                # 프롬프트 인코딩
                text_encoder_output = self.text_encoder(
                    self.tokenizer(
                        "a photo of character sticker",
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to("cuda")
                ).last_hidden_state  # [B, 77, 768]
                
                # LAION encoder에서 두 종류의 임베딩을 동시에 얻기
                encoder_output = self.text_encoder_2(
                    self.tokenizer_2(
                        "a photo of character sticker",
                        padding="max_length",
                        max_length=self.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to("cuda"),
                    output_hidden_states=True,
                )
                
                # hidden states는 sequence 임베딩용 [B, 77, 1280]
                text_encoder_output_laion = encoder_output.last_hidden_state  
                
                # text_embeds는 pooled output 대신 사용 [B, 1280]
                text_embeds = encoder_output.text_embeds  
                
                # visual tokens를 CLIP과 LAION 공간으로 projection
                clip_tokens, laion_tokens = self.visual_projector(visual_tokens)
                
                # Image adapter로 sketch에서 토큰 생성
                clip_adapted, laion_adapted = self.image_adapter(sketch_image)  # 각각 [B, 4, 768], [B, 4, 1280]
                
                # Cross Attention 적용
                self.cross_attn = CrossAttention(dim=768).to("cuda", dtype=torch.float32)
                self.cross_attn_laion = CrossAttention(dim=1280).to("cuda", dtype=torch.float32)
                
                # CLIP 토큰들의 Cross Attention
                clip_attended = self.cross_attn(
                    query=clip_adapted,      # [B, 4, 768]
                    key_value=clip_tokens    # [B, 192, 768] 
                )
                
                # LAION 토큰들의 Cross Attention
                laion_attended = self.cross_attn_laion(
                    query=laion_adapted,     # [B, 4, 1280]
                    key_value=laion_tokens   # [B, 192, 1280] 
                )
                # print("============================================")
                # print(f"clip_attended shape: {clip_attended.shape}")
                # print(f"laion_attended shape: {laion_attended.shape}")
                # print(f"text_encoder_output shape: {text_encoder_output.shape}")
                # print(f"text_encoder_output_laion shape: {text_encoder_output_laion.shape}")
                # 각각의 임베딩을 text encoder 출력과 결합
                clip_embeds = torch.cat([text_encoder_output, clip_attended], dim=1)    # [B, 81, 768]
                laion_embeds = torch.cat([text_encoder_output_laion, laion_attended], dim=1)  # [B, 77, 1280]
                
                # 최종 결합
                combined_embeds = torch.cat([clip_embeds, laion_embeds], dim=-1)  # [B, 81, 2048] # 이게 prompt로 들어가는거같음
                
                # 랜덤 noise latents 생성
                latents = torch.randn(
                    (1, 4, 96, 96),
                    device="cuda",
                    dtype=torch.float32
                )
                
                # timestep 생성
                timesteps = self.noise_scheduler.timesteps
                timestep = timesteps[random.randint(0, len(timesteps) - 1)]
                timestep = torch.tensor([timestep], device="cuda", dtype=torch.float32)
                
                # 노이즈 생성 및 noisy latents 생성
                noise = torch.randn_like(latents, dtype=torch.float32)
                noisy_latents = self.noise_scheduler.add_noise(
                    latents,
                    noise,
                    timestep  # 이제 1-d tensor
                )
                
                # ControlNet에 sketch 이미지 입력
                down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
                    noisy_latents,
                    timestep,
                    encoder_hidden_states=combined_embeds,
                    controlnet_cond=sticker_image,  # sketch가 아닌 sticker_image를 사용
                    return_dict=False,
                )
                
                # ControlNet 출력 상태 확인
                print("\nControlNet 출력 상태:")
                print(f"down_block_res_samples requires_grad: {[x.requires_grad for x in down_block_res_samples]}")
                print(f"mid_block_res_sample requires_grad: {mid_block_res_sample.requires_grad}")
                
                # UNet에 전달
                original_size = (768, 768)  # 768x768로 변경
                crop_size = (768, 768)     # 768x768로 변경
                target_size = (768, 768)   # 768x768로 변경
                crop_coords = (0, 0)         # (x, y)

                time_ids = torch.tensor([
                    original_size[0],  # original height
                    original_size[1],  # original width
                    crop_size[0],     # crop height
                    crop_size[1],     # crop width
                    crop_coords[0],   # crop x
                    crop_coords[1],   # crop y
                ], dtype=torch.float32).unsqueeze(0).to("cuda")  # [1, 6]
                
                noise_pred = self.pipeline.unet(
                    noisy_latents,
                    timestep,
                    encoder_hidden_states=combined_embeds,
                    added_cond_kwargs={
                        "text_embeds": text_embeds,
                        "time_ids": time_ids
                    },
                ).sample
                
                # loss 계산 전 tensor 상태 확인
                print(f"\nBefore loss calculation:")
                print(f"noise_pred: requires_grad={noise_pred.requires_grad}, dtype={noise_pred.dtype}")
                print(f"noise: requires_grad={noise.requires_grad}, dtype={noise.dtype}")

                # noise_pred와 noise의 dtype 불일치 해결
                noise_pred = noise_pred.float()
                noise = noise.float()
                loss = F.mse_loss(noise_pred, noise, reduction="mean")

                # loss 계산 후 상태 확인
                print(f"\nAfter loss calculation:")
                print(f"Loss value: {loss.item()}")
                print(f"Loss requires_grad: {loss.requires_grad}")
                print(f"Loss grad_fn: {loss.grad_fn}")
                if torch.isnan(loss):
                    print("Warning: Loss is NaN!")
                    print(f"noise_pred dtype: {noise_pred.dtype}")
                    print(f"noise dtype: {noise.dtype}")
                
                # backward with scaler
                scaler.scale(loss).backward()
                
                # gradient clipping 전에 unscale
                if step % 5 == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.pipeline.controlnet.parameters(), max_norm=1.0)
                    
                    # optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                
                # Learning rate scheduler
                scheduler.step(loss)
            
            # 매 100 스텝마다 이미지 생성 및 로깅
            if step % 1000 == 0:
                try:
                    self.pipeline.controlnet.eval()
                    self.image_adapter.eval()
                    self.visual_projector.eval()
                    
                    with torch.no_grad():
                        test_tokens = visual_tokens[0:1].clone().to(dtype=torch.float32)
                        
                        # 프롬프트 인코딩
                        text_encoder_output = self.text_encoder(
                            self.tokenizer(
                                "a photo of character sticker",
                                padding="max_length",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt",
                            ).input_ids.to("cuda")
                        ).last_hidden_state.to(dtype=torch.float32)

                        encoder_output = self.text_encoder_2(
                            self.tokenizer_2(
                                "a photo of character sticker",
                                padding="max_length",
                                max_length=self.tokenizer_2.model_max_length,
                                truncation=True,
                                return_tensors="pt",
                            ).input_ids.to("cuda"),
                            output_hidden_states=True,
                        )
                        
                        text_encoder_output_laion = encoder_output.last_hidden_state.to(dtype=torch.float32)
                        text_embeds = encoder_output.text_embeds.to(dtype=torch.float32)

                        # Visual tokens 처리
                        clip_tokens, laion_tokens = self.visual_projector(test_tokens)
                        clip_adapted, laion_adapted = self.image_adapter(sketch_image.to(dtype=torch.float32))

                        # Cross Attention
                        clip_attended = self.cross_attn(
                            query=clip_adapted,
                            key_value=clip_tokens
                        )
                        
                        laion_attended = self.cross_attn_laion(
                            query=laion_adapted,
                            key_value=laion_tokens
                        )

                        # 임베딩 결합
                        clip_embeds = torch.cat([text_encoder_output, clip_attended], dim=1)
                        laion_embeds = torch.cat([text_encoder_output_laion, laion_attended], dim=1)
                        combined_embeds = torch.cat([clip_embeds, laion_embeds], dim=-1).to(dtype=torch.float32)

                        # 랜덤 noise latents 생성
                        latents = torch.randn(
                            (1, 4, 96, 96),
                            device="cuda",
                            dtype=torch.float32
                        )

                        # timestep 생성
                        timesteps = self.noise_scheduler.timesteps
                        timestep = timesteps[random.randint(0, len(timesteps) - 1)]
                        timestep = torch.tensor([timestep], device="cuda", dtype=torch.float32)

                        # 노이즈 생성 및 noisy latents 생성
                        noise = torch.randn_like(latents, dtype=torch.float32)
                        noisy_latents = self.noise_scheduler.add_noise(
                            latents,
                            noise,
                            timestep
                        )

                        # ControlNet에 sketch 이미지 입력
                        down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
                            noisy_latents.to(dtype=torch.float32),
                            timestep.to(dtype=torch.float32),
                            encoder_hidden_states=combined_embeds.to(dtype=torch.float32),
                            controlnet_cond=sticker_image.to(dtype=torch.float32),
                            return_dict=False,
                        )

                        # UNet에 전달
                        original_size = (768, 768)  # 768x768로 변경
                        crop_size = (768, 768)     # 768x768로 변경
                        target_size = (768, 768)   # 768x768로 변경
                        crop_coords = (0, 0)

                        time_ids = torch.tensor([
                            original_size[0],
                            original_size[1],
                            crop_size[0],
                            crop_size[1],
                            crop_coords[0],
                            crop_coords[1],
                        ], dtype=torch.float32).unsqueeze(0).to("cuda")

                        noise_pred = self.pipeline.unet(
                            noisy_latents,
                            timestep,
                            encoder_hidden_states=combined_embeds,
                            added_cond_kwargs={
                                "text_embeds": text_embeds,
                                "time_ids": time_ids
                            },
                        ).sample

                        # VAE 디코딩 전 latents 처리 수정
                        print("\nProcessing latents:")
                        # 원본 latents 상태 확인
                        print(f"Original latents: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
                        
                        # 정규화 및 스케일링
                        latents = (latents - latents.mean()) / latents.std() * 0.18215
                        latents = torch.clamp(latents, -2.0, 2.0)  # 더 엄격한 클리핑
                        print(f"Normalized latents: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
                        
                        # VAE 디코딩
                        try:
                            torch.cuda.empty_cache()
                            # float32로 통일
                            latents = latents.to(dtype=torch.float32)
                            
                            # 디코딩 전 추가 체크
                            print(f"Pre-decode latents range: ({latents.min().item():.4f}, {latents.max().item():.4f})")
                            
                            # 단계별 디코딩
                            vae_output = self.pipeline.vae.decode(latents)
                            vae_output = vae_output.sample
                            print(f"Raw VAE output range: ({vae_output.min().item():.4f}, {vae_output.max().item():.4f})")
                            
                            # 엄격한 클리핑
                            vae_output = torch.clamp(vae_output, -1.0, 1.0)
                            print(f"Clipped VAE output range: ({vae_output.min().item():.4f}, {vae_output.max().item():.4f})")
                            
                            # 이미지 변환
                            image = (vae_output / 2 + 0.5).clamp(0, 1)
                            print(f"Final image tensor range: ({image.min().item():.4f}, {image.max().item():.4f})")
                            
                            # 메모리 정리
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"VAE decoding error: {str(e)}")
                            print(traceback.format_exc())
                            continue
                        
                        # numpy 변환 및 이미지 저장
                        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                        image = (image * 255).round().astype(np.uint8)[0]
                        
                        # 최종 이미지 상태 확인
                        print(f"Final numpy array shape: {image.shape}")
                        print(f"Final numpy array range: ({image.min()}, {image.max()})")
                        
                        generated_images = Image.fromarray(image)

                        # 모든 학습 모델 저장
                        model_save_dir = os.path.join(self.result_dir, 'models')
                        os.makedirs(model_save_dir, exist_ok=True)
                        
                        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        model_save_path = os.path.join(model_save_dir, f'model_step_{step}_{current_time}.pt')
                        
                        # 세 모델의 상태 모두 저장
                        torch.save({
                            'step': step,
                            'controlnet_state_dict': self.pipeline.controlnet.state_dict(),
                            'image_adapter_state_dict': self.image_adapter.state_dict(),
                            'visual_projector_state_dict': self.visual_projector.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'loss': loss.item(),
                        }, model_save_path)
                        
                        # wandb 로깅
                        wandb.log({
                            "step": step,
                            "loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "generated_image": wandb.Image(generated_images, caption=f"Step {step}"),
                            "visual_tokens": wandb.Image(visual_tokens[0].cpu(), caption="Input Tokens"),
                        })
                        
                        print(f"\nStep {step}:")
                        print(f"- Model saved at: {model_save_path}")
                        print(f"- Current loss: {loss.item():.4f}")
                    
                    # 다시 train 모드로 전환
                    self.pipeline.controlnet.train()
                    self.image_adapter.train()
                    self.visual_projector.train()
                    
                except Exception as e:
                    print(f"Error in generation: {str(e)}")
                    print(traceback.format_exc())
                    continue
            
            # 기본 loss 로깅
            wandb.log({"training_loss": loss.item()})
            
            # 메모리 관리
            if step % 5 == 0:
                torch.cuda.empty_cache()

            # Gradient 모니터링 추가 (디버깅용)
            if step % 100 == 0:
                for name, param in self.pipeline.controlnet.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"{name} gradient norm: {grad_norm}")

    def test(self, pose_image, target_image, prompt="a photo of character sticker", save_path=None):
        self.controlnet.eval()
        self.unet.eval()
        self.vae.eval()
        
        with torch.no_grad():
            # 포즈 이미지 전처리
            if isinstance(pose_image, str):
                pose_image = Image.open(pose_image).convert("RGB")
                pose_image = self.transform(pose_image).unsqueeze(0).to("cuda")
            
            # 프겟 이미지 전처리
            if isinstance(target_image, str):
                target_image = Image.open(target_image).convert("RGB")
                target_image = self.transform(target_image).unsqueeze(0).to("cuda")
            
            # 프롬프트 인코딩
            prompt_embeds, negative_embeds = self.encode_prompt(prompt, "", "cuda")
            
            # VAE를 통한 타겟 이미지의 latent 추출
            latents = self.vae.encode(target_image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
            # 노이즈 제거 스케줄러 설정
            self.noise_scheduler.set_timesteps(50)
            
            # 디노이징 과정
            for t in self.noise_scheduler.timesteps:
                # ControlNet으로 조건부 특징 추출
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=pose_image,
                    return_dict=False,
                )
                
                
                # UNet으로 노이즈 예측
                noise_pred = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # 노이즈 제거 스텝
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            
            # VAE로 이미지 디코딩
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = Image.fromarray((image * 255).astype(np.uint8))
            
            # 이미지 저장
            if save_path:
                image.save(save_path)
            
            return image

    def encode_prompt(self, prompt, negative_prompt, device):
        if isinstance(prompt, list):
            prompt = prompt[0]
        if isinstance(negative_prompt, list):
            negative_prompt = negative_prompt[0]

        # Tokenize prompts
        prompt_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        negative_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        # Get prompt embeddings from text encoder
        prompt_embeds = self.text_encoder(prompt_tokens)[0]
        negative_embeds = self.text_encoder(negative_tokens)[0]

        # Get pooled embeddings from text_encoder_2
        prompt_embeds_2 = self.text_encoder_2(prompt_tokens, output_hidden_states=True)
        negative_embeds_2 = self.text_encoder_2(negative_tokens, output_hidden_states=True)

        # Concatenate embeddings
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2.hidden_states[-2]], dim=-1)
        negative_embeds = torch.cat([negative_embeds, negative_embeds_2.hidden_states[-2]], dim=-1)
        prompt_embeds = torch.cat([negative_embeds, prompt_embeds])

        # Get pooled embeddings
        pooled_prompt_embeds = torch.cat([
            negative_embeds_2.text_embeds,
            prompt_embeds_2.text_embeds,
        ])

        return prompt_embeds, pooled_prompt_embeds

    def __del__(self):
        # 학습 종료 시 wandb 종료
        wandb.finish()

if __name__ == '__main__':
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Currently allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # opt 설정
    class Options:
        def __init__(self):
            self.data_path = "/workspace/data/changhyun/dataset/"
            self.output_dir = "/workspace/data/changhyun/projects/emoji_generation/output"
            self.batchsize = 1  # 배치 사이즈 줄임

    opt = Options()
    main = Main(opt)
    main.train(epoch=10)  # 테스트 실행

