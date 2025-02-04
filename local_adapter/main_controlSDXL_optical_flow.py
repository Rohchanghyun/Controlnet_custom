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
from network import MGN, Image_adapter, Base_adapter, ImageTokenAdapter, VisualTokenProjector, TokenImageAdapter
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


class Main():
    def __init__(self, opt):
        super().__init__()
        
        # result_dir 초기화 추가
        self.result_dir = os.path.join(opt.output_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.result_dir, exist_ok=True)
        
        print("\n=== 모델 초기화 시작 ===")
        
        # model_id 추가
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.controlnet_model_id = "xinsir/controlnet-scribble-sdxl-1.0"
        print("\n1. SDXL pipeline + ControlNet 초기화 중...")
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=ControlNetModel.from_pretrained(
                self.controlnet_model_id,
                torch_dtype=torch.float32,
                use_safetensors=True  # .bin 파일 사용 허용
            ),
            scheduler=self.noise_scheduler,
            torch_dtype=torch.float32
        ).to('cuda')
        

        
        print("\n2. 모델 컴포넌트 GPU 이동 중...")
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
        self.noise_scheduler_inference = EulerDiscreteScheduler.from_pretrained(
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
        
        print("4. CUDA 메모리 초기화 중...")
        torch.cuda.empty_cache()
        print(f"- 사용 가능한 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"- 현재 할당된 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("\n5. 메모리 최적화 설정 적용 중...")
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()
        print("- 메모리 최적화 설정 완료")
        print(f"- 최종 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"- 남은 GPU 메모리: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
        
        print("\n=== 모델 초기화 완료 ===\n")
        
        self.device = torch.device('cuda')

        # TokenImageAdapter 추가
        self.token_image_adapter = TokenImageAdapter().to("cuda", dtype=torch.float32)

        # __init__ 메소드에서
        self.pipeline = self.pipeline.to(dtype=torch.float32)
        self.controlnet = self.controlnet.to(dtype=torch.float32)
        self.text_encoder = self.text_encoder.to(dtype=torch.float32)
        self.text_encoder_2 = self.text_encoder_2.to(dtype=torch.float32)

    def get_time_ids(self):
        # SDXL에서 사용하는 time_ids 생성
        original_size = (768, 768)  # 원본 이미지 크기
        target_size = (768, 768)    # 타겟 이미지 크기
        crops_coords_top_left = (0, 0)  # 크롭 좌표
        
        add_time_ids = torch.tensor([
            original_size[0],        # original height
            original_size[1],        # original width
            target_size[0],          # target height
            target_size[1],          # target width
            crops_coords_top_left[0], # crop top
            crops_coords_top_left[1], # crop left
        ], dtype=torch.float32)
        
        # 배치 차원 추가하고 GPU로 이동
        add_time_ids = add_time_ids.unsqueeze(0).to("cuda")
        
        return add_time_ids

    def train(self, epoch):
        print("\n=== 학습 초기화 시작 ===")
        print(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        print("\n1. wandb 초기화 중...")
        wandb.init(
            project="controlnet-sticker",
            name=f"training_run_no_ca_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
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

        
        print("- TokenImageAdapter 초기화")
        self.token_image_adapter = TokenImageAdapter().to("cuda", dtype=torch.float32)
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
            {'params': self.token_image_adapter.parameters()}
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
        self.noise_scheduler_inference.set_timesteps(num_inference_steps=50)  # 예: 50 steps
        
        # Image 저장 디렉토리
        image_save_dir = os.path.join(self.result_dir, 'generated_images')
        os.makedirs(image_save_dir, exist_ok=True)

        for step, batch in enumerate(tqdm(self.data.token_sticker_loader)):
            # 각 배치 시작 시 optimizer 초기화
            optimizer.zero_grad()
            
            # 데이터 준비
            visual_tokens = batch["visual_tokens"][:, :, :768].to("cuda", dtype=torch.float32)
            sketch_image = batch["sketch_image"].to("cuda", dtype=torch.float32)
            sticker_image = batch["sticker_image"].to("cuda", dtype=torch.float32)

            sketch_image = (sketch_image / 127.5) - 1.0  # [0,255] → [-1,1]
            sticker_image = (sticker_image / 127.5) - 1.0  # [0,255] → [-1,1]

            # 두 text encoder의 출력을 올바르게 결합
            prompt_embeds = self.text_encoder(
                self.tokenizer(
                    "a photo of character sticker",
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to("cuda")
            ).last_hidden_state

            # negative prompt도 포함
            negative_prompt_embeds = self.text_encoder(
                self.tokenizer(
                    "",  # empty negative prompt
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to("cuda")
            ).last_hidden_state

            # Text encoder 2
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

            negative_prompt_embeds_2 = self.text_encoder_2(
                self.tokenizer_2(
                    "",
                    padding="max_length",
                    max_length=self.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to("cuda"),
                output_hidden_states=True,
            )

            # SDXL 형식으로 올바르게 결합
            prompt_embeds = torch.cat([
                torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                torch.cat([negative_prompt_embeds_2.hidden_states[-2], encoder_output.hidden_states[-2]], dim=0)
            ], dim=-1)

            # TokenImageAdapter를 사용하여 visual tokens와 sketch image 결합
            adapted_image = self.token_image_adapter(visual_tokens, sketch_image)
            #print("adapted_image:", adapted_image)  # None인지 확인
            #print("adapted_image shape:", adapted_image.shape if adapted_image is not None else "None")
            
            # 먼저 timestep 정의
            timesteps = self.noise_scheduler.timesteps
            timestep = timesteps[random.randint(0, len(timesteps) - 1)]
            timestep = torch.tensor([timestep], device="cuda", dtype=torch.float32)

            # sticker image를 VAE로 인코딩
            latents = self.pipeline.vae.encode(sticker_image).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor
            
            # noise 추가
            noise = torch.randn_like(latents)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timestep)
            
            # Text encoder 2의 출력 처리
            text_embeds = encoder_output.text_embeds

            # UNet에 전달
            time_ids = self.get_time_ids()
            
            added_cond_kwargs={
                "text_embeds": text_embeds,
                "time_ids": time_ids
            }
            
            # ControlNet forward pass
            down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
                noisy_latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=adapted_image,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )

            noise_pred = self.pipeline.unet(
                noisy_latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # loss 계산
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            # loss가 valid한지 확인
            if not torch.isfinite(loss):
                print(f"Warning: Loss is {loss.item()}, skipping batch")
                continue

            # Backward pass
            loss.backward()
            
            # 매 5 스텝마다 optimizer step
            if (step + 1) % 5 == 0:
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.pipeline.controlnet.parameters(),
                        self.token_image_adapter.parameters()
                    ),
                    max_norm=1.0
                )
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Learning rate scheduler update
            scheduler.step(loss)

            # 로깅
            if step % 1000 == 0:
                print(f"\nStep {step}:")
                print(f"Loss: {loss.item():.4f}")
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
                print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
                # 가중치 저장
                checkpoint_dir = os.path.join(self.result_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # ControlNet 저장
                controlnet_path = os.path.join(checkpoint_dir, f'controlnet_step_{step}.pt')
                torch.save(
                    self.pipeline.controlnet.state_dict(),
                    controlnet_path
                )
                print(f"ControlNet saved to: {controlnet_path}")
                print(f"File exists: {os.path.exists(controlnet_path)}")
                print(f"File size: {os.path.getsize(controlnet_path) / 1024 / 1024:.2f} MB")
                
                # TokenImageAdapter 저장
                adapter_path = os.path.join(checkpoint_dir, f'token_image_adapter_step_{step}.pt')
                torch.save(
                    self.token_image_adapter.state_dict(),
                    adapter_path
                )
                print(f"TokenImageAdapter saved to: {adapter_path}")
                print(f"File exists: {os.path.exists(adapter_path)}")
                print(f"File size: {os.path.getsize(adapter_path) / 1024 / 1024:.2f} MB")

        
                with torch.no_grad():
                    print("sketch_image shape:", sketch_image.shape)
                    latent = self.pipeline.vae.encode(sketch_image.to(dtype=torch.float32)).latent_dist.sample()
                    latent = latent * self.pipeline.vae.config.scaling_factor  # 스케일링 적용

                    noise = torch.randn_like(latent)
                    noisy_latent = self.pipeline.scheduler.add_noise(latent, noise, timestep)

                    adapted_image = self.token_image_adapter(visual_tokens, sketch_image).to("cuda")
                    adapted_image = adapted_image.clamp(-1, 1)  # 명시적 클리핑

                    # 프롬프트 인코딩
                    prompt="a photo of character sticker"
                    prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, "", "cuda")
                    
                    added_cond_kwargs = {
                        "text_embeds": torch.zeros((2, 1280), device="cuda"),
                        "time_ids": self.get_time_ids()
                    }
                    
                    # # latents 크기를 64x64로 변경
                    # latents = torch.randn(
                    #     (1, self.pipeline.unet.config.in_channels, 96, 96),  # 128x128 -> 64x64
                    #     device="cuda",
                    # )
                    
                    # 노이즈 스케줄러 설정
                    self.noise_scheduler_inference.set_timesteps(30)
                    latents = latents * self.noise_scheduler_inference.init_noise_sigma
                    
                    # 디노이징 루프
                    for t in self.noise_scheduler_inference.timesteps:
                        # ControlNet으로부터 residual 얻기
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            latents,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_cond=adapted_image,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )
                        

                        noise_pred = self.pipeline.unet(
                            latents,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            down_block_additional_residuals=down_block_res_samples,  # down block 무시
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample
                        
                        # 노이즈 제거 스텝
                        latents = self.noise_scheduler_inference.step(noise_pred, t, latents).prev_sample
                    
                    # 이미지 디코딩
                    image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
                    image = Image.fromarray((image * 255).astype(np.uint8))

                    wandb.log({
                        "step": step,
                        "loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "generated_image": wandb.Image(image, caption=f"Step {step}"),
                        "input_sketch": wandb.Image(sketch_image[0], caption="Input Sketch"),
                        "adapted_image": wandb.Image(adapted_image[0], caption="Adapted Image")
                    })

            # 메모리 관리
            if step % 5 == 0:
                torch.cuda.empty_cache()

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
            prompt="a photo of character sticker"
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
            print("decoded image shape:", image.shape)
            decoded_image = (image / 2 + 0.5).clamp(0, 1)
            decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).float().numpy()

            if decoded_image.shape[0] == 1:
                decoded_image = decoded_image.squeeze(0)  # (1, H, W, C) -> (H, W, C)

            decoded_image = (decoded_image * 255).astype(np.uint8)
            decoded_image = Image.fromarray(decoded_image)
            
            # 이미지 저장
            if save_path:
                image.save(save_path)
            
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
            self.batchsize = 2  # 배치 사이즈 줄임

    opt = Options()
    main = Main(opt)
    main.train(epoch=10)  # 테스트 실행

