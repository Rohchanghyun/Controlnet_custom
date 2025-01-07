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
    StableDiffusionControlNetPipeline
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
import datetime
import traceback

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
    def __init__(self, opt):
        self.opt = opt
        
        # SD 1.5 모델 ID로 변경
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.controlnet_model_id = "lllyasviel/sd-controlnet-openpose"
        
        # 스케줄러 설정
        self.noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler",
        )
        
        torch.cuda.set_device(0)
        
        # VAE를 float32로 로드
        vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32  # float16 대신 float32 사용
        )
        
        # ControlNet 파이프라인 설정
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=ControlNetModel.from_pretrained(
                self.controlnet_model_id,
                torch_dtype=torch.float32  # float16 대신 float32 사용
            ),
            vae=vae,  # 명시적으로 float32 VAE 사용
            scheduler=self.noise_scheduler,
            torch_dtype=torch.float32,  # 전체 파이프라인을 float32로 설정
            safety_checker=None,
            requires_safety_checker=False
        ).to('cuda')
        
        # VAE의 스케일링 팩터 확인
        print(f"VAE scaling factor: {self.pipeline.vae.config.scaling_factor}")
        
        # 결과 디렉토리 설정
        self.result_dir = opt.output_dir
        os.makedirs(self.result_dir, exist_ok=True)

    def train(self, epoch):
        wandb.init(
            project="controlnet-sticker",
            name=f"training_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "learning_rate": 1e-5,
                "architecture": "ControlNet-SD1.5",
                "dataset": "sticker-pose",
                "epochs": epoch,
            }
        )

        # 메이터 로더 초기화
        self.data = Data(self.opt)
        
        # 메모리 절약을 위한 설정
        torch.cuda.empty_cache()
        
        # 모든 모델 컴포넌트를 GPU로 이동
        self.pipeline.text_encoder.to("cuda")
        self.pipeline.vae.to("cuda")
        self.pipeline.unet.to("cuda")
        self.pipeline.controlnet.to("cuda")
        
        # gradient checkpointing 활성화
        self.pipeline.controlnet.enable_gradient_checkpointing()
        
        # 학습 모드로 설정
        self.pipeline.controlnet.train()
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        
        optimizer = torch.optim.AdamW(
            self.pipeline.controlnet.parameters(), 
            lr=5e-6,  # 1e-5에서 낮춤
            weight_decay=1e-2  # 가중치 정규화 추가
        )
        
        # 학습률 스케줄러 추가
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=50,
            min_lr=1e-6
        )
        
        # 학습을 위한 스케줄러 설정
        self.noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler",
            num_train_timesteps=1000
        )
        self.noise_scheduler.set_timesteps(1000)
        
        # 이미지 저장 디렉토리 생성
        image_save_dir = os.path.join(self.result_dir, 'generated_images')
        os.makedirs(image_save_dir, exist_ok=True)
        
        for step, batch in enumerate(tqdm(self.data.sticker_pose_loader)):
            with torch.amp.autocast('cuda'):
                pose_images = batch["pose_images"].to("cuda", dtype=torch.float16)
                target_images = batch["target_images"].to("cuda", dtype=torch.float16)
                prompt = batch["prompt"]
                
                # 수정된 prompt encoding 방식
                text_encoder_output = self.pipeline.text_encoder(
                    self.pipeline.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.pipeline.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to("cuda")
                )
                prompt_embeds = text_encoder_output[0]
                
                # VAE encoding
                target_latents = self.pipeline.vae.encode(target_images).latent_dist.sample()
                target_latents = target_latents * self.pipeline.vae.config.scaling_factor
                
                # 노이즈 추가 (수정된 부분)
                noise = torch.randn_like(target_latents)
                # 타임스텝 선택 수정
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=target_latents.device
                ).long()
                
                # 노이즈 추가
                noisy_latents = self.noise_scheduler.add_noise(
                    target_latents,
                    noise,
                    timesteps
                )
                
                # ControlNet 추론
                down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=pose_images,
                    return_dict=False
                )
                
                # UNet 추론
                noise_pred = self.pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # 손실 계산
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
            # Gradient Clipping 추가
            torch.nn.utils.clip_grad_norm_(self.pipeline.controlnet.parameters(), max_norm=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 매 100 스텝마다 이미지 생성 및 로깅
            if step % 100 == 0:
                try:
                    self.pipeline.controlnet.eval()
                    
                    with torch.no_grad():
                        test_pose = pose_images[0:1].clone().float()
                        test_prompt = prompt[0] if isinstance(prompt, list) else prompt
                        
                        # 생성 시 더 많은 inference steps 사용
                        generated_images = self.pipeline(
                            prompt=test_prompt,
                            image=test_pose,
                            num_inference_steps=50,  # 30에서 증가
                            guidance_scale=7.5,
                            negative_prompt="low quality, worst quality, bad anatomy",  # 네거티브 프롬프트 추가
                        ).images[0]
                        
                        # wandb 로깅에 타겟 이미지 추가
                        target_img = target_images[0].cpu().float()
                        target_img = (target_img * 0.5 + 0.5).clamp(0, 1)  # denormalize
                        target_img = target_img.numpy().transpose(1, 2, 0)
                        target_img = (target_img * 255).astype(np.uint8)
                        
                        wandb.log({
                            "step": step,
                            "loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]['lr'],  # 현재 학습률 로깅
                            "generated_image": wandb.Image(generated_images, caption=f"Step {step}: {test_prompt}"),
                            "pose_image": wandb.Image(pose_images, caption="Input Pose"),
                            "target_image": wandb.Image(target_img, caption="Target Image"),
                        })
                    
                    self.pipeline.controlnet.train()
                    
                    # 스케줄러 업데이트
                    scheduler.step(loss)
                    
                except Exception as e:
                    print(f"Error in generation: {str(e)}")
                    print(traceback.format_exc())
                    continue
            
            # 기본 loss 로깅 (매 스텝)
            wandb.log({"training_loss": loss.item()})
            
            # 더 자주 캐시 비우기
            if step % 5 == 0:  # 10에서 5로 변경
                torch.cuda.empty_cache()
            
            # 로깅 및 체크포인트 저장 로직...

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

        prompt_tokens2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
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

        negative_tokens2 = self.tokenizer_2(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        # Get prompt embeddings from both text encoders
        prompt_embeds = self.text_encoder(prompt_tokens, return_dict=False)[0]
        prompt_embeds_2 = self.text_encoder_2(prompt_tokens2, return_dict=False)[0]

        # Get negative embeddings from both text encoders
        negative_embeds = self.text_encoder(negative_tokens, return_dict=False)[0]
        negative_embeds_2 = self.text_encoder_2(negative_tokens2, return_dict=False)[0]

        # Concatenate the embeddings from both text encoders
        prompt_embeds = torch.concat([prompt_embeds, prompt_embeds_2], dim=-1)
        negative_embeds = torch.concat([negative_embeds, negative_embeds_2], dim=-1)

        # Concatenate negative and positive embeddings
        prompt_embeds = torch.cat([negative_embeds, prompt_embeds])

        # Return without pooled embeddings
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
            self.data_path = "/workspace/data/changhyun/dataset/character_emoticon_data"  # 실제 데이터 경로로 수정
            self.output_dir = "/workspace/data/changhyun/projects/emoji_generation/output"  # 출력 경로
            self.batchsize = 4

    opt = Options()
    main = Main(opt)
    main.train(epoch=10)  # 테스트 실행

