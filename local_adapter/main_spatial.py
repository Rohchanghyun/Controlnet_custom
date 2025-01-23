import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler
)
from data import Data
import wandb
import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image

class Main():
    def __init__(self, opt):
        self.opt = opt
        
        # SDXL 모델 ID
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.controlnet_model_id = "diffusers/controlnet-canny-sdxl-1.0"
        
        # 스케줄러 설정
        self.noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler"
        )
        
        # VAE 로드 (float32)
        vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32
        )
        
        # SDXL ControlNet 파이프라인 설정
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=ControlNetModel.from_pretrained(
                self.controlnet_model_id,
                torch_dtype=torch.float32
            ),
            vae=vae,
            scheduler=self.noise_scheduler,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to('cuda')
        
        # 결과 디렉토리 설정
        self.result_dir = opt.output_dir
        os.makedirs(self.result_dir, exist_ok=True)

    def train(self, epoch):
        wandb.init(
            project="controlnet-sticker-sdxl",
            name=f"sdxl_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "learning_rate": 1e-5,
                "architecture": "ControlNet-SDXL",
                "dataset": "sticker-pose",
                "epochs": epoch,
            }
        )

        # 데이터 로더 초기화
        self.data = Data(self.opt)
        
        # 메모리 관리
        torch.cuda.empty_cache()
        
        # 모델 컴포넌트를 GPU로 이동
        self.pipeline.text_encoder.to("cuda")
        self.pipeline.text_encoder_2.to("cuda")  # SDXL은 두 개의 텍스트 인코더 사용
        self.pipeline.vae.to("cuda")
        self.pipeline.unet.to("cuda")
        self.pipeline.controlnet.to("cuda")
        
        # gradient checkpointing 활성화
        self.pipeline.controlnet.enable_gradient_checkpointing()
        
        # 학습 모드 설정
        self.pipeline.controlnet.train()
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(
            self.pipeline.controlnet.parameters(),
            lr=1e-5,
            weight_decay=1e-2
        )
        
        # 학습률 스케줄러
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            min_lr=1e-6
        )

        # 학습을 위한 스케줄러 설정
        self.noise_scheduler.set_timesteps(1000)
        
        # 이미지 저장 디렉토리 생성
        image_save_dir = os.path.join(self.result_dir, 'generated_images')
        os.makedirs(image_save_dir, exist_ok=True)
        
        for step, batch in enumerate(tqdm(self.data.sticker_pose_loader)):
            with torch.amp.autocast('cuda'):
                pose_images = batch["pose_images"].to("cuda")
                target_images = batch["target_images"].to("cuda")
                prompt = batch["prompt"]
                
                # SDXL의 두 텍스트 인코더를 위한 임베딩 생성
                prompt_embeds = self.pipeline._encode_prompt(
                    prompt,
                    device="cuda",
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=None,
                )
                
                # VAE encoding
                target_latents = self.pipeline.vae.encode(target_images).latent_dist.sample()
                target_latents = target_latents * self.pipeline.vae.config.scaling_factor
                
                # 노이즈 추가
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=target_latents.device
                ).long()
                
                noisy_latents = self.noise_scheduler.add_noise(
                    target_latents,
                    noise,
                    timesteps
                )
                
                # ControlNet과 UNet 추론
                down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=pose_images,
                    return_dict=False
                )
                
                noise_pred = self.pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # 손실 계산
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.pipeline.controlnet.parameters(), max_norm=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 매 100 스텝마다 이미지 생성 및 로깅
            if step % 100 == 0:
                try:
                    self.pipeline.controlnet.eval()
                    
                    with torch.no_grad():
                        test_pose = pose_images[0:1].clone()
                        test_prompt = prompt[0] if isinstance(prompt, list) else prompt
                        
                        # 이미지 생성
                        generated_images = self.pipeline(
                            prompt=test_prompt,
                            image=test_pose,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            negative_prompt="low quality, worst quality, bad anatomy",
                        ).images[0]
                        
                        # wandb 로깅
                        target_img = target_images[0].cpu()
                        target_img = (target_img * 0.5 + 0.5).clamp(0, 1)
                        target_img = target_img.numpy().transpose(1, 2, 0)
                        target_img = (target_img * 255).astype(np.uint8)
                        
                        wandb.log({
                            "step": step,
                            "loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "generated_image": wandb.Image(generated_images, caption=f"Step {step}: {test_prompt}"),
                            "pose_image": wandb.Image(test_pose.cpu(), caption="Input Pose"),
                            "target_image": wandb.Image(target_img, caption="Target Image"),
                        })
                        
                        # 이미지 저장
                        save_path = os.path.join(image_save_dir, f"step_{step}.png")
                        generated_images.save(save_path)
                    
                    self.pipeline.controlnet.train()
                    
                except Exception as e:
                    print(f"Error in generation: {str(e)}")
                    continue
            
            # 학습률 스케줄러 업데이트
            scheduler.step(loss)
            
            # 기본 loss 로깅
            wandb.log({"training_loss": loss.item()})
            
            # 메모리 관리
            if step % 5 == 0:
                torch.cuda.empty_cache()
            
            # 체크포인트 저장 (매 1000 스텝마다)
            if step % 1000 == 0:
                checkpoint_dir = os.path.join(self.result_dir, f"checkpoint_{step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.pipeline.controlnet.save_pretrained(checkpoint_dir)
