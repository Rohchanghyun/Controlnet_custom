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

import random

import wandb

import torch.nn as nn
import itertools

# LoRA를 적용한 Custom Attention Processor
class LoRAAttnProcessor(nn.Module):
    """LoRA attention processor for self-attention"""
    def __init__(self, hidden_size, cross_attention_dim=None, rank=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        
        # LoRA layers for self attention
        self.lora_k = nn.Linear(hidden_size, rank, bias=False)
        self.lora_v = nn.Linear(hidden_size, rank, bias=False)
        self.lora_k_proj = nn.Linear(rank, hidden_size, bias=False)
        self.lora_v_proj = nn.Linear(rank, hidden_size, bias=False)
            
    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        # LoRA 적용
        key = key + self.lora_k_proj(self.lora_k(key))
        value = value + self.lora_v_proj(self.lora_v(value))

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
    def __init__(self, hidden_size, cross_attention_dim, rank=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        
        # Cross attention projection
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        
        # LoRA layers for cross attention
        self.lora_k = nn.Linear(hidden_size, rank, bias=False)
        self.lora_v = nn.Linear(hidden_size, rank, bias=False)
        self.lora_k_proj = nn.Linear(rank, hidden_size, bias=False)
        self.lora_v_proj = nn.Linear(rank, hidden_size, bias=False)
            
    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        
        # Cross attention projection
        key = self.to_k_ip(encoder_hidden_states)
        value = self.to_v_ip(encoder_hidden_states)
        
        # LoRA 적용
        key = key + self.lora_k_proj(self.lora_k(key))
        value = value + self.lora_v_proj(self.lora_v(value))

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
    def __init__(self, extractor, image_adapter):
        self.data = Data()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader
        self.testset = self.data.testset

        self.image_adapter = image_adapter
        self.extractor = extractor

        if hasattr(opt, 'extractor_weight') and opt.extractor_weight:
            print(f"Loading pretrained weights from {opt.extractor_weight}")
            self.extractor.load_state_dict(torch.load(opt.extractor_weight))

        self.model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler"
        )
        
        torch.cuda.set_device(0)
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=self.noise_scheduler,
            torch_dtype=torch.float32
        ).to('cuda')
        
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.noise_scheduler = self.pipeline.scheduler
        
        del self.pipeline
        torch.cuda.empty_cache()
        
        self.vae.eval()
        
        # UNet의 attention processor 수정
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            
            print(f"Processing {name}:")
            print(f"cross_attention_dim: {cross_attention_dim}")
            print(f"hidden_size: {hidden_size}")
            
            # device 설정 부분 수정
            device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES가 설정되어 있으므로 항상 0번을 사용
            
            # processor 초기화 부분
            if name.endswith("attn1.processor"):
                processor = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=None,
                    rank=128
                ).to(device)
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                processor = LoRAIPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=128
                ).to(device)
                processor.load_state_dict(weights, strict=False)
            
            attn_procs[name] = processor
            
        self.unet.set_attn_processor(attn_procs)
        
        # LoRA 파라미터 수집
        self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
        
        # Optimizer에 LoRA 파라미터도 포함
        params_to_optimize = [
            {'params': self.image_adapter.parameters()},
            {'params': self.adapter_modules.parameters()}
        ]
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )
        
        # 전체 학습 스텝 수 계산
        total_steps = len(self.train_loader) * opt.epoch
        
        # Learning rate scheduler를 스텝 기준으로 수정
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,  # 전체 스텝 수 기준
            eta_min=1e-6
        )
        
        self.result_dir = opt.output_dir
        os.makedirs(self.result_dir, exist_ok=True)
        
    def train(self, epoch):
        # 결과 저장할 디렉토리 생성
        os.makedirs(f'{opt.output_dir}/results', exist_ok=True)
        
        self.unet.eval()
        self.image_adapter.train()
        total_loss = 0
        
        total_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f'Batch 0/{total_batches}', ncols=100)
        
        for batch, (inputs, labels, captions) in enumerate(pbar):
            pbar.set_description(f'Batch {batch+1}/{total_batches}')
            inputs = inputs.to('cuda')
            images = inputs
            labels = labels.to('cuda')
            
            # VAE 인코딩 - 96x96 latent 생성
            with torch.no_grad():
                latents = self.vae.encode(images).latent_dist.sample()  # [batch_size, 4, 96, 96]
                latents = latents * self.vae.config.scaling_factor
            
            # 노이즈 생성 - 96x96 크기로 맞춤
            noise = torch.randn_like(latents)  # [batch_size, 4, 96, 96]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (images.shape[0],), device='cuda'
            ).long()
            
            # 노이즈 추가
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)  # [batch_size, 4, 96, 96]
            
            outputs = self.extractor(inputs)
            embedding_feature = outputs[0]
            projected_image_embedding = self.image_adapter(embedding_feature)
            
            # 임베딩 차원의 10%를 랜덤하게 0으로 만듦
            batch_size, seq_length, hidden_dim = projected_image_embedding.shape
            mask_size = int(hidden_dim * 0.1)  # 10% 계산
            
            for b in range(batch_size):
                for s in range(seq_length):
                    # 각 시퀀스마다 랜덤하게 인덱스 선택
                    zero_indices = torch.randperm(hidden_dim)[:mask_size]
                    projected_image_embedding[b, s, zero_indices] = 0
            
            with torch.no_grad():
                text_input_ids = self.tokenizer(
                    captions,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to('cuda')
                
                text_embeddings = self.text_encoder(text_input_ids)[0]
            
            conditions = torch.cat([text_embeddings, projected_image_embedding], dim=1) # concat 하는 순서 중요
            
            # UNet으로 노이즈 예측
            noise_pred = self.unet(
                noisy_latents,  # [batch_size, 4, 96, 96]
                timesteps,
                encoder_hidden_states=conditions
            ).sample
            
            loss = F.mse_loss(noise_pred, noise)
                
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            self.lr_scheduler.step()
            
            total_loss += loss.item()
            
            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch
            })
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
                        # 매 1000 배치마다 이미지 생성  저장
            if batch % 1000 == 0:
                self.extractor.eval()
                self.image_adapter.eval()
                
                # 테스트용 이미지와 프롬프트
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
                    
                    text_embeddings = self.text_encoder(text_input_ids)[0]
                    conditions = torch.cat([text_embeddings, projected_image_embedding], dim=1)
                    
                    # Unconditional 임베딩 생성
                    uncond_text_input = self.tokenizer(
                        "",
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to("cuda")
                    
                    uncond_feature = torch.zeros((projected_image_embedding.shape[0], 2048)).to('cuda')
                    uncond_image_input = self.image_adapter(uncond_feature)
                    
                    uncond_embeddings = self.text_encoder(uncond_text_input.input_ids)[0]
                    uncond_embeddings = uncond_embeddings.repeat(projected_image_embedding.shape[0], 1, 1)
                    uncond_conditions = torch.cat([uncond_embeddings, uncond_image_input], dim=1)
                    
                    # 노이즈에서 이미지 생성
                    latents = torch.randn((1, 4, 64, 64), device='cuda')
                    
                    # EulerAncestral 스케줄러 설정
                    euler_scheduler = EulerAncestralDiscreteScheduler.from_config(self.noise_scheduler.config)
                    euler_scheduler.set_timesteps(30)
                    timesteps = euler_scheduler.timesteps
                    guidance_scale = 7.5
                    
                    for t in timesteps:
                        latent_model_input = euler_scheduler.scale_model_input(latents, t)
                        
                        # Unconditional 예측
                        noise_pred_uncond = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=uncond_conditions
                        ).sample
                        
                        # Conditional 예측
                        noise_pred_text = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=conditions
                        ).sample
                        
                        # CFG 적용
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                        # 스케줄러 스텝
                        latents = euler_scheduler.step(
                            model_output=noise_pred,
                            timestep=t,
                            sample=latents
                        ).prev_sample
                
                # VAE 디코딩
                with torch.no_grad():
                    latents = 1 / self.vae.config.scaling_factor * latents
                    image = self.vae.decode(latents).sample
                
                # 미지 후처리 및 저장
                image = (image / 2 + 0.5).clamp(0, 1)
                image = (image * 255).clamp(0, 255).round()
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = image.astype(np.uint8)[0]
                
                result_image = Image.fromarray(image)
                result_image.save(f'{opt.output_dir}/results/generated_image_batch_{batch}.png')
                
                # wandb 로깅
                original_image = sample_image.cpu().squeeze(0).permute(1, 2, 0)
                original_image = ((original_image * 0.5 + 0.5) * 255).clamp(0, 255).round().numpy().astype(np.uint8)
                original_image = Image.fromarray(original_image)
                
                wandb.log({
                    "original_image": wandb.Image(original_image, caption=f"Original - {sample_caption}"),
                    "generated_image": wandb.Image(result_image, caption=f"Generated - {sample_caption}"),
                    "caption": sample_caption,
                    "batch": batch
                })
                
                # 가중치 저장
                image_adapter_path = f'{opt.output_dir}/image_adapter/image_adapter_epoch_{epoch}_batch_{batch}.pt'
                adapter_modules_path = f'{opt.output_dir}/adapter_modules/adapter_modules_epoch_{epoch}_batch_{batch}.pt'
                
                os.makedirs(os.path.dirname(image_adapter_path), exist_ok=True)
                os.makedirs(os.path.dirname(adapter_modules_path), exist_ok=True)
                
                torch.save(self.image_adapter.state_dict(), image_adapter_path)
                torch.save(self.adapter_modules.state_dict(), adapter_modules_path)
                
                print(f'Image adapter weights saved to {image_adapter_path}')
                print(f'Adapter modules weights saved to {adapter_modules_path}')
                    
                self.image_adapter.train()
                self.adapter_modules.train()
        
        avg_loss = total_loss / len(self.train_loader)
        wandb.log({
            "epoch_avg_loss": avg_loss,
            "epoch": epoch
        })
        print(f"\nEpoch {epoch}/{opt.epoch}, Average Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    # wandb 초기화 전에 GPU 설정 확인
    
    import torch
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Currently allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    extractor = MGN().to('cuda')
    image_adapter = Image_adapter(extractor_dim=2048, hidden_dim=1024, clip_embeddings_dim=opt.clip_embeddings_dim).to('cuda')
    main = Main(extractor, image_adapter)

    if opt.mode == 'train':
        # wandb 초기화
        wandb.init(
            project="Controlnet",
            config=vars(opt),
            name=f"sd_image_adapter_cfg_ip_euler_adapter_zero_embedding_drop_text_first_{opt.clip_embeddings_dim}_{opt.epoch}_{opt.lr}"
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