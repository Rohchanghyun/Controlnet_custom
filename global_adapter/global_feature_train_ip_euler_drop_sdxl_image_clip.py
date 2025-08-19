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
import traceback

import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F

from torchvision import transforms

from diffusers import StableDiffusionXLPipeline, DDPMScheduler, EulerAncestralDiscreteScheduler

from opt import opt
from data import Data
from network import MGN, Concat_ImageAdapter_3Layers
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from sklearn.manifold import TSNE

from PIL import Image

import wandb

import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint

import open_clip

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

        # 모델 ID를 opt에서 가져오기
        self.model_id = opt.model_id if hasattr(opt, 'model_id') and opt.model_id else "stable-diffusion-v1-5"
        
        torch.cuda.set_device(0)
        
        # fp16으로 모델 로드 변경
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to('cuda')

        # 반드시 pipeline의 scheduler config를 사용해서 생성
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler"
        )
        # noise_scheduler 정보 출력
        print("Noise Scheduler 정보:")
        print(f"  클래스: {type(self.noise_scheduler)}")
        if hasattr(self.noise_scheduler, 'config'):
            print(f"  Config: {self.noise_scheduler.config}")
        else:
            print("  config 속성이 없습니다.")
        # SDXL은 두 개의 텍스트 인코더를 사용
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        # self.noise_scheduler = self.pipeline.scheduler  # 이미 위에서 할당함
        
        del self.pipeline
        torch.cuda.empty_cache()
        
        # 그래디언트 체크포인팅 활성화 - 메모리 사용량 크게 줄임
        self.unet.enable_gradient_checkpointing()
        
        # 불필요한 모듈은 CPU로 이동
        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        
        self.vae.eval()
        
        
        # Optimizer에서 LoRA 부분 제거
        params_to_optimize = [
            {'params': self.image_adapter.parameters()}
            # LoRA 파라미터 제거
            # {'params': self.adapter_modules.parameters()}
        ]
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=opt.lr * 0.1,  # 학습률을 1/10로 감소
            weight_decay=opt.weight_decay,
            eps=1e-7,  # epsilon 값을 기본값 1e-8에서 증가
            betas=(0.9, 0.95)  # beta2 값을 0.999에서 0.95로 감소하여 학습 안정화
        )
        
        # 전체 학습 스텝 수 계산
        total_steps = len(self.train_loader) * opt.epoch
        
        # Learning rate scheduler를 워밍업을 포함한 형태로 수정
        num_warmup_steps = int(len(self.train_loader) * 0.1)  # 전체 데이터의 10%를 워밍업에 사용

        from torch.optim.lr_scheduler import LambdaLR

        def get_lr_schedule_with_warmup(optimizer, num_warmup_steps, total_steps):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    # 워밍업 기간 동안 선형적으로 증가
                    return float(current_step) / float(max(1, num_warmup_steps))
                else:
                    # 워밍업 이후 코사인 감소
                    progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            return LambdaLR(optimizer, lr_lambda)

        # 필요한 math 모듈 import 추가
        import math

        # 기존 스케줄러 대신 워밍업이 포함된 스케줄러 사용
        self.lr_scheduler = get_lr_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            total_steps=total_steps
        )
        
        self.result_dir = opt.output_dir
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.clip_model = self.clip_model.to("cuda").eval()
        
        # fp16 학습을 위한 GradScaler 추가
        self.scaler = GradScaler(
            init_scale=2**8,          # 초기 스케일을 2**10에서 2**8로 감소
            growth_factor=1.5,        # 스케일 증가 비율 감소
            backoff_factor=0.5,       # 스케일 감소 비율 유지
            growth_interval=200,      # 스케일 증가 간격 증가
            enabled=True
        )
        
    def train(self, epoch):
        # 결과 저장할 디렉토리 생성
        os.makedirs(f'{opt.output_dir}/results', exist_ok=True)
        
        self.unet.eval()
        self.image_adapter.train()
        total_loss = 0
        
        total_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f'Batch 0/{total_batches}', ncols=100)
        
        # 배치 크기 확인 및 필요시 조정
        batch_size = next(iter(self.train_loader))[0].shape[0]

        
        for batch, (inputs, labels, captions) in enumerate(pbar):
            pbar.set_description(f'Batch {batch+1}/{total_batches}')
            inputs = inputs.to('cuda')
            images = inputs
            labels = labels.to('cuda')
            self.extractor = self.extractor.to('cuda')
            self.image_adapter = self.image_adapter.to('cuda')
            # 이미지 정규화 및 fp16 변환
            with torch.no_grad():
                # VAE 인코딩에는 fp32 사용 (fp16 대신 full precision 사용)
                self.vae.to('cuda')
                
                # 이미지 값 범위 확인
                print("Original image range:", images.min().item(), images.max().item())
                
                # [-1, 1] 범위로 정규화 (만약 이미 정규화되어 있지 않다면)
                if images.min() < -1.1 or images.max() > 1.1:
                    print("이미지를 [-1, 1] 범위로 정규화합니다")
                    images = torch.clamp(images, -1.0, 1.0)
                
                # VAE 인코딩을 위해 fp32로 유지 (fp16으로 변환하지 않음)
                images_fp32 = images.to(dtype=torch.float32)
                print("Images shape and dtype:", images_fp32.shape, images_fp32.dtype)
                print("Images min/max:", images_fp32.min().item(), images_fp32.max().item())
                
                # VAE도 fp32로 변환
                self.vae = self.vae.to(dtype=torch.float32)
                
                # fp32에서 VAE 인코딩 수행
                latents = self.vae.encode(images_fp32).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
                # 결과 출력 및 NaN 검사
                print(f"latents shape: {latents.shape}")
                print(f"latents min/max: {latents.min().item()}/{latents.max().item()}")
                print(f"latents mean: {latents.mean().item()}, std: {latents.std().item()}")
                
                # VAE 인코딩 후 latent를 fp16으로 변환
                latents = latents.to(dtype=torch.float16)
                
                # latents 값 범위 확인 및 제한
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print("경고: VAE 출력에 NaN 또는 Inf가 있습니다!")
                    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
                
                self.vae.to('cpu')  # 인코딩 후 CPU로 이동
                
                # CLIP 이미지 특징 추출
                normalized_images = (images + 1) / 2.0  # [-1, 1] -> [0, 1] 범위로 변환
                images_np = normalized_images.cpu().permute(0, 2, 3, 1).numpy()
                clip_images = []
                for img_np in images_np:
                    # 넘파이 배열을 PIL 이미지로 변환
                    pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
                    # CLIP 전처리 적용
                    clip_img = self.clip_transform(pil_img)
                    clip_images.append(clip_img)
                clip_images = torch.stack(clip_images).to('cuda')
                clip_image_features = self.clip_model.encode_image(clip_images)

                
                # === CLIP 이미지 feature 통계 wandb 기록 ===
                clip_img_feat = clip_image_features.detach().cpu().float()
                clip_img_feat_min = float(clip_img_feat.min())
                clip_img_feat_max = float(clip_img_feat.max())
                clip_img_feat_mean = float(clip_img_feat.mean())
                clip_img_feat_std = float(clip_img_feat.std())

                wandb.log({
                    "clip_img_feat/min": clip_img_feat_min,
                    "clip_img_feat/max": clip_img_feat_max,
                    "clip_img_feat/mean": clip_img_feat_mean,
                    "clip_img_feat/std": clip_img_feat_std,
                })
                torch.cuda.empty_cache()
            
            # 노이즈 생성 - ControlNeXt 스타일로 변경
            noise = torch.randn_like(latents, device=latents.device)
            
            # 타임스텝 생성 - 명시적으로 dtype 지정
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (images.shape[0],), device=latents.device
            ).long()
            
            # 노이즈 추가 - 예외 처리 추가
            noisy_latents_list = []
            for i in range(latents.shape[0]):
                try:
                    print(f"noisey_latents shape: {latents[i:i+1].shape}, noise shape: {noise[i:i+1].shape}, timesteps shape: {timesteps[i:i+1].shape}")
                    noisy_latent = self.noise_scheduler.add_noise(
                        latents[i:i+1], noise[i:i+1], timesteps[i:i+1]
                    )
                except Exception as e:
                    print(f"노이즈 추가 중 오류 (i={i}): {e}")
                    traceback.print_exc()  # 전체 에러 트레이스백 출력
                    noisy_latent = latents[i:i+1] + noise[i:i+1]
                noisy_latents_list.append(noisy_latent)
            noisy_latents = torch.cat(noisy_latents_list, dim=0)
            
            # 결과 안정성 확인
            if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                print("경고: noisy_latents에 NaN 또는 Inf가 있습니다!")
                noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0)
            
            outputs = self.extractor(inputs)
            embedding_feature = outputs[0] # [2,2048]
            
            
            projected_image_embedding = self.image_adapter(embedding_feature, clip_image_features)
            
            
            # 임베딩 차원의 10%를 랜덤하게 0으로 만듦
            batch_size, seq_length, hidden_dim = projected_image_embedding.shape
            mask_size = int(hidden_dim * 0.1)  # 10% 계산
            
            for b in range(batch_size):
                for s in range(seq_length):
                    zero_indices = torch.randperm(hidden_dim)[:mask_size]
                    projected_image_embedding[b, s, zero_indices] = 0
            
            # SDXL은 두 개의 텍스트 인코더를 사용하므로 프롬프트 인코딩 방식을 변경
            with torch.no_grad():
                # 필요할 때만 텍스트 인코더를 GPU로 이동
                self.text_encoder.to('cuda')
                self.text_encoder_2.to('cuda')
                
                # fp16으로 명시적 변환
                self.text_encoder = self.text_encoder.to(dtype=torch.float16)
                self.text_encoder_2 = self.text_encoder_2.to(dtype=torch.float16)
                
                # 첫 번째 텍스트 인코더
                text_input_ids = self.tokenizer(
                    captions,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to('cuda')
                
                # 두 번째 텍스트 인코더
                text_input_ids_2 = self.tokenizer_2(
                    captions,
                    padding="max_length",
                    max_length=self.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to('cuda')
                
                # SDXL 추가 조건부 정보 (원본 이미지 해상도, 크롭 정보 등)
                original_size = (768, 768)  # 원본 이미지 크기 (가정)
                target_size = (1024, 1024)    # 타겟 이미지 크기 (가정)
                crops_coords_top_left = (0, 0)  # 크롭 좌표
                
                # 첫 번째 텍스트 인코더 임베딩
                prompt_embeds = self.text_encoder(
                    text_input_ids,
                    output_hidden_states=True,
                ).hidden_states[-2]
                
                # 두 번째 텍스트 인코더 임베딩
                prompt_embeds_2 = self.text_encoder_2(
                    text_input_ids_2,
                    output_hidden_states=True,
                ).hidden_states[-2]
                
                # 임베딩 정보 출력하여 크기 확인
                # print(f"prompt_embeds shape: {prompt_embeds.shape}")
                # print(f"prompt_embeds_2 shape: {prompt_embeds_2.shape}")
                # print(f"projected_image_embedding shape: {projected_image_embedding.shape}")
                
                # 두 임베딩 결합
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
                # print(f"Combined prompt_embeds shape: {prompt_embeds.shape}")
                
                # 추가 조건부 정보 인코딩
                add_text_embeds = self.text_encoder_2(
                    text_input_ids_2,
                    output_hidden_states=True,
                ).hidden_states[-1]
                
                # 중요: 첫 번째 토큰(CLS)만 사용하여 차원 조정
                if len(add_text_embeds.shape) == 3:
                    add_text_embeds = add_text_embeds[:, 0, :]
                
                # 추가 시간 임베딩을 위한 정보
                add_time_ids = torch.tensor(
                    [list(original_size) + list(crops_coords_top_left) + list(target_size)],
                    dtype=torch.float32
                ).to('cuda').repeat(batch_size, 1)
                
                # 사용 후 CPU로 이동
                self.text_encoder.to('cpu')
                self.text_encoder_2.to('cpu')
                torch.cuda.empty_cache()
            
            
            conditions = torch.cat([prompt_embeds, projected_image_embedding], dim=1)
            
            # 입력 검증 추가
            if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                print("경고: noisy_latents에 NaN 또는 Inf가 있습니다!")
                noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0)

            if torch.isnan(conditions).any() or torch.isinf(conditions).any():
                print("경고: conditions에 NaN 또는 Inf가 있습니다!")
                conditions = torch.nan_to_num(conditions, nan=0.0, posinf=1.0, neginf=-1.0)
            

            # 모든 입력 텐서를 명시적으로 float16으로 변환 (dtype 일치 확보)
            conditions = conditions.to(dtype=torch.float16)
            noisy_latents = noisy_latents.to(dtype=torch.float16)
            add_text_embeds = add_text_embeds.to(dtype=torch.float16)
            add_time_ids = add_time_ids.to(dtype=torch.float16)
            
            # UNet도 확실하게 float16으로 설정
            self.unet = self.unet.to(dtype=torch.float16)
            
            # UNet에 추가 조건부 정보 전달
            # UNet에 입력하기 직전에 모두 같은 device로 맞추기
            device = latents.device
            conditions = conditions.to(device)
            noisy_latents = noisy_latents.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device)
            timesteps = timesteps.to(device)

            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=conditions,
                added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            ).sample
            
            # 노이즈 예측 후 NaN 체크 강화
            if torch.isnan(noise_pred).any():
                print("경고: noise_pred에 NaN이 있습니다!")
                # NaN 디버깅을 위한 중간값 출력
                print(f"Noisy latents range: {noisy_latents.min().item()} to {noisy_latents.max().item()}")
                print(f"Timesteps: {timesteps}")
                print(f"Conditions shape: {conditions.shape}")
                print(f"Conditions min/max: {conditions.min().item()}/{conditions.max().item()}")
                print(f"Projected image embedding min/max: {projected_image_embedding.min().item()}/{projected_image_embedding.max().item()}")
                
                # NaN을 0으로 대체
                noise_pred = torch.where(torch.isnan(noise_pred), torch.zeros_like(noise_pred), noise_pred)
            
            # 손실 계산을 fp32로 변경
            with autocast(dtype=torch.float16, enabled=False):  # fp32로 계산
                loss = F.mse_loss(noise_pred, noise)
            
            # NaN 손실 체크 및 처리
            if torch.isnan(loss).any():
                print("경고: 손실이 NaN입니다! 이번 배치는 건너뜁니다.")
                loss = torch.tensor(0.0, device='cuda', requires_grad=True)
        
            # fp16 스케일러를 사용한 역전파 및 옵티마이저 스텝
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 그래디언트 클리핑 값 감소
            torch.nn.utils.clip_grad_norm_(self.image_adapter.parameters(), max_norm=0.5)  # 1.0에서 0.5로 감소

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.lr_scheduler.step()
            
            total_loss += loss.item()
            
            
            # image adapter 결과 feature 통계 기록
            img_feat = projected_image_embedding.detach().cpu().float()
            img_feat_min = float(img_feat.min())
            img_feat_max = float(img_feat.max())
            img_feat_mean = float(img_feat.mean())
            img_feat_std = float(img_feat.std())

            # 텍스트 condition feature 통계 기록
            txt_feat = prompt_embeds.detach().cpu().float()
            txt_feat_min = float(txt_feat.min())
            txt_feat_max = float(txt_feat.max())
            txt_feat_mean = float(txt_feat.mean())
            txt_feat_std = float(txt_feat.std())

            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch,
                "img_feat/min": img_feat_min,
                "img_feat/max": img_feat_max,
                "img_feat/mean": img_feat_mean,
                "img_feat/std": img_feat_std,
                "txt_feat/min": txt_feat_min,
                "txt_feat/max": txt_feat_max,
                "txt_feat/mean": txt_feat_mean,
                "txt_feat/std": txt_feat_std,
            })

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # --- 이미지 어댑터 및 텍스트 임베딩 통계 wandb 로깅 ---
            def log_tensor_stats_wandb(name, tensor, step):
                t = tensor.detach().float().cpu().flatten().numpy()
                # 분위수로 적절한 범위 계산 (1%~99%)
                q_low, q_high = np.quantile(t, [0.01, 0.99])
                # 범위 제한해서 시각화
                plt.figure(figsize=(6, 4))
                plt.hist(t, bins=64, color='royalblue', alpha=0.8, range=(q_low, q_high))
                plt.title(f"{name} Histogram (batch {step})")
                plt.xlabel(f"Value (clipped: {q_low:.2f} ~ {q_high:.2f})")
                plt.ylabel("Frequency")
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                wandb.log({f"{name}/hist_image": wandb.Image(img)})
                plt.close()

            # projected_image_embedding, prompt_embeds 통계 로깅 (학습 배치마다)
            log_tensor_stats_wandb("train/projected_image_embedding", projected_image_embedding, batch)
            log_tensor_stats_wandb("train/prompt_embeds", prompt_embeds, batch)

            # 매 1000 배치마다 validation
            if batch % 1000 == 0:
                # validation 시작 전
                self.extractor.eval()
                self.image_adapter.eval()
                self.text_encoder.eval()
                self.text_encoder_2.eval()
                self.unet.eval()
                self.vae.eval()

                import gc
                def cleanup_memory():
                    torch.cuda.empty_cache()
                    gc.collect()

                # === opt.py에서 validation 인자 사용 ===
                val_image_path = opt.validation_image_path
                val_caption = opt.validation_caption
                num_samples = opt.validation_num_samples
                guidance_scale_1 = opt.guidance_scale_1
                guidance_scale_2 = opt.guidance_scale_2
                inference_height = opt.inference_height
                inference_width = opt.inference_width
                num_inference_steps = opt.num_inference_steps

                # step별 폴더 생성
                step_dir = os.path.join(self.result_dir, f'validation_results/step_{batch}')
                os.makedirs(step_dir, exist_ok=True)

                # 이미지 로드 및 전처리
                val_image = Image.open(val_image_path).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((inference_height, inference_width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                sample_image = transform(val_image).unsqueeze(0).to('cuda')
                sample_caption = val_caption

                # 이미지 특징 추출
                with torch.no_grad():
                    outputs = self.extractor(sample_image)
                    embedding_feature = outputs[0].clone()
                    image_adapter = self.image_adapter.to(dtype=torch.float32)
                    embedding_feature = embedding_feature.to(dtype=torch.float32)
                    clip_image = self.clip_transform(val_image).unsqueeze(0).to("cuda")
                    clip_image_features = self.clip_model.encode_image(clip_image)
                    clip_image_features = clip_image_features.detach().clone()
                    clip_image_features = clip_image_features.to(dtype=torch.float32)
                    projected_image_embedding = image_adapter(embedding_feature, clip_image_features)
                    projected_image_embedding = projected_image_embedding.detach().clone()
                    
                    uncond_feature = torch.zeros((projected_image_embedding.shape[0], 2048), dtype=torch.float32).to('cuda')
                    empty_pil = Image.new("RGB", (224, 224), (0, 0, 0))
                    empty_clip_image = self.clip_transform(empty_pil).unsqueeze(0).to("cuda")
                    uncond_clip_feature = self.clip_model.encode_image(empty_clip_image)
                    uncond_clip_feature = uncond_clip_feature.to(dtype=torch.float32)
                    uncond_image_input = image_adapter(uncond_feature, uncond_clip_feature)
                    uncond_image_input = uncond_image_input.detach().clone()
                del outputs, embedding_feature, clip_image, uncond_feature, uncond_clip_feature
                cleanup_memory()

                # 텍스트 인코더 준비
                text_encoder = self.text_encoder.to('cuda', dtype=torch.float16)
                text_encoder_2 = self.text_encoder_2.to('cuda', dtype=torch.float16)
                unet = self.unet
                vae = self.vae

                original_size = (inference_height, inference_width)
                target_size = (inference_height, inference_width)
                crops_coords_top_left = (0, 0)

                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                    text_input_ids = self.tokenizer(
                        sample_caption,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to('cuda')
                    text_input_ids_2 = self.tokenizer_2(
                        sample_caption,
                        padding="max_length",
                        max_length=self.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to('cuda')

                    prompt_embeds_1 = text_encoder(
                        text_input_ids,
                        output_hidden_states=True,
                    ).hidden_states[-2]
                    prompt_embeds_2 = text_encoder_2(
                        text_input_ids_2,
                        output_hidden_states=True,
                    ).hidden_states[-2]
                    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
                    add_text_embeds = text_encoder_2(
                        text_input_ids_2,
                        output_hidden_states=True,
                    ).hidden_states[-1]
                    if len(add_text_embeds.shape) == 3:
                        add_text_embeds = add_text_embeds[:, 0, :]

                    add_time_ids = torch.tensor(
                        [list(original_size) + list(crops_coords_top_left) + list(target_size)],
                        dtype=torch.float32
                    ).to('cuda')

                    # Unconditional 임베딩 생성
                    uncond_text_input = self.tokenizer(
                        "",
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to("cuda")
                    uncond_text_input_2 = self.tokenizer_2(
                        "",
                        padding="max_length",
                        max_length=self.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to("cuda")
                    uncond_embeds_1 = text_encoder(
                        uncond_text_input.input_ids,
                        output_hidden_states=True,
                    ).hidden_states[-2]
                    uncond_embeds_2 = text_encoder_2(
                        uncond_text_input_2.input_ids,
                        output_hidden_states=True,
                    ).hidden_states[-2]
                    uncond_embeds = torch.cat([uncond_embeds_1, uncond_embeds_2], dim=-1)
                    uncond_add_text_embeds = text_encoder_2(
                        uncond_text_input_2.input_ids,
                        output_hidden_states=True,
                    ).hidden_states[-1]
                    if len(uncond_add_text_embeds.shape) == 3:
                        uncond_add_text_embeds = uncond_add_text_embeds[:, 0, :]

                    prompt_embeds = prompt_embeds.detach().clone()
                    add_text_embeds = add_text_embeds.detach().clone()
                    uncond_embeds = uncond_embeds.detach().clone()
                    uncond_add_text_embeds = uncond_add_text_embeds.detach().clone()
                    add_time_ids = add_time_ids.detach().clone()

                text_encoder = text_encoder.to('cpu')
                text_encoder_2 = text_encoder_2.to('cpu')
                del text_encoder, text_encoder_2, text_input_ids, text_input_ids_2
                del uncond_text_input, uncond_text_input_2
                cleanup_memory()

                prompt_embeds = prompt_embeds.to(dtype=torch.float16)
                projected_image_embedding = projected_image_embedding.to(dtype=torch.float16)
                uncond_embeds = uncond_embeds.to(dtype=torch.float16)
                uncond_image_input = uncond_image_input.to(dtype=torch.float16)
                add_text_embeds = add_text_embeds.to(dtype=torch.float16)
                uncond_add_text_embeds = uncond_add_text_embeds.to(dtype=torch.float16)

                # 여러 이미지 생성
                wandb_images = []
                projected_image_embeddings_list = []
                for sample_idx in range(num_samples):
                    full_conditions = torch.cat([prompt_embeds, projected_image_embedding], dim=1)
                    image_conditions = torch.cat([uncond_embeds, projected_image_embedding], dim=1)
                    full_uncond_conditions = torch.cat([uncond_embeds, uncond_image_input], dim=1)

                    latent_height = inference_height // 8
                    latent_width = inference_width // 8
                    latents = torch.randn((1, 4, latent_height, latent_width), device='cuda', dtype=torch.float16) * 0.9

                    self.noise_scheduler.set_timesteps(num_inference_steps)
                    timesteps = self.noise_scheduler.timesteps

                    unet = unet.to(dtype=torch.float16).to('cuda')

                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                        for t in timesteps:
                            latent_model_input = self.noise_scheduler.scale_model_input(latents, t)
                            batch_add_time_ids = add_time_ids
                            if add_time_ids.shape[0] != latent_model_input.shape[0]:
                                batch_add_time_ids = add_time_ids.repeat(latent_model_input.shape[0], 1)

                            noise_pred_uncond_full = unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=full_uncond_conditions,
                                added_cond_kwargs={"text_embeds": uncond_add_text_embeds, "time_ids": batch_add_time_ids}
                            ).sample

                            noise_pred_image = unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=image_conditions,
                                added_cond_kwargs={"text_embeds": uncond_add_text_embeds, "time_ids": batch_add_time_ids}
                            ).sample

                            noise_pred_full = unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=full_conditions,
                                added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": batch_add_time_ids}
                            ).sample

                            noise_pred = noise_pred_uncond_full + guidance_scale_1 * (noise_pred_image - noise_pred_uncond_full) + guidance_scale_2 * (noise_pred_full - noise_pred_image)

                            if torch.isnan(noise_pred).any():
                                print("경고: inference 중 noise_pred에 NaN이 있습니다!")
                                noise_pred = torch.where(torch.isnan(noise_pred), torch.zeros_like(noise_pred), noise_pred)

                            latents = self.noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=t,
                                sample=latents
                            ).prev_sample

                    cleanup_memory()

                    vae = vae.to(dtype=torch.float32).to('cuda')
                    latents_fp32 = latents.to(dtype=torch.float32)
                    latents_fp32 = 1 / vae.config.scaling_factor * latents_fp32
                    del latents
                    cleanup_memory()
                    with torch.no_grad():
                        image = vae.decode(latents_fp32).sample
                    del latents_fp32
                    cleanup_memory()

                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = (image * 255).clamp(0, 255).round()
                    image = image.cpu().permute(0, 2, 3, 1).numpy()
                    image = image.astype(np.uint8)[0]

                    result_image = Image.fromarray(image)
                    output_path = os.path.join(step_dir, f'generated_sample_{sample_idx}.png')
                    result_image.save(output_path)
                    print(f"이미지 저장 완료: {output_path}")

                    wandb_images.append(wandb.Image(result_image, caption=f"Generated {sample_idx+1}: {sample_caption}"))

                    # adapter weight 저장 (원하면 한 번만 저장)
                    if sample_idx == 0:
                        adapter_weight_path = os.path.join(step_dir, f"adapter_step_{batch}.pth")
                        torch.save(self.image_adapter.state_dict(), adapter_weight_path)
                        print(f"Adapter weight 저장 완료: {adapter_weight_path}")

                    # projected_image_embedding 저장
                    projected_image_embeddings_list.append(projected_image_embedding.detach().cpu().numpy())

                # --- wandb에 adapter embedding 통계 및 분포 시각화 ---
                projected_image_embeddings_np = np.concatenate(projected_image_embeddings_list, axis=0)
                wandb.log({
                    "validation/projected_image_embedding_min": float(projected_image_embeddings_np.min()),
                    "validation/projected_image_embedding_max": float(projected_image_embeddings_np.max()),
                    "validation/projected_image_embedding_mean": float(projected_image_embeddings_np.mean()),
                    "validation/projected_image_embedding_std": float(projected_image_embeddings_np.std()),
                    "validation/batch": batch
                })

                # 히스토그램 wandb로 로깅
                wandb.log({
                    "validation/projected_image_embedding_hist": wandb.Histogram(projected_image_embeddings_np)
                })

                # t-SNE 시각화 (2D)
                try:
                    tsne = TSNE(n_components=2, random_state=42)
                    tsne_result = tsne.fit_transform(projected_image_embeddings_np.reshape(projected_image_embeddings_np.shape[0], -1))
                    plt.figure(figsize=(6, 6))
                    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
                    plt.title("projected_image_embedding t-SNE")
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    wandb.log({"validation/projected_image_embedding_tsne": wandb.Image(buf)})
                    plt.close()
                except Exception as e:
                    print(f"t-SNE 시각화 실패: {e}")

                # wandb 로깅 (원본 이미지 포함)
                original_image = sample_image.cpu().squeeze(0).permute(1, 2, 0)
                original_image = ((original_image * 0.5 + 0.5) * 255).clamp(0, 255).round().numpy().astype(np.uint8)
                original_image = Image.fromarray(original_image)

                wandb.log({
                    "validation/original_image": wandb.Image(original_image, caption=f"Original"),
                    "validation/generated_images": wandb_images,
                    "validation/caption": sample_caption,
                    "validation/batch": batch
                })

                # 모듈들 다시 GPU로 복귀
                self.extractor = self.extractor.to('cuda')
                self.image_adapter = self.image_adapter.to('cuda')
                self.text_encoder = self.text_encoder.to('cuda')
                self.text_encoder_2 = self.text_encoder_2.to('cuda')
                self.unet = self.unet.to('cuda')
                self.vae = self.vae.to('cuda')
                cleanup_memory()

                avg_loss = total_loss / len(self.train_loader)
                wandb.log({
                    "epoch_avg_loss": avg_loss,
                    "epoch": epoch
                })
                print(f"\nEpoch {epoch}/{opt.epoch}, Average Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    # PYTORCH_CUDA_ALLOC_CONF 환경 변수 설정
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 메모리 현황 출력
    import torch
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Currently allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # 메모리 정리
    torch.cuda.empty_cache()
    
    extractor = MGN().to('cuda')
    # Image_adapter 클래스를 SDXL에 맞게 수정
    image_adapter = Concat_ImageAdapter_3Layers().to('cuda')
    main = Main(extractor, image_adapter)

    if opt.mode == 'train':
        # wandb 초기화
        wandb.init(
            project="Controlnet",
            config=vars(opt),
            name=f"sdxl_MLP_{opt.clip_embeddings_dim}_{opt.epoch}_{opt.lr}"
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