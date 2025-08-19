import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import random
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import gc

from diffusers import StableDiffusionXLPipeline, DDPMScheduler, EulerAncestralDiscreteScheduler
from torch.cuda.amp import autocast

from opt import opt
from data import Data
from network import MGN, Image_adapter_SDXL

import wandb

class Validator:
    def __init__(self, extractor, image_adapter, main_obj, output_dir=None, use_wandb=True):
        """
        메모리 효율적인 검증 클래스
        
        Args:
            extractor: 특징 추출기 모델
            image_adapter: 이미지 어댑터 모델
            main_obj: 메인 학습 객체 (모든 모델과 토크나이저가 포함됨)
            output_dir: 결과 저장 경로
            use_wandb: wandb 사용 여부
        """
        # 학습 객체에서 필요한 모든 객체 참조 (복사하지 않음)
        self.main_obj = main_obj  
        self.test_loader = main_obj.test_loader
        self.image_adapter = image_adapter
        self.extractor = extractor
        self.output_dir = output_dir if output_dir else main_obj.result_dir
        self.use_wandb = use_wandb
        
        # 결과 저장 디렉토리 생성
        os.makedirs(f'{self.output_dir}/validation_results', exist_ok=True)
    
    def _safe_clamp(self, tensor, min_val=-10.0, max_val=10.0):
        """텐서의 값 범위를 안전하게 제한"""
        return torch.clamp(tensor, min=min_val, max=max_val)
    
    def _cleanup_memory(self):
        """적극적인 메모리 정리"""
        torch.cuda.empty_cache()
        gc.collect()
    
    def validate(self, epoch=0, step=0, num_samples=1, guidance_scale_1=7.5, guidance_scale_2=1.5):
        """메모리 효율적인 검증 함수"""
        self._cleanup_memory()
        
        # 평가 모드로 설정
        self.extractor.eval()
        self.image_adapter.eval()
        
        # 저장 경로
        save_dir = f'{self.output_dir}/validation_results'
        os.makedirs(save_dir, exist_ok=True)
        
        # 검증 이미지 해상도 감소 (메모리 절약)
        inference_height = 768  # 원래 1024
        inference_width = 768   # 원래 1024
        
        # 타임스텝 수 감소
        num_inference_steps = 15  # 원래 20
        
        for sample_idx in range(num_samples):
            # 메모리 정리
            self._cleanup_memory()
            
            # 테스트 데이터셋에서 랜덤 이미지 선택
            random_idx = random.randint(0, len(self.test_loader.dataset) - 1)
            sample_image, _, sample_caption = self.test_loader.dataset[random_idx]
            sample_image = sample_image.unsqueeze(0).to('cuda')
            
            # 이미지 임베딩 추출 및 프로젝션
            with torch.no_grad():
                outputs = self.extractor(sample_image)
                embedding_feature = outputs[0]
                
                # 메모리 정리
                del outputs
                self._cleanup_memory()
                
                # CLIP 모델이 있는 경우 (global_feature_train_ip_euler_drop_sdxl_image_clip.py)
                if hasattr(self.main_obj, 'clip_model'):
                    # CLIP 이미지 특징 추출
                    normalized_images = (sample_image + 1) / 2.0
                    images_np = normalized_images.cpu().permute(0, 2, 3, 1).numpy()
                    clip_images = []
                    for img_np in images_np:
                        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
                        clip_img = self.main_obj.clip_tensor_transform(pil_img)
                        clip_images.append(clip_img)
                    clip_images = torch.stack(clip_images).to('cuda')
                    clip_image_features = self.main_obj.clip_model.encode_image(clip_images)
                    projected_image_embedding = self.image_adapter(embedding_feature, clip_image_features)
                    
                    # 사용 후 메모리 정리
                    del normalized_images, images_np, clip_images, clip_image_features
                    self._cleanup_memory()
                else:
                    # 기본 이미지 어댑터 사용
                    projected_image_embedding = self.image_adapter(embedding_feature)
                
                # 임베딩 정규화
                projected_image_embedding = F.normalize(projected_image_embedding, p=2, dim=-1)
                
                # CPU에 샘플 이미지 저장 후 GPU 메모리 해제
                sample_image_cpu = sample_image.cpu()
                del sample_image
                self._cleanup_memory()
                
                # 텍스트 인코더를 GPU로 이동 (fp16으로 설정)
                self.main_obj.text_encoder.to('cuda', dtype=torch.float16)
                self.main_obj.text_encoder_2.to('cuda', dtype=torch.float16)
                
                # SDXL 텍스트 임베딩 생성
                text_input_ids = self.main_obj.tokenizer(
                    sample_caption,
                    padding="max_length",
                    max_length=self.main_obj.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to('cuda')
                
                text_input_ids_2 = self.main_obj.tokenizer_2(
                    sample_caption,
                    padding="max_length",
                    max_length=self.main_obj.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to('cuda')
                
                # 추가 조건부 정보 - 해상도 감소된 값 사용
                original_size = (inference_height, inference_width)
                target_size = (inference_height, inference_width)
                crops_coords_top_left = (0, 0)
                
                # 첫 번째 텍스트 인코더
                prompt_embeds_1 = self.main_obj.text_encoder(
                    text_input_ids,
                    output_hidden_states=True,
                ).hidden_states[-2]
                
                # 두 번째 텍스트 인코더
                prompt_embeds_2 = self.main_obj.text_encoder_2(
                    text_input_ids_2,
                    output_hidden_states=True,
                ).hidden_states[-2]
                
                # 결합
                prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
                
                # 메모리 정리
                del prompt_embeds_1, prompt_embeds_2
                self._cleanup_memory()
                
                # 추가 임베딩
                add_text_embeds = self.main_obj.text_encoder_2(
                    text_input_ids_2,
                    output_hidden_states=True,
                ).hidden_states[-1]
                
                # 첫 번째 토큰(CLS)만 사용하여 차원 조정
                if len(add_text_embeds.shape) == 3:
                    add_text_embeds = add_text_embeds[:, 0, :]
                
                add_time_ids = torch.tensor(
                    [list(original_size) + list(crops_coords_top_left) + list(target_size)],
                    dtype=torch.float16  # 직접 fp16으로 생성
                ).to('cuda')
                
                # Unconditional 임베딩 생성
                uncond_text_input = self.main_obj.tokenizer(
                    "",
                    padding="max_length",
                    max_length=self.main_obj.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to("cuda")
                
                uncond_text_input_2 = self.main_obj.tokenizer_2(
                    "",
                    padding="max_length",
                    max_length=self.main_obj.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to("cuda")
                
                # 조건부 없는 임베딩
                if hasattr(self.main_obj, 'clip_model'):
                    uncond_feature = torch.zeros((projected_image_embedding.shape[0], 2048), dtype=torch.float16).to('cuda')
                    dummy_clip_feature = torch.zeros_like(projected_image_embedding, dtype=torch.float16).to('cuda')
                    uncond_image_input = self.image_adapter(uncond_feature, dummy_clip_feature)
                    del dummy_clip_feature
                else:
                    uncond_feature = torch.zeros((projected_image_embedding.shape[0], 2048), dtype=torch.float16).to('cuda')
                    uncond_image_input = self.image_adapter(uncond_feature)
                
                # unconditional 임베딩도 정규화
                uncond_image_input = F.normalize(uncond_image_input, p=2, dim=-1)
                
                # Unconditional 텍스트 임베딩
                uncond_embeds_1 = self.main_obj.text_encoder(
                    uncond_text_input.input_ids,
                    output_hidden_states=True,
                ).hidden_states[-2]
                
                uncond_embeds_2 = self.main_obj.text_encoder_2(
                    uncond_text_input_2.input_ids,
                    output_hidden_states=True,
                ).hidden_states[-2]
                
                uncond_embeds = torch.cat([uncond_embeds_1, uncond_embeds_2], dim=-1)
                
                # 메모리 정리
                del uncond_embeds_1, uncond_embeds_2
                self._cleanup_memory()
                
                # Unconditional 추가 임베딩
                uncond_add_text_embeds = self.main_obj.text_encoder_2(
                    uncond_text_input_2.input_ids,
                    output_hidden_states=True,
                ).hidden_states[-1]
                
                # 차원 조정
                if len(uncond_add_text_embeds.shape) == 3:
                    uncond_add_text_embeds = uncond_add_text_embeds[:, 0, :]
                
                # 텍스트 인코더를 다시 CPU로 이동
                self.main_obj.text_encoder.to('cpu')
                self.main_obj.text_encoder_2.to('cpu')
                self._cleanup_memory()
                
                # 차원 불일치 해결
                if prompt_embeds.shape[-1] != projected_image_embedding.shape[-1]:
                    adapter_projection = nn.Linear(
                        projected_image_embedding.shape[-1], 
                        prompt_embeds.shape[-1]
                    ).to(device=projected_image_embedding.device, dtype=torch.float16)
                    
                    # 가중치 초기화
                    nn.init.normal_(adapter_projection.weight, std=0.02)
                    nn.init.zeros_(adapter_projection.bias)
                    
                    # 차원 변환
                    projected_image_embedding = adapter_projection(projected_image_embedding)
                    uncond_image_input = adapter_projection(uncond_image_input)
                    
                    # Linear 레이어 사용 후 정리
                    del adapter_projection
                    self._cleanup_memory()
                
                # 모든 임베딩을 float16으로 변환
                prompt_embeds = prompt_embeds.to(dtype=torch.float16)
                projected_image_embedding = projected_image_embedding.to(dtype=torch.float16)
                uncond_embeds = uncond_embeds.to(dtype=torch.float16)
                uncond_image_input = uncond_image_input.to(dtype=torch.float16)
                add_text_embeds = add_text_embeds.to(dtype=torch.float16)
                uncond_add_text_embeds = uncond_add_text_embeds.to(dtype=torch.float16)
                
                # 조건 결합
                # Full Conditional
                full_conditions = torch.cat([prompt_embeds, projected_image_embedding], dim=1)
                
                # Image Condition
                image_conditions = torch.cat([uncond_embeds, projected_image_embedding], dim=1)
                
                # Full unconditional
                full_uncond_conditions = torch.cat([uncond_embeds, uncond_image_input], dim=1)
                
                # Text Unconditional
                text_uncond_conditions = torch.cat([uncond_embeds, projected_image_embedding], dim=1)
                
                # NaN 및 Inf 체크 및 클리핑
                for tensor_name, tensor in [
                    ("full_conditions", full_conditions),
                    ("image_conditions", image_conditions),
                    ("full_uncond_conditions", full_uncond_conditions),
                    ("text_uncond_conditions", text_uncond_conditions)
                ]:
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        print(f"경고: {tensor_name}에 NaN 또는 Inf가 있습니다!")
                        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 텐서 클리핑 적용
                full_conditions = self._safe_clamp(full_conditions)
                image_conditions = self._safe_clamp(image_conditions)
                full_uncond_conditions = self._safe_clamp(full_uncond_conditions)
                text_uncond_conditions = self._safe_clamp(text_uncond_conditions)
                
                # 노이즈에서 이미지 생성 (fp16으로) - 해상도 감소
                latent_height = inference_height // 8
                latent_width = inference_width // 8
                latents = torch.randn((1, 4, latent_height, latent_width), device='cuda', dtype=torch.float16) * 0.9
                
                # EulerAncestral 스케줄러 설정 - 타임스텝 수 감소
                euler_scheduler = EulerAncestralDiscreteScheduler.from_config(self.main_obj.noise_scheduler.config)
                euler_scheduler.set_timesteps(num_inference_steps)
                timesteps = euler_scheduler.timesteps
                
                # UNet을 fp16으로 설정
                self.main_obj.unet = self.main_obj.unet.to('cuda', dtype=torch.float16)
                
                # 디퓨전 프로세스 수행 - 메모리 효율적인 방식으로 변경
                with torch.no_grad(), autocast(dtype=torch.float16):
                    for t in timesteps:
                        latent_model_input = euler_scheduler.scale_model_input(latents, t)
                        
                        # 모델 입력 클리핑
                        latent_model_input = self._safe_clamp(latent_model_input)
                        
                        # add_time_ids가 올바른 배치 크기를 가지도록 조정
                        batch_add_time_ids = add_time_ids
                        if add_time_ids.shape[0] != latent_model_input.shape[0]:
                            batch_add_time_ids = add_time_ids.repeat(latent_model_input.shape[0], 1)
                        
                        # 현재 배치 처리 - 각 조건 순차적으로 처리 후 즉시 정리
                        # Unconditional 예측
                        noise_pred_uncond_full = self.main_obj.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=full_uncond_conditions,
                            added_cond_kwargs={"text_embeds": uncond_add_text_embeds, "time_ids": batch_add_time_ids}
                        ).sample
                        
                        # Text Unconditional 예측
                        noise_pred_uncond_text = self.main_obj.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_uncond_conditions,
                            added_cond_kwargs={"text_embeds": uncond_add_text_embeds, "time_ids": batch_add_time_ids}
                        ).sample
                        
                        # CFG 계산 시작 - 첫 부분
                        noise_pred = noise_pred_uncond_full.clone()
                        
                        # 메모리 정리
                        del noise_pred_uncond_full
                        self._cleanup_memory()
                        
                        # Image Conditional 예측
                        noise_pred_image = self.main_obj.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=image_conditions,
                            added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": batch_add_time_ids}
                        ).sample
                        
                        # CFG 두번째 부분 계산
                        noise_pred = noise_pred + guidance_scale_1 * (noise_pred_image - noise_pred_uncond_text)
                        
                        # 메모리 정리
                        del noise_pred_image
                        self._cleanup_memory()
                        
                        # Full Conditional 예측
                        noise_pred_full = self.main_obj.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=full_conditions,
                            added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": batch_add_time_ids}
                        ).sample
                        
                        # CFG 마지막 부분
                        noise_pred = noise_pred + guidance_scale_2 * (noise_pred_full - noise_pred_uncond_text)
                        
                        # 메모리 정리
                        del noise_pred_full, noise_pred_uncond_text
                        self._cleanup_memory()
                        
                        # 노이즈 예측 결과 확인
                        if torch.isnan(noise_pred).any():
                            print("경고: inference 중 noise_pred에 NaN이 있습니다!")
                            noise_pred = torch.where(torch.isnan(noise_pred), torch.zeros_like(noise_pred), noise_pred)
                        
                        # 스케줄러 스텝
                        latents = euler_scheduler.step(
                            model_output=noise_pred,
                            timestep=t,
                            sample=latents
                        ).prev_sample
                        
                        # 메모리 정리
                        del noise_pred
                        self._cleanup_memory()
                
                # UNet을 CPU로 이동 
                self.main_obj.unet.to('cpu')
                self._cleanup_memory()
                
                # VAE 디코딩 - CPU→GPU 이동 및 fp32로 변환
                self.main_obj.vae.to('cuda', dtype=torch.float32)
                latents_fp32 = latents.to(dtype=torch.float32)
                latents_fp32 = 1 / self.main_obj.vae.config.scaling_factor * latents_fp32
                
                # 메모리 정리
                del latents
                self._cleanup_memory()
                
                # fp32로 디코딩 수행 - 작은 청크로 나누어 처리
                with torch.no_grad():
                    image = self.main_obj.vae.decode(latents_fp32).sample
                
                # VAE를 다시 CPU로 이동
                self.main_obj.vae.to('cpu')
                
                # 이미지 후처리 및 저장
                image = (image / 2 + 0.5).clamp(0, 1)
                image = (image * 255).clamp(0, 255).round()
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = image.astype(np.uint8)[0]
                
                result_image = Image.fromarray(image)
                result_image.save(f'{save_dir}/validation_sample_{sample_idx}_epoch_{epoch}_step_{step}.png')
                
                # 메모리 정리
                del image, latents_fp32
                self._cleanup_memory()
                
                # wandb 로깅
                if self.use_wandb:
                    original_image = sample_image_cpu.squeeze(0).permute(1, 2, 0)
                    original_image = ((original_image * 0.5 + 0.5) * 255).clamp(0, 255).round().numpy().astype(np.uint8)
                    original_image = Image.fromarray(original_image)
                    
                    wandb.log({
                        f"validation/original_image_{sample_idx}": wandb.Image(original_image, caption=f"Original - {sample_caption}"),
                        f"validation/generated_image_{sample_idx}": wandb.Image(result_image, caption=f"Generated - {sample_caption}"),
                        "validation/caption": sample_caption,
                        "epoch": epoch,
                        "step": step
                    })
                    
                    # 메모리 정리
                    del original_image
                    self._cleanup_memory()
        
        # 가중치 저장
        weights_dir = f'{self.output_dir}/image_adapter'
        os.makedirs(weights_dir, exist_ok=True)
        image_adapter_path = f'{weights_dir}/image_adapter_epoch_{epoch}_step_{step}.pt'
        torch.save(self.image_adapter.state_dict(), image_adapter_path)
        print(f'검증 완료: 이미지 어댑터 가중치가 {image_adapter_path}에 저장되었습니다.')
        
        return image_adapter_path

if __name__ == '__main__':
    # 환경 변수 설정
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 메모리 현황 출력
    print(f"사용 가능한 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"현재 할당된 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    # wandb 초기화 (필요한 경우)
    use_wandb = True
    if use_wandb and not wandb.run:
        wandb.init(
            project="Controlnet-Validation",
            config=vars(opt),
            name=f"validation_SDXL_{opt.clip_embeddings_dim}"
        )
    
    # 모델 초기화
    extractor = MGN().to('cuda')
    image_adapter = Image_adapter_SDXL(
        extractor_dim=2048, 
        hidden_dim=1024, 
        clip_embeddings_dim=opt.clip_embeddings_dim
    ).to('cuda')
    
    # 가중치 로드 (필요한 경우)
    if hasattr(opt, 'extractor_weight') and opt.extractor_weight:
        print(f"{opt.extractor_weight}에서 추출기 가중치 로드 중...")
        extractor.load_state_dict(torch.load(opt.extractor_weight))
    
    if hasattr(opt, 'adapter_weight') and opt.adapter_weight:
        print(f"{opt.adapter_weight}에서 어댑터 가중치 로드 중...")
        image_adapter.load_state_dict(torch.load(opt.adapter_weight))
    
    # 검증 수행
    validator = Validator(
        extractor=extractor,
        image_adapter=image_adapter,
        main_obj=opt,
        output_dir=opt.output_dir,
        use_wandb=use_wandb
    )
    
    validator.validate(
        epoch=opt.epoch if hasattr(opt, 'epoch') else 0,
        step=opt.step if hasattr(opt, 'step') else 0,
        num_samples=opt.num_validation_samples if hasattr(opt, 'num_validation_samples') else 1
    )