import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import gc
from network import MGN, SimpleCrossSelfAttentionStack1D_transpose
from torch.cuda.amp import autocast
import open_clip
import matplotlib.pyplot as plt

# 인자 파서 설정
parser = argparse.ArgumentParser(description='SDXL Image Adapter Validation')
parser.add_argument('--extractor_weight', type=str, required=True, 
                    help='특징 추출기 가중치 경로')
parser.add_argument('--adapter_weight', type=str, required=True, 
                    help='이미지 어댑터 가중치 경로')
parser.add_argument('--image_path', type=str, required=True,
                    help='검증에 사용할 이미지 경로')
parser.add_argument('--caption', type=str, default="",
                    help='이미지에 대한 캡션 (빈 문자열이면 기본 프롬프트 사용)')
parser.add_argument('--output_dir', type=str, required=True,
                    help='결과 저장 경로')
parser.add_argument('--clip_embeddings_dim', type=int, default=8,
                    help='CLIP 임베딩 차원 (기본값: 8)')
parser.add_argument('--num_samples', type=int, default=3,
                    help='생성할 이미지 수 (기본값: 3)')
parser.add_argument('--guidance_scale_1', type=float, default=7.5,
                    help='첫 번째 가이던스 스케일 (기본값: 7.5)')
parser.add_argument('--guidance_scale_2', type=float, default=1.5,
                    help='두 번째 가이던스 스케일 (기본값: 1.5)')
parser.add_argument('--model_id', type=str, default="neta-art/neta-xl-2.0",
                    help='SDXL 모델 ID (기본값: neta-art/neta-xl-2.0)')
parser.add_argument('--vae_model_path', type=str, default="madebyollin/sdxl-vae-fp16-fix",
                    help='VAE 모델 경로 (기본값: madebyollin/sdxl-vae-fp16-fix)')
parser.add_argument('--use_wandb', action='store_true',
                    help='wandb 사용 여부')
parser.add_argument('--image_size', type=int, default=768,
                    help='이미지 크기 (기본값: 512)')
parser.add_argument('--inference_height', type=int, default=1024,
                    help='추론 높이 (기본값: 768, 원래는 1024)')
parser.add_argument('--inference_width', type=int, default=1024,
                    help='추론 너비 (기본값: 768, 원래는 1024)')
parser.add_argument('--num_inference_steps', type=int, default=50, 
                    help='추론 스텝 수 (기본값: 15, 원래는 20)')

args = parser.parse_args()

# 필요한 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 메모리 정리 함수
def cleanup_memory():
    """적극적인 메모리 정리"""
    torch.cuda.empty_cache()
    gc.collect()

# 이미지 변환 함수 정의
def load_and_transform_image(image_path, size=512):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((size, size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 이미지 로드 및 변환
    image = Image.open(image_path)
    transformed_image = transform(image).unsqueeze(0).to('cuda')
    return transformed_image

# 텐서 값 클리핑 함수
def _safe_clamp(tensor, min_val=-10.0, max_val=10.0):
    """텐서의 값 범위를 안전하게 제한"""
    return torch.clamp(tensor, min=min_val, max=max_val)

# wandb 초기화 (필요한 경우)
if args.use_wandb:
    import wandb
    if not wandb.run:
        wandb.init(
            project="Controlnet-Validation",
            config=vars(args),
            name=f"validation_SDXL_{args.clip_embeddings_dim}"
        )

# 저장 경로 설정
save_dir = os.path.join(args.output_dir, 'validation_results')
os.makedirs(save_dir, exist_ok=True)

# 메모리 현황 출력
print(f"사용 가능한 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"현재 할당된 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# 초기 메모리 정리
cleanup_memory()

print(f"이미지 {args.image_path} 로드 중...")
sample_image = load_and_transform_image(args.image_path, args.image_size)
sample_caption = args.caption if args.caption else "A vibrant digital art piece with dynamic shapes and colors"

# Step 1: 이미지 특징 추출 단계
print("1. 이미지 특징 추출 단계 시작...")
cleanup_memory()

# MGN 추출기 모델 초기화
print("MGN 추출기 모델 초기화 중...")
extractor = MGN().to('cuda')
print(f"{args.extractor_weight}에서 추출기 가중치 로드 중...")
extractor.load_state_dict(torch.load(args.extractor_weight))
extractor.eval()

# 이미지 어댑터 모델 초기화
print("이미지 어댑터 모델 초기화 중...")
image_adapter = SimpleCrossSelfAttentionStack1D_transpose(
    extractor_dim=2048, 
    hidden_dim=1024, 
    clip_embeddings_dim=args.clip_embeddings_dim
).to('cuda')
print(f"{args.adapter_weight}에서 어댑터 가중치 로드 중...")
image_adapter.load_state_dict(torch.load(args.adapter_weight))
image_adapter.eval()

# 1. open_clip import 및 CLIP 모델 로드
clip_model, _, clip_transform = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
clip_model = clip_model.to("cuda").eval()

# 이미지 피처 추출
print("이미지 특징 추출 중...")
with torch.no_grad():
    # MGN feature 추출
    outputs = extractor(sample_image)
    embedding_feature = outputs[0].clone()
    
    # CLIP feature 추출
    pil_image = Image.open(args.image_path).convert("RGB")
    clip_image = clip_transform(pil_image).unsqueeze(0).to("cuda")
    clip_feature = clip_model.encode_image(clip_image)
    
    # 메모리 정리
    del outputs
    cleanup_memory()
    
    # 모든 모델이 동일한 dtype을 사용하도록 설정 (float32)
    image_adapter = image_adapter.to(dtype=torch.float32)
    embedding_feature = embedding_feature.to(dtype=torch.float32)
    clip_feature = clip_feature.to(dtype=torch.float32)
    
    # 2개 feature를 image_adapter에 입력
    projected_image_embedding, attention_maps = image_adapter(embedding_feature, clip_feature, return_attention=True)
    
    # 임베딩 정규화
    projected_image_embedding = F.normalize(projected_image_embedding, p=2, dim=-1)
    projected_image_embedding = 1000 * projected_image_embedding
    # 추가 feature 생성 (uncond) - 같은 dtype 사용
    uncond_feature = torch.zeros((projected_image_embedding.shape[0], 2048), dtype=torch.float32).to('cuda')
    empty_pil = Image.new("RGB", (224, 224), (0, 0, 0))
    empty_clip_image = clip_transform(empty_pil).unsqueeze(0).to("cuda")
    uncond_clip_feature = clip_model.encode_image(empty_clip_image)
    uncond_clip_feature = uncond_clip_feature.to(dtype=torch.float32)
    uncond_image_input = image_adapter(uncond_feature, uncond_clip_feature)
    uncond_image_input = F.normalize(uncond_image_input, p=2, dim=-1)
    
    # 임베딩 추출 완료 후 CPU로 이동
    projected_image_embedding = projected_image_embedding.detach().clone()
    print(f"projected_image_embedding.shape: {projected_image_embedding.shape}")
    print(f"clip_feature.shape: {clip_feature.shape}")
    uncond_image_input = uncond_image_input.detach().clone()

# extractor와 adapter를 메모리에서 해제
print("MGN 추출기와 어댑터 모델 해제 중...")
extractor = extractor.to('cpu')
image_adapter = image_adapter.to('cpu')
del extractor, image_adapter, embedding_feature, uncond_feature, clip_feature, uncond_clip_feature
cleanup_memory()

print(f"특징 추출 후 메모리 사용량: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# Step 2: SDXL 텍스트 인코더 단계
print("2. 텍스트 인코더 단계 시작...")
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, EulerAncestralDiscreteScheduler

# SDXL 모델 로드 시 메모리 최적화
print(f"SDXL 모델 {args.model_id} 로드 중...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    args.model_id, 
    torch_dtype=torch.float16,
    use_safetensors=True
)

# VAE 모델 로드 (별도 지정된 경우)
if args.vae_model_path:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        args.vae_model_path,
        torch_dtype=torch.float16
    )
    pipe.vae = vae

# 필요한 컴포넌트 저장
tokenizer = pipe.tokenizer
tokenizer_2 = pipe.tokenizer_2
text_encoder = pipe.text_encoder.to('cuda', dtype=torch.float16)
text_encoder_2 = pipe.text_encoder_2.to('cuda', dtype=torch.float16)
unet = pipe.unet
vae = pipe.vae

# 메모리 사용량 최적화: gradient checkpointing 활성화
unet.enable_gradient_checkpointing()

# 노이즈 스케줄러 설정
noise_scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 추출 후 파이프라인 해제
del pipe
cleanup_memory()

# 추가 조건부 정보 - 해상도 감소
original_size = (args.inference_height, args.inference_width)
target_size = (args.inference_height, args.inference_width)
crops_coords_top_left = (0, 0)

# 텍스트 임베딩 생성
print("텍스트 임베딩 생성 중...")
with torch.no_grad(), autocast(dtype=torch.float16):
    # SDXL 텍스트 임베딩 생성
    text_input_ids = tokenizer(
        sample_caption,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to('cuda')
    
    text_input_ids_2 = tokenizer_2(
        sample_caption,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to('cuda')
    
    # 텍스트 임베딩 생성
    prompt_embeds_1 = text_encoder(
        text_input_ids,
        output_hidden_states=True,
    ).hidden_states[-2]
    
    prompt_embeds_2 = text_encoder_2(
        text_input_ids_2,
        output_hidden_states=True,
    ).hidden_states[-2]
    
    # 결합
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    
    # 메모리 정리
    del prompt_embeds_1, prompt_embeds_2
    cleanup_memory()
    
    # 추가 임베딩
    add_text_embeds = text_encoder_2(
        text_input_ids_2,
        output_hidden_states=True,
    ).hidden_states[-1]
    
    # 첫 번째 토큰(CLS)만 사용하여 차원 조정
    if len(add_text_embeds.shape) == 3:
        add_text_embeds = add_text_embeds[:, 0, :]
    
    add_time_ids = torch.tensor(
        [list(original_size) + list(crops_coords_top_left) + list(target_size)],
        dtype=torch.float16
    ).to('cuda')
    
    # Unconditional 임베딩 생성
    uncond_text_input = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to("cuda")
    
    uncond_text_input_2 = tokenizer_2(
        "",
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Unconditional 텍스트 임베딩
    uncond_embeds_1 = text_encoder(
        uncond_text_input.input_ids,
        output_hidden_states=True,
    ).hidden_states[-2]
    
    uncond_embeds_2 = text_encoder_2(
        uncond_text_input_2.input_ids,
        output_hidden_states=True,
    ).hidden_states[-2]
    
    uncond_embeds = torch.cat([uncond_embeds_1, uncond_embeds_2], dim=-1)
    
    # 메모리 정리
    del uncond_embeds_1, uncond_embeds_2
    cleanup_memory()
    
    # Unconditional 추가 임베딩
    uncond_add_text_embeds = text_encoder_2(
        uncond_text_input_2.input_ids,
        output_hidden_states=True,
    ).hidden_states[-1]
    
    # 차원 조정
    if len(uncond_add_text_embeds.shape) == 3:
        uncond_add_text_embeds = uncond_add_text_embeds[:, 0, :]

    # 텍스트 임베딩 저장
    prompt_embeds = prompt_embeds.detach().clone()
    add_text_embeds = add_text_embeds.detach().clone()
    uncond_embeds = uncond_embeds.detach().clone()
    uncond_add_text_embeds = uncond_add_text_embeds.detach().clone()
    add_time_ids = add_time_ids.detach().clone()

# 텍스트 인코더 메모리 해제
print("텍스트 인코더 모델 해제 중...")
text_encoder = text_encoder.to('cpu')
text_encoder_2 = text_encoder_2.to('cpu')
del text_encoder, text_encoder_2, text_input_ids, text_input_ids_2
del uncond_text_input, uncond_text_input_2
cleanup_memory()

print(f"텍스트 인코딩 후 메모리 사용량: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# 모든 임베딩을 float16으로 변환
prompt_embeds = prompt_embeds.to(dtype=torch.float16)
projected_image_embedding = projected_image_embedding.to(dtype=torch.float16)
uncond_embeds = uncond_embeds.to(dtype=torch.float16)
uncond_image_input = uncond_image_input.to(dtype=torch.float16)
add_text_embeds = add_text_embeds.to(dtype=torch.float16)
uncond_add_text_embeds = uncond_add_text_embeds.to(dtype=torch.float16)

# 차원 불일치 해결
if prompt_embeds.shape[-1] != projected_image_embedding.shape[-1]:
    print("임베딩 차원 조정 중...")
    import torch.nn as nn
    adapter_projection = nn.Linear(
        projected_image_embedding.shape[-1], 
        prompt_embeds.shape[-1]
    ).to(device=projected_image_embedding.device, dtype=torch.float16)
    
    # 가중치 초기화
    nn.init.normal_(adapter_projection.weight, std=0.02)
    nn.init.zeros_(adapter_projection.bias)
    
    # 차원 변환
    with torch.no_grad():
        projected_image_embedding = adapter_projection(projected_image_embedding)
        uncond_image_input = adapter_projection(uncond_image_input)
    
    # 메모리 해제
    del adapter_projection
    cleanup_memory()

# 여러 이미지 생성
for sample_idx in range(args.num_samples):
    print(f"샘플 {sample_idx+1}/{args.num_samples} 생성 중...")
    
    # 조건 결합
    # Full Conditional
    full_conditions = torch.cat([prompt_embeds, projected_image_embedding], dim=1)
    
    # Image Condition
    image_conditions = torch.cat([uncond_embeds, projected_image_embedding], dim=1)
    
    # Full unconditional
    full_uncond_conditions = torch.cat([uncond_embeds, uncond_image_input], dim=1)
    
    
    # NaN 및 Inf 체크 및 클리핑
    for tensor_name, tensor in [
        ("full_conditions", full_conditions),
        ("image_conditions", image_conditions),
        ("full_uncond_conditions", full_uncond_conditions)
    ]:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"경고: {tensor_name}에 NaN 또는 Inf가 있습니다!")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 텐서 클리핑 적용
    # clamp 전
    before = full_conditions.clone()
    # clamp 적용
    full_conditions = _safe_clamp(full_conditions)
    # clamp된 값 개수 계산
    num_clamped = ((before != full_conditions).sum()).item()
    print(f"full_conditions에서 clamp된 값 개수: {num_clamped}")

    image_conditions = _safe_clamp(image_conditions)
    full_uncond_conditions = _safe_clamp(full_uncond_conditions)
    
    # 노이즈 크기 감소 - 해상도 감소
    latent_height = args.inference_height // 8
    latent_width = args.inference_width // 8
    latents = torch.randn((1, 4, latent_height, latent_width), device='cuda', dtype=torch.float16) * 0.9
    
    # EulerAncestral 스케줄러 설정 - 스텝 수 감소
    noise_scheduler.set_timesteps(args.num_inference_steps)
    timesteps = noise_scheduler.timesteps
    
    # UNet을 GPU로 이동
    unet = unet.to(dtype=torch.float16).to('cuda')
    
    # 디퓨전 프로세스 수행 - 메모리 효율적인 방식으로 변경
    print("이미지 생성 중...")
    with torch.no_grad(), autocast(dtype=torch.float16):
        for t in timesteps:
            latent_model_input = noise_scheduler.scale_model_input(latents, t)
            
            # 모델 입력 클리핑
            latent_model_input = _safe_clamp(latent_model_input)
            
            # add_time_ids가 올바른 배치 크기를 가지도록 조정
            batch_add_time_ids = add_time_ids
            if add_time_ids.shape[0] != latent_model_input.shape[0]:
                batch_add_time_ids = add_time_ids.repeat(latent_model_input.shape[0], 1)
            
            # 현재 배치 처리 - 각 조건 순차적으로 처리 후 즉시 정리
            # Unconditional 예측
            noise_pred_uncond_full = unet(
                latent_model_input,
                t,
                encoder_hidden_states=full_uncond_conditions,
                added_cond_kwargs={"text_embeds": uncond_add_text_embeds, "time_ids": batch_add_time_ids}
            ).sample
            
            # Image conditional 예측
            noise_pred_image = unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_conditions,
                added_cond_kwargs={"text_embeds": uncond_add_text_embeds, "time_ids": batch_add_time_ids}
            ).sample
            
            # Full Conditional 예측
            noise_pred_full = unet(
                latent_model_input,
                t,
                encoder_hidden_states=full_conditions,
                added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": batch_add_time_ids}
            ).sample
            
            # CFG 계산 시작 - 첫 부분
            noise_pred = noise_pred_uncond_full + args.guidance_scale_1 * (noise_pred_image - noise_pred_uncond_full) + args.guidance_scale_2 * (noise_pred_full - noise_pred_image)
            
            
            # 노이즈 예측 결과 확인
            if torch.isnan(noise_pred).any():
                print("경고: inference 중 noise_pred에 NaN이 있습니다!")
                noise_pred = torch.where(torch.isnan(noise_pred), torch.zeros_like(noise_pred), noise_pred)
            
            # 스케줄러 스텝
            latents = noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latents
            ).prev_sample
            
            # 메모리 정리
            del noise_pred
            cleanup_memory()
    
    # unet을 CPU로 이동
    unet = unet.to('cpu')
    if sample_idx == args.num_samples - 1:
        del unet
    cleanup_memory()
    
    # VAE 디코딩
    print("VAE 디코딩 중...")
    vae = vae.to(dtype=torch.float32).to('cuda')
    
    # VAE 디코딩 - fp32로 변환해서 처리
    latents_fp32 = latents.to(dtype=torch.float32)
    latents_fp32 = 1 / vae.config.scaling_factor * latents_fp32
    
    # 메모리 정리
    del latents
    cleanup_memory()
    
    # fp32로 디코딩 수행
    with torch.no_grad():
        image = vae.decode(latents_fp32).sample
    
    # VAE 메모리 해제
    vae = vae.to('cpu')
    del latents_fp32
    cleanup_memory()
    
    # 이미지 후처리 및 저장
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).clamp(0, 255).round()
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = image.astype(np.uint8)[0]
    
    result_image = Image.fromarray(image)
    output_path = f'{save_dir}/generated_sample_{sample_idx}.png'
    result_image.save(output_path)
    print(f"이미지 저장 완료: {output_path}")

    # === prompt_embeds, projected_image_embedding 값 범위 출력 ===
    def print_tensor_stats(name, tensor):
        t = tensor.detach().cpu().float()
        print(f"{name} - min: {t.min().item():.6f}, max: {t.max().item():.6f}, mean: {t.mean().item():.6f}, std: {t.std().item():.6f}")

    print_tensor_stats("prompt_embeds", prompt_embeds)
    print_tensor_stats("projected_image_embedding", projected_image_embedding)
    print_tensor_stats("full_conditions", full_conditions)

    # === 각 샘플별 attention map 저장 ===
    last_attn = attention_maps[0]  # 마지막 layer
    attn_map = last_attn[0, 0].detach().cpu().numpy()  # 첫 배치, 첫 head
    print(f"Attention map shape: {attn_map.shape}")
    # plt.imshow(attn_map, cmap='viridis')
    # plt.colorbar()
    # plt.title(f"Cross Attention Map (샘플 {sample_idx+1}, 마지막 layer, head 0)")
    # output_attn_path = os.path.join(save_dir, f"cross_attention_map_{sample_idx}.png")
    # plt.savefig(output_attn_path)
    # plt.close()
    # print(f"Attention map이 {output_attn_path}에 저장되었습니다.")
    # # === 여기까지 추가 ===

    # wandb 로깅
    if args.use_wandb:
        original_image = sample_image.cpu().squeeze(0).permute(1, 2, 0)
        original_image = ((original_image * 0.5 + 0.5) * 255).clamp(0, 255).round().numpy().astype(np.uint8)
        original_image = Image.fromarray(original_image)
        
        wandb.log({
            f"validation/original_image": wandb.Image(original_image, caption=f"Original"),
            f"validation/generated_image_{sample_idx}": wandb.Image(result_image, caption=f"Generated - {sample_caption}"),
            "validation/caption": sample_caption
        })
    
    # 마지막 샘플이 아니라면 메모리 정리
    if sample_idx < args.num_samples - 1:
        del image, result_image
        cleanup_memory()

# 이미지 원본 메모리 해제
del sample_image, projected_image_embedding, uncond_image_input
del prompt_embeds, add_text_embeds, uncond_embeds, uncond_add_text_embeds, add_time_ids
del full_conditions, image_conditions, full_uncond_conditions
if 'vae' in locals():
    del vae
cleanup_memory()

print(f"검증 완료! 결과가 {save_dir}에 저장되었습니다.")