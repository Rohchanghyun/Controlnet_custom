import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from network import MGN, Image_adapter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb

def generate_image_with_adapter():
    # extractor 로드
    extractor = MGN().to('cuda')
    extractor.load_state_dict(torch.load("/workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt"))

    # Image Adapter 로드 (사전 학습된 모델 가정)
    image_adapter = Image_adapter().to("cuda")
    image_adapter.load_state_dict(torch.load("/workspace/data/changhyun/output/global_adapter/256/256_resume_adapter_best/image_adapter_2000.pt"))
    
    text_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # 이미지 로드 및 전처리
    input_image = Image.open("/workspace/data/changhyun/dataset/emoji_data/test/Eren Yeager/Image_18.jpg")
    input_image = input_image.convert('RGB')  # RGB 식으로 변환
    input_image = input_image.resize((256, 256), Image.LANCZOS)  # 256x256으로 리사이즈
    
    # PIL 이미지를 텐서로 변환
    image_tensor = torch.from_numpy(np.array(input_image)).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)로 변환
    image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가
    image_tensor = image_tensor / 255.0  # 정규화
    image_tensor = image_tensor.to("cuda")
    # CLIP 이미지 인코더를 통해 이미지 특징 추출
    extractor.eval()
    with torch.no_grad():
        image_features = extractor(image_tensor)
        image_features = image_features[0]
    
    # Image Adapter를 통해 특징 변환
    adapted_features = image_adapter(image_features)
    
        # 텍스트 임베딩 생성
    with torch.no_grad():
        # 캡션 텍스트를 전처리
        text_inputs = processor(text="A character with black hair and brown leather jacket", return_tensors="pt", padding=True).to('cuda')
        text_embedding = text_encoder.get_text_features(**text_inputs)
        
    print(text_inputs)    
    print(text_embedding.shape)
    
    pdb.set_trace()
    # Cross Attention 레이어 정의
    class CrossAttention(nn.Module):
        def __init__(self, dim, out_dim):
            super().__init__()
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, out_dim)
            self.scale = dim ** -0.5

        def forward(self, x, context):
            q = self.query(x)
            k = self.key(context)
            v = self.value(context)
            
            attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attention = F.softmax(attention, dim=-1)
            out = torch.matmul(attention, v)
            return out

    # Cross Attention 인스턴스 생성
    cross_attention = CrossAttention(dim=512, out_dim=768).to('cuda')  # dim은 임베딩 차원에 맞게 조정
    
    # Cross Attention 적용
    fused_features = cross_attention(adapted_features, text_embedding)
    
    # fused_features 형태 변환 [1, 768] -> [1, 77, 768]
    fused_features = fused_features.unsqueeze(1)  # [1, 1, 768]
    fused_features = fused_features.repeat(1, 77, 1)  # [1, 77, 768]로 복제
    
    # float16으로 변환
    fused_features = fused_features.to(dtype=torch.float16)
    
    print("Fused features shape:", fused_features.shape)  # 확인용
    
    # SD 1.5 모델 로드
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")
    
    # 노이즈 생성 (float16으로 변환)
    generator = torch.Generator(device='cuda')  # CUDA generator 생성
    generator.manual_seed(0)  # 시드 설정
    
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, 64, 64),
        generator=generator,  # CUDA generator 사용
        device="cuda",
        dtype=torch.float16
    )
    
    # 스케줄러 설정
    scheduler = pipe.scheduler
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps.to(device="cuda", dtype=torch.long)
    
    # alphas_cumprod를 CUDA로 이동
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device="cuda")
    
    # 디노이징 과정
    for t in timesteps:
        # latents를 모델에 입력할 형태로 확장
        latent_model_input = scheduler.scale_model_input(latents, t)
        
        # fused_features를 condition으로 사용
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=fused_features
            ).sample
        
        # 스케줄러 스텝
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # 이미지 생성
    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    
    # 후처리
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])
    
    # 이미지 저장
    image.save("generated_image.png")


if __name__ == "__main__":
    generate_image_with_adapter()
