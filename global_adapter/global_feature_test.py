import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from network import MGN, Image_adapter,Image_adapter_77_768, Base_adapter, Base_adapter_77_768

def generate_image_from_text(prompt, output_path="generated_image.png", seed=None):
    # 시드 설정
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None

    # CLIP 텍스트 인코더와 토크나이저 초기화
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    text_encoder_large = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    tokenizer_large = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    extractor = MGN().to('cuda')
    extractor.load_state_dict(torch.load("./best/model_2000.pt"))
    extractor.eval()
    
    #image_adapter = Image_adapter().to("cuda")
    image_adapter = Image_adapter_77_768().to("cuda")
    image_adapter.load_state_dict(torch.load("./result/image_adapter_77_768/image_adapter/image_adapter_200_best.pt"))
    image_adapter.eval()
    
    projector = Base_adapter_77_768().to('cuda')  
    projector.load_state_dict(torch.load("/workspace/mnt/sda/changhyun/Controlnet_custom/result/projector_77_768/projector/projector_100.pt"))
    projector.eval()
    
    # Stable Diffusion 파이프라인 초기화
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # 텍스트 임베딩 생성
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to("cuda")
    
    # CLIP 텍스트 인코더로 텍스트 임베딩 추출
    with torch.no_grad():
        text_embeddings = text_encoder(inputs.input_ids)[0]
    with torch.no_grad():
        text_embeddings_large = text_encoder_large(inputs.input_ids)[0]
    
    
    print(f"text_embeddings shape: {text_embeddings.shape}")
    
    input_image = Image.open("/workspace/mnt/sda/changhyun/dataset/top15character/test/Eren Yeager/Image_11.jpg")
    input_image = input_image.convert('RGB').resize((256, 256), Image.LANCZOS)
    
    image_tensor = torch.from_numpy(np.array(input_image)).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to("cuda")
    
    # Feature extraction 및 adaptation
    with torch.no_grad():
        image_features = extractor(image_tensor)[0]
        adapted_features = image_adapter(image_features)
        #fused_embeddings = projector(adapted_features, text_embeddings_large)
        #fused_embeddings = 0.3 * adapted_features + 0.7 * text_embeddings_large
        fused_embeddings = torch.cat([adapted_features, text_embeddings_large], dim=1)
        fused_embeddings = fused_embeddings.to(dtype=torch.float16)
    
    print(f"image_features shape: {image_features.shape}")
    print(f"adapted_features shape: {adapted_features.shape}")
    print(f"fused_embeddings shape: {fused_embeddings.shape}")
    
    # Positive embedding 생성
    with torch.no_grad():
        text_embeddings_large = text_encoder_large(inputs.input_ids)[0]
        fused_embeddings = torch.cat([adapted_features, text_embeddings_large], dim=1)
        fused_embeddings = fused_embeddings.to(dtype=torch.float16)

    # Negative embedding 생성 (빈 프롬프트 사용)
    uncond_input = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        uncond_embeddings_large = text_encoder_large(uncond_input.input_ids)[0]
        # Positive embedding과 같은 형식으로 negative embedding 생성
        uncond_fused = torch.cat([adapted_features, uncond_embeddings_large], dim=1)
        uncond_fused = uncond_fused.to(dtype=torch.float16)

    # 두 임베딩을 모두 pipeline에 전달
    image = pipeline(
        prompt_embeds=fused_embeddings,
        negative_prompt_embeds=uncond_fused,
        num_inference_steps=50,
        guidance_scale=2.5,
        generator=generator
    ).images[0]
    
    # 이미지 저장
    image.save(output_path)
    return image

# 사용 예시
if __name__ == "__main__":
    prompt = "A man with black hair and brown leather jacket holding a knife"
    generated_image = generate_image_from_text(prompt, "768_concat.png", seed=77)
