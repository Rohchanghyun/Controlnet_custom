import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from PIL import Image
import numpy as np
from network import MGN, Image_adapter, Image_adapter_77_768, Base_adapter, Base_adapter_77_768

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
    
    # extractor = MGN().to('cuda')
    # extractor.load_state_dict(torch.load("/workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt"))
    # extractor.eval()
    
    # #image_adapter = Image_adapter().to("cuda")
    # image_adapter = Image_adapter_77_768().to("cuda")
    # image_adapter.load_state_dict(torch.load("/workspace/data/changhyun/output/global_adapter/256/256_resume_adapter_best/image_adapter_2000.pt"))
    # image_adapter.eval()
    
    # projector = Base_adapter_77_768().to('cuda')  
    # projector.load_state_dict(torch.load("/workspace/data/changhyun/output/global_adapter/256/256_resume_adapter_best/projector_2000.pt"))
    # projector.eval()
    
    # CLIP 이미지 인코더 초기화
    vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    vision_encoder.eval()
    
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
        # image_features = extractor(image_tensor)[0]
        # adapted_features = image_adapter(image_features)
        
        # 이미지를 CLIP 이미지 인코더의 입력 형식에 맞게 전처리
        image_tensor_clip = torch.nn.functional.interpolate(image_tensor, size=(224, 224), mode='bicubic')
        vision_outputs = vision_encoder(image_tensor_clip)
        image_embeddings = vision_outputs.last_hidden_state  # [1, 50, 768]
        
        # # CLS 토큰 제거
        # image_embeddings = image_embeddings[:, 1:, :]  # [1, 49, 768]
        
        
        print(f"image_embeddings shape: {image_embeddings.shape}")
        # 기존 fusion 로직과 결합
        #fused_embeddings = torch.cat([text_embeddings_large, image_embeddings], dim=1)
        #fused_embeddings = torch.cat([text_embeddings_large, image_embeddings], dim=1)
        fused_embeddings = fused_embeddings.to(dtype=torch.float16)
    
    # print(f"image_features shape: {image_features.shape}")
    # print(f"adapted_features shape: {adapted_features.shape}")
    # print(f"fused_embeddings shape: {fused_embeddings.shape}")
    
    # 이미지 생성 - prompt 파라미터 제거
    image = pipeline(
        prompt_embeds=fused_embeddings,
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
    generated_image = generate_image_from_text(prompt, "768_clip_text.png", seed=42)
