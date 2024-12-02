import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from network import MGN, Image_adapter, Base_adapter
import os
import glob

def generate_image_from_text(prompt, output_path="generated_image.png", seed=None):
    # 시드 설정

    # CLIP 텍스트 인코더와 토크나이저 초기화
    text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    
    extractor = MGN().to('cuda')
    extractor.load_state_dict(torch.load("/workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt", weights_only=True))
    extractor.eval()
    
    image_adapter = Image_adapter().to("cuda")
    image_adapter.load_state_dict(torch.load("/workspace/data/changhyun/output/global_adapter/256/256_resume_adapter_best/image_adapter_2000.pt", weights_only=True))
    image_adapter.eval()
    
    projector = Base_adapter().to('cuda')  
    projector.load_state_dict(torch.load("/workspace/data/changhyun/output/projector/projector/projector_16000.pt", weights_only=True))
    projector.eval()
    
    
    # CLIP 텍스트 인코더로 텍스트 임베딩 추출
    with torch.no_grad():
        # 캡션 텍스트를 전처리
        text_inputs = processor(text=prompt, return_tensors="pt", padding=True).to('cuda')
        text_outputs = text_encoder.text_model(**text_inputs)
        text_embedding = text_outputs.last_hidden_state
        text_embedding = text_embedding.squeeze()[1:-1]
    


    image_folder = "/workspace/data/changhyun/projects/emoji_generation/Controlnet_custom/attention_image/"
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    
    adapted_features_list = []

    for image_path in image_files:
        input_image = Image.open(image_path)
        input_image = input_image.convert('RGB').resize((256, 256), Image.LANCZOS)
        
        image_tensor = torch.from_numpy(np.array(input_image)).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to("cuda")
    
        # Feature extraction 및 adaptation
        with torch.no_grad():
            image_features = extractor(image_tensor)[0]
            adapted_features = image_adapter(image_features)
            adapted_features_list.append(adapted_features)
    
    # 든 이미지의 adapted_features를 스택
    all_adapted_features = torch.stack(adapted_features_list, dim=0)
    all_adapted_features = all_adapted_features.squeeze(1)
    
    print(f"text_embeddings shape: {text_embedding.shape}")
    print(f"all_adapted_features shape: {all_adapted_features.shape}")
    
    attention_embeddings = torch.cat((text_embedding, all_adapted_features), dim=0)

    print(f"attention_embeddings shape: {attention_embeddings.shape}")

    # Self-attention 계산
    attention_weights = torch.matmul(attention_embeddings, attention_embeddings.transpose(0, 1))
    attention_weights = attention_weights / torch.sqrt(torch.tensor(512.0))  # 스케일링
    
    
    # 수치적 안정성을 위해 매우 작은 값들을 0으로 처리
    attention_weights = torch.where(attention_weights < 1e-10, torch.zeros_like(attention_weights), attention_weights)
    
    # Attention map 시각화
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(15, 12))  # 레이블이 더 길어질 수 있으므로 크기 증가
    attention_map = attention_weights.cpu().numpy()
    
    # 텍스트 토큰 레이블 생성 (프롬프트를 단어로 분리)
    text_labels = prompt.split()
    
    # 이미지 파일 레이블 생성 (파일명에서 확장자 제거)
    image_labels = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    
    # 모든 레이블 결합
    labels = text_labels + image_labels
    
    # 히트맵 생성
    sns.heatmap(attention_map, 
                xticklabels=labels,
                yticklabels=labels,
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'},
                annot=True,  # 숫자 표시 활성화
                fmt='.2f',   # 소수점 2자리까지 표시
                annot_kws={'size': 8})  # 숫자의 폰트 크기 설정
    
    plt.title('Self-Attention Map')
    plt.xlabel('Token Position')
    plt.ylabel('Token Position')
    
    # x축 레이블 회전하여 가독성 향상
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 레이블이 잘리지 않도록 여백 조정
    plt.tight_layout()
    
    # 결과 저장
    attention_map_path = output_path.replace('.png', '_attention_map.png')
    plt.savefig(attention_map_path, bbox_inches='tight')
    plt.close()
    
    print(f"Attention map saved to: {attention_map_path}")
    
# 사용 예시
if __name__ == "__main__":
    prompt = "A man with black hair and brown leather jacket holding a knife"
    generated_image = generate_image_from_text(prompt, "attention_map.png")
