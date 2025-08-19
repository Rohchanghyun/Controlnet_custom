from transformers import CLIPProcessor, CLIPModel

# Hugging Face에서 사전 학습된 ControlNet 모델을 불러옵니다.
model_name = "lllyasviel/sd-controlnet-canny"  # 원하는 모델 이름으로 변경 가능
clip_model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# 예제 입력
image = ...  # 이미지 데이터
text = "예제 텍스트"

# 입력 데이터 처리
inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

# 모델 실행
outputs = clip_model(**inputs)
