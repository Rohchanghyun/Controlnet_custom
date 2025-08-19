import torch
from diffusers import StableDiffusionXLPipeline
import sys
import os

# SDXL 모델과 pretrained weight 불러오기 (HuggingFace)
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
model = pipe.unet

output_path = os.path.join(os.path.dirname(__file__), "architecture_output.txt")

with open(output_path, "w", encoding="utf-8") as f:
    # 모델 architecture 출력
    print(model, file=f)

    # Attention layer 찾기 및 weight 값 출력 (예시: 첫 번째 attention layer)
    for name, module in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower():
            print(f"\n{name}: {module}", file=f)
            # weight가 있는 경우 출력
            if hasattr(module, "to_q"):
                print("to_q.weight:", module.to_q.weight.data, file=f)
            if hasattr(module, "to_k"):
                print("to_k.weight:", module.to_k.weight.data, file=f)
            if hasattr(module, "to_v"):
                print("to_v.weight:", module.to_v.weight.data, file=f)
            break  # 첫 번째 attention만 예시로 출력

    # 전체 attention layer를 모두 보고 싶으면 break 제거