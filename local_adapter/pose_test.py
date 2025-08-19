from PIL import Image
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers import EulerAncestralDiscreteScheduler, PNDMScheduler

def generate_image_from_condition(input_image_path, prompt, negative_prompt=""):
    # ControlNet 모델 로드
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float16
    )

    # Stable Diffusion 파이프라인 설정
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    
    # GPU 사용 설정
    pipe.to("cuda")
    
    # Euler Ancestral sampler 설정
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    
    
    # 입력 이미지 로드 및 전처리 (768x768로 변경)
    input_image = Image.open(input_image_path)
    input_image = input_image.resize((768, 768))
    
    # 이미지 생성
    output_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        num_inference_steps=40,
        guidance_scale=7.5,
        height=768,
        width=768
    ).images[0]
    
    return output_image

# 사용 예시
if __name__ == "__main__":
    input_image_path = "./test.png"
    prompt = "a man at the beach"
    negative_prompt = "low quality, blurry, distorted, bad anatomy, ugly, poor details"
    
    generated_image = generate_image_from_condition(input_image_path, prompt, negative_prompt)
    generated_image.save("generated_output.png")
