CUDA_VISIBLE_DEVICES=0 python run_controlnext.py \
 --pretrained_model_name_or_path="admruul/anything-v3.0" \
 --output_dir="examples/oddoong" \
 --validation_image "examples/oddoong/condition_0.png" \
 --validation_prompt "oddoong, white_penguin, simple_design, large_eyes, yellow_beak, holding_money, offering_coin, cartoon_style, round_body, minimalistic_features" \
 --negative_prompt "PBH" "PBH"\
 --controlnet_model_name_or_path pretrained/deepfashion_multiview/controlnet.safetensors \
 #--lora_path lora/yuanshen/genshin_124.safetensors \
 #(Optional, less generality, stricter control)--unet_model_name_or_path pretrained/deepfashion_multiview/unet.safetensors