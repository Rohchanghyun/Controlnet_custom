# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 1234 train_controlnext.py \
#  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#  --output_dir="checkpoints" \
#  --dataset_name=fusing/fill50k \
#  --resolution=512 \
#  --learning_rate=1e-5 \
#  --validation_image "examples/conditioning_image_1.png" "examples/conditioning_image_2.png" \
#  --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
#  --checkpoints_total_limit 3 \
#  --checkpointing_steps 400 \
#  --validation_steps 400 \
#  --num_train_epochs 4 \
#  --train_batch_size=6 \
#  --controlnext_scale 0.35 \
#  --save_load_weights_increaments 


CUDA_VISIBLE_DEVICES=1 accelerate launch train_controlnext.py \
--pretrained_model_name_or_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
--pretrained_vae_model_name_or_path "admruul/anything-v3.0" \
--use_safetensors \
--output_dir "/workspace/data/changhyun/output/ControlNeXt" \
--num_train_epochs 5000 \
--checkpointing_steps 500 \
--checkpoints_total_limit 10 \
--logging_dir "logs" \
--resolution 512 \
--gradient_checkpointing \
--set_grads_to_none \
--proportion_empty_prompts 0.2 \
--controlnet_scale_factor 1.0 \
--mixed_precision fp16 \
--enable_xformers_memory_efficient_attention \
--data_path "/workspace/data/changhyun/dataset/emoji_data" \
--image_column "image_path" \
--conditioning_image_column "conditioning_image" \
--caption_column "caption" \
--validation_prompt "A man with red hair and a black coat" \
--validation_image "/workspace/data/changhyun/dataset/emoji_data/train/Shanks/1.png" \
#--resume_from_checkpoint "train/example/pokemon_canny/chekpoints/checkpoint-10000" \