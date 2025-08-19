export HF_DATASETS_ONLINE=1
export HF_HUB_ONLINE=1
accelerate launch train_controlnext.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
--pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
--variant fp16 \
--use_safetensors \
--output_dir "train/examples/sticker_directory_test_canny_pretrained_vae_vidit_1" \
--logging_dir "logs" \
--train_batch_size 2 \
--dataloader_num_workers 4 \
--resolution 1024 \
--gradient_checkpointing \
--set_grads_to_none \
--proportion_empty_prompts 0.2 \
--mixed_precision fp16 \
--controlnet_scale_factor 1.0 \
--enable_xformers_memory_efficient_attention \
--dataset_name "Nahrawy/VIDIT-Depth-ControlNet" \
--image_column "image" \
--conditioning_image_column "depth_map" \
--caption_column "caption" \
--validation_prompt "a stone tower on a rocky island" \
--validation_image "examples/vidit_depth/condition_0.png"