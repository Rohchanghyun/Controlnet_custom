export HF_DATASETS_ONLINE=1
export HF_HUB_ONLINE=1
accelerate launch train_controlnext.py --pretrained_model_name_or_path "neta-art/neta-xl-2.0" \
--pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
--use_safetensors \
--output_dir "train/examples/photo_sketch_high" \
--logging_dir "logs" \
--train_batch_size 6 \
--dataloader_num_workers 4 \
--resolution 1024 \
--num_train_epochs 400 \
--report_to wandb \
--tracker_run_name "photo_sketch_high_quality" \
--checkpointing_steps 100000 \
--validation_steps 1000 \
--validation_guidance_scale 10.0 \
--gradient_checkpointing \
--set_grads_to_none \
--proportion_empty_prompts 0.2 \
--mixed_precision fp16 \
--controlnet_scale_factor 1.0 \
--enable_xformers_memory_efficient_attention \
--train_data_dir "/workspace/data/changhyun/dataset/emo_sketch" \
--image_column "image" \
--conditioning_image_column "conditioning_image" \
--caption_column "text" \
--validation_prompt "bear-like character, laughing, sticker style , white background " \
--validation_image "examples/photo_sketch/condition_0.png"