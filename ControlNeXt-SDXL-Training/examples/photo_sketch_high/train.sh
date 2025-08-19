export HF_DATASETS_ONLINE=1
export HF_HUB_ONLINE=1
accelerate launch train_controlnext.py --pretrained_model_name_or_path "neta-art/neta-xl-2.0" \
--pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
--use_safetensors \
--output_dir "train/examples/photo_sketch_condition_test_tag_2" \
--logging_dir "logs" \
--train_batch_size 4 \
--dataloader_num_workers 4 \
--resolution 1024 \
--num_train_epochs 400 \
--report_to wandb \
--tracker_run_name "photo_sketch_condition_test_tag_2" \
--checkpointing_steps 100000 \
--validation_steps 100 \
--gradient_checkpointing \
--gradient_accumulation_steps 4 \
--set_grads_to_none \
--proportion_empty_prompts 0.1 \
--mixed_precision fp16 \
--controlnet_scale_factor 1.0 \
--enable_xformers_memory_efficient_attention \
--train_data_dir "/workspace/data/changhyun/dataset/emo_sketch" \
--image_column "image" \
--conditioning_image_column "conditioning_image" \
--caption_column "text" \
--validation_guidance_scale 7.5 \
--validation_prompt "bear-like character, laughing, blue colored character,sticker style , white background, high quality " \
--validation_image "examples/photo_sketch_high/condition_0.png"