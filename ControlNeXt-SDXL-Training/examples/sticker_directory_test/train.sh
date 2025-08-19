export HF_DATASETS_ONLINE=1
export HF_HUB_ONLINE=1
accelerate launch train_controlnext.py --pretrained_model_name_or_path "neta-art/neta-xl-2.0" \
--pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
--use_safetensors \
--output_dir "train/examples/sticker_directory_test_canny_condition_512" \
--logging_dir "logs" \
--train_batch_size 4 \
--dataloader_num_workers 4 \
--resolution 512 \
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
--validation_prompt "A image of a bear-like character laughing in sticker style" \
--validation_image "examples/sticker_directory_test/condition_4.png"