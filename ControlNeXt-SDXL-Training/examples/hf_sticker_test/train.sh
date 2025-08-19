export HF_DATASETS_ONLINE=1
export HF_HUB_ONLINE=1
accelerate launch train_controlnext_no_training.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
--pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
--use_safetensors \
--output_dir "train/examples/hf_sticker_test" \
--logging_dir "logs" \
--train_batch_size 6 \
--dataloader_num_workers 4 \
--resolution 1024 \
--num_train_epochs 400 \
--report_to wandb \
--checkpointing_steps 100000 \
--validation_steps 1000 \
--gradient_checkpointing \
--set_grads_to_none \
--proportion_empty_prompts 0.2 \
--mixed_precision fp16 \
--controlnet_scale_factor 1.0 \
--enable_xformers_memory_efficient_attention \
--dataset_name "sambob/fscoco_sketch" \
--image_column "image" \
--conditioning_image_column "guide" \
--caption_column "text" \
--validation_prompt "A giraffe is standing in the middle of a field" \
--validation_image "examples/hf_sticker_test/condition_0.png"