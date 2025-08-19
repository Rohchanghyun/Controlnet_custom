
# GPU 설정 (사용할 GPU 번호)
export CUDA_VISIBLE_DEVICES=2
python run_validation.py \
  --extractor_weight "/workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt" \
  --adapter_weight "/workspace/data/changhyun/output/sdxl_image_adapter_cfg_ip_euler_zero_adapter_drop_SDXL_stabilityai" \
  --image_path "/workspace/data/changhyun/projects/emoji_generation/Controlnet_custom/total_model/examples/emo_sketch/original.png" \
  --caption "rabbit-like character" \
  --output_dir "./validation/no_clip/filter_4/768_1024_50_3_7.5_rabbit_character" \
  --clip_embeddings_dim 8 \
  --num_samples 10 \
  --guidance_scale_1 3 \
  --guidance_scale_2 7.5 \
  --model_id "cagliostrolab/animagine-xl-3.0" \
  --vae_model_path "madebyollin/sdxl-vae-fp16-fix" \
 