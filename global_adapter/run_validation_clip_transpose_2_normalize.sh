
# GPU 설정 (사용할 GPU 번호)
export CUDA_VISIBLE_DEVICES=1
python run_validation_clip_transpose_2_normalize.py \
  --extractor_weight "/workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt" \
  --adapter_weight "/workspace/data/changhyun/output/sdxl_image_adapter_cfg_ip_euler_zero_adapter_drop_SDXL_image_clip_attention_8_filter_1024attention_transpose/image_adapter_epoch1.pth" \
  --image_path "/workspace/data/changhyun/projects/emoji_generation/Controlnet_custom/global_adapter/test_classified_animation_U+1F489_CAACAgUAAxUAAWSdjxggx1pfuRceq2D5_tA3P2qnAAKDAAOF8sQ41Vj-gvgl5p8vBA_2x.png" \
  --caption "animation, bear-like character, happy, waving hand, front view close-up, sticker style, white background" \
  --output_dir "./validation/clip/filter_4/neta_768_1024_50_3_7.5_attention_norm_otherimage" \
  --clip_embeddings_dim 8 \
  --num_samples 100 \
  --guidance_scale_1 3 \
  --guidance_scale_2 7.5 \
  --model_id "neta-art/neta-xl-2.0" \
  --vae_model_path "madebyollin/sdxl-vae-fp16-fix" \
 