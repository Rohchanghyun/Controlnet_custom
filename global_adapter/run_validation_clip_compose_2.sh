
# GPU 설정 (사용할 GPU 번호)
export CUDA_VISIBLE_DEVICES=1
python run_validation_clip_compose.py \
  --extractor_weight "/workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt" \
  --adapter_weight "/workspace/data/changhyun/output/sdxl_MLP_4_compose/validation_results/step_9000/adapter_step_9000.pth" \
  --image_path "/workspace/data/changhyun/projects/emoji_generation/Controlnet_custom/total_model/examples/emo_sketch/original.png" \
  --caption "animation, bear-like character, happy, waving hand, front view close-up, sticker style, white background" \
  --output_dir "./validation/compose/filter_4/768_1024_50_3_7.5_beaer" \
  --clip_embeddings_dim 4 \
  --num_samples 10 \
  --guidance_scale_1 3 \
  --guidance_scale_2 7.5 \
  --model_id "cagliostrolab/animagine-xl-3.0" \
  --vae_model_path "madebyollin/sdxl-vae-fp16-fix" \
 