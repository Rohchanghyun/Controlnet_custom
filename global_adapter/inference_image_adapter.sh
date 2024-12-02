CUDA_VISIBLE_DEVICES=0 python inference.py \
--mode evaluate \
--data_path /workspace/data/changhyun/dataset/emoji_data \
--extractor_weight /workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt \
--image_adapter_weight /workspace/data/changhyun/output/sd_image_adapter_cfg_77/image_adapter/image_adapter_7600.pt \
--output_dir /workspace/data/changhyun/output/inference/512_euler_input_256_null_token_uncond \
--batchid 8 \
--batchimage 2 \
--epoch 10000 \
--lr 2e-4 