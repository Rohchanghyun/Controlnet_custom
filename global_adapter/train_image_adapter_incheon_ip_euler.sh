CUDA_VISIBLE_DEVICES=0 python global_feature_train_ip_euler.py \
--mode train \
--data_path /workspace/data/changhyun/dataset/danbooru_safe/ \
--extractor_weight /workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt \
--output_dir /workspace/data/changhyun/output/sd_image_adapter_cfg_8_512_ip_euler_zero_adapter \
--clip_embeddings_dim 8 \
--batchid 1 \
--batchimage 2 \
--epoch 2 \
--lr 2e-4 