CUDA_VISIBLE_DEVICES=1 python global_feature_train_sd_euler.py \
--mode train \
--data_path /workspace/data/changhyun/dataset/danbooru_safe/ \
--extractor_weight /workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt \
--output_dir /workspace/data/changhyun/output/sd_image_adapter_cfg_4_768_euler_zero_adapter \
--clip_embeddings_dim 4 \
--batchid 1 \
--batchimage 2 \
--epoch 2 \
--lr 2e-4 