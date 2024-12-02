CUDA_VISIBLE_DEVICES=3 python global_feature_train_sd_ddpm.py \
--mode train \
--data_path /workspace/data/changhyun/dataset/danbooru_safe/ \
--extractor_weight /workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt \
--output_dir /workspace/data/changhyun/output/sd_image_adapter_cfg_2_768 \
--clip_embeddings_dim 2 \
--batchid 1 \
--batchimage 2 \
--epoch 2 \
--lr 2e-4 