CUDA_VISIBLE_DEVICES=3 python main_controlSDXL.py \
--mode train \
--data_path /workspace/data/changhyun/dataset/pose_attention/ \
--extractor_weight /workspace/data/changhyun/output/global_adapter/256/extractor/model_2000.pt \
--output_dir /workspace/data/changhyun/output/pose_sketch_attention \
--clip_embeddings_dim 4 \
--batchid 1 \
--batchimage 4 \
--epoch 10 \
--lr 2e-4 