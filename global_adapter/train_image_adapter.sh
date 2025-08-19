CUDA_VISIBLE_DEVICES=1 python global_feature_train_sd.py \
--mode train \
--data_path ../dataset/top15character \
--extractor_weight ./best/model_2000.pt \
--output_dir ./result/sd_image_adapter_20_768 \
--batchid 8 \
--batchimage 2 \
--epoch 3000 \
--lr 2e-4 