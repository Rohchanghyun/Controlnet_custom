CUDA_VISIBLE_DEVICES=1 python global_feature_train_77_768.py \
--mode train \
--data_path ../dataset/top15character \
--extractor_weight ./best/model_2000.pt \
--image_adapter_weight ./result/image_adapter_77_768/image_adapter/image_adapter_200_best.pt \
--output_dir ./result/projector_77_768 \
--batchid 8 \
--batchimage 2 \
--epoch 10000 \
--lr 2e-4 