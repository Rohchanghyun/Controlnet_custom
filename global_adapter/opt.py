import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="../../../dataset/emoji_data",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis', 'tsne'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='/workspace/dataset/top15character/Elric Edward/Image_2.jpg',
                    help='path to the image you want to query')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--extractor_weight',
                    default="../../../output/global_adapter/256/extractor/model_2000.pt",
                    help='load weights ')

parser.add_argument('--image_adapter_weight',
                    default="None",
                    help='load weights ')

parser.add_argument('--epoch',
                    type=int,
                    default=2000,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    type=float,
                    default=2e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    type=list,
                    default=[320, 380],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument('--clip_embeddings_dim',
                    type=int,
                    default=4,
                    help='clip embeddings dim')

parser.add_argument("--batchid",
                    type=int,
                    default=8,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    type=int,
                    default=8,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    type=int,
                    default=8,
                    help='the batch size for test')

parser.add_argument("--optimizer",
                    default='adamw',
                    help='optimizer')

parser.add_argument("--output_dir",
                    default='../../../output/256/',
                    help='output_dir')

opt = parser.parse_args()
